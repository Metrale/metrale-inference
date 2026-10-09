// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextDsaLayer`, one GLM-5.3 DSA block: the projections, the indexer
//! cache write, token selection and the NoPE MLA gather-attend over the selected tokens.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - `decode_k` fails before its first launch when the indexer cache is behind the sequence
//!   (`len < seq_len`) or `k` is 0 or above the workspace's `max_rows`. An indexer cache
//!   ahead of the sequence is rewound to `seq_len` first.
//! - `indexer_forward` checks that the indexer cache has room (`ensure_room`) before it
//!   writes.
//!
//! Decode, end to end:
//!
//! ```text
//! hidden ─┬─ q_a_proj ─ RMSNorm ─┬─ q_absorb ────────────── Q (latent space)
//!         │                      └─ indexer.wq_b ────────── q_idx  ─┐
//!         ├─ indexer.wk ─ LayerNorm(w,b) ─ state.k_normed ──────────┤
//!         ├─ compress_gate ─────────────── state.gate ──────────────┼─ select_tokens
//!         ├─ weights_proj ──────────────── head weights ────────────┘        │
//!         └─ kv_a_proj ─ RMSNorm ─ FP8 ─── paged latent cache                │
//!                                                                            ▼
//!                                            glm5next_dsa_mla_decode_fp8 (gather)
//! ```
//!
//! # Kernel choices the shapes do not check
//!
//! * The norms use `rms_norm_vanilla`, `x * rms * w`. The `rms_norm` kernel (module `norm`)
//!   computes `x * rms * (1 + w)` with the same signature.
//! * `indexer.k_norm` is a LayerNorm with a bias: `nllb_layernorm_bf16(x, w, b, …)`.
//! * `weights_proj` carries `index_heads^-0.5` from load ([`Glm5NextDsaWeights`]), because
//!   `dsa_index_scores` does not apply it.
//! * Q reaches the decode kernel as `q_absorb`: `q_b_proj` multiplied through `kv_b_proj`'s
//!   K half, so it is in the `kv_lora_rank`-wide latent space the kernel dots against. The
//!   raw `q_b_proj` is `qk_head_dim` (256) per head.

use anyhow::{Context, Result};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::attend::{DsaDecodeInputs, DsaDecodePaging, Glm5NextDsaDecodeKernel, decode_attention};
use super::paged::{IndexerCache, IndexerRowsDev};
use super::select::{DsaSelectInputs, select_tokens};
use super::state::Glm5NextDsaState;
use super::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_layers::layer::{ForwardContext, LayerState, TransformerLayer};

mod decode_k;
mod decode_rows;
mod host_rows;
mod indexer_place;
mod kernels;
mod prefill_rows;
mod proj_gemm;
mod row_ops;
mod rows;
mod select_rows;
mod workspace;

pub use decode_rows::DsaRowSpan;
pub use indexer_place::IndexerPlace;
pub use kernels::Glm5NextDsaLayerKernels;
use proj_gemm::gemm;
pub use workspace::Glm5NextDsaWorkspace;
pub(crate) use workspace::batch_select_enabled;

/// 2026-09-25: One DSA block's weights on the device, already sharded for this rank.
pub struct Glm5NextDsaWeights {
    pub q_a_proj: DevicePtr,
    pub q_a_layernorm: DevicePtr,
    /// 2026-09-25: `[local_heads * kv_lora_rank, q_lora_rank]` BF16: `q_b_proj` absorbed
    /// through `kv_b_proj`'s K half, so Q arrives in latent space.
    pub q_absorb: DevicePtr,
    pub kv_a_proj: DevicePtr,
    pub kv_a_layernorm: DevicePtr,
    /// 2026-09-25: `[hidden, local_heads * kv_lora_rank]` BF16, row-parallel; the caller
    /// all-reduces the output.
    ///
    /// `o_proj` with `kv_b_proj`'s V half folded in (`build::absorb_o`), because the decode
    /// kernel's output is in latent space, `kv_lora_rank` per head. The checkpoint `o_proj`
    /// is `v_head_dim` per head (256 against 512 on GLM-5.3).
    pub o_absorb: DevicePtr,
    // 2026-09-25: The indexer weights below are replicated on every rank (`tp.rs`).
    pub wk: DevicePtr,
    pub k_norm_weight: DevicePtr,
    pub k_norm_bias: DevicePtr,
    pub compress_gate: DevicePtr,
    pub wq_b: DevicePtr,
    /// 2026-09-25: Multiplied by `index_heads^-0.5` at load; `dsa_index_scores` does not
    /// apply it.
    pub weights_proj: DevicePtr,
    /// 2026-09-25: `[index_kpool, index_head_dim]` FP32; the checkpoint stores BF16.
    pub ape: DevicePtr,
}

pub struct Glm5NextDsaLayer {
    pub cfg: Glm5NextDsaConfig,
    pub weights: Glm5NextDsaWeights,
    pub kernels: Glm5NextDsaLayerKernels,
    pub select_kernels: Glm5NextDsaKernels,
    pub decode_kernel: Glm5NextDsaDecodeKernel,
    pub workspace: Glm5NextDsaWorkspace,
    /// 2026-09-25: Index in the model's layer stack; used only in error messages.
    pub layer_idx: usize,
    /// 2026-09-25: Index into the KV pool (`kv_cache.k_pool_ptr`): the loader counts DSA
    /// layers only, so it runs 0..11 on GLM-5.3 while `layer_idx` runs to 45. The MTP drafter
    /// uses 0.
    pub attn_layer_idx: usize,
    pub rms_eps: f32,
    /// 2026-09-25: FP8 latent-cache scale. The decode reads with it and the latent write
    /// takes `1/scale`.
    pub kv_scale: f32,
    /// 2026-09-25: Use the workspace's `bt`/`sl` rather than allocating them per call. The
    /// loaders set it true unless `METRALE_GLM_DSA_ALLOC_PER_STEP=1`.
    pub persist_bt: bool,
    /// 2026-10-09: Where `alloc_state` puts a sequence's indexer rows. The text stack's
    /// loader sets `paged::text_stack_indexer_cache()`; the MTP drafter's layer is flat.
    pub indexer_cache: IndexerCache,
}

impl Glm5NextDsaLayer {
    /// 2026-10-09: Every BF16-output projection this layer launches through `proj_gemm::gemm`,
    /// as `(weight, n, k, name)` with the shapes the forward passes, for
    /// `--dense-quantization fp8`. The FP32-output `wq_b` and `weights_proj` are not listed.
    pub fn dense_projections(&self) -> Vec<(DevicePtr, usize, usize, &'static str)> {
        let c = &self.cfg;
        let w = &self.weights;
        let latent = c.local_heads * c.kv_lora_rank;
        vec![
            (w.q_a_proj, c.q_lora_rank, c.hidden, "dsa.q_a_proj"),
            (w.q_absorb, latent, c.q_lora_rank, "dsa.q_absorb"),
            (w.kv_a_proj, c.kv_lora_rank, c.hidden, "dsa.kv_a_proj"),
            (w.o_absorb, c.hidden, latent, "dsa.o_absorb"),
            (w.wk, c.index_head_dim, c.hidden, "dsa.indexer.wk"),
            (
                w.compress_gate,
                c.index_head_dim,
                c.hidden,
                "dsa.indexer.compress_gate",
            ),
        ]
    }

    /// 2026-09-25: Project `hidden` into indexer cache row `state.len()`, then advance by one.
    ///
    /// With [`IndexerPlace::Device`], `k_normed` and `gate` go to the workspace staging rows
    /// and `dsa_indexer_store` places them at the device-side position; with
    /// [`IndexerPlace::Host`] they are written straight into the cache row. Also leaves this
    /// row's selector head weights in the workspace. Fails before any write when the cache is
    /// full.
    pub fn indexer_forward(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        place: IndexerPlace<'_>,
        stream: u64,
    ) -> Result<()> {
        self.indexer_forward_with(gpu, hidden, state, kv_cache, place, true, stream)
    }

    /// 2026-10-09: [`Self::indexer_forward`], with the selector head weights computed only
    /// when `head_weights` is set; `decode_rows` computes them for all its rows at once.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn indexer_forward_with(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        state: &mut Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        place: IndexerPlace<'_>,
        head_weights: bool,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Checked before any write: everything below writes row `state.len()`,
        // which is past the end of the buffers when the cache is full.
        state.ensure_room(1)?;
        let d = self.cfg.index_head_dim;
        let pos = state.len();
        let w = &self.workspace;
        let (k_dst, gate_dst) = match place {
            IndexerPlace::Device { .. } => (w.stage_k, w.stage_gate),
            IndexerPlace::Host { block_table } => {
                self.host_row_ptrs(state, kv_cache, block_table)?
            }
        };

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.wk,
            k_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;
        KernelLaunch::new(gpu, self.select_kernels.k_norm)
            .grid([1, 1, 1])
            .block([d.min(1024) as u32, 1, 1])
            .shared_mem((d.min(1024) * 4) as u32)
            .arg_ptr(k_dst)
            .arg_ptr(self.weights.k_norm_weight)
            .arg_ptr(self.weights.k_norm_bias)
            .arg_u32(1)
            .arg_u32(d as u32)
            .arg_f32(self.rms_eps)
            .launch(stream)?;

        gemm(
            gpu,
            self.kernels.gemm,
            self.kernels.gemv,
            self.kernels.gemv_batchm,
            hidden,
            self.weights.compress_gate,
            gate_dst,
            1,
            d,
            self.cfg.hidden,
            stream,
        )?;

        // 2026-09-25: Selector head weights from the layer input, FP32 out. `weights_proj` is
        // `[index_heads, hidden]`; the reference computes
        // `weights_proj(hidden) * index_heads**-0.5`
        // (`glm5next_dsa_ref/gen_dsa_indexer_golden.py`), and the scale is already in the
        // weight (`build.rs` transform 2).
        if head_weights {
            gemm(
                gpu,
                self.kernels.gemm_f32,
                self.kernels.gemv_f32,
                // 2026-09-25: M = 1; the batched rows use `gemv_batchm_f32` (`decode_rows`).
                KernelHandle(0),
                hidden,
                self.weights.weights_proj,
                self.workspace.head_weights,
                1,
                self.cfg.index_heads,
                self.cfg.hidden,
                stream,
            )?;
        }

        self.store_indexer_row(gpu, state, kv_cache, place, pos, d, 0, stream)
    }

    /// 2026-09-25: The selector query projection and the selection for one query row, written
    /// to row `row` of the selection output.
    ///
    /// `decode_k` calls it right after that row's indexer write, so the geometry is planned
    /// at the row's own cache length; `attend_rows` then attends all rows in one launch.
    #[allow(clippy::too_many_arguments)]
    fn select_row(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        state: &Glm5NextDsaState,
        rows: IndexerRowsDev,
        q_pos_dev: DevicePtr,
        replay_safe: bool,
        // 2026-10-09: `(q_idx, head_weights)` this row's selector query and head weights
        // when the caller computed them already (`decode_rows`, several rows in one launch);
        // `None` computes the query here and reads the head weights `indexer_forward` left.
        pre: Option<(DevicePtr, DevicePtr)>,
        stream: u64,
    ) -> Result<()> {
        let w = &self.workspace;
        let geom = state.geometry(&self.cfg, 1)?;

        let (q, weights) = match pre {
            Some(p) => p,
            None => {
                // 2026-09-25: The selector query `q_idx`, FP32 out.
                gemm(
                    gpu,
                    self.kernels.gemm_f32,
                    self.kernels.gemv_f32,
                    // 2026-09-25: M = 1; see `pre` for the batched rows.
                    KernelHandle(0),
                    w.q_resid.offset(row * self.cfg.q_lora_rank * 2),
                    self.weights.wq_b,
                    w.q_idx,
                    1,
                    self.cfg.index_heads * self.cfg.index_head_dim,
                    self.cfg.q_lora_rank,
                    stream,
                )?;
                // 2026-09-25: `head_weights` comes from `indexer_forward`, which has the
                // layer input.
                (w.q_idx, w.head_weights)
            }
        };

        let inputs = DsaSelectInputs {
            rows,
            ape: self.weights.ape,
            q,
            weights,
            q_pos: q_pos_dev,
            // 2026-09-25: All 1s, set once in `Glm5NextDsaWorkspace::new`.
            q_mask: w.q_mask,
            first_key: 0,
            geom_dev: if replay_safe {
                w.geom_dev
            } else {
                DevicePtr::NULL
            },
        };
        // 2026-09-25: On the replay-safe path the grid and the shared-memory request are set
        // at the context ceiling and the live extents come from `geom_dev`, so one graph
        // serves every context length.
        let launch = if replay_safe {
            super::select::DsaSelectLaunch::Ceiling {
                max_pools: super::select::contiguous_pool_count(
                    self.cfg.index_kpool,
                    super::state::max_dsa_context(&self.cfg),
                ),
            }
        } else {
            super::select::DsaSelectLaunch::Exact
        };
        let t = crate::glm5next_layer::profile::start();
        // 2026-09-25: Row `row` of the `[max_rows, out_width]` selection output. The kernels
        // run one query row (`q_rows == 1`) into this row's slot, so `attend_rows` reads all
        // rows in one launch.
        select_tokens(
            gpu,
            &self.select_kernels,
            &self.cfg,
            &geom,
            &inputs,
            &w.select.row(row, &self.cfg),
            launch,
            stream,
        )?;
        use crate::glm5next_layer::profile;
        profile::end(profile::DSA_SELECT, t, gpu, stream);
        Ok(())
    }

    /// 2026-09-25: The gather-attend for all `rows` query rows in one launch.
    ///
    /// Grid y is the row. Each row reads the paged latent cache through its own `q_abs` row,
    /// selection row and `seq_lens` entry, and the block-table row at stride
    /// `max_blocks_per_seq` (0 when the rows share one table).
    #[allow(clippy::too_many_arguments)]
    fn attend_rows(
        &self,
        gpu: &dyn GpuBackend,
        rows: usize,
        state: &Glm5NextDsaState,
        kv_cache: &PagedKvCache,
        block_table_dev: DevicePtr,
        seq_lens_dev: DevicePtr,
        paging: &DsaDecodePaging,
        stream: u64,
    ) -> Result<()> {
        use crate::glm5next_layer::profile;
        let w = &self.workspace;
        // 2026-09-25: `decode_attention` reads only `out_width` and `q_rows` from it.
        let geom = state.geometry(&self.cfg, rows)?;
        let paging = DsaDecodePaging {
            num_seqs: rows,
            ..*paging
        };
        let t = profile::start();
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        decode_attention(
            gpu,
            self.decode_kernel,
            &self.cfg,
            &geom,
            &paging,
            &DsaDecodeInputs {
                q: w.q_abs,
                k_cache: pool,
                v_cache: pool, // 2026-09-25: absorbed NoPE MLA: K and V are the same latent
                out: w.attn_out,
                block_tables: block_table_dev,
                seq_lens: seq_lens_dev,
                sel_indices: w.select.tokens(),
                k_scale: self.kv_scale,
                v_scale: self.kv_scale,
                split_ws: w.split_ws,
            },
            stream,
        )?;
        profile::end(profile::DSA_ATTEND, t, gpu, stream);
        Ok(())
    }
}

impl TransformerLayer for Glm5NextDsaLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(self.alloc_dsa_state(gpu)?))
    }

    /// 2026-09-25: Frees a `Glm5NextDsaState`; any other state type is left alone. The
    /// composite `Glm5NextLayer` has its own override that does the same.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(dsa) = state.as_any_mut().downcast_mut::<Glm5NextDsaState>() {
            dsa.free(gpu)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_k(
            hidden,
            1,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            // 2026-09-25: `is_prefill`: a single-token decode is not a prefill sub-chunk.
            false,
        )
    }
}

impl metrale_model_layers::layer::LayerCapabilities for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerWeightSetup for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerWriteOnAccept for Glm5NextDsaLayer {}

impl metrale_model_layers::layer::LayerGraphHooks for Glm5NextDsaLayer {
    /// 2026-09-25: Fails when a replay writing indexer rows up to `seq_len + k` would pass the
    /// cache capacity; the error carries the context "DSA replay pre-check".
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .ensure_room_through(seq_len + k)
            .with_context(|| {
                format!("DSA replay pre-check (before launch_graph, seq_len {seq_len} + k {k})")
            })
    }

    /// 2026-09-25: A replayed graph writes the indexer rows (positions from device memory)
    /// without running `decode`, so this moves the host-side row counter to where `decode_k`
    /// would have left it (`Glm5NextDsaState::sync_to`).
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        state
            .as_any_mut()
            .downcast_mut::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!("Glm5NextDsaLayer got a state that is not Glm5NextDsaState")
            })?
            .sync_to(seq_len, k)
    }
}

impl metrale_model_layers::layer::LayerAuxState for Glm5NextDsaLayer {}
impl metrale_model_layers::layer::LayerSplitPrefill for Glm5NextDsaLayer {}
impl metrale_model_layers::circuit_exec::CircuitBindings for Glm5NextDsaLayer {}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod bt_trim_tests;

// 2026-10-09: The mock rig the `decode_rows` and host-row staging tests share.
#[cfg(test)]
mod decode_rows_fixture;
