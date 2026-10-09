// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextLayer`, the GLM-5.3 decoder layer behind `TransformerLayer`: a KDA or
//! DSA mixer, a dense or routed-MoE MLP, and the mHC hyper-connection.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every norm launches `rms_norm_vanilla` (`x * w / rms`), never `rms_norm`, which scales by
//!   `1 + w`.
//! - Highway slots are per token: `decode` uses slot 0, `prefill` gives prompt token `t` slot
//!   `ctx.hc_row_offset + t`, `decode_batched` gives verify row `t` slot `t`, and
//!   (2026-10-08) `decode_multi_seq` gives sequence row `r` slot `ctx.hc_row_offset + r`.
//! - The last text layer collapses the highway with `hc_head_mean`, which takes no weights.
//! - Any all-reduce of a mixer or MLP output happens before `hc_post` folds that output into
//!   the highway.
//!
//! One mHC site, as `forward_one` and `forward_k` run it:
//!
//! ```text
//! layer 0 only:  hc_expand(hidden) -> streams            [hc_mult, hidden] FP32
//! each site:     hc_pre(streams) -> y (into hidden), post, comb
//!                rms_norm_vanilla(y, site norm) -> normed
//!                sublayer(normed) -> out
//!                hc_post(out, residual = streams, post, comb) -> streams
//! last layer:    hc_head_mean(streams) -> hidden
//! ```
//!
//! `hc_pre` only reads `streams`, so `hc_post` takes the same buffer as its residual and writes
//! its result over it.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_dsa::layer::Glm5NextDsaLayer;
use crate::glm5next_dsa::state::Glm5NextDsaState;
use crate::glm5next_kda::{Glm5NextKdaConfig, Glm5NextKdaLayer, Glm5NextKdaWorkspace, KdaSeqState};
use crate::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_dense, forward_moe};
use crate::glm5next_mlp::weights::{Glm5NextDenseMlpWeights, Glm5NextMoeWeights};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_model_layers::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use metrale_model_layers::layer::{
    LayerAuxState, LayerCapabilities, LayerGraphHooks, LayerSplitPrefill, LayerWeightSetup,
    LayerWriteOnAccept,
};
// 2026-09-25: GLM's own mHC launchers, not DeepSeek-V4's `ops::hc_pre`/`ops::hc_post`.
use crate::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_expand, glm_hc_post, glm_hc_pre,
    hc_head_mean,
};

pub mod state;

pub mod profile;
pub use state::alloc_kda_ssm_state;

mod levers;
mod steps;
mod types;
pub(crate) mod wide_gemv;
pub use levers::prefill_rows;
pub(crate) use levers::{
    PREFILL_ROWS, cublas_wide_proj, dsa_batch_qidx, multi_seq_chunk_rows, multi_seq_chunks,
};
pub use steps::{GroupSpan, group_spans};
pub use types::{Glm5NextLayer, Glm5NextMhc, Glm5NextMixer, Glm5NextMlpSite};

impl TransformerLayer for Glm5NextLayer {
    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(match &self.mixer {
            // 2026-09-25: On the model path a KDA layer gets SSM pool addresses instead
            // (`uses_ssm_pool`); this builds a zeroed, pool-free state for any other caller.
            Glm5NextMixer::Kda { cfg, .. } => Box::new(alloc_kda_ssm_state(gpu, cfg)?),
            Glm5NextMixer::Dsa(l) => Box::new(Glm5NextDsaState::alloc(gpu, &l.cfg)?),
        })
    }

    /// 2026-09-25: Frees a `Glm5NextDsaState`'s device buffers; the indexer cache is sized for
    /// the configured maximum context (`max_dsa_context`), not the prompt. Any other state is
    /// left alone, whatever the mixer: on the model path a KDA layer's `SsmLayerState` holds SSM
    /// pool addresses, which this layer does not own.
    fn release_state(&self, state: &mut dyn LayerState, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(dsa) = state.as_any_mut().downcast_mut::<Glm5NextDsaState>() {
            dsa.free(gpu)?;
        }
        Ok(())
    }

    /// 2026-10-08: A DSA layer's padding row gets a view of its workspace's padding indexer
    /// buffers (`Glm5NextDsaWorkspace::pad_state`) instead of a cache sized for the whole
    /// context, which `alloc_state` would allocate on every padded step and nothing would
    /// free. A KDA layer's padding row uses the SSM pool's dummy slot, which the model builds
    /// itself; this answers `alloc_state` for it.
    fn alloc_pad_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        match &self.mixer {
            Glm5NextMixer::Dsa(l) => Ok(Box::new(l.workspace.pad_state(&l.cfg))),
            Glm5NextMixer::Kda { .. } => self.alloc_state(gpu),
        }
    }

    /// 2026-10-08: One decode token for each of `num_seqs` sequences in one batched step
    /// (`steps/multi_seq.rs`): row `r` of `hidden` is sequence `r`, with its own highway slot,
    /// state, `seq_lens[r]` and metadata row. The DSA mixer reads the block tables from
    /// `ctx.attn_metadata`, so `block_tables` is not read; `residual` is not either, since
    /// the highway is the residual.
    #[allow(clippy::too_many_arguments)]
    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_multi(hidden, num_seqs, states, kv_cache, seq_lens, ctx, stream)
    }

    /// 2026-10-09: The batched speculative verify: `ks[i]` rows for each of `n_seqs`
    /// sequences, sequence-major, through `forward_verify_multi`. Each sequence's rows are
    /// the rows its own `decode_batched` verify runs (`steps/verify_multi.rs`). `wy_tables`
    /// (a GDN layer's) and `residual` (the highway is the residual) are not read.
    #[allow(clippy::too_many_arguments)]
    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        seq_lens: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if ks.len() != n_seqs {
            bail!(
                "GLM layer {}: a {n_seqs}-sequence batched verify got {} row counts",
                self.layer_idx,
                ks.len()
            );
        }
        self.forward_verify_multi(hidden, ks, seq_lens, states, kv_cache, ctx, stream)
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_one(
            hidden,
            residual,
            0,
            state,
            kv_cache,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            ctx,
            stream,
        )
    }

    /// 2026-09-25: Prefill with the highway indexed by token: prompt token `t` uses highway slot
    /// `ctx.hc_row_offset + t` (2026-10-08: the offset is 0 except in the fused decode +
    /// prefill step, where the decode rows hold the slots below it), so every token's streams
    /// are still there when the next layer reads them. The trait's default runs each token
    /// through `decode`, which uses slot 0 for all of them.
    ///
    /// Errors when `ctx.hc_row_offset + num_tokens` exceeds `ctx.buffers.max_batch_tokens()`,
    /// the number of highway slots in the buffer arena.
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len_start: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let cap = ctx.buffers.max_batch_tokens();
        // 2026-10-08: In a fused decode + prefill step the chunk's highway rows start after
        // the decode rows, at `ctx.hc_row_offset` (0 on every other caller).
        let slot0 = ctx.hc_row_offset;
        if slot0 + num_tokens > cap {
            bail!(
                "GLM layer {}: prefill of {num_tokens} tokens from highway slot {slot0} exceeds \
                 the {cap}-token mHC highway the buffer arena was sized for; each token needs \
                 its own slot",
                self.layer_idx
            );
        }
        // 2026-09-25: A text layer runs the prompt in sub-chunks of `prefill_rows()` tokens
        // through `forward_k`. The MTP block (`mhc: None`) walks token by token, because
        // `forward_k` refuses a layer without a highway.
        let rows = if self.mhc.is_some() {
            prefill_rows().min(cap)
        } else {
            1
        };
        if rows > 1 {
            let mut t = 0usize;
            while t < num_tokens {
                let k = rows.min(num_tokens - t);
                self.forward_k(
                    hidden.offset(t * self.hidden * 2),
                    k,
                    state,
                    kv_cache,
                    seq_len_start + t,
                    block_table,
                    ctx,
                    stream,
                    false,
                    slot0 + t,
                    // 2026-09-25: `is_prefill` is true only for this caller.
                    // This IS the prefill sub-chunk caller.
                    true,
                )?;
                t += k;
            }
            return Ok(());
        }
        for t in 0..num_tokens {
            let off = t * self.hidden * 2;
            self.forward_one(
                hidden.offset(off),
                residual.offset(off),
                slot0 + t,
                state,
                kv_cache,
                seq_len_start + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: K tokens of one sequence in one call, the speculative-verify body:
    /// `forward_k` over highway slots `0..K`, taking KDA state snapshots. The trait's default
    /// calls `decode` per token, and `decode` uses highway slot 0 for every token.
    #[allow(clippy::too_many_arguments)]
    fn decode_batched(
        &self,
        hidden: DevicePtr,
        _residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: For a KDA layer, row `t < K - 1` writes intermediate `t`, and
        // `rollback_ssm_states_dispatch` restores intermediate `num_accepted - 1`.
        // 2026-10-08: Under `--ssm-rollback-mode replay` the pool has no intermediates; the
        // slot's checkpoint and verify record stand in for them (`replay_verify_prepare`).
        let mut replay = None;
        if let (Glm5NextMixer::Kda { layer, .. }, true) = (&self.mixer, num_tokens > 1) {
            let st = self.kda_state(state)?;
            let snapshots = st.h_state_intermediates.len() + 1 >= num_tokens
                && st.conv_state_intermediates.len() + 1 >= num_tokens;
            if !snapshots {
                let Some(r) = self.replay_verify_prepare(layer, st, num_tokens, ctx, stream)?
                else {
                    // 2026-09-25: Error rather than run without the intermediates: a rejected
                    // draft could then not be rewound.
                    bail!(
                        "GLM layer {}: a {num_tokens}-token verify needs {} per-token state \
                         snapshots but the pool has h={} conv={}, and no replay record \
                         (--ssm-rollback-mode replay). With none, this is the \
                         self-speculative / ngram path on a model whose MTP pool was never \
                         sized; with too few, --num-drafts exceeds the pool's tier.",
                        self.layer_idx,
                        num_tokens - 1,
                        st.h_state_intermediates.len(),
                        st.conv_state_intermediates.len(),
                    );
                };
                replay = Some(r);
            }
        }

        self.forward_k(
            hidden,
            num_tokens,
            state,
            kv_cache,
            seq_len,
            block_table,
            ctx,
            stream,
            // 2026-10-08: A replay-mode verify takes no snapshots; it records after the forward.
            replay.is_none(),
            0,
            // 2026-09-25: `is_prefill` is false here.
            // A speculative verify, NOT a prefill sub-chunk: true here would give an eager
            // verify the prefill-only batched DSA selector (`batch_select_enabled`).
            false,
        )?;
        if let (Some(record), Glm5NextMixer::Kda { layer, ws, .. }) = (replay, &self.mixer) {
            layer.record_verify_rows(ctx.gpu, ws, num_tokens - 1, &record, stream)?;
        }
        Ok(())
    }

    /// 2026-10-08: The DFlash drafter reads each target layer's completed output averaged
    /// over the `hc_mult` streams. After `forward_one` / `forward_k` return, the FFN site's
    /// `hc_post` has folded the MLP output into the highway, so highway slot `r` holds this
    /// layer's completed streams for row `r` (decode: slot 0; prefill and verify: slot `t`
    /// for row `t`) until the next layer's `hc_pre` reads them; `hidden` instead holds the
    /// FFN site's pre-mix `y` (the last layer alone collapses the highway into it).
    /// `hc_head_mean` is that unweighted mean, one launch per row so each row lands at its
    /// strided slot. The MTP block (`mhc: None`) has no highway; its `hidden` is the
    /// completed output.
    fn dflash_tap_rows(
        &self,
        gpu: &dyn GpuBackend,
        buffers: &metrale_gpu_runtime::buffers::BufferArena,
        src_row0: usize,
        rows: usize,
        dst: DevicePtr,
        dst_row_stride_bytes: usize,
        stream: u64,
    ) -> Result<bool> {
        let Some(mhc) = self.mhc.as_ref() else {
            return Ok(false);
        };
        let (h, hc) = (self.hidden, mhc.hc_mult);
        for r in 0..rows {
            hc_head_mean(
                gpu,
                mhc.kernels.hc_head,
                buffers.hc_streams().offset((src_row0 + r) * hc * h * 4),
                dst.offset(r * dst_row_stride_bytes),
                1,
                h as u32,
                hc as u32,
                stream,
            )?;
        }
        Ok(true)
    }
}

impl LayerCapabilities for Glm5NextLayer {
    /// 2026-09-25: Was true while `decode_multi_seq` was the trait default, which ran every
    /// sequence through `decode`'s highway slot 0 and the DSA mixer's metadata row 0.
    /// 2026-10-08: False for a text layer: `decode_multi_seq` (`forward_multi`) gives each row
    /// its own slot, state and metadata row. True for the MTP block (no hyper-connection),
    /// which `forward_multi` refuses; the model never puts that block in its layer list.
    fn decode_multi_seq_unsupported(&self) -> bool {
        self.mhc.is_none()
    }

    /// 2026-10-09: True. Every per-step input of the decode is read from device buffers the
    /// model uploads before a replay (the DSA position, slot, `seq_len` and block table from
    /// the metadata, the KDA state from the slot-keyed pool), and `reduce_partial` issues the
    /// collectives on the capturing stream; the batched decode already replays the same
    /// launches with a communicator.
    fn decode_graph_with_comm(&self) -> bool {
        true
    }

    /// 2026-10-08: True: a DSA row selects over its own sequence's indexer cache
    /// (`Glm5NextDsaLayer::decode_rows`) and a KDA layer has no index, so the model's mHC +
    /// sparse-index per-sequence rule does not apply to this layer.
    fn decode_multi_seq_selects_index_per_row(&self) -> bool {
        true
    }

    /// 2026-09-25: Was true while this layer had no `decode_verify_multi`.
    /// 2026-10-09: False for a text layer (`steps/verify_multi.rs`); true for the MTP block,
    /// which has no hyper-connection and which `forward_spans` refuses.
    fn decode_verify_multi_unsupported(&self) -> bool {
        self.mhc.is_none()
    }

    /// 2026-10-09: True: the batched verify issues the same collectives on every rank (one
    /// all-reduce per mixer and MLP site per row group) and keeps no rank-local choice, so
    /// the worker ranks can run it from the batch rank 0 announces.
    fn batch_verify_across_ranks(&self) -> bool {
        true
    }

    /// 2026-09-25: True. A DSA layer's per-sequence state comes from `gpu.alloc` in
    /// `alloc_state`, so the addresses a captured graph holds belong to one sequence.
    fn graph_stale_on_new_sequence(&self) -> bool {
        true
    }

    /// 2026-09-25: True for a KDA layer. Its recurrent and conv state then come from the SSM
    /// pool, and with MTP on, so do the checkpoint and per-token intermediate pointers that a
    /// speculative rollback restores (`meta.rs`).
    fn uses_ssm_pool(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    /// 2026-09-25: True for a KDA layer, whose mixer carries recurrent state.
    fn is_ssm_layer(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }

    /// 2026-10-08: True for a KDA layer: `decode_batched` checkpoints and records a
    /// replay-mode verify, and `ssm_replay_commit` rebuilds the accepted state.
    fn supports_ssm_replay(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Kda { .. })
    }
}

impl LayerWeightSetup for Glm5NextLayer {}
impl LayerWriteOnAccept for Glm5NextLayer {
    /// 2026-10-08: A KDA layer's replay commit (`Glm5NextKdaLayer::commit_replay`); `Ok(false)`
    /// on a DSA layer, which keeps no recurrent state.
    fn ssm_replay_commit(
        &self,
        gpu: &dyn GpuBackend,
        state: &mut dyn LayerState,
        accepted: usize,
        k_rows: usize,
        stream: u64,
    ) -> Result<bool> {
        let Glm5NextMixer::Kda { layer, ws, .. } = &self.mixer else {
            return Ok(false);
        };
        let st = self.kda_state(state)?;
        let (live, checkpoint, record) = self.replay_parts(layer, st)?;
        layer.commit_replay(
            gpu,
            &live,
            &checkpoint,
            &record,
            accepted,
            k_rows,
            ws,
            stream,
        )?;
        Ok(true)
    }
}

impl LayerGraphHooks for Glm5NextLayer {
    /// 2026-09-25: A graph replay runs only kernels, so the host-side length of the DSA indexer
    /// cache is reconciled here (`Glm5NextDsaState::sync_to`). The inner `Glm5NextDsaLayer`
    /// implements this too, but the model's layer list holds this composite, so this impl is the
    /// one called. KDA keeps no host-side state.
    fn sync_replayed_step(
        &self,
        state: &mut dyn LayerState,
        seq_len: usize,
        k: usize,
    ) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => self.dsa_state(state)?.sync_to(seq_len, k),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }

    /// 2026-09-25: Checks before a graph replay that the DSA indexer cache can hold `seq_len + k`
    /// rows. As with `sync_replayed_step`, this composite impl is the one the model calls.
    fn check_replay_room(&self, state: &dyn LayerState, seq_len: usize, k: usize) -> Result<()> {
        match &self.mixer {
            Glm5NextMixer::Dsa(_) => state
                .as_any()
                .downcast_ref::<Glm5NextDsaState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "GLM layer {}: a DSA mixer was handed state that is not a \
                         Glm5NextDsaState",
                        self.layer_idx
                    )
                })?
                .ensure_room_through(seq_len + k)
                // 2026-09-25: `ensure_room_through` raises the same error for every caller;
                // the context names this route.
                .with_context(|| {
                    format!(
                        "DSA replay pre-check (layer {}, before launch_graph, seq_len \
                         {seq_len} + k {k})",
                        self.layer_idx
                    )
                }),
            Glm5NextMixer::Kda { .. } => Ok(()),
        }
    }
}

impl LayerAuxState for Glm5NextLayer {
    /// 2026-09-25: True for a DSA layer: its indexer cache is the state `snapshot_aux` and
    /// `restore_aux` carry. A KDA layer's state is in the SSM pool (`uses_ssm_pool`) and is not
    /// carried here.
    fn has_aux_state(&self) -> bool {
        matches!(self.mixer, Glm5NextMixer::Dsa(_))
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            return Ok(None);
        }
        let st = state
            .as_any()
            .downcast_ref::<Glm5NextDsaState>()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "GLM layer {}: a DSA mixer was handed state that is not a Glm5NextDsaState",
                    self.layer_idx
                )
            })?;
        Ok(Some(st.snapshot_blob(gpu, stream)?))
    }

    /// 2026-09-25: Errors on a KDA layer, whose state travels with the SSM snapshot. On a DSA
    /// layer it restores the blob through `Glm5NextDsaState::restore_blob`; `apply_aux_states`
    /// propagates any error.
    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !matches!(self.mixer, Glm5NextMixer::Dsa(_)) {
            bail!(
                "GLM layer {}: restore_aux on a KDA layer — KDA state is pool-backed and \
                 travels with the SSM snapshot, so a blob addressed here is a routing bug",
                self.layer_idx
            );
        }
        self.dsa_state(state)?.restore_blob(blob, gpu, stream)
    }
}

impl LayerSplitPrefill for Glm5NextLayer {}
impl metrale_model_layers::circuit_exec::CircuitBindings for Glm5NextLayer {}

#[cfg(test)]
mod tests;
