// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Glm5NextMtpHead`: GLM-5.3's MTP block as a [`DraftProposer`], one draft token per `forward_one`.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - The FP8 copy of the draft head is read only on the vocab-sharded sweep; every other
//!   sweep reads the BF16 `lm_head`.
//!
//! ```text
//! x = eh_proj( concat( enorm(embed[token]), hnorm(target_hidden) ) )   [1, 2H] -> [1, H]
//! x = layers.{num_hidden_layers}(x)      DSA + routed MoE, plain residual (no mHC)
//! logits = lm_head( shared_head.norm(x) )              the target's own head
//! draft  = argmax(logits)
//! ```
//!
//! The block is sharded like a text layer: its routed experts are split across EP ranks and
//! its DSA `o_proj` is row-parallel, so its forward needs the communicator (`needs_comm`).

use anyhow::{Result, bail};
use parking_lot::Mutex;
use std::any::Any;

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::glm5next_dsa::state::Glm5NextDsaState;
use metrale_model_layers::layer::{ForwardContext, LayerState};

use crate::weight_loader::Glm5NextMtpModule;
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::{DraftProposer, ProposerState};
use metrale_model_layers::weight_map::DenseWeight;

mod graph;
mod head_xchg;
mod init;
mod proposer;

/// 2026-10-11: The drafter graph's per-state bytes, for the per-sequence reserve.
pub use graph::state_bytes as graph_state_bytes;
/// 2026-10-11: Bytes of the draft head's exchange buffer, one per sequence state.
pub use head_xchg::BUFFER_BYTES as HEAD_XCHG_BUFFER_BYTES;
/// 2026-10-11: The `METRALE_GLM_MTP_EH_TP` lever, for the ranks' agreement check.
pub use init::eh_tp_enabled;

/// 2026-09-25: Per-sequence drafter state: the block's own indexer cache and KV block table.
pub struct Glm5NextMtpProposerState {
    dsa: Glm5NextDsaState,
    /// 2026-09-25: Rows the drafter has written. `propose` and `after_verify` roll it back.
    seq_len: usize,
    block_table: Vec<u32>,
    /// 2026-09-25: How many drafts the last `propose` wrote, so `after_verify` knows what
    /// to trim.
    last_drafted: usize,
    /// 2026-09-25: Scratch: `[2, hidden]` BF16 concat, `[hidden]` BF16 block input,
    /// `[vocab]` BF16 logits, `[1]` u32 argmax.
    concat: DevicePtr,
    x: DevicePtr,
    logits: DevicePtr,
    arg: DevicePtr,
    /// 2026-09-25: BF16 lanes for the vocab-sharded head's cross-rank pick: each rank's
    /// max logit, then three base-256 digits of each rank's token id (2026-10-11:
    /// `head_xchg`, sized for its `MAX_RANKS`).
    head_xchg: DevicePtr,
    /// 2026-09-25: Set by `free_state`, which is a no-op once it is set.
    released: bool,
    /// 2026-10-11: The drafter graph's metadata and graph (`METRALE_GLM_MTP_GRAPH`); `None`
    /// runs every draft eagerly.
    graph: Option<graph::DrafterGraph>,
}

impl ProposerState for Glm5NextMtpProposerState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct Glm5NextMtpHead {
    module: Glm5NextMtpModule,
    embed_tokens: DenseWeight,
    lm_head: DenseWeight,
    /// 2026-09-25: A one-layer pool of the drafter's own, so its entries can be trimmed
    /// independently of the target's.
    kv_cache: Mutex<PagedKvCache>,
    rms_norm_k: KernelHandle,
    gemv_k: KernelHandle,
    argmax_k: KernelHandle,
    hidden: usize,
    vocab: usize,
    max_seq_len: usize,
    /// 2026-09-25: Vocab shard of the shared `lm_head` this rank sweeps:
    /// `[head_v0, head_v0 + head_n)`. `head_n == vocab` (and `head_v0 == 0`) unless
    /// `init::draft_head_shard` splits the vocab across the `head_world` ranks.
    head_world: usize,
    head_rank: usize,
    /// 2026-10-11: This rank's `eh_proj` output rows `(r0, n)` under `METRALE_GLM_MTP_EH_TP`
    /// (`init::eh_shard`); `None` computes them all.
    eh_shard: Option<(usize, usize)>,
    head_v0: usize,
    head_n: usize,
    /// 2026-09-25: FP8 E4M3 copy of this rank's vocab shard of `lm_head`, for drafting only.
    /// The target verifies every draft with its own head, so this copy can change the
    /// acceptance rate but not an emitted token. `None` when `METRALE_GLM_MTP_HEAD_FP8=0`,
    /// when the `gemv_fp8w` kernel is absent, or when quantisation fails.
    /// Measured 2026-08-29 (nsys): 2.66 ms -> ~1.33 ms per draft sweep.
    head_fp8: Option<metrale_model_layers::weight_map::Fp8DenseWeight>,
    gemv_fp8w_k: KernelHandle,
    /// 2026-10-11: `(glm5next_mtp_xchg_pack, glm5next_mtp_xchg_pick)` when
    /// `METRALE_GLM_MTP_XCHG_DEVICE=1` and the build has them: the vocab-sharded head's exchange
    /// on the device (`exchange_on_device`). `None` keeps the host exchange.
    xchg_k: Option<(KernelHandle, KernelHandle)>,
}

/// 2026-09-25: Rows the GLM drafter can ever be asked for: the served context, clamped to
/// the rows its DSA indexer cache reserves. `max_dsa_context` is `max_context` (the serve's
/// `--max-seq-len`) rounded down to whole `index_kpool` pools, so the cap follows the config.
fn drafter_context_rows(max_seq_len: usize, cfg: &crate::glm5next_dsa::Glm5NextDsaConfig) -> usize {
    max_seq_len.min(crate::glm5next_dsa::state::max_dsa_context(cfg))
}

impl Glm5NextMtpHead {
    fn norm(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([1, 1, 1])
            .block([(n.min(1024)) as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(n as u32)
            .arg_f32(self.module.layer.rms_eps)
            .launch(stream)
    }

    /// 2026-09-25: `x = eh_proj(concat)`. 2026-10-11: under `METRALE_GLM_MTP_EH_TP` with a
    /// communicator, this rank computes its output rows into a zeroed `x` and one all-reduce of
    /// the row assembles them; each row is the same GEMV over the same input either way (a block
    /// computes four rows independently), so `x` keeps its bits.
    fn eh_proj(
        &self,
        gpu: &dyn GpuBackend,
        st: &Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden;
        if let (Some(comm), Some((r0, n))) = (ctx.comm, self.eh_shard) {
            gpu.memset_async(st.x, 0, h * 2, stream)?;
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                st.concat,
                &DenseWeight {
                    weight: self.module.eh_proj.weight.offset(r0 * 2 * h * 2),
                },
                st.x.offset(r0 * 2),
                n as u32,
                (2 * h) as u32,
                stream,
            )?;
            return comm.all_reduce_async(st.x.0, h * 2, stream);
        }
        ops::dense_gemv(
            gpu,
            self.gemv_k,
            st.concat,
            &self.module.eh_proj,
            st.x,
            h as u32,
            (2 * h) as u32,
            stream,
        )
    }

    /// 2026-10-11: `(n, v0, sharded)`: the vocab rows this rank's draft head sweeps. Sharded
    /// only with a communicator and a vocab split across ranks; otherwise the sweep covers the
    /// whole vocab.
    fn head_range(&self, ctx: &ForwardContext) -> (usize, usize, bool) {
        let sharded = ctx.comm.is_some() && self.head_n != self.vocab;
        if sharded {
            (self.head_n, self.head_v0, true)
        } else {
            (self.vocab, 0, false)
        }
    }

    /// 2026-10-11: One draft's device work, from the embedding row `embed_src` and the target
    /// hidden `hidden_in` to the argmax in `st.arg`: the input norms and `eh_proj`, the block
    /// (advancing `seq_len`), `shared_head.norm`, the head sweep and the argmax. Eager, or the
    /// body of the drafter graph (`graph.rs`).
    fn forward_body(
        &self,
        embed_src: DevicePtr,
        hidden_in: DevicePtr,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        self.norm(gpu, embed_src, self.module.enorm, st.concat, h, stream)?;
        self.norm(
            gpu,
            hidden_in,
            self.module.hnorm,
            st.concat.offset(h * 2),
            h,
            stream,
        )?;
        self.eh_proj(gpu, st, ctx, stream)?;

        // 2026-09-25: The block writes its output back over `st.x` (plain residual, in place).
        if skip_block() {
            st.seq_len += 1;
        } else {
            let mut kv = self.kv_cache.lock();
            let dsa_state: &mut dyn LayerState = &mut st.dsa;
            self.module.layer.decode_one_for_drafter(
                st.x,
                dsa_state,
                &mut kv,
                st.seq_len,
                &mut st.block_table,
                ctx,
                stream,
            )?;
            drop(kv);
            st.seq_len += 1;
        }
        if skip_head() {
            return Ok(());
        }
        // 2026-09-25: `shared_head.norm`, then the target's own `lm_head`: the MTP layer carries
        // no head of its own.
        self.norm(gpu, st.x, self.module.final_norm, st.x, h, stream)?;
        let (n, v0, sharded) = self.head_range(ctx);
        // 2026-09-25: The FP8 copy covers only `[head_v0, head_v0 + head_n)`, so it serves the
        // sharded sweep alone; an unsharded sweep reads the BF16 head.
        match self.head_fp8.filter(|_| sharded && n == self.head_n) {
            Some(q) => ops::dense_gemv_fp8w(
                gpu,
                self.gemv_fp8w_k,
                st.x,
                &q,
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
            None => ops::dense_gemv(
                gpu,
                self.gemv_k,
                st.x,
                &DenseWeight {
                    weight: self.lm_head.weight.offset(v0 * h * 2),
                },
                st.logits,
                n as u32,
                h as u32,
                stream,
            )?,
        }
        ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, n as u32, stream)?;
        if let (Some(k), Some(comm)) = (self.device_xchg(ctx), ctx.comm) {
            self.exchange_on_device(gpu, comm, k, st, v0, stream)?;
        }
        Ok(())
    }

    /// 2026-10-11: The device exchange's kernels when this propose runs it: the lever and kernels
    /// (`xchg_k`) and a sharded head with a communicator.
    fn device_xchg(&self, ctx: &ForwardContext) -> Option<(KernelHandle, KernelHandle)> {
        let (_, _, sharded) = self.head_range(ctx);
        self.xchg_k.filter(|_| sharded && ctx.comm.is_some())
    }

    /// 2026-10-11: The `(max, argmax)` exchange of the host tail (`head_xchg::pack` / `pick`) on
    /// the device: pack this rank's lanes from `st.arg` and its logit, all-reduce them, and
    /// leave the winning global id in `st.arg`. No host step, so it runs inside the drafter
    /// graph and the ranks meet at its all-reduce straight from their own head sweeps.
    fn exchange_on_device(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn metrale_comm::CommBackend,
        (pack_k, pick_k): (KernelHandle, KernelHandle),
        st: &Glm5NextMtpProposerState,
        v0: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, pack_k)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(st.arg)
            .arg_ptr(st.logits)
            .arg_ptr(st.head_xchg)
            .arg_u32(self.head_world as u32)
            .arg_u32(self.head_rank as u32)
            .arg_u32(v0 as u32)
            .launch(stream)?;
        comm.all_reduce_async(
            st.head_xchg.0,
            head_xchg::xchg_bytes(self.head_world),
            stream,
        )?;
        KernelLaunch::new(gpu, pick_k)
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(st.head_xchg)
            .arg_u32(self.head_world as u32)
            .arg_ptr(st.arg)
            .launch(stream)
    }

    /// 2026-09-25: One draft token. Advances `seq_len` by one.
    #[allow(clippy::too_many_arguments)]
    fn forward_one(
        &self,
        token: u32,
        hidden_in: DevicePtr,
        // 2026-10-11: The first draft of this propose (it reads the target's hidden row).
        first: bool,
        position: usize,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<u32> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        if position >= self.max_seq_len {
            bail!(
                "GLM MTP drafter: position {position} is past the {} it was sized for",
                self.max_seq_len
            );
        }
        // 2026-09-25: The embedding row is read by pointer: `embed_tokens` is `[vocab, hidden]`
        // BF16 and a row is contiguous, so there is nothing to gather.
        let embed_row = self.embed_tokens.weight.offset(token as usize * h * 2);
        // 2026-10-11: `METRALE_GLM_MTP_GRAPH`: the same launches, replayed from a graph
        // (`graph.rs`); the readback and exchange below stay eager.
        if graph::eligible(self, st, first) {
            self.forward_graphed(embed_row, hidden_in, st, ctx, stream)?;
        } else {
            self.forward_body(embed_row, hidden_in, st, ctx, stream)?;
        }
        // 2026-09-25: Timing arm `METRALE_GLM_MTP_SKIP=head`: the head is skipped and the
        // draft is token 0, so `propose` times the block alone. `=block` skips the block instead.
        if skip_head() {
            return Ok(0);
        }
        let (_, v0, sharded) = self.head_range(ctx);
        let mut out = [0u8; 4];
        if self.device_xchg(ctx).is_some() {
            // 2026-10-11: The device exchange left the global id in `st.arg`: one read.
            gpu.copy_d2h(st.arg, &mut out)?;
            return Ok(u32::from_le_bytes(out));
        }
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.arg, &mut out)?;
        let local = u32::from_le_bytes(out) as usize;
        let Some(comm) = ctx.comm.filter(|_| sharded) else {
            return Ok((v0 + local) as u32);
        };
        // 2026-09-25: Exchange (max, argmax) through one all-reduce of BF16 lanes: each rank's
        // max logit, then three base-256 digits of each rank's global index (`head_xchg` has
        // the layout and why a SUM all-reduce delivers it untouched). The logit lane is the
        // head's own BF16 output, so it round-trips bit for bit.
        let mut lb = [0u8; 2];
        gpu.copy_d2h(st.logits.offset(local * 2), &mut lb)?;
        let mut pack = head_xchg::pack(self.head_world, self.head_rank, lb, v0 + local)?;
        gpu.copy_h2d(&pack, st.head_xchg)?;
        comm.all_reduce_async(st.head_xchg.0, pack.len(), stream)?;
        gpu.synchronize(stream)?;
        gpu.copy_d2h(st.head_xchg, &mut pack)?;
        // 2026-09-25: A tie goes to the lower rank, identically on every rank, so the ranks'
        // drafter KV streams cannot diverge on a tie.
        head_xchg::pick(self.head_world, &pack)
    }

    /// 2026-09-25: Append `tokens.len() - 1` drafter context rows: row `r` is pair key
    /// `row_base + r` = `(embed(tokens[r + 1]), hiddens row r)`. Used for both the whole-prompt
    /// prefill and the catch-up feed; the only difference between them is `row_base`.
    ///
    /// The row space is dense: a row's KV slot, indexer row and RoPE position are all
    /// `seq_len` (see `Glm5NextDsaLayer::write_kv_row`), so every pair key from 0 up is
    /// written and slot == key.
    ///
    /// Unless `METRALE_GLM_MTP_PREFILL_FULL=1`, a context row runs only `input_norm` and the
    /// DSA cache writes (`drafter_write_kv_row`): no MoE, no attention, no `lm_head`, because
    /// its block output is discarded.
    #[allow(clippy::too_many_arguments)]
    fn rows_impl(
        &self,
        tokens: &[u32],
        hiddens: DevicePtr,
        row_base: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<usize> {
        let st = match state
            .as_any_mut()
            .downcast_mut::<Glm5NextMtpProposerState>()
        {
            Some(s) => s,
            None => return Ok(0),
        };
        // 2026-09-25: Rows must append exactly at the drafter's current length, or the dense
        // row space grows a hole and every later RoPE position is wrong.
        if st.seq_len != row_base || tokens.len() < 2 {
            return Ok(0);
        }
        let h = self.hidden;
        let rows = tokens.len() - 1;
        if row_base + rows > self.max_seq_len {
            return Ok(0);
        }
        let gpu = ctx.gpu;
        let dbg = metrale_model_layers::speculative::mtp_refeed_debug();
        let prefill_full = std::env::var("METRALE_GLM_MTP_PREFILL_FULL")
            .ok()
            .as_deref()
            == Some("1");
        let mut kv = self.kv_cache.lock();
        let Glm5NextMtpProposerState {
            dsa,
            seq_len,
            block_table,
            concat,
            x,
            ..
        } = st;
        for r in 0..rows {
            let embed_row = self
                .embed_tokens
                .weight
                .offset(tokens[r + 1] as usize * h * 2);
            self.norm(gpu, embed_row, self.module.enorm, *concat, h, stream)?;
            self.norm(
                gpu,
                hiddens.offset(r * h * 2),
                self.module.hnorm,
                concat.offset(h * 2),
                h,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                self.gemv_k,
                *concat,
                &self.module.eh_proj,
                *x,
                h as u32,
                (2 * h) as u32,
                stream,
            )?;
            let dsa_state: &mut dyn LayerState = dsa;
            // 2026-09-25: Diagnostic arm `METRALE_GLM_MTP_PREFILL_FULL=1`: build the row through
            // the full block path a propose uses (`decode_one_for_drafter`) instead of the
            // cache-only write.
            if prefill_full {
                self.module.layer.decode_one_for_drafter(
                    *x,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            } else {
                self.module.layer.drafter_write_kv_row(
                    *x,
                    dsa_state,
                    &mut kv,
                    *seq_len,
                    block_table,
                    ctx,
                    stream,
                )?;
            }
            if dbg {
                let fp = metrale_model_layers::speculative::hidden_fingerprint(
                    gpu,
                    hiddens.offset(r * h * 2),
                    h,
                );
                tracing::info!(
                    "GLM_MTP_DBG ctx row slot={} key={} tok={} fp_hidden={fp:016x}",
                    *seq_len,
                    row_base + r,
                    tokens[r + 1],
                );
            }
            *seq_len += 1;
        }
        Ok(rows)
    }
}

/// 2026-09-25: `METRALE_GLM_MTP_SKIP=head`: stop the drafter after the block, before
/// `shared_head.norm`, the `lm_head` gemv, the argmax and the D2H. A timing arm; read once
/// per process.
fn skip_head() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_SKIP").ok().as_deref() == Some("head"))
}

/// 2026-09-25: `METRALE_GLM_MTP_SKIP=block`: skip the MTP block itself and run only the
/// head. A timing arm; read once per process.
fn skip_block() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_SKIP").ok().as_deref() == Some("block"))
}

#[cfg(test)]
#[path = "glm5next_mtp_head_tests.rs"]
mod a59_sizing_tests;
