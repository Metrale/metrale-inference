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
//!     target_hidden: hc_mean(streams), or under `METRALE_MTP_TARGET_POSTNORM=1` its target
//!     final norm, model.norm(hc_mean(streams)), the row vLLM's GLM MTP layer is fed
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

mod batch;
mod batch_check;
mod batch_graph;
mod catchup;
mod ctx_rows;
mod draft_io;
mod draft_lp;
mod graph;
mod head_xchg;
mod hidden_half;
mod init;
mod kv_pool;
mod proposer;
mod rows;

/// 2026-10-11: The batched propose's lever, its row limit, and the block workspaces' rows.
pub use batch::{PROPOSE_BATCH_MAX, drafter_block_rows, propose_batch_enabled};
/// 2026-10-11: Rows the MTP block's DSA workspace is built for: the batched propose's rows and
/// a context-row pass (`ctx_rows`).
pub fn drafter_dsa_rows() -> usize {
    drafter_block_rows().max(ctx_rows::CTX_ROWS_CHUNK)
}
/// 2026-10-11: The drafter graph's per-state bytes, for the per-sequence reserve.
pub use graph::state_bytes as graph_state_bytes;
/// 2026-10-11: Bytes of the draft head's exchange buffer, one per sequence state.
pub use head_xchg::BUFFER_BYTES as HEAD_XCHG_BUFFER_BYTES;
/// 2026-10-11: The host exchange, the oracle of the device kernels (`glm53_mtp_xchg_gate`).
pub use head_xchg::{
    MAX_RANKS as HEAD_XCHG_MAX_RANKS, pack as head_xchg_pack, pack_lp as head_xchg_pack_lp,
    pick as head_xchg_pick, pick_lp as head_xchg_pick_lp, xchg_bytes as head_xchg_bytes,
    xchg_lp_bytes as head_xchg_lp_bytes,
};
/// 2026-10-11: The `METRALE_GLM_MTP_EH_TP` lever, for the ranks' agreement check.
pub use init::eh_tp_enabled;
/// 2026-10-11: Bytes of one sequence's drafter KV region, for the per-sequence reserve.
pub use kv_pool::drafter_kv_bytes_per_seq;

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
    /// 2026-10-11: This state's region of the drafter KV pool (`kv_pool`); `block_table` is
    /// its run of blocks. `None` when the pool had no free region: the state drafts nothing
    /// rather than sharing another sequence's rows.
    kv_region: Option<usize>,
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
    /// independently of the target's. 2026-10-11: One region per live state (`kv_regions`).
    kv_cache: Mutex<PagedKvCache>,
    kv_regions: kv_pool::DrafterKvRegions,
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
    /// 2026-10-11: The batched propose (`METRALE_GLM_MTP_PROPOSE_BATCH`); `None` proposes per
    /// sequence.
    batch: Option<batch::ProposeBatch>,
    /// 2026-10-11: `(glm5next_mtp_xchg_pack, glm5next_mtp_xchg_pick)` when
    /// `METRALE_GLM_MTP_XCHG_DEVICE=1` and the build has them: the vocab-sharded head's exchange
    /// on the device (`exchange_on_device`). `None` keeps the host exchange.
    xchg_k: Option<(KernelHandle, KernelHandle)>,
    /// 2026-10-11: The context rows' pass scratch (`ctx_rows`); `None` runs them one at a time.
    ctx_rows: Option<Mutex<ctx_rows::CtxRowsScratch>>,
    /// 2026-10-11: The draft log-probability kernels (`draft_lp`).
    lp_k: draft_lp::LpKernels,
    /// 2026-10-11: The target's own final norm (`model.language_model.norm.weight`, BF16
    /// `[hidden]`, the pointer its `final_norm_apply` reads), applied to target-hidden rows
    /// under `ModelLevers::mtp_target_postnorm` (`hidden_half`).
    target_final_norm: DevicePtr,
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
        self.norm_rows(gpu, x, w, out, n, 1, stream)
    }

    /// 2026-10-11: [`Self::norm`] over `rows` contiguous rows of `n` (one block per row; each
    /// row's result is the one-row launch's).
    #[allow(clippy::too_many_arguments)]
    fn norm_rows(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        n: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(gpu, self.rms_norm_k)
            .grid([rows as u32, 1, 1])
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
    // 2026-10-11: `track_caller`, so the feed levers' first-application log names the caller
    // (the eager draft or the drafter graph's capture), not this body.
    #[track_caller]
    fn forward_body(
        &self,
        // 2026-10-11: The token's embedding row; `None` zeroes the embedding half (pair key 0
        // under `METRALE_MTP_ZERO_POS0`, `hidden_half::zeroes_embedding`).
        embed_src: Option<DevicePtr>,
        hidden_in: DevicePtr,
        // 2026-10-11: `hidden_in` is a target row (the first draft), not the drafter's own
        // chain output (`hidden_half`).
        target_row: bool,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = self.hidden;
        self.embed_half(gpu, embed_src, st.concat, stream)?;
        // 2026-10-11: `st.x` is free until `eh_proj` writes it, so it stages the target norm.
        self.hidden_half(ctx, hidden_in, target_row, st.x, st.concat, stream)?;
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
        // 2026-10-11: Under the confidence stop the argmax also writes the draft's
        // log-probability, at `arg + 4` (`draft_lp`).
        let lp = self.single_lp_ready(ctx).then_some(st.arg.offset(4));
        match lp {
            Some(lp) => ops::argmax_bf16_batch_lp(
                gpu,
                self.lp_k.argmax_lp,
                st.logits,
                st.arg,
                lp,
                n as u32,
                1,
                n as u32,
                stream,
            )?,
            None => ops::argmax_bf16(gpu, self.argmax_k, st.logits, st.arg, n as u32, stream)?,
        }
        if let (Some(k), Some(comm)) = (self.device_xchg(ctx), ctx.comm) {
            self.exchange_rows(
                gpu,
                comm,
                k,
                st.arg,
                st.logits,
                0,
                st.head_xchg,
                1,
                lp,
                v0,
                stream,
            )?;
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
    /// the device, for `rows` rows: pack each row's lanes from `arg[r]` and its logit (row `r`
    /// of `logits` starts `r * logits_row_bytes` in), all-reduce every row's lanes at once, and
    /// leave each winning global id in `arg[r]`. No host step, so it runs inside the drafter
    /// graph and the ranks meet at its all-reduce straight from their own head sweeps. Row `r`'s
    /// lanes are `xchg + r * xchg_bytes(world)`, so the rows are one contiguous all-reduce.
    /// 2026-10-11: With `lp` (row `r`'s shard log-probability at `lp + 4r`, from
    /// `argmax_bf16_batch_lp`) the lanes also carry each rank's shard log-sum-exp
    /// (`head_xchg::pack_lp`, `xchg_lp_bytes` per row) and the winner's whole-vocab
    /// log-probability replaces it at `lp + 4r`.
    #[allow(clippy::too_many_arguments)]
    fn exchange_rows(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn metrale_comm::CommBackend,
        (pack_k, pick_k): (KernelHandle, KernelHandle),
        arg: DevicePtr,
        logits: DevicePtr,
        logits_row_bytes: usize,
        xchg: DevicePtr,
        rows: usize,
        lp: Option<DevicePtr>,
        v0: usize,
        stream: u64,
    ) -> Result<()> {
        let world = self.head_world as u32;
        let xb = match lp {
            Some(_) => head_xchg::xchg_lp_bytes(self.head_world),
            None => head_xchg::xchg_bytes(self.head_world),
        };
        for r in 0..rows {
            let launch = match lp {
                Some(lp) => KernelLaunch::new(gpu, self.lp_k.pack_lp)
                    .arg_ptr(arg.offset(r * 4))
                    .arg_ptr(logits.offset(r * logits_row_bytes))
                    .arg_ptr(lp.offset(r * 4)),
                None => KernelLaunch::new(gpu, pack_k)
                    .arg_ptr(arg.offset(r * 4))
                    .arg_ptr(logits.offset(r * logits_row_bytes)),
            };
            launch
                .grid([1, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(xchg.offset(r * xb))
                .arg_u32(world)
                .arg_u32(self.head_rank as u32)
                .arg_u32(v0 as u32)
                .launch(stream)?;
        }
        comm.all_reduce_async(xchg.0, rows * xb, stream)?;
        for r in 0..rows {
            let launch = KernelLaunch::new(
                gpu,
                if lp.is_some() {
                    self.lp_k.pick_lp
                } else {
                    pick_k
                },
            )
            .grid([1, 1, 1])
            .block([32, 1, 1])
            .arg_ptr(xchg.offset(r * xb))
            .arg_u32(world)
            .arg_ptr(arg.offset(r * 4));
            match lp {
                Some(lp) => launch.arg_ptr(lp.offset(r * 4)).launch(stream)?,
                None => launch.launch(stream)?,
            }
        }
        Ok(())
    }

    /// 2026-09-25: One draft token. Advances `seq_len` by one. 2026-10-11: With its top-1
    /// log-probability under the confidence stop (`draft_lp`), NaN otherwise.
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
    ) -> Result<(u32, f32)> {
        self.draft_on_device(token, hidden_in, first, position, st, ctx, stream)?;
        self.read_draft(st, ctx, stream)
    }

    /// 2026-10-11: The device work of one draft (`forward_one` without its readback): the
    /// global id is left in `st.arg` when [`Self::draft_stays_on_device`] holds.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn draft_on_device(
        &self,
        token: u32,
        hidden_in: DevicePtr,
        first: bool,
        position: usize,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
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
        // 2026-10-11: This draft's pair key is `position - 1` (its hidden is position - 1's).
        let zero = position
            .checked_sub(1)
            .is_some_and(|key| hidden_half::zeroes_embedding(ctx.levers.mtp_zero_pos0, key));
        let embed = (!zero).then_some(embed_row);
        match embed {
            Some(e) if graph::eligible(self, st, first) => {
                self.forward_graphed(e, hidden_in, st, ctx, stream)
            }
            _ => self.forward_body(embed, hidden_in, first, st, ctx, stream),
        }
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
