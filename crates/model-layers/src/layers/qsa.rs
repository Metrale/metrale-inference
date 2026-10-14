// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Qwen3.8-Flash-Next QSA indexer: sparse-attention token selection.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - Keys are ingested in position order: `prefill_ingest` and
//!   `decode_select` refuse a start position other than `ingested`.
//! - At or below `inert_bound()` (`budget + ratio - 1`) visible tokens every
//!   visible token is selected, so the dense path is used unchanged.
//!
//! Reference: `Qwen4ExpTextQSAIndexer`. The attention layer's input is
//! projected to `n_heads` query heads plus one raw key per token. The visible
//! prefix is grouped into `ratio`-token blocks whose keys are mean-pooled,
//! k_layernormed and roped at the block's first position. Each query attends
//! the top `budget / ratio` blocks by `sum_h relu(q_h . k_b) / sqrt(hd)`, plus
//! the incomplete tail.
//!
//! Raw keys are ingested during prefill and decode. 2026-10-13 (L-K9a): only the
//! incomplete block's raw keys are kept per sequence (`raw_tail`, a `ratio`-row ring);
//! a complete block is pooled as soon as its last key arrives and its raw keys are not
//! read again, so prefill pools each slab out of one per-layer scratch. Decode selection gathers
//! the selected K/V into a contiguous scratch that, through an identity block
//! table, the paged decode attention reads as a cache. Prefill selection
//! (`prefill_select`, in `qsa_select.rs`) overwrites the attention context of
//! the rows past the bound. Selection takes a host top-k over scores read back
//! from the device, so a layer with an indexer reports
//! `decode_graph_unsupported` (the batched decode) and, per step,
//! `decode_graph_unsupported_at` past the bound (`qsa_decode.rs`).

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

#[path = "qsa_decode.rs"]
mod qsa_decode;
#[path = "qsa_select.rs"]
mod qsa_select;
#[path = "qsa_snapshot.rs"]
mod qsa_snapshot;
#[cfg(test)]
#[path = "qsa_tests.rs"]
mod tests;

/// 2026-09-25: One decode step's selection: contiguous NHD `k/v` scratch + identity table.
pub struct QsaSelection {
    pub k_scratch: DevicePtr,
    pub v_scratch: DevicePtr,
    pub table_dev: DevicePtr,
    pub seq_len_dev: DevicePtr,
    pub n_sel: u32,
    pub max_blocks: u32,
}

pub struct QsaSeqState {
    /// 2026-09-25: Tokens ingested (contiguous from 0). 2026-10-13: the raw keys of the
    /// last `ingested % ratio` of them are in `raw_tail`; the rest are pooled.
    ingested: usize,
    /// 2026-09-25: Complete `ratio`-token blocks pooled into `block_keys`.
    pooled: usize,
    /// 2026-10-13: `ingested` when the current decode step began (`begin_step`), which
    /// `rearm_step` restores after a CUDA-graph capture that recorded the step but ran
    /// nothing.
    step_start: usize,
    /// 2026-09-25: Pages of the identity block table already uploaded to the
    /// layer's `table_dev` by `decode_select`.
    table_len: usize,
    /// 2026-10-13: `[ratio, hd]` BF16 ring: the incomplete block's raw indexer keys,
    /// row `pos % ratio` (L-K9a; was `[max_tokens, hd]`, 113 MB a sequence at 36,864).
    raw_tail: DevicePtr,
    /// 2026-09-25: `[max_tokens/ratio, hd]` BF16: this sequence's pooled block keys.
    block_keys: DevicePtr,
}

pub struct QsaIndexer {
    // 2026-09-25: `[(n_heads+1)*hd, hidden]` BF16 row-major, then the two
    // `[hd]` norm weights.
    qk_proj_w: DevicePtr,
    q_norm_w: DevicePtr,
    k_norm_w: DevicePtr,

    n_heads: u32,
    hd: u32,
    ratio: u32,
    budget: u32,
    block_topk: u32,
    rot: u32,
    theta: f32,
    eps: f32,
    hidden: u32,
    nkv_attn: u32,
    hd_attn: u32,
    max_tokens: usize,

    k_pool_k: KernelHandle,
    /// 2026-10-13: `qsa_ingest_step`, the decode ingest at the device position.
    k_ingest_k: KernelHandle,
    k_qprep_k: KernelHandle,
    k_score_k: KernelHandle,
    k_gather_k: KernelHandle,
    k_qprep_rows_k: KernelHandle,
    k_score_rows_k: KernelHandle,
    /// 2026-09-25: Tensor-core row scorer, bound with `try_kernel`: a null handle
    /// on a target without it, and `prefill_select` then uses the scalar
    /// `qsa_score_rows`.
    k_score_rows_tc_k: KernelHandle,
    k_prefill_attn_k: KernelHandle,
    /// 2026-10-14: `qsa_prefill_attn_tc`, zero off Hopper or with `METRALE_QSA_ATTN_TC=0`.
    k_prefill_attn_tc_k: KernelHandle,
    /// 2026-10-13: `dsa_topk_pools`, the device top-k of the prefill selection.
    k_topk_pools_k: KernelHandle,

    // 2026-09-25: Scratch shapes: `qk_scratch` [INGEST_SLAB, (n_heads+1)*hd]
    // BF16; 2026-10-13: `slab_raw` [INGEST_SLAB + ratio, hd] BF16;
    // `q_post` [n_heads, hd] F32; `scores_dev` [max_tokens/ratio] F32;
    // `sel_dev` [budget+ratio] i32; `k_scratch`/`v_scratch`
    // [budget+ratio, nkv_attn, hd_attn] BF16; `table_dev`
    // [ceil((budget+ratio)/8)] i32, enough for any block_size >= 8;
    // `seq_len_dev` [1] i32.
    qk_scratch: DevicePtr,
    /// 2026-10-13: One prefill slab's raw keys behind the sequence's incomplete block,
    /// from that block's first position: what `qsa_block_pool` pools them from.
    slab_raw: DevicePtr,
    q_post: DevicePtr,
    scores_dev: DevicePtr,
    sel_dev: DevicePtr,
    k_scratch: DevicePtr,
    v_scratch: DevicePtr,
    table_dev: DevicePtr,
    seq_len_dev: DevicePtr,
    /// 2026-09-25: The sequence's block table, `[ceil(max_tokens/8)]` i32,
    /// uploaded from the caller's host slice on each `prefill_select` call.
    prefill_table_dev: DevicePtr,
}

/// 2026-09-25: Prefill ingest GEMM slab (rows), bounding `qk_scratch`.
const INGEST_SLAB: usize = 2048;

/// 2026-10-13: Device bytes of one sequence's QSA carry for an indexer of head dim `hd`
/// and compress ratio `ratio` (> 0) sized for `max_tokens`: the `ratio`-row raw-tail ring
/// plus `max_tokens / ratio` block keys, BF16. What `new_seq_state` allocates and the
/// serve's preflight reserve charges.
pub fn seq_state_bytes_for(max_tokens: usize, hd: usize, ratio: usize) -> usize {
    ratio * hd * 2 + max_tokens / ratio * hd * 2
}

impl QsaIndexer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        qk_proj_w: DevicePtr,
        q_norm_w: DevicePtr,
        k_norm_w: DevicePtr,
        n_heads: usize,
        hd: usize,
        ratio: usize,
        budget: usize,
        rot: usize,
        theta: f32,
        eps: f32,
        hidden: usize,
        nkv_attn: usize,
        hd_attn: usize,
        max_tokens: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        anyhow::ensure!(
            ratio > 0 && budget.is_multiple_of(ratio),
            "QSA: budget % ratio != 0"
        );
        // 2026-10-13: `max_tokens` sizes every sequence's raw and block keys; the
        // serve derives it from `--max-seq-len` (`qsa_tokens::qsa_max_tokens`).
        anyhow::ensure!(max_tokens > 0, "QSA: max_tokens must be positive");
        let block_topk = budget / ratio;
        let qk_width = (n_heads + 1) * hd;
        let sel_cap = budget + ratio;
        Ok(Self {
            qk_proj_w,
            q_norm_w,
            k_norm_w,
            n_heads: n_heads as u32,
            hd: hd as u32,
            ratio: ratio as u32,
            budget: budget as u32,
            block_topk: block_topk as u32,
            rot: rot as u32,
            theta,
            eps,
            hidden: hidden as u32,
            nkv_attn: nkv_attn as u32,
            hd_attn: hd_attn as u32,
            max_tokens,
            k_pool_k: gpu.kernel("qsa_indexer", "qsa_block_pool")?,
            k_ingest_k: gpu.kernel("qsa_indexer", "qsa_ingest_step")?,
            k_qprep_k: gpu.kernel("qsa_indexer", "qsa_qprep")?,
            k_score_k: gpu.kernel("qsa_indexer", "qsa_score")?,
            k_gather_k: gpu.kernel("qsa_indexer", "qsa_gather")?,
            k_qprep_rows_k: gpu.kernel("qsa_indexer", "qsa_qprep_rows")?,
            k_score_rows_k: gpu.kernel("qsa_indexer", "qsa_score_rows")?,
            k_score_rows_tc_k: crate::layers::try_kernel(gpu, "qsa_indexer", "qsa_score_rows_tc"),
            k_prefill_attn_k: gpu.kernel("qsa_indexer", "qsa_prefill_attn")?,
            k_prefill_attn_tc_k: ops::QsaPrefillAttn::resolve_tc(gpu),
            k_topk_pools_k: gpu.kernel("dsa_indexer", "dsa_topk_pools")?,
            qk_scratch: gpu.alloc(INGEST_SLAB * qk_width * 2)?,
            slab_raw: gpu.alloc((INGEST_SLAB + ratio) * hd * 2)?,
            q_post: gpu.alloc(n_heads * hd * 4)?,
            scores_dev: gpu.alloc(max_tokens / ratio * 4)?,
            sel_dev: gpu.alloc(sel_cap * 4)?,
            k_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            v_scratch: gpu.alloc(sel_cap * nkv_attn * hd_attn * 2)?,
            table_dev: gpu.alloc(sel_cap.div_ceil(8) * 4)?,
            seq_len_dev: gpu.alloc(4)?,
            prefill_table_dev: gpu.alloc(max_tokens.div_ceil(8) * 4)?,
        })
    }

    /// 2026-09-25: One sequence's indexer carry: counters + raw/pooled key buffers.
    /// The launch scratch stays in the layer and is shared by every sequence.
    /// 2026-10-13: Its device bytes are [`Self::seq_state_bytes`].
    pub fn new_seq_state(&self, gpu: &dyn GpuBackend) -> Result<QsaSeqState> {
        let hd = self.hd as usize;
        let ratio = self.ratio as usize;
        Ok(QsaSeqState {
            ingested: 0,
            pooled: 0,
            step_start: 0,
            table_len: 0,
            // 2026-10-13: `seq_state_bytes_for` counts exactly these two.
            raw_tail: gpu.alloc(ratio * hd * 2)?,
            block_keys: gpu.alloc(self.max_tokens / ratio * hd * 2)?,
        })
    }

    /// 2026-10-13: Device bytes one sequence's carry holds (`new_seq_state`).
    pub fn seq_state_bytes(&self) -> usize {
        seq_state_bytes_for(self.max_tokens, self.hd as usize, self.ratio as usize)
    }

    /// 2026-09-25: Release one sequence's indexer carry.
    ///
    /// `QsaSeqState` holds bare `DevicePtr`s, so dropping the struct frees
    /// nothing; the raw-tail and block-key buffers are freed only here.
    ///
    /// Idempotent: each pointer is nulled once its free is attempted, so a
    /// second call (or a release after a partial failure) cannot double-free.
    /// A failure on the first buffer still attempts the second; the first
    /// error is returned.
    pub fn release_seq_state(&self, st: &mut QsaSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        let mut first_err: Option<anyhow::Error> = None;
        for p in [&mut st.raw_tail, &mut st.block_keys] {
            if p.is_null() {
                continue;
            }
            if let Err(e) = gpu.free(*p) {
                first_err.get_or_insert(e);
            }
            *p = DevicePtr(0);
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// 2026-09-25: The largest visible prefix whose selection is all-visible:
    /// with at most `budget + ratio - 1` visible tokens there are at most
    /// `budget / ratio` complete blocks, so every block is selected.
    pub fn inert_bound(&self) -> usize {
        (self.budget + self.ratio - 1) as usize
    }

    fn qk_width(&self) -> usize {
        (self.n_heads as usize + 1) * self.hd as usize
    }

    /// 2026-09-25: Ingest `num_tokens` prefill tokens starting at `seq_start`:
    /// project qk, store the raw keys, pool newly complete blocks.
    /// 2026-10-13: Per slab, the incomplete block's keys (the ring) and the slab's keys
    /// go back to back into `slab_raw`, every complete block there is pooled, and the
    /// keys of the new incomplete block go back to the ring: the same keys pooled by the
    /// same kernel as from a sequence-long buffer, so the block keys are bit-identical.
    /// `seq_start == 0` resets the counters. Errors when `seq_start` is not
    /// the ingested count (a prefix-cache skip) or the total exceeds
    /// `max_tokens` (the serve's `--max-seq-len`).
    pub fn prefill_ingest(
        &self,
        st: &mut QsaSeqState,
        hidden: DevicePtr,
        num_tokens: usize,
        seq_start: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if seq_start == 0 {
            st.ingested = 0;
            st.pooled = 0;
        }
        anyhow::ensure!(
            seq_start == st.ingested,
            "QSA: prefill chunk starts at {seq_start} but {} tokens are \
             ingested — a prefix-cache skip bypassed the indexer. Serve \
             qwen4_exp with the prefix cache disabled until QSA learns to \
             re-ingest cached prefixes.",
            st.ingested
        );
        anyhow::ensure!(
            seq_start + num_tokens <= self.max_tokens,
            "QSA: {} tokens exceeds the indexer's max_tokens={} (--max-seq-len)",
            seq_start + num_tokens,
            self.max_tokens
        );

        let hd = self.hd as usize;
        let ratio = self.ratio as usize;
        let row = hd * 2;
        let qkw = self.qk_width();
        let mut off = 0usize;
        while off < num_tokens {
            let ts = INGEST_SLAB.min(num_tokens - off);
            ops::cublas_bf16_proj_dense(
                hidden.offset((off) * self.hidden as usize * 2),
                self.qk_proj_w,
                self.qk_scratch,
                ts as u32,
                qkw as u32,
                self.hidden,
                stream,
            )
            .context("QSA qk projection (prefill)")?;
            let first = seq_start + off;
            let tail = first % ratio;
            if tail > 0 {
                gpu.copy_d2d_async(st.raw_tail, self.slab_raw, tail * row, stream)?;
            }
            // 2026-09-25: Raw key = the last hd columns of each row.
            gpu.copy_d2d_2d_async(
                self.qk_scratch.offset(self.n_heads as usize * hd * 2),
                qkw * 2,
                self.slab_raw.offset(tail * row),
                row,
                row,
                ts,
                stream,
            )?;
            let complete = (tail + ts) / ratio;
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                self.slab_raw,
                self.k_norm_w,
                st.block_keys,
                (first / ratio) as u32,
                complete as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            let rest = (tail + ts) % ratio;
            if rest > 0 {
                gpu.copy_d2d_async(
                    self.slab_raw.offset(complete * ratio * row),
                    st.raw_tail,
                    rest * row,
                    stream,
                )?;
            }
            off += ts;
        }
        st.ingested = seq_start + num_tokens;
        st.pooled = st.ingested / ratio;
        Ok(())
    }

    // 2026-09-25: `prefill_select` is in `qsa_select.rs`; 2026-10-13: `decode_select`
    // is in `qsa_decode.rs`.
}
