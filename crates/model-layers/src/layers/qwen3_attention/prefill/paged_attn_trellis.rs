// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Chunked-prefill attention over a `Trellis`-family paged cache: the layer's
//! trellis state launches its causal prefill kernel over the chunk's block table.
//!
//! Owner: model-layers (attention).
//! Invariants: runs only for a layer that built trellis state; a sliding window is refused by
//! that state with a coded error.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use super::super::trellis_kv::{TrellisKv, TrellisPrefill};
use super::paged_attn::{PagedAttnArgs, PagedAttnOutcome};
use crate::layer::ForwardContext;

impl Qwen3AttentionLayer {
    /// 2026-10-18: Runs the chunk's attention into `args.attn_out` through `t` and returns
    /// `Continue`.
    pub(super) fn prefill_attention_paged_trellis(
        &self,
        t: &TrellisKv,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        args: &PagedAttnArgs,
    ) -> Result<PagedAttnOutcome> {
        let (k, v) = self.kv_sides(kv_cache)?;
        let p = TrellisPrefill {
            q: args.q_contiguous,
            out: args.attn_out,
            block_table: args.meta.block_table,
            seq_len_start: u32::try_from(args.seq_len_start)?,
            n: args.n,
            nq: args.nq,
            nkv: args.nkv,
            hd: args.hd,
            bs: u32::try_from(args.bs)?,
            inv_sqrt_d: args.inv_sqrt_d,
            sliding_window: self.sliding_window.unwrap_or(0),
            k,
            v,
        };
        t.prefill(ctx.gpu, &p, args.stream)?;
        Ok(PagedAttnOutcome::Continue)
    }
}
