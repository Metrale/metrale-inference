// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: PLE over the rows of a multi-sequence decode step: one host hash
//! and one row gather for every live sequence, the projections and the gate
//! over all rows, the dilated conv per row against its own carry, one highway
//! add. `next_ids` is the host half every PLE entry point shares (`forward`,
//! `prestage`, `forward_rows`).
//!
//! Owner: model-layers (PLE).
//! Invariants:
//! - `next_ids` is called with a complete history (`context_len` tokens); the
//!   GPU entry points reset an incomplete carry first.
//! - `forward_rows` injects into row `i` of `highway` exactly what `forward_row`
//!   injects for that sequence alone: the gate and the conv are per token, the
//!   projections have no row-dependent reduction, and each row's conv runs
//!   against its own state.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::ids::{PleIdDims, ple_ngram_ids};
use super::{PleLayer, PleSeqState};
use crate::layer::ForwardContext;

/// 2026-10-13: The ids of `tokens` appended to `history`, hashed together and
/// sliced to the new tokens (the reference's `[:, -T:]`), flat `[T * heads]`;
/// `history` advances to the last `context_len` tokens of the window.
pub(in crate::layers::ple) fn next_ids(
    dims: &PleIdDims,
    history: &mut Vec<u32>,
    tokens: &[u32],
) -> Vec<u64> {
    let mut window = std::mem::take(history);
    window.extend_from_slice(tokens);
    let all = ple_ngram_ids(dims, &window);
    let flat = all[all.len() - tokens.len()..]
        .iter()
        .flat_map(|r| r.iter().copied())
        .collect();
    let keep = dims.context_len();
    *history = window[window.len() - keep..].to_vec();
    flat
}

impl PleLayer {
    /// 2026-10-13: The host half of one batched decode step, before its CUDA-graph
    /// replay or capture: every row's ids hashed, one slot upload for the batch (row `i`
    /// at slot row `i`), every history advanced. `forward_rows` (or a replayed graph,
    /// after which `replayed` clears the stage) consumes it.
    pub fn prestage_rows(
        &self,
        carries: &mut [&mut PleSeqState],
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let n = carries.len();
        anyhow::ensure!(
            tokens.len() == n,
            "PLE prestage rows: {} tokens for {n} carries",
            tokens.len()
        );
        anyhow::ensure!(
            n <= self.max_tokens,
            "PLE prestage rows: {n} rows overflow the {}-row slot buffer",
            self.max_tokens
        );
        let mut flat = Vec::with_capacity(n * self.dims.ngram_heads());
        for (st, &tok) in carries.iter_mut().zip(tokens) {
            if st.history.len() != self.dims.context_len() {
                self.reset(st, gpu, stream)?;
            }
            flat.extend(next_ids(&self.dims, &mut st.history, &[tok]));
        }
        let va = self.gather_host(&flat, gpu, stream)?;
        for (i, st) in carries.iter_mut().enumerate() {
            st.prestaged_va = Some(va);
            st.last_staged_va = va;
            st.staged_row = Some(i);
        }
        Ok(())
    }

    /// 2026-10-13: A replayed CUDA graph ran this carry's staged step, so the stage is
    /// spent.
    pub fn replayed(&self, st: &mut PleSeqState) {
        st.prestaged_va = None;
    }

    /// 2026-10-13: Inject one decode token per sequence into `highway`
    /// `[n, hc_mult * hidden]` FP32 (row `i` is `carries[i]`'s), in place.
    /// `tokens[i]` is sequence `i`'s token. It consumes the rows `prestage_rows`
    /// staged; unstaged, it hashes and uploads itself, which graph capture refuses.
    pub fn forward_rows(
        &self,
        carries: &mut [&mut PleSeqState],
        highway: DevicePtr,
        tokens: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let n = carries.len();
        anyhow::ensure!(
            tokens.len() == n,
            "PLE rows: {} tokens for {n} carries",
            tokens.len()
        );
        anyhow::ensure!(
            n <= self.max_tokens,
            "PLE: {n} rows exceed the {} this layer was sized for. Raise \
             METRALE_PLE_MAX_TOKENS or lower the decode batch.",
            self.max_tokens
        );
        if n == 0 {
            return Ok(());
        }
        let c = self.hc_mult * self.hidden;
        let heads = self.dims.ngram_heads();
        let gpu = ctx.gpu;

        // 2026-10-13: Rows staged by `prestage_rows` for this step are consumed: their
        // slots are uploaded and their histories advanced, so only the kernels remain,
        // which a CUDA graph may record. A single-sequence stage is a leftover of an
        // earlier step and is dropped, as `forward_row` drops it.
        let staged: Vec<Option<u64>> = carries
            .iter_mut()
            .enumerate()
            .map(|(i, st)| {
                let va = st.prestaged_va.take();
                va.filter(|_| st.staged_row == Some(i))
            })
            .collect();
        if let Some(Some(va)) = staged.first().copied() {
            anyhow::ensure!(
                staged.iter().all(|s| *s == Some(va)),
                "PLE rows: the batch was staged in part ({} of {n} rows)",
                staged.iter().filter(|s| s.is_some()).count()
            );
            self.gather_embed(va, n, heads, gpu, stream)?;
        } else {
            anyhow::ensure!(
                staged.iter().all(Option::is_none),
                "PLE rows: the batch was staged in part (row 0 is not)"
            );
            anyhow::ensure!(
                !ctx.graph_capture,
                "PLE rows: the pageable slot upload cannot run inside CUDA graph capture; \
                 stage the rows first (`prestage_rows`)"
            );
            let mut flat = Vec::with_capacity(n * heads);
            for (st, &tok) in carries.iter_mut().zip(tokens) {
                if st.history.len() != self.dims.context_len() {
                    self.reset(st, gpu, stream)?;
                }
                flat.extend(next_ids(&self.dims, &mut st.history, &[tok]));
            }
            self.gather(&flat, n, heads, gpu, stream)?;
        }

        self.project(n, gpu, stream)?;
        self.gate(highway, n, gpu, stream)?;
        for (i, st) in carries.iter().enumerate() {
            let off = i * c * 4;
            self.conv(
                self.gated_normed.offset(off),
                self.gated.offset(off),
                st.conv,
                self.out.offset(off),
                1,
                gpu,
                stream,
            )?;
        }
        self.add_highway(highway, n, gpu, stream)
    }
}
