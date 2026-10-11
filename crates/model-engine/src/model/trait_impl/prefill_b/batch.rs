// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched prefill for N concurrent streams, `prefill_batch_chunk_dispatch`.
//!
//! A batch that `kernel_batched_eligible` accepts, with `METRALE_Q12_BATCHED`
//! not set to 0 or false, goes to the kernel-batched path in `batch_kernel.rs`.
//! Otherwise, or when that path returns `NotAdmitted`, the streams run one
//! after another through the same step functions as `prefill_chunk_dispatch`,
//! with the KV-cache lock taken once for the whole loop.
//!
//! Owner: model-engine.
//! Invariants:
//! - In the per-stream loop a stream whose prefill fails gets
//!   `DevicePtr::NULL` and the remaining streams still run.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::batch_kernel::KernelBatchResult;
use super::proc_range::ProcRange;
use super::upload_meta::MetaLayout;
use crate::traits::{Model, PrefillSlice, SequenceState};

impl TransformerModel {
    /// 2026-09-25: Batched-prefill dispatch for N concurrent streams.
    ///
    /// On the per-stream path the result is parallel to `streams`: the stream's
    /// last-token logits pointer when its chunk is the last, `DevicePtr::NULL`
    /// otherwise or when the stream failed.
    ///
    /// `row_base` shifts each finishing stream's logits row to `row_base +
    /// stream_idx`. It is 0 for a prefill-only step and the number of decode
    /// rows inside a mixed step, whose rows `0..n_decode` belong to decode
    /// lanes (see `Model::prefill_batch_chunk_rows`).
    pub(in crate::model) fn prefill_batch_chunk_dispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        let n = streams.len();
        // 2026-09-25: `METRALE_NO_PREFILL_ROW_SHIFT` set to 1 or true forces
        // `row_base = 0`, which puts prefill logits on the decode lanes' rows.
        // When the logits buffer cannot hold rows `row_base..row_base + n`,
        // `row_base` falls back to 0 with a warning rather than writing past it.
        let shift_disabled = std::env::var("METRALE_NO_PREFILL_ROW_SHIFT")
            .map(|v| v == "1" || v.to_lowercase() == "true")
            .unwrap_or(false);
        let row_base = if shift_disabled { 0 } else { row_base };
        let logits_rows = self.buffers.sizes().logits / (self.config.vocab_size * 2);
        let row_base = if row_base + n > logits_rows {
            tracing::warn!(
                "Batched prefill: logits arena holds {logits_rows} rows, need \
                 row_base={row_base}+n={n}; falling back to row_base=0 \
                 (decode lanes may alias prefill rows this step)"
            );
            0
        } else {
            row_base
        };
        tracing::debug!(
            target: "metrale::q12",
            n = n,
            "prefill_batch_chunk_dispatch entry"
        );
        if n == 0 {
            return Ok(Vec::new());
        }
        // 2026-10-05: A single stream runs the single-stream prefill into its logits row, then
        // the eager drafter prefill, as `impl_forward`'s single-stream prefill does.
        if n == 1 {
            let s = &mut streams[0];
            let logits = self.prefill_chunk_dispatch_row(
                s.prompt_tokens,
                s.seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                row_base,
                stream,
            )?;
            self.try_eager_drafter_prefill(s.seq, s.is_last_chunk, stream);
            return Ok(vec![logits]);
        }

        let arena_cap = self.buffers.max_batch_tokens();
        for (i, s) in streams.iter().enumerate() {
            if s.chunk_len > arena_cap {
                anyhow::bail!(
                    "Batched prefill stream {i} chunk_len={} exceeds arena \
                     capacity {arena_cap}. Reduce --max-prefill-tokens.",
                    s.chunk_len
                );
            }
        }

        // 2026-09-25: Kernel-batched path, when `kernel_batched_eligible`
        // accepts the batch. `NotAdmitted` falls through to the per-stream loop
        // below; an error is returned to the caller. `METRALE_Q12_BATCHED` set
        // to 0 or false disables the path; unset or any other value leaves it on.
        let q12_batched_enabled = std::env::var("METRALE_Q12_BATCHED")
            .map(|v| v != "0" && v.to_lowercase() != "false")
            .unwrap_or(true);
        if q12_batched_enabled && self.kernel_batched_eligible(streams) {
            tracing::debug!(
                target: "metrale::q12",
                n = n,
                chunk_len = streams[0].chunk_len,
                is_last_chunk = streams[0].is_last_chunk,
                "Q12 kernel-batched dispatch attempt"
            );
            match self.prefill_batch_chunk_kernel_batched(streams, stream, row_base) {
                Ok(KernelBatchResult::Completed(v)) => {
                    // 2026-09-25: Logged at info, once per completed kernel-batched
                    // prefill. `total_tokens` is the sum of `chunk_len`.
                    let total: usize = streams.iter().map(|s| s.chunk_len).sum();
                    tracing::info!(
                        target: "metrale::q12",
                        n = n,
                        total_tokens = total,
                        "Q12 kernel-batched prefill dispatched (fused large-M)"
                    );
                    return Ok(v);
                }
                Ok(KernelBatchResult::NotAdmitted) => {
                    tracing::info!(
                        target: "metrale::q12",
                        "Q12 kernel-batched cache plan not admitted → falling back to per-stream"
                    );
                }
                // 2026-09-25: An admitted batch can already own KV and sequence
                // state, and a per-stream retry would allocate or restore it a
                // second time, so the error is returned.
                Err(e) => return Err(e),
            }
        } else if !q12_batched_enabled {
            tracing::trace!(
                target: "metrale::q12",
                "Q12 kernel-batched disabled via METRALE_Q12_BATCHED=0"
            );
        } else {
            // 2026-09-25: Ineligible: log the batch shape, at info when varlen
            // batching is on (`varlen_prefill_enabled`) and at debug otherwise.
            let chunk_lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
            let chunk_starts: Vec<usize> = streams.iter().map(|s| s.chunk_start).collect();
            let total: usize = chunk_lens.iter().sum();
            if super::batch_kernel::varlen_prefill_enabled() {
                tracing::info!(
                    target: "metrale::q12",
                    n = n,
                    chunk_lens = ?chunk_lens,
                    chunk_starts = ?chunk_starts,
                    total = total,
                    arena_cap = self.buffers.max_batch_tokens(),
                    head_dim = self.config.head_dim,
                    model_type = self.config.model_type.as_str(),
                    "Q12 kernel-batched ineligible — falling back to per-stream"
                );
            } else {
                tracing::debug!(
                    target: "metrale::q12",
                    n = n,
                    chunk_lens = ?chunk_lens,
                    chunk_starts = ?chunk_starts,
                    total = total,
                    arena_cap = self.buffers.max_batch_tokens(),
                    head_dim = self.config.head_dim,
                    model_type = self.config.model_type.as_str(),
                    "Q12 kernel-batched ineligible — falling back to per-stream"
                );
            }
        }

        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        Ok(self.prefill_streams_serial(streams, row_base, stream))
    }

    /// 2026-10-05: The streams one after another, each by the single-stream prefill into logits
    /// row `row_base + i`. A stream that fails gets NULL logits and the others go on.
    pub(in crate::model) fn prefill_streams_serial(
        &self,
        streams: &mut [PrefillSlice<'_>],
        row_base: usize,
        stream: u64,
    ) -> Vec<DevicePtr> {
        let mut logits_out: Vec<DevicePtr> = Vec::with_capacity(streams.len());
        for (stream_idx, slice) in streams.iter_mut().enumerate() {
            // 2026-09-25: A stream whose prefill fails gets NULL logits and the
            // loop goes on with the others.
            // 2026-10-05: Each stream runs the single-stream prefill
            // (`prefill_chunk_dispatch_row`: its tail split and in-pass SSM capture included) into
            // its own logits row, then the single-stream eager drafter prefill
            // (`impl_forward`), so a stream's bits do not depend on how it was scheduled. This
            // path used to re-implement the pass without the tail split.
            let is_last_chunk = slice.is_last_chunk;
            let stream_res = self.prefill_chunk_dispatch_row(
                slice.prompt_tokens,
                slice.seq,
                slice.chunk_start,
                slice.chunk_len,
                is_last_chunk,
                row_base + stream_idx,
                stream,
            );
            if stream_res.is_ok() {
                self.try_eager_drafter_prefill(slice.seq, is_last_chunk, stream);
            }
            match stream_res {
                Ok(l) => logits_out.push(l),
                Err(e) => {
                    tracing::error!(
                        "Batched prefill fallback: stream {stream_idx} failed: {e:#} \
                         — isolating (NULL logits; only this stream fails, batch continues)"
                    );
                    logits_out.push(DevicePtr::NULL);
                }
            }
        }
        logits_out
    }
}
