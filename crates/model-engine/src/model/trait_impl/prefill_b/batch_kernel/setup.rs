// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The per-stream setup of a batched prefill wave (moved out of
//! `prefill_batch_chunk_kernel_batched` unchanged): KV pre-flight, prefix reservation, buffer
//! zeroing, then per stream at its packed offset the embed, prefix lookup, block allocation,
//! processing range and metadata upload. Shared by the kernel-batched wave and the exact wave
//! (`wave.rs`).
//!
//! Owner: model-engine.
//! Invariants:
//! - Both `Return` outcomes come before any buffer is zeroed or any sequence is changed.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::super::types::TransformerModel;
use super::super::proc_range::ProcRange;
use super::phases::{Flow, PerStreamMeta};
use super::{KernelBatchResult, varlen_prefill_enabled};
use crate::traits::PrefillSlice;

/// 2026-10-05: A finished setup: per-stream results, the scratch cursor past the per-stream
/// metadata, the packed row count, the shared MRoPE flag and the packed buffers' bases.
pub(in crate::model) struct StreamSetup {
    pub(in crate::model) per_stream: Vec<PerStreamMeta>,
    pub(in crate::model) scratch_cursor: usize,
    pub(in crate::model) running_proc_off: usize,
    pub(in crate::model) use_mrope: bool,
    pub(in crate::model) hidden_base: DevicePtr,
    pub(in crate::model) residual_base: DevicePtr,
}

/// 2026-10-05: The setup's outcome: ready, or a result the caller returns as is.
pub(in crate::model) enum SetupFlow {
    Ready(StreamSetup),
    Return(KernelBatchResult),
}

impl TransformerModel {
    /// 2026-10-05: The setup of `streams` on `stream`, under the caller's KV lock.
    pub(in crate::model) fn batched_stream_setup(
        &self,
        streams: &mut [PrefillSlice<'_>],
        kv_cache: &mut PagedKvCache,
        stream: u64,
    ) -> Result<SetupFlow> {
        let n = streams.len();
        let chunk_len = streams[0].chunk_len;
        let is_last_chunk = streams[0].is_last_chunk;
        let h = self.config.hidden_size;
        let dtype_bytes = 2usize;
        let varlen = varlen_prefill_enabled();
        // 2026-09-25: Stream b's hidden rows are packed at Σ `proc_count` of
        // the earlier streams, the layout the staged `BatchedAttnMetadata` and
        // `GdnPrefillBuffers.total_len` use. `proc_count` is known only after
        // `proc_range`, so the offset is a running sum advanced at the end of
        // each setup iteration.
        let mut running_proc_off = 0usize;
        let arena_cap_tokens = self.buffers.max_batch_tokens();
        // 2026-09-25: Per-stream scratch slots are sized for the longest chunk,
        // so a longer stream's metadata fits its slot.
        let max_chunk_len = streams
            .iter()
            .map(|s| s.chunk_len)
            .max()
            .unwrap_or(chunk_len);


        // 2026-09-25: Allocation pre-flight. Setup allocates KV blocks stream by
        // stream (`ensure_blocks_through_prefill`), and an error there fails the
        // whole admitted batch (batch.rs returns it). So the batch's total block
        // need is checked first, evicting prefix-cache blocks if needed; if it
        // still does not fit, the batch is not admitted and runs per-stream.
        // This runs before the prefix reservation, so a decline has nothing to
        // release. Block counts use the read-only `peek_matched_tokens` probe.
        if let Flow::Return(r) = self.kv_preflight(streams, kv_cache) {
            return Ok(SetupFlow::Return(r));
        }

        // 2026-09-25: The prefix reservation runs while no sequence has been
        // changed. A refused reservation has released its radix references
        // and the batch runs per-stream; an admitted one is consumed by the
        // setup loop instead of a second cache walk.
        let reserved_prefix_matches =
            match self.prefill_b_reserve_batched_prefix_matches(streams, kv_cache.block_size()) {
                Some(matches) => matches,
                None => return Ok(SetupFlow::Return(KernelBatchResult::NotAdmitted)),
            };

        // 2026-09-25: Zero the shared buffers once for the whole batch, with the
        // rule `prefill_chunk_dispatch` uses.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if streams[0].chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let hidden_base = self.buffers.hidden_states();
        let _residual_base = self.buffers.residual();

        // 2026-09-25: Setup, stream by stream, at packed offsets. Each stream's
        // metadata goes to its own scratch slice; the stacked
        // `BatchedAttnMetadata` is staged after all of them.
        let mut per_stream: Vec<PerStreamMeta> = Vec::with_capacity(n);
        // 2026-09-25: Chunk-0 batches are admitted by `prefill_batched_chunk_zero_admitted`, so
        // the paged upload fires on the same condition; otherwise a chunk-0
        // stream's `block_table_dev` would stay NULL.
        let force_paged_first_chunk = streams[0].chunk_start == 0
            && metrale_model_layers::layers::ops::prefill_batched_chunk_zero_admitted();

        let mut use_mrope: Option<bool> = None;
        let mut needs_paged: Option<bool> = None;

        // 2026-09-25: Per-stream metadata slot: 16 bytes per token of the
        // longest chunk plus 64, and at least 4096 bytes. The admission sizing
        // (`q12_batched_scratch_bytes_varlen`) uses the same formula.
        let per_stream_meta_bytes = ((max_chunk_len * 16) + 64).max(4096);
        // 2026-09-25: The scratch cursor starts after the MoE top-k staging
        // area, sized for Σ `chunk_len` with varlen (packed like the hidden rows)
        // and for n times the longest chunk otherwise.
        let moe_scratch_tokens = if varlen {
            streams.iter().map(|slice| slice.chunk_len).sum()
        } else {
            max_chunk_len * n
        };
        let moe_scratch_bytes = moe_scratch_tokens * self.config.num_experts_per_tok * 4 * 2;
        let mut scratch_cursor = (moe_scratch_bytes + 63) & !63;

        for (b, slice) in streams.iter_mut().enumerate() {
            let tokens = slice.prompt_tokens;
            let chunk_start = slice.chunk_start;
            // 2026-09-25: This stream's chunk length; streams differ only with varlen.
            let cl = slice.chunk_len;
            let total = tokens.len();
            let seq = &mut *slice.seq;

            // 2026-09-25: Embed at this stream's packed offset. On a partial
            // cache hit `proc_range` re-embeds the processed suffix at the same
            // offset. An embed longer than `proc_count` spills into the next
            // streams' regions, which those streams write later, so each region
            // is last written by its own stream.
            let proc_off_b = running_proc_off;
            let hidden_b = hidden_base.offset(proc_off_b * h * dtype_bytes);
            // 2026-09-25: With `METRALE_Q12_EFFECTIVE_ARENA=1` the probed cached
            // prefix is not embedded: `proc_range` re-embeds the uncached suffix
            // at this offset, and admission charged the arena for the suffix
            // only.
            let embed_skip = if super::eligible::effective_arena_charge_enabled() {
                let bs_probe = kv_cache.block_size();
                self.prefix_cache
                    .peek_matched_tokens(tokens, bs_probe, seq.adapter_id)
                    .saturating_sub(chunk_start)
                    .min(cl)
            } else {
                0
            };
            if embed_skip < cl {
                self.prefill_b_embed_chunk_at(
                    tokens,
                    chunk_start + embed_skip,
                    cl - embed_skip,
                    hidden_b,
                    stream,
                )?;
            }

            // 2026-09-25: Prefix-cache lookup, EP agreement and Marconi restore. A
            // chunk-0 stream with prefix caching on consumes its reservation.
            let reserved_match = if self.prefix_cache.is_active() && chunk_start == 0 {
                Some(reserved_prefix_matches[b].clone())
            } else {
                None
            };
            let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
                tokens,
                seq,
                chunk_start,
                total,
                kv_cache,
                stream,
                reserved_match,
            )?;

            let bs = kv_cache.block_size();
            let end_pos = chunk_start + cl;
            let blocks_needed = (end_pos - 1) / bs + 1;
            crate::model::block_mgmt::ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;

            // 2026-09-25: Processing range. This stream's hidden offset is passed
            // so a cache-hit re-embed lands in its own region.
            let (proc_start, proc_count, effective_seq_len_start) = match self
                .prefill_b_proc_range(
                    tokens,
                    seq,
                    chunk_start,
                    cl,
                    is_last_chunk,
                    kv_write_start,
                    marconi_skip,
                    hidden_b,
                    stream,
                )? {
                ProcRange::Compute {
                    proc_start,
                    proc_count,
                    effective_seq_len_start,
                } => (proc_start, proc_count, effective_seq_len_start),
                ProcRange::EarlyReturn(_) => anyhow::bail!(
                    "kernel-batched: stream {b} early-returned during proc_range \
                         — eligibility check missed this. Caller should fall back."
                ),
            };

            // 2026-09-25: Every stream must match stream 0's
            // `effective_seq_len_start`, and, without varlen, its `proc_count`;
            // otherwise the batch fails with an error.
            if b > 0 {
                if !varlen && per_stream[0].proc_count != proc_count {
                    anyhow::bail!(
                        "kernel-batched: stream {b} proc_count={} differs from \
                         stream 0 proc_count={}. Caller should fall back.",
                        proc_count,
                        per_stream[0].proc_count
                    );
                }
                if per_stream[0].effective_seq_len_start != effective_seq_len_start {
                    anyhow::bail!(
                        "kernel-batched: stream {b} effective_seq_len_start={} \
                         differs from stream 0={}. Caller should fall back.",
                        effective_seq_len_start,
                        per_stream[0].effective_seq_len_start
                    );
                }
            }

            let meta_base = self.buffers.scratch().offset(scratch_cursor);
            // 2026-09-25: The upload may use scratch from `scratch_cursor` to its
            // end; the cursor then advances by `per_stream_meta_bytes`.
            let meta_region_bytes = self.buffers.scratch_bytes().saturating_sub(scratch_cursor);
            let layout = self.prefill_b_upload_meta_at(
                tokens,
                seq,
                chunk_start,
                cl,
                proc_start,
                proc_count,
                effective_seq_len_start,
                kv_cache,
                meta_base,
                meta_region_bytes,
                stream,
            )?;
            if layout.needs_paged || force_paged_first_chunk {
                self.prefill_b_upload_paged(
                    seq,
                    total,
                    proc_start,
                    proc_count,
                    meta_base,
                    layout.slot_offset,
                    kv_cache,
                    stream,
                )?;
            }
            scratch_cursor += per_stream_meta_bytes;

            // 2026-09-25: Stream 0 sets the MRoPE and paged flags; a later stream
            // that differs fails the batch.
            match (use_mrope, layout.use_mrope) {
                (None, m) => use_mrope = Some(m),
                (Some(prev), m) if prev != m => {
                    anyhow::bail!("kernel-batched: stream {b} use_mrope={m} mismatch with stream 0")
                }
                _ => {}
            }
            match (needs_paged, layout.needs_paged) {
                (None, p) => needs_paged = Some(p),
                (Some(prev), p) if prev != p => anyhow::bail!(
                    "kernel-batched: stream {b} needs_paged={p} mismatch with stream 0"
                ),
                _ => {}
            }

            let kv_write_start_eff = if marconi_skip { 0 } else { kv_write_start };
            let (block_table_dev, seq_len_dev) = if layout.needs_paged || force_paged_first_chunk {
                let page_meta = seq.chunked_prefill_meta.as_ref().unwrap();
                (page_meta.block_table, page_meta.seq_len)
            } else {
                (DevicePtr::NULL, DevicePtr::NULL)
            };
            let num_blocks = seq.block_table.len();

            per_stream.push(PerStreamMeta {
                chunk_start,
                proc_start,
                proc_count,
                effective_seq_len_start,
                kv_write_start_eff,
                block_table_dev,
                seq_len_dev,
                num_blocks,
                proc_off: proc_off_b,
                kv_write_start,
                marconi_skip,
                layout,
            });
            // 2026-09-25: Admission charged the arena from probed matches, and an
            // eviction during this setup can shrink a later stream's match, so
            // the real packed offset is checked against the arena before it
            // advances. The error is returned to the caller.
            if running_proc_off + proc_count > arena_cap_tokens {
                anyhow::bail!(
                    "Q12 batched staging overran the arena: stream {b} needs                      {proc_count} tokens at offset {running_proc_off} > cap                      {arena_cap_tokens} (prefix match shrank after pre-flight)"
                );
            }
            running_proc_off += proc_count;
        }
        Ok(SetupFlow::Ready(StreamSetup {
            per_stream,
            scratch_cursor,
            running_proc_off,
            use_mrope: use_mrope.unwrap_or(false),
            hidden_base,
            residual_base: _residual_base,
        }))
    }
}
