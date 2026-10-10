// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Chunked prefill, `prefill_chunk_dispatch`: one chunk of a prompt through the step modules under `prefill_b/`.
//!
//! Steps, in order: `embed_chunk` (embedding and vision-pad overlay),
//! `prefix_lookup` (prefix cache, EP agreement, Marconi restore), `proc_range`
//! (processing range; may return early), `upload_meta` and `upload_paged`
//! (metadata upload), `forward_layers`, then `finalize_last` on the last chunk
//! or `save_checkpoint` on any other. Once taken, the `kv_cache` lock is held
//! for the rest of the chunk and passed to each step as `&mut`. The multi-stream
//! path is in `batch.rs` and `batch_kernel.rs`.
//!
//! Owner: model-engine.
//! Invariants:
//! - A chunk that returns `Ok`, including a fully cached one, appends its tokens
//!   to `seq.tokens` and sets `seq.seq_len` to `chunk_start + chunk_len`.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::{Model, SequenceState};

mod batch;
mod batch_kernel;
#[cfg(test)]
mod batch_kernel_tests;
mod batched_layer;
mod embed_chunk;
mod exact_leaf;
mod finalize_last;
mod forward_layers;
mod h_state_ptrs;
mod midchunk_capture;
mod prefix_lookup;
mod prefix_reserve;
mod proc_range;
mod prompt_logprobs;
mod save_checkpoint;
mod snap_agree;
#[cfg(test)]
mod snap_agree_tests;
mod spans;
pub(in crate::model) mod spans_wire;
mod stage_batched;
mod upload_meta;
mod upload_paged;

impl TransformerModel {
    /// 2026-09-27: The token at which this prompt's prefill is split so that an SSM
    /// snapshot lands at `prefill_plan::tail_split_point`: `Some` for an SSM model with
    /// snapshots and prefix caching on and no vision pads in the prompt.
    /// `METRALE_NO_TAIL_SPLIT=1` turns the split off.
    pub(in crate::model) fn prefill_tail_split_dispatch(&self, tokens: &[u32]) -> Option<usize> {
        if self.config.num_ssm_layers() == 0
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || std::env::var("METRALE_NO_TAIL_SPLIT").as_deref() == Ok("1")
            || self.tokens_have_vision_pad(tokens)
        {
            return None;
        }
        crate::prefill_plan::tail_split_point(tokens.len(), self.kv_cache.lock().block_size())
    }

    pub(super) fn prefill_chunk_dispatch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        assert!(
            chunk_start + chunk_len <= total,
            "chunk_start({chunk_start}) + chunk_len({chunk_len}) > total({total})"
        );

        // 2026-09-25: Tail-checkpoint split. A later turn's prefix match is
        // block-aligned, and a snapshot deeper than the match cannot be
        // restored. A last chunk that spans the split point
        // (`prefill_tail_split_dispatch`) is split there once, so
        // `prefill_b_save_checkpoint` saves a snapshot at it. 2026-09-27: the
        // scheduler ends a non-last chunk at the same point
        // (`prefill_plan::plan_chunk_len`), so only a last chunk can span it
        // here. The split does not depend on radix contents, so a prompt is
        // processed in the same passes cold and warm, and on every rank.
        //
        // 2026-10-01: A pass that computes from below the split point captures the SSM
        // state there inside the pass instead (`prepare_cut_capture`), and at the tail
        // boundary one block above it, so the prompt tail is not run as a second pass
        // over every layer's weights. Measured on GB10 (dgx3, Qwen3.6-35B-A3B-FP8,
        // 299-token prompt): the second pass covered 27 tokens and took 68.6 of 229.7 ms
        // of prefill, about 60% of the expert weights read again. A request that restores
        // at or past the split point (a warm repeat, which now restores at the tail
        // boundary and replays 2 to 16 tokens) keeps the two-pass path. The passes then
        // differ cold and warm, but the bits do not: the replayed rows run the same
        // kernels in both (`replay_tail`, `prefill_row_invariant`).
        if is_last_chunk
            && let Some(cut) = self.prefill_tail_split_dispatch(tokens)
            && cut > chunk_start
        {
            if self.inpass_cut_capture_supported(seq)
                && !self.prefill_restores_through(tokens, seq, chunk_start, cut, stream)?
            {
                return self.prefill_chunk_pass(
                    tokens,
                    seq,
                    chunk_start,
                    chunk_len,
                    true,
                    Some(cut),
                    stream,
                );
            }
            self.prefill_chunk_dispatch(
                tokens,
                seq,
                chunk_start,
                cut - chunk_start,
                false,
                stream,
            )?;
            return self.prefill_chunk_dispatch(tokens, seq, cut, total - cut, true, stream);
        }
        self.prefill_chunk_pass(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            None,
            stream,
        )
    }

    /// 2026-10-01: Whether this prefill's chunk-0 prefix lookup restored SSM state at or
    /// past `cut`, running the lookup now when chunk 0 is this call; the chunk then
    /// replays that decision (`prefill_b_prefix_lookup`, its re-entry rule). Such a
    /// request keeps the two-pass split, whose first part is fully cached and returns
    /// early. The lookup's outcome decides, not the radix match: a match the lookup does
    /// not restore from (a short prompt, another session) recomputes from token 0, and
    /// must do so in the same single pass as a cold request.
    fn prefill_restores_through(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        cut: usize,
        stream: u64,
    ) -> Result<bool> {
        if chunk_start == 0 && !seq.prefix_lookup_applied {
            let mut kv_cache = self.kv_cache.lock();
            self.prefill_b_prefix_lookup(
                tokens,
                seq,
                0,
                tokens.len(),
                &mut kv_cache,
                stream,
                None,
            )?;
        }
        Ok(seq.prefix_lookup_skip && seq.marconi_skip_to >= cut)
    }

    /// 2026-10-01: Whether a prefill pass may capture the tail split point's SSM state
    /// in-pass for `seq`. Needs: the mid-chunk capture switch on
    /// (`--no-ssm-tail-midchunk` turns it off, `ssm_tail_midchunk_enabled`); a build
    /// other than `metrale_scale`, whose capture is `prepare_midchunk_capture`; prefill
    /// kernels chosen without regard to row count (`prefill_row_invariant`), so a warm
    /// request replaying the tail gives its rows the cold pass's bits; every layer
    /// honouring the replay-tail split (`supports_replay_tail_split`); no aux layer
    /// state, which a snapshot takes at the end of a pass; one rank; and an FP32 h state,
    /// so the captured bytes equal what `SsmSnapshotPool::save` writes after a pass
    /// ending there.
    fn inpass_cut_capture_supported(&self, seq: &SequenceState) -> bool {
        !cfg!(metrale_scale)
            && metrale_gpu_runtime::ssm_tail_midchunk_enabled()
            && !metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled()
            && !self.seq_ssm_h_is_f16(seq)
            && !self.requires_aux_state()
            && !self.multi_rank_protocol_active()
            && metrale_model_layers::layers::prefill_row_invariant()
            && self.layers.iter().all(|l| l.supports_replay_tail_split())
    }

    /// 2026-10-01: One prefill pass over `tokens[chunk_start..chunk_start + chunk_len]`.
    /// `cut_capture` is the tail split point when the pass is to capture its SSM state
    /// in-pass (`prefill_chunk_dispatch`).
    fn prefill_chunk_pass(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        chunk_len: usize,
        is_last_chunk: bool,
        cut_capture: Option<usize>,
        stream: u64,
    ) -> Result<DevicePtr> {
        let total = tokens.len();
        let arena_cap = self.buffers.max_batch_tokens();
        if chunk_len > arena_cap {
            anyhow::bail!(
                "Prefill chunk ({chunk_len} tokens) exceeds buffer arena capacity ({arena_cap} tokens). \
                 Reduce --max-prefill-tokens or prompt length."
            );
        }

        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        // 2026-09-25: With `comm` set, every buffer is zeroed on every chunk.
        // Otherwise only the first chunk zeroes, and only the prefill
        // essentials; later chunks rely on the embedding and the layer forward
        // writing each buffer before it is read.
        if self.comm.is_some() {
            self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        } else if chunk_start == 0 {
            self.buffers
                .zero_prefill_essentials(self.gpu.as_ref(), stream)?;
        }

        let mut kv_cache = self.kv_cache.lock();

        // 2026-09-25: Embed the chunk and overlay vision-pad positions.
        self.prefill_b_embed_chunk(tokens, chunk_start, chunk_len, stream)?;

        // 2026-09-25: Prefix-cache lookup, EP agreement and Marconi snapshot restore.
        let (kv_write_start, marconi_skip) = self.prefill_b_prefix_lookup(
            tokens,
            seq,
            chunk_start,
            total,
            &mut kv_cache,
            stream,
            None,
        )?;

        if std::env::var("METRALE_SSM_SAVE_DUMP").is_ok() {
            self.ssm_pool.debug_state_checksum(
                seq.slot_idx,
                self.gpu.as_ref(),
                stream,
                &format!("chunk_entry start={chunk_start} len={chunk_len} kvws={kv_write_start}"),
            );
        }

        let bs = kv_cache.block_size();
        let end_pos = chunk_start + chunk_len;
        let blocks_needed = (end_pos - 1) / bs + 1;
        super::super::block_mgmt::ensure_blocks_through_prefill(
            seq,
            blocks_needed - 1,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            stream,
            self.levers.kv_poison,
        )?;

        // 2026-09-25: Processing range for this chunk; a fully cached chunk
        // returns early.
        let (proc_start, proc_count, effective_seq_len_start) = match self.prefill_b_proc_range(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            is_last_chunk,
            kv_write_start,
            marconi_skip,
            // 2026-09-25: A single stream's hidden rows start at the buffer base.
            self.buffers.hidden_states(),
            stream,
        )? {
            proc_range::ProcRange::Compute {
                proc_start,
                proc_count,
                effective_seq_len_start,
            } => (proc_start, proc_count, effective_seq_len_start),
            proc_range::ProcRange::EarlyReturn(ptr) => {
                // 2026-09-25: A fully cached chunk still records its tokens:
                // decode-checkpoint registration and the radix insert in
                // `cache_sequence` read `seq.tokens`, paired with the full block
                // table.
                seq.tokens
                    .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
                seq.seq_len = chunk_start + chunk_len;
                seq.last_decode_ckpt_block = seq.tokens.len() / bs;
                return Ok(ptr);
            }
        };

        // 2026-09-25: Upload positions (MRoPE when enabled) and slot metadata.
        let upload_meta::MetaLayout {
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
        } = self.prefill_b_upload_meta(
            tokens,
            seq,
            chunk_start,
            chunk_len,
            proc_start,
            proc_count,
            effective_seq_len_start,
            &kv_cache,
            stream,
        )?;
        // 2026-10-01: A pass that captures at `cut` runs its tail rows' attention as a paged
        // call (`replay_tail` in qwen3_attention), which reads the block table, so the
        // paged metadata is uploaded even for a first chunk. Its slot fill writes the
        // same slots the first-chunk upload above wrote.
        let needs_paged = needs_paged
            || cut_capture.is_some_and(|c| proc_start < c && c < proc_start + proc_count);

        // 2026-09-25: Paged metadata (block table and `seq_len`).
        if needs_paged {
            self.prefill_b_upload_paged(
                seq,
                total,
                proc_start,
                proc_count,
                meta_base,
                slot_offset,
                &kv_cache,
                stream,
            )?;
        }

        self.gpu.synchronize(stream)?;

        // 2026-09-25: Mid-chunk tail SSM capture is planned before the forward
        // pass, which uses the plan. `None` (flag off, or the pass does not span
        // `tb`, among other cases) means no capture.
        let cut_plan = cut_capture.and_then(|cut| {
            self.prepare_cut_capture(seq, &mut kv_cache, proc_start, proc_count, cut, total)
        });
        let cut_captured = cut_plan.is_some();
        let midcap_plan = match cut_plan {
            Some(plan) => Some(plan),
            None => self.prepare_midchunk_capture(
                tokens,
                seq,
                &mut kv_cache,
                proc_start,
                proc_count,
                stream,
            ),
        };

        // 2026-09-25: Forward through all layers.
        self.prefill_b_forward_layers(
            seq,
            &mut kv_cache,
            chunk_start,
            chunk_len,
            is_last_chunk,
            proc_count,
            effective_seq_len_start,
            kv_write_start,
            marconi_skip,
            meta_base,
            slot_offset,
            pos_stream_bytes,
            use_mrope,
            needs_paged,
            midcap_plan.as_ref(),
            stream,
        )?;

        // 2026-09-25: Register the captured slots once the pass has written the
        // `tb` state into them.
        // 2026-10-01: An in-pass split-point capture is registered after the pass, as
        // `prefill_b_save_checkpoint` registers a pass that ends there: the split point,
        // then the deeper tail boundary when the plan has it, so the prompt's tail
        // checkpoint is the deeper one.
        if let Some(plan) = midcap_plan.as_ref() {
            if cut_captured {
                if let (Some(tb_early), Some(slot)) = (plan.tb_early, plan.snap_slot_early) {
                    self.prefill_b_register_checkpoint(
                        tokens,
                        seq,
                        &mut kv_cache,
                        tb_early,
                        slot,
                        stream,
                    )?;
                }
                self.prefill_b_register_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    plan.tb,
                    plan.snap_slot,
                    stream,
                )?;
            } else {
                self.finalize_midchunk_capture(tokens, seq, plan);
            }
        }

        // 2026-09-25: Append this chunk's tokens; the early-return arm above
        // appends them itself.
        seq.tokens
            .extend_from_slice(&tokens[chunk_start..chunk_start + chunk_len]);
        seq.seq_len = chunk_start + chunk_len;
        // 2026-09-25: Prime the decode-checkpoint gate; after the last chunk it
        // holds the prompt's full-block count (see prefill_a.rs).
        seq.last_decode_ckpt_block = seq.tokens.len() / bs;

        // 2026-09-25: Prompt logprobs are projected while this chunk's hidden
        // rows are live, before `prefill_b_finalize_last` overwrites the norm
        // output and logits. A no-op unless `seq.collect_prompt_logprobs` is
        // set.
        self.collect_prompt_logprobs_chunk(
            tokens,
            seq,
            chunk_start,
            proc_start,
            proc_count,
            stream,
        )?;

        if is_last_chunk {
            // 2026-09-25: Final norm, LM head, prefix-cache insert and snapshot save.
            self.prefill_b_finalize_last(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                proc_count,
                stream,
            )
        } else {
            // 2026-09-25: Intermediate Marconi checkpoint.
            self.prefill_b_save_checkpoint(
                tokens,
                seq,
                &mut kv_cache,
                chunk_start,
                chunk_len,
                stream,
            )?;
            Ok(DevicePtr::NULL)
        }
    }
}
