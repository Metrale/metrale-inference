// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched prefill: one chunk of each of several sequences in one
//! pass over the layers (`LayerSplitPrefill::prefill_spans`), announced to the worker ranks
//! as `EP_CMD_PREFILL_SPANS` (`spans_wire.rs`) so every rank runs the same pass.
//!
//! Owner: model-engine prefill.
//! Invariants:
//! - On a multi-rank serve `prefill_batch_chunk` takes this path for every batch; it refuses a
//!   batch it cannot run before anything is sent, so a refusal never leaves a worker waiting.
//! - Every rank runs `prefill_spans_run` with the same sequences, chunks and order: the
//!   per-sequence prefix lookup (its rank agreement included), the layer pass, then each
//!   sequence's finalize, in sequence order.
//! - Each sequence gets what its single-sequence chunk gives it outside the layers: the
//!   prefix lookup, block allocation, processing range, `tokens` / `seq_len`, and the last
//!   chunk's logits (row `row_base + i`) or the intermediate checkpoint.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{ForwardContext, PrefillSpan};

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::spans_wire::{
    EP_CMD_PREFILL_SPANS, SpansHeader, decode_spans_header, encode_spans_header, spans_batches,
    spans_header_words, split_prompts,
};
use crate::traits::{ModelForward, PrefillSlice, SequenceState};

impl TransformerModel {
    /// 2026-10-09: Whether this serve can run the multi-rank batched prefill at all: a
    /// multi-rank world on the v2 protocol (the slots travel in the payload), every layer
    /// implementing `prefill_spans`, and none of the per-chunk extras the pass leaves out: a
    /// live prefix cache (a restore changes a chunk's rows), DFlash or MTP prompt capture,
    /// LoRA, token overlays, or the high-speed-swap window.
    pub(in crate::model) fn prefill_spans_supported_dispatch(&self) -> bool {
        self.multi_rank_protocol_active()
            && self.ep_protocol_v2
            && !self.layers.is_empty()
            && self.layers.iter().all(|l| l.prefill_spans_supported())
            && !self.prefix_cache.is_active()
            && self.dflash_capture_layers.is_empty()
            && self.mtp_prefill_hidden.is_null()
            && self.lora.is_none()
            && self.overlays.is_none()
            && self.kv_cache.lock().config().cache_blocks_per_seq.is_none()
    }

    /// 2026-10-09: Rank 0's batched prefill on a multi-rank serve: check the batch, then for
    /// each sub-batch that fits the arena announce it and run the pass. Returns one entry per
    /// stream: its logits (row `row_base + i`) when its chunk is its last, else NULL.
    pub(in crate::model) fn prefill_spans_dispatch(
        &self,
        streams: &mut [PrefillSlice<'_>],
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        ensure!(
            self.prefill_spans_supported_dispatch(),
            "multi-rank batched prefill: this serve cannot run it (prefill_spans_supported)"
        );
        let n = streams.len();
        let logits_rows = self.buffers.sizes().logits / (self.config.vocab_size * 2);
        ensure!(
            row_base + n <= logits_rows,
            "multi-rank batched prefill: logits rows {row_base}..{} pass the {logits_rows}-row \
             arena",
            row_base + n
        );
        let adapter = streams.first().map(|s| s.seq.adapter_id);
        for (i, s) in streams.iter().enumerate() {
            ensure!(
                s.chunk_len > 0 && s.chunk_start + s.chunk_len <= s.prompt_tokens.len(),
                "multi-rank batched prefill: stream {i} chunk {}+{} of a {}-token prompt",
                s.chunk_start,
                s.chunk_len,
                s.prompt_tokens.len()
            );
            // 2026-10-09: The worker derives `is_last` from the chunk bounds
            // (`SpansHeader::is_last`); a caller that says otherwise would split the ranks.
            ensure!(
                s.is_last_chunk == (s.chunk_start + s.chunk_len >= s.prompt_tokens.len()),
                "multi-rank batched prefill: stream {i} marks is_last={} for chunk {}+{} of {}",
                s.is_last_chunk,
                s.chunk_start,
                s.chunk_len,
                s.prompt_tokens.len()
            );
            ensure!(
                !self.tokens_have_vision_pad(s.prompt_tokens)
                    && s.seq.collect_prompt_logprobs.is_none()
                    && Some(s.seq.adapter_id) == adapter,
                "multi-rank batched prefill: stream {i} has image rows, collects prompt \
                 logprobs or routes to another adapter; it needs the single-sequence prefill"
            );
        }
        let lens: Vec<usize> = streams.iter().map(|s| s.chunk_len).collect();
        let mut out = Vec::with_capacity(n);
        for range in spans_batches(&lens, self.buffers.max_batch_tokens())? {
            let sub = &mut streams[range.clone()];
            let header = SpansHeader {
                row_base: row_base + range.start,
                seq_ids: sub.iter().map(|s| s.seq.slot_idx as u32).collect(),
                chunk_start: sub.iter().map(|s| s.chunk_start).collect(),
                chunk_len: sub.iter().map(|s| s.chunk_len).collect(),
                full_len: sub.iter().map(|s| s.prompt_tokens.len()).collect(),
            };
            let words = encode_spans_header(&header)?;
            let prompts: Vec<u32> = sub
                .iter()
                .flat_map(|s| s.prompt_tokens.iter().copied())
                .collect();
            self.ep_broadcast_seq_and_cmd(0, EP_CMD_PREFILL_SPANS, true)?;
            self.ep_broadcast_u32(sub.len() as u32)?;
            self.ep_broadcast_tokens(&words)?;
            self.ep_broadcast_tokens(&prompts)?;
            out.extend(self.prefill_spans_run(sub, header.row_base)?);
        }
        Ok(out)
    }

    /// 2026-10-09: Worker side of `EP_CMD_PREFILL_SPANS`: read the batch, run the pass rank 0
    /// runs, then normalise each sequence's SSM state as the single-sequence prefill handler
    /// does after a chunk.
    pub(in crate::model) fn ep_worker_prefill_spans(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        let max = slots.len();
        if !(1..=max).contains(&n) {
            bail!("ep_worker_prefill_spans: {n} sequences (1..={max})");
        }
        let words = self.ep_broadcast_tokens(&vec![0u32; spans_header_words(n)])?;
        let header = decode_spans_header(n, max, &words)?;
        let flat = self.ep_broadcast_tokens(&vec![0u32; header.prompt_words()])?;
        let prompts = split_prompts(&header, &flat)?;
        let mut refs = super::super::verify_batch_ep::ordered_slot_refs(
            slots,
            &header.seq_ids,
            "ep_worker_prefill_spans",
        )?;
        {
            let mut slices: Vec<PrefillSlice<'_>> = refs
                .iter_mut()
                .zip(&prompts)
                .enumerate()
                .map(|(i, (seq, p))| PrefillSlice {
                    prompt_tokens: p,
                    seq,
                    chunk_start: header.chunk_start[i],
                    chunk_len: header.chunk_len[i],
                    is_last_chunk: header.is_last(i),
                })
                .collect();
            self.prefill_spans_run(&mut slices, header.row_base)?;
        }
        let stream = self.gpu.default_stream();
        for seq in refs.iter() {
            if let Err(e) = self.normalize_ssm_states(seq, stream) {
                tracing::warn!("Worker SSM state normalization failed: {e:#}");
            }
        }
        Ok(true)
    }

    /// 2026-10-09: The pass every rank runs (module invariants). `streams` fit the arena
    /// together; stream `i`'s rows start at hidden row `Σ chunk_len[..i]`.
    pub(in crate::model) fn prefill_spans_run(
        &self,
        streams: &mut [PrefillSlice<'_>],
        row_base: usize,
    ) -> Result<Vec<DevicePtr>> {
        let stream = self.gpu.default_stream();
        let h = self.config.hidden_size;
        let hidden = self.buffers.hidden_states();
        let total: usize = streams.iter().map(|s| s.chunk_len).sum();
        ensure!(
            total <= self.buffers.max_batch_tokens(),
            "prefill spans: {total} rows exceed the {}-row arena",
            self.buffers.max_batch_tokens()
        );
        // 2026-10-09: As every multi-rank chunk does (`prefill_chunk_pass`).
        self.buffers.zero_all(self.gpu.as_ref(), stream)?;
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut offs = Vec::with_capacity(streams.len());
        let mut off = 0usize;
        for (i, s) in streams.iter_mut().enumerate() {
            offs.push(off);
            let dst = hidden.offset(off * h * 2);
            let (tokens, seq) = (s.prompt_tokens, &mut *s.seq);
            self.prefill_b_embed_chunk_at(tokens, s.chunk_start, s.chunk_len, dst, stream)?;
            let (kv_write_start, skip) = self.prefill_b_prefix_lookup(
                tokens,
                seq,
                s.chunk_start,
                tokens.len(),
                &mut kv_cache,
                stream,
                None,
            )?;
            ensure!(
                !skip,
                "prefill spans: stream {i} restored a cached prefix; the pass runs whole chunks"
            );
            let blocks_needed = (s.chunk_start + s.chunk_len - 1) / bs + 1;
            super::super::super::block_mgmt::ensure_blocks_through_prefill(
                seq,
                blocks_needed - 1,
                &mut kv_cache,
                self.prefix_cache.as_ref(),
                self.gpu.as_ref(),
                stream,
                self.levers.kv_poison,
            )?;
            match self.prefill_b_proc_range(
                tokens,
                seq,
                s.chunk_start,
                s.chunk_len,
                s.is_last_chunk,
                kv_write_start,
                skip,
                dst,
                stream,
            )? {
                ProcRange::Compute {
                    proc_start,
                    proc_count,
                    ..
                } if proc_start == s.chunk_start && proc_count == s.chunk_len => {}
                _ => bail!("prefill spans: stream {i} would not process its whole chunk"),
            }
            off += s.chunk_len;
        }
        self.gpu.synchronize(stream)?;
        let adapter_slot = streams.first().map_or(-1, |s| s.seq.adapter_slot);
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            // 2026-10-09: A `prefill_spans` layer stages its own per-row metadata.
            attn_metadata: None,
            profile: false,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: self.routed_slot_layers(adapter_slot),
            midchunk_capture: None,
            moe_lora_route: self.moe_lora_route(adapter_slot),
        };
        for (l, layer) in self.layers.iter().enumerate() {
            let mut spans: Vec<PrefillSpan<'_>> = streams
                .iter_mut()
                .map(|s| {
                    let seq = &mut *s.seq;
                    PrefillSpan {
                        state: seq.layer_states[l].as_mut(),
                        block_table: &mut seq.block_table,
                        seq_len_start: s.chunk_start,
                        rows: s.chunk_len,
                    }
                })
                .collect();
            layer
                .prefill_spans(hidden, &mut spans, &mut kv_cache, &ctx, stream)
                .map_err(|e| anyhow::anyhow!("Prefill spans layer {l} failed: {e}"))?;
        }
        let mut out = Vec::with_capacity(streams.len());
        for (i, s) in streams.iter_mut().enumerate() {
            let (tokens, seq) = (s.prompt_tokens, &mut *s.seq);
            let end = s.chunk_start + s.chunk_len;
            seq.tokens.extend_from_slice(&tokens[s.chunk_start..end]);
            seq.seq_len = end;
            // 2026-10-09: Prime the decode-checkpoint gate, as `prefill_chunk_pass` does.
            seq.last_decode_ckpt_block = seq.tokens.len() / bs;
            out.push(if s.is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    &mut kv_cache,
                    s.chunk_start,
                    s.chunk_len,
                    s.chunk_len,
                    offs[i],
                    row_base + i,
                    stream,
                )?
            } else {
                self.prefill_b_save_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    s.chunk_start,
                    s.chunk_len,
                    stream,
                )?;
                DevicePtr::NULL
            });
        }
        Ok(out)
    }
}
