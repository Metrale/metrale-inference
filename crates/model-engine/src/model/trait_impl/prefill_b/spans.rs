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
//! - 2026-10-11: With a live prefix cache a sequence's rows are its processing range (from its
//!   restore point; none for a fully cached non-last chunk), and a last chunk that computes
//!   across the tail split point captures there in-pass (`prepare_cut_capture`), as
//!   `prefill_chunk_dispatch` does; the layers split the sequence's rows at the capture points.

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{ForwardContext, PrefillSpan};

use super::super::super::types::TransformerModel;
use super::midchunk_capture::MidCapturePlan;
use super::proc_range::ProcRange;
use super::spans_wire::{
    EP_CMD_PREFILL_SPANS, SpansHeader, decode_spans_header, encode_spans_header, spans_batches,
    spans_header_words, split_prompts,
};
use crate::traits::{ModelForward, PrefillSlice, SequenceState};

impl TransformerModel {
    /// 2026-10-09: The rows one batched prefill step should carry, or `None` when this serve
    /// cannot run the multi-rank batched prefill at all. It runs on a multi-rank world on the v2
    /// protocol (the slots travel in the payload), with every layer implementing
    /// `prefill_spans`, and none of the per-chunk extras the pass leaves out: DFlash or MTP
    /// prompt capture, LoRA, token overlays, or the high-speed-swap window. 2026-10-11: A live
    /// prefix cache only where every layer splits its whole pass at an in-pass capture and the
    /// serve captures in-pass (`inpass_cut_capture_serve_ok`): a stream's chunk then runs from
    /// its restore point, and the tail split point is captured inside the pass, as its
    /// single-sequence prefill does. The rows are the narrowest layer's row group
    /// (`prefill_spans_rows`), at most the arena.
    pub(in crate::model) fn prefill_spans_rows_dispatch(&self) -> Option<usize> {
        let ok = self.multi_rank_protocol_active()
            && self.ep_protocol_v2
            && (!self.prefix_cache.is_active()
                || (self.splits_whole_pass_at_capture() && self.inpass_cut_capture_serve_ok()))
            && self.dflash_capture_layers.is_empty()
            && self.lora.is_none()
            && self.overlays.is_none()
            && self.kv_cache.lock().config().cache_blocks_per_seq.is_none();
        if !ok {
            return None;
        }
        let rows = self
            .layers
            .iter()
            .map(|l| l.prefill_spans_rows())
            .collect::<Option<Vec<usize>>>()?;
        let narrowest = rows.into_iter().min()?;
        Some(narrowest.min(self.buffers.max_batch_tokens()).max(1))
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
            self.prefill_spans_rows_dispatch().is_some(),
            "multi-rank batched prefill: this serve cannot run it (prefill_spans_rows)"
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
            self.spans_normalize(sub.iter().map(|s| &*s.seq));
        }
        Ok(out)
    }

    /// 2026-10-14: Clamp each sequence's SSM state after a batched pass, as the single-sequence
    /// handlers do after a chunk. Rank 0 (`prefill_spans_dispatch`) and the workers
    /// (`ep_worker_prefill_spans`) both call it, so every rank's state takes the same clamp;
    /// before, only the workers clamped. A failure is logged, as the workers always logged it.
    fn spans_normalize<'s>(&self, seqs: impl IntoIterator<Item = &'s SequenceState>) {
        let stream = self.gpu.default_stream();
        for seq in seqs {
            if let Err(e) = self.normalize_ssm_states(seq, stream) {
                tracing::warn!("SSM state normalization after a batched prefill failed: {e:#}");
            }
        }
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
        self.spans_normalize(refs.iter().map(|s| &**s));
        Ok(true)
    }

    /// 2026-10-09: The pass every rank runs (module invariants). `streams` fit the arena
    /// together; stream `i`'s rows start at hidden row `Σ chunk_len[..i]`. 2026-10-11: At the
    /// sum of the earlier streams' computed rows, which a restore shortens.
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
        // 2026-10-11: Per stream: its first hidden row, its computed rows (`None` for a fully
        // cached non-last chunk) and its in-pass capture plan.
        let mut offs = Vec::with_capacity(streams.len());
        let mut ranges: Vec<Option<(usize, usize)>> = Vec::with_capacity(streams.len());
        let mut plans: Vec<Option<MidCapturePlan>> = Vec::with_capacity(streams.len());
        let mut off = 0usize;
        for s in streams.iter_mut() {
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
            let range = match self.prefill_b_proc_range(
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
                    effective_seq_len_start,
                } => {
                    ensure!(
                        effective_seq_len_start == proc_start,
                        "prefill spans: processing range {proc_start}+{proc_count} starts its \
                         positions at {effective_seq_len_start}"
                    );
                    Some((proc_start, proc_count))
                }
                ProcRange::EarlyReturn(_) => None,
            };
            plans.push(match range {
                Some((proc_start, proc_count)) => {
                    self.spans_cut_capture(s, &mut kv_cache, proc_start, proc_count)?
                }
                None => None,
            });
            off += range.map_or(0, |(_, n)| n);
            ranges.push(range);
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
            settings: &self.settings,
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
        let counters: Vec<std::sync::atomic::AtomicUsize> = plans
            .iter()
            .map(|_| std::sync::atomic::AtomicUsize::new(0))
            .collect();
        let captures: Vec<_> = plans
            .iter()
            .zip(&counters)
            .map(|(p, c)| p.as_ref().map(|p| p.as_capture(c)))
            .collect();
        for (l, layer) in self.layers.iter().enumerate() {
            let mut spans: Vec<PrefillSpan<'_>> = streams
                .iter_mut()
                .zip(&ranges)
                .zip(&captures)
                .filter_map(|((s, range), capture)| {
                    let (proc_start, proc_count) = (*range)?;
                    let seq = &mut *s.seq;
                    Some(PrefillSpan {
                        state: seq.layer_states[l].as_mut(),
                        block_table: &mut seq.block_table,
                        seq_len_start: proc_start,
                        rows: proc_count,
                        capture: capture.as_ref(),
                    })
                })
                .collect();
            if spans.is_empty() {
                break;
            }
            layer
                .prefill_spans(hidden, &mut spans, &mut kv_cache, &ctx, stream)
                .map_err(|e| anyhow::anyhow!("Prefill spans layer {l} failed: {e}"))?;
        }
        let mut out = Vec::with_capacity(streams.len());
        for (i, s) in streams.iter_mut().enumerate() {
            let (tokens, seq) = (s.prompt_tokens, &mut *s.seq);
            let end = s.chunk_start + s.chunk_len;
            // 2026-10-11: The MTP drafter's whole-prompt capture, as the single-sequence pass
            // takes it after its layers (`prefill_b/forward_layers.rs`), stream by stream in batch
            // order: a chunk 0 claims the shared capture, a later chunk appends while it still
            // owns it, so the batch leaves it as the same prefills one after another would.
            // Before, a serve with an MTP drafter could not batch prefills at all.
            if let Some((proc_start, proc_count)) = ranges[i] {
                self.try_mtp_prefill_capture_from(
                    seq,
                    proc_start,
                    proc_count,
                    hidden.offset(offs[i] * h * 2),
                    stream,
                )?;
            }
            if let Some(plan) = plans[i].as_ref() {
                self.register_cut_capture(tokens, seq, &mut kv_cache, plan, stream)?;
            }
            seq.tokens.extend_from_slice(&tokens[s.chunk_start..end]);
            seq.seq_len = end;
            // 2026-10-09: Prime the decode-checkpoint gate, as `prefill_chunk_pass` does.
            seq.last_decode_ckpt_block = seq.tokens.len() / bs;
            // 2026-10-11: The eager drafter prefill while this stream still owns the capture,
            // before the next stream's capture replaces it (the single-sequence wrappers run it
            // after their pass, `impl_forward.rs`). It writes no buffer the other streams' rows
            // are in (`DraftProposer::prefill_uses_shared_buffers`).
            self.try_eager_drafter_prefill(seq, s.is_last_chunk, stream);
            let Some((_, proc_count)) = ranges[i] else {
                // 2026-10-11: A fully cached non-last chunk, as `prefill_chunk_pass` returns it.
                out.push(DevicePtr::NULL);
                continue;
            };
            out.push(if s.is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    &mut kv_cache,
                    s.chunk_start,
                    s.chunk_len,
                    proc_count,
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

    /// 2026-10-11: Stream `s`'s in-pass capture over its computed rows
    /// `[proc_start, proc_start + proc_count)`, where `prefill_chunk_dispatch` would capture:
    /// a last chunk from below the tail split point (`prefill_tail_split_dispatch`) that
    /// computes across it. Errors where that dispatch would run two passes instead, which
    /// `prefill_spans_rows_dispatch` rules out for the serve.
    fn spans_cut_capture(
        &self,
        s: &PrefillSlice<'_>,
        kv_cache: &mut metrale_cache::kv_cache::PagedKvCache,
        proc_start: usize,
        proc_count: usize,
    ) -> Result<Option<MidCapturePlan>> {
        let total = s.prompt_tokens.len();
        let Some(cut) = s
            .is_last_chunk
            .then(|| self.prefill_tail_split_at(s.prompt_tokens, kv_cache.block_size()))
            .flatten()
            .filter(|&cut| {
                cut > s.chunk_start && proc_start < cut && cut < proc_start + proc_count
            })
        else {
            return Ok(None);
        };
        ensure!(
            self.inpass_cut_capture_supported(&*s.seq),
            "prefill spans: a {total}-token prompt computes across its tail split point {cut} \
             and cannot capture it in-pass; it needs the single-sequence prefill"
        );
        Ok(self.prepare_cut_capture(&*s.seq, kv_cache, proc_start, proc_count, cut, total))
    }
}
