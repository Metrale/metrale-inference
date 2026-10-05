// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The exact wave prefill: a wave of concurrently arriving prompts whose rows get
//! the bits of their single-stream prefill (`prefill_chunk_dispatch`), with every layer's FFN
//! (the MoE, most of a pass's weight bytes) run once for the whole wave.
//!
//! Per layer, each stream runs the layer's mixer half (`LayerSplitPrefill::prefill_mixer`: the
//! attention or GDN block, its residual add and the post-mixer norm) with that stream's own
//! single-stream context: its positions, slots and block table, its in-pass tail capture plan,
//! its K/V write floor. The mixer leaves the stream's FFN input in `norm_output`, copied to the
//! wave's staging rows. Then `prefill_ffn_rows` runs the FFN half once over all rows. Every
//! op of the FFN half computes a row from that row alone (row-invariant prefill kernels), so a
//! row gets the bits of the single-stream pass. After the layers, each stream gets the
//! single-stream epilogue: MTP capture, checkpoint registration, finalize, eager drafter.
//!
//! Owner: model-engine.
//! Invariants:
//! - Admission (`wave_exact_admits`) runs before any sequence is changed; it returns false
//!   for every case the single-stream pass handles differently (a two-pass tail split, LoRA,
//!   DFlash capture, prompt logprobs, multi-rank, a layer without the mixer/FFN split).
//! - The per-stream context equals `prefill_b_forward_layers`' for the same stream.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, MidchunkCapture};

use super::super::super::super::types::TransformerModel;
use super::super::midchunk_capture::MidCapturePlan;
use super::{KernelBatchResult, SetupFlow, StreamSetup};
use crate::traits::PrefillSlice;

/// 2026-10-05: `METRALE_PREFILL_WAVE_EXACT` (presence): batched prefill waves run the exact
/// wave instead of the kernel-batched layers. Read once per process.
pub(in crate::model) fn wave_exact_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_PREFILL_WAVE_EXACT").is_some())
}

impl TransformerModel {
    /// 2026-10-05: Whether the exact wave can give every stream its single-stream bits.
    fn wave_exact_admits(&self, streams: &[PrefillSlice<'_>]) -> bool {
        let single_rank = self.comm.is_none() && !self.multi_rank_protocol_active();
        let layers_split = self.layers.iter().all(|l| l.supports_wave_prefill());
        let plain = self.lora.is_none() && self.dflash_capture_layers.is_empty();
        let streams_ok = streams.iter().all(|s| {
            let cut = if s.is_last_chunk {
                self.prefill_tail_split_dispatch(s.prompt_tokens)
            } else {
                None
            };
            // 2026-10-05: A last chunk spanning the tail split point runs the in-pass capture
            // single-stream only when it is supported; otherwise it is split into two passes,
            // which a wave does not reproduce.
            let split_ok = cut
                .is_none_or(|c| c <= s.chunk_start || self.inpass_cut_capture_supported(&*s.seq));
            split_ok && s.seq.collect_prompt_logprobs.is_none()
        });
        single_rank && layers_split && plain && streams_ok
    }

    /// 2026-10-05: The staging rows for the wave's FFN input, at least `bytes`, kept for later
    /// waves and freed with the model (`drop.rs`).
    fn wave_ffn_staging(&self, bytes: usize) -> Result<DevicePtr> {
        let mut slot = self.wave_ffn_staging.lock();
        match *slot {
            Some((ptr, cap)) if cap >= bytes => Ok(ptr),
            prev => {
                if let Some((ptr, _)) = prev {
                    self.gpu.free(ptr)?;
                }
                let ptr = self.gpu.alloc(bytes)?;
                *slot = Some((ptr, bytes));
                Ok(ptr)
            }
        }
    }

    /// 2026-10-05: The exact wave prefill of `streams` (see the module header). `NotAdmitted`
    /// leaves every sequence unchanged.
    pub(in crate::model) fn prefill_batch_chunk_wave_exact(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<KernelBatchResult> {
        if !self.wave_exact_admits(streams) {
            return Ok(KernelBatchResult::NotAdmitted);
        }
        let n = streams.len();
        let is_last_chunk = streams[0].is_last_chunk;
        let h = self.config.hidden_size;
        let row_bytes = h * 2;
        let mut kv_cache = self.kv_cache.lock();
        let StreamSetup {
            per_stream,
            running_proc_off,
            hidden_base,
            residual_base,
            ..
        } = match self.batched_stream_setup(streams, &mut kv_cache, stream)? {
            SetupFlow::Ready(s) => s,
            SetupFlow::Return(r) => return Ok(r),
        };
        self.gpu.synchronize(stream)?;
        let staging = self.wave_ffn_staging(running_proc_off * row_bytes)?;

        // 2026-10-05: Per stream, as `prefill_chunk_pass`: the tail-capture plan (in-pass cut
        // capture, else the mid-chunk capture), and whether the pass needs paged metadata.
        let mut plans: Vec<(Option<MidCapturePlan>, bool, bool)> = Vec::with_capacity(n);
        for (b, slice) in streams.iter_mut().enumerate() {
            let m = &per_stream[b];
            let tokens = slice.prompt_tokens;
            let total = tokens.len();
            let cut = if is_last_chunk {
                self.prefill_tail_split_dispatch(tokens)
                    .filter(|&c| c > slice.chunk_start)
            } else {
                None
            };
            anyhow::ensure!(
                m.proc_count > 1 || m.effective_seq_len_start == 0,
                "exact wave: stream {b} is a single-token pass after position 0"
            );
            anyhow::ensure!(
                cut.is_none()
                    || !(slice.seq.prefix_lookup_skip
                        && slice.seq.marconi_skip_to >= cut.unwrap_or(0)),
                "exact wave: stream {b} restored through its tail split point"
            );
            let needs_paged = m.layout.needs_paged
                || cut.is_some_and(|c| m.proc_start < c && c < m.proc_start + m.proc_count);
            anyhow::ensure!(
                !needs_paged || !m.block_table_dev.is_null(),
                "exact wave: stream {b} needs paged metadata the setup did not upload"
            );
            let cut_plan = cut.and_then(|c| {
                self.prepare_cut_capture(
                    slice.seq,
                    &mut kv_cache,
                    m.proc_start,
                    m.proc_count,
                    c,
                    total,
                )
            });
            let cut_captured = cut_plan.is_some();
            let plan = match cut_plan {
                Some(p) => Some(p),
                None => self.prepare_midchunk_capture(
                    tokens,
                    slice.seq,
                    &mut kv_cache,
                    m.proc_start,
                    m.proc_count,
                    stream,
                ),
            };
            plans.push((plan, cut_captured, needs_paged));
        }

        self.wave_layers(
            streams,
            &per_stream,
            &plans,
            &mut kv_cache,
            hidden_base,
            residual_base,
            staging,
            running_proc_off,
            stream,
        )?;

        // 2026-10-05: The single-stream epilogue, stream by stream, in `prefill_chunk_pass`'s
        // order: MTP capture (end of the layer loop), checkpoint registration, tokens, finalize,
        // then the eager drafter prefill (`impl_forward`).
        let mut logits_out = Vec::with_capacity(n);
        for (b, slice) in streams.iter_mut().enumerate() {
            let m = &per_stream[b];
            let tokens = slice.prompt_tokens;
            let (chunk_start, cl) = (slice.chunk_start, slice.chunk_len);
            let seq = &mut *slice.seq;
            self.try_mtp_prefill_capture_from(
                seq,
                m.effective_seq_len_start,
                m.proc_count,
                hidden_base.offset(m.proc_off * row_bytes),
                stream,
            )?;
            let (plan, cut_captured, _) = &plans[b];
            if let Some(plan) = plan.as_ref() {
                if *cut_captured {
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
            let bs = kv_cache.block_size();
            seq.tokens
                .extend_from_slice(&tokens[chunk_start..chunk_start + cl]);
            seq.seq_len = chunk_start + cl;
            seq.last_decode_ckpt_block = seq.tokens.len() / bs;
            let logits = if is_last_chunk {
                self.prefill_b_finalize_last_at(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    cl,
                    m.proc_count,
                    m.proc_off,
                    row_base + b,
                    stream,
                )?
            } else {
                self.prefill_b_save_checkpoint(
                    tokens,
                    seq,
                    &mut kv_cache,
                    chunk_start,
                    cl,
                    stream,
                )?;
                DevicePtr::NULL
            };
            self.try_eager_drafter_prefill(seq, is_last_chunk, stream);
            logits_out.push(logits);
        }
        Ok(KernelBatchResult::Completed(logits_out))
    }

    /// 2026-10-05: The layer loop of the exact wave: per layer, each stream's mixer with its
    /// single-stream context, then the FFN half once over every stream's rows.
    #[allow(clippy::too_many_arguments)]
    fn wave_layers(
        &self,
        streams: &mut [PrefillSlice<'_>],
        per_stream: &[super::PerStreamMeta],
        plans: &[(Option<MidCapturePlan>, bool, bool)],
        kv_cache: &mut PagedKvCache,
        hidden_base: DevicePtr,
        residual_base: DevicePtr,
        staging: DevicePtr,
        total_rows: usize,
        stream: u64,
    ) -> Result<()> {
        let row_bytes = self.config.hidden_size * 2;
        // 2026-10-05: One mid-chunk capture counter per stream, as one per single-stream pass.
        let counters: Vec<std::sync::atomic::AtomicUsize> = (0..streams.len())
            .map(|_| std::sync::atomic::AtomicUsize::new(0))
            .collect();
        let ffn_ctx = self.wave_ctx(None, None, false);
        for (i, layer) in self.layers.iter().enumerate() {
            for (b, slice) in streams.iter_mut().enumerate() {
                let m = &per_stream[b];
                let (plan, _, needs_paged) = &plans[b];
                let seq = &mut *slice.seq;
                let attn_metadata = self.wave_attn_metadata(m, seq.block_table.len(), *needs_paged);
                let midchunk_capture = plan.as_ref().map(|p| MidchunkCapture {
                    cap_local: p.cap_local,
                    h_dsts: &p.h_dsts,
                    conv_dsts: &p.conv_dsts,
                    h_bytes: p.h_bytes,
                    conv_bytes: p.conv_bytes,
                    ssm_layer_counter: &counters[b],
                    cap_local_early: p.cap_local_early,
                    h_dsts_early: &p.h_dsts_early,
                    conv_dsts_early: &p.conv_dsts_early,
                    replay_tail: p.replay_tail,
                });
                let ctx = self.wave_ctx(Some(attn_metadata), midchunk_capture, m.marconi_skip);
                // 2026-10-05: The K/V write floor of `prefill_b_forward_layers`.
                let kv_write_start = if m.marconi_skip {
                    seq.cached_prefix_tokens
                        .saturating_sub(m.effective_seq_len_start)
                        .min(m.proc_count)
                } else {
                    m.kv_write_start
                };
                let off = m.proc_off * row_bytes;
                layer
                    .prefill_mixer(
                        hidden_base.offset(off),
                        residual_base.offset(off),
                        m.proc_count,
                        seq.layer_states[i].as_mut(),
                        kv_cache,
                        m.effective_seq_len_start,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        kv_write_start,
                        &ctx,
                        stream,
                    )
                    .map_err(|e| anyhow::anyhow!("exact wave layer {i} stream {b} mixer: {e}"))?;
                self.gpu.copy_d2d_async(
                    self.buffers.norm_output(),
                    staging.offset(off),
                    m.proc_count * row_bytes,
                    stream,
                )?;
            }
            layer
                .prefill_ffn_rows(hidden_base, staging, total_rows, &ffn_ctx, stream)
                .map_err(|e| anyhow::anyhow!("exact wave layer {i} FFN: {e}"))?;
        }
        Ok(())
    }

    /// 2026-10-05: A stream's attention metadata as `prefill_b_forward_layers` builds it, from
    /// the stream's own metadata upload.
    fn wave_attn_metadata(
        &self,
        m: &super::PerStreamMeta,
        max_blocks_per_seq: usize,
        needs_paged: bool,
    ) -> AttnMetadataDev {
        let l = &m.layout;
        let (positions_h, positions_w) = if l.use_mrope {
            (
                l.meta_base.offset(l.pos_stream_bytes),
                l.meta_base.offset(l.pos_stream_bytes * 2),
            )
        } else {
            (l.meta_base, l.meta_base)
        };
        let (block_table, seq_len) = if needs_paged {
            (m.block_table_dev, m.seq_len_dev)
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        AttnMetadataDev {
            positions: l.meta_base,
            positions_h,
            positions_w,
            slot: l.meta_base.offset(l.slot_offset),
            seq_len,
            block_table,
            max_blocks_per_seq: max_blocks_per_seq as u32,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    }

    /// 2026-10-05: The context of a wave step; a mixer passes its stream's metadata and plan,
    /// the FFN half neither (and no token ids, so a hash-routed MoE fails instead of reading
    /// one stream's ids for all).
    fn wave_ctx<'a>(
        &'a self,
        attn_metadata: Option<AttnMetadataDev>,
        midchunk_capture: Option<MidchunkCapture<'a>>,
        gdn_exact_replay: bool,
    ) -> ForwardContext<'a> {
        let token_ids = attn_metadata.is_some().then(|| self.buffers.token_ids());
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay,
            gdn_write_on_accept: false,
            token_ids,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture,
            moe_lora_route: metrale_model_layers::layer::MoeLoraRoute::Refuse,
        }
    }
}
