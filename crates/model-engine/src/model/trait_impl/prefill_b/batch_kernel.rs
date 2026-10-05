// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel-batched prefill, `prefill_batch_chunk_kernel_batched`: N streams share one loop over the layers.
//!
//! Setup runs per stream at packed offsets in the shared buffers (embed,
//! prefix lookup, block allocation, processing range, metadata upload). One
//! loop over the layers then calls `prefill_ssm_batched_layer` or
//! `prefill_attn_batched_layer`, and each stream is finalized (last chunk) or
//! checkpointed. The admission rules are in `eligible.rs`.
//!
//! Owner: model-engine.
//! Invariants:
//! - Both `NotAdmitted` returns come before any buffer is zeroed or any
//!   sequence is changed.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use super::proc_range::ProcRange;
use super::stage_batched::PerStreamStageInfo;
use super::upload_meta::MetaLayout;

mod eligible;
mod phases;
mod setup;

use phases::{Flow, PerStreamMeta};
pub(in crate::model) use setup::{SetupFlow, StreamSetup};

// 2026-09-25: Re-exported for the sibling modules and `batch_kernel_tests.rs`.
use eligible::first_chunk_batched_enabled;
pub(in crate::model) use eligible::{
    batched_reserve_hybrid_ssm_ok, cache_batch_matches_compatible, check_kernel_batched_eligible,
    config_is_mla, varlen_prefill_enabled,
};

use crate::traits::{Model, PrefillSlice, SequenceState};
use metrale_model_layers::layer::{
    BatchedAttnMetadata, ForwardContext, GdnPrefillBuffers, LayerState, TransformerLayer,
};

pub(in crate::model) enum KernelBatchResult {
    Completed(Vec<DevicePtr>),
    NotAdmitted,
}

impl TransformerModel {
    /// 2026-09-25: Kernel-batched prefill of `streams`; the caller must have
    /// checked `kernel_batched_eligible`.
    ///
    /// Returns `NotAdmitted` when the batch's KV need cannot be met or the
    /// prefix reservation is refused. Returns an error when a later per-stream
    /// check fails (for example a differing `proc_count` without varlen),
    /// possibly after sequences were changed.
    ///
    /// `row_base` shifts each stream's logits row clear of the decode lanes
    /// in a mixed step; the caller bounds-checks it against the arena.
    pub(in crate::model) fn prefill_batch_chunk_kernel_batched(
        &self,
        streams: &mut [PrefillSlice<'_>],
        stream: u64,
        row_base: usize,
    ) -> Result<KernelBatchResult> {
        let n = streams.len();
        let chunk_len = streams[0].chunk_len;
        let is_last_chunk = streams[0].is_last_chunk;
        let varlen = varlen_prefill_enabled();
        let stream = if self.multi_rank_protocol_active() {
            self.gpu.default_stream()
        } else {
            stream
        };

        let mut kv_cache = self.kv_cache.lock();
        let StreamSetup {
            per_stream,
            mut scratch_cursor,
            running_proc_off,
            use_mrope,
            hidden_base,
            residual_base: _residual_base,
        } = match self.batched_stream_setup(streams, &mut kv_cache, stream)? {
            SetupFlow::Ready(s) => s,
            SetupFlow::Return(r) => return Ok(r),
        };

        self.gpu.synchronize(stream)?;

        // 2026-09-25: Stage `BatchedAttnMetadata`, then run the layer loop.
        let proc_count = per_stream[0].proc_count;
        let seq_lens_start = per_stream[0].effective_seq_len_start;

        let streams_info: Vec<PerStreamStageInfo<'_>> = streams
            .iter()
            .zip(per_stream.iter())
            .map(|(slice, m)| PerStreamStageInfo {
                proc_start: m.proc_start,
                proc_count: m.proc_count,
                block_table_dev: m.block_table_dev,
                seq_len_dev: m.seq_len_dev,
                num_blocks: m.num_blocks,
                seq: &*slice.seq,
            })
            .collect();

        let meta = self.stage_batched_attn_metadata(
            &streams_info,
            &kv_cache,
            use_mrope,
            scratch_cursor,
            stream,
        )?;
        let stage_size = meta.staged_bytes;
        scratch_cursor += stage_size;

        // 2026-09-25: Return an error if the `h_state_ptrs` slot (n x 8 bytes)
        // would run past scratch.
        let scratch_bytes = self.buffers.sizes().scratch;
        let projected_usage = scratch_cursor + (n * std::mem::size_of::<u64>());
        if projected_usage > scratch_bytes {
            anyhow::bail!(
                "kernel-batched prefill scratch overflow: projected {} bytes \
                 > scratch capacity {} bytes (n={n}, chunk_len={chunk_len}, \
                 proc_count={proc_count}). Falling back to per-stream.",
                projected_usage,
                scratch_bytes
            );
        }

        let gdn_bufs = GdnPrefillBuffers {
            qkv: self.gdn_buf_qkv,
            gate_beta: self.gdn_buf_gate_beta,
            output: self.gdn_buf_out,
            z: self.gdn_buf_z,
            // 2026-09-25: Packed token count: Σ `proc_count` with varlen, and
            // `proc_count * n` otherwise, where every stream has stream 0's
            // `proc_count`.
            total_len: if varlen {
                running_proc_off
            } else {
                proc_count * n
            },
        };

        // 2026-09-25: `attn_metadata` is None: the batched dispatchers take the
        // `BatchedAttnMetadata` as an argument.
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: self.profile,
            comm: self.comm_ref(),
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            // 2026-09-25: None: the streams of one batch may use different
            // adapters.
            routed_lora_layers: None,
            midchunk_capture: None,
            // 2026-09-25: Refuse: a MoE layer with an adapter installed returns an
            // error (`moe_route_gate`) rather than apply one adapter to every
            // packed row. Without a MoE adapter the route is not consulted.
            moe_lora_route: metrale_model_layers::layer::MoeLoraRoute::Refuse,
        };

        // 2026-09-25: Scratch offset of the `h_state_ptrs` array, staged per SSM
        // layer.
        let h_state_ptrs_off = scratch_cursor;

        let kv_write_starts: Vec<usize> = per_stream.iter().map(|m| m.kv_write_start_eff).collect();

        self.run_batched_layers(
            streams,
            &per_stream,
            hidden_base,
            _residual_base,
            &mut kv_cache,
            &kv_write_starts,
            seq_lens_start,
            &meta,
            &gdn_bufs,
            h_state_ptrs_off,
            &ctx,
            stream,
        )?;

        self.codispatch_btcheck(streams, n);

        // 2026-09-25: Finalize or checkpoint each stream.
        let logits_out = self.finalize_batched_streams(
            streams,
            &per_stream,
            &mut kv_cache,
            is_last_chunk,
            row_base,
            n,
            stream,
        )?;

        Ok(KernelBatchResult::Completed(logits_out))
    }
}

// 2026-09-25: Unit tests for `check_kernel_batched_eligible` are in
// `batch_kernel_tests.rs`, which prefill_b.rs mounts.
