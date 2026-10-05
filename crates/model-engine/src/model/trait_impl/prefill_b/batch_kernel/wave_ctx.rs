// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The per-stream attention metadata and the forward contexts of an exact wave
//! (`wave.rs`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, MidchunkCapture};

use super::super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-05: A stream's attention metadata as `prefill_b_forward_layers` builds it, from
    /// the stream's own metadata upload.
    pub(super) fn wave_attn_metadata(
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
    pub(super) fn wave_ctx<'a>(
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
