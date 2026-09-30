// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The host steps `forward_one` shares with the circuit's draft program: the step's
//! attention metadata, the draft token the argmax left, and the row bookkeeping.
//!
//! Owner: model-layers (MTP head).
//! Invariants: `finish_row` runs once per drafter row written.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::{MtpHead, MtpProposerState};
use crate::layer::ForwardContext;
use crate::layers::mtp_meta::{MTP_META_OFFSET, pack_mtp_attn_meta};
use crate::layers::ops;

impl MtpHead {
    /// 2026-09-29: Grow `state`'s block table in the draft cache to cover its next row and
    /// upload the step's attention metadata; returns its address and block-table width.
    pub(in crate::layers::mtp_head) fn upload_draft_meta(
        &self,
        kv_cache: &mut metrale_cache::kv_cache::PagedKvCache,
        state: &mut MtpProposerState,
        position: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, u32)> {
        let bs = kv_cache.block_size();
        let blocks_needed = (state.seq_len / bs) + 1;
        while state.block_table.len() < blocks_needed {
            state.block_table.push(kv_cache.alloc_block()?);
        }
        let meta_base = ctx.buffers.scratch().offset(MTP_META_OFFSET);
        let block_idx = state.block_table[state.seq_len / bs];
        let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
        // 2026-09-25: The metadata grows with the block table, so its bound is the rest
        // of the scratch arena past `MTP_META_OFFSET`; `pack_mtp_attn_meta` fails
        // rather than write past it.
        let meta_buf = pack_mtp_attn_meta(
            position as u32,
            global_slot,
            (state.seq_len + 1) as i32,
            &state.block_table,
            ctx.buffers.scratch_bytes().saturating_sub(MTP_META_OFFSET),
        )?;
        ctx.gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;
        Ok((meta_base, state.block_table.len() as u32))
    }

    /// 2026-09-29: The draft token the argmax left at `out_ptr`: with `draft_embed_target`,
    /// its embedding goes there and its id to `draft_token_id_dev`, and 0 is returned;
    /// otherwise the id is copied back.
    pub(in crate::layers::mtp_head) fn draft_token(
        &self,
        ctx: &ForwardContext,
        out_ptr: DevicePtr,
        draft_embed_target: Option<DevicePtr>,
        stream: u64,
    ) -> Result<u32> {
        if let Some(embed_target) = draft_embed_target {
            ops::embed_from_argmax(
                ctx.gpu,
                self.embed_from_argmax_k,
                out_ptr,
                self.embed_tokens.weight,
                embed_target,
                self.draft_token_id_dev,
                ctx.config.hidden_size as u32,
                stream,
            )?;
            Ok(0)
        } else {
            let mut buf = [0u8; 4];
            ctx.gpu.copy_d2h(out_ptr, &mut buf)?;
            Ok(u32::from_le_bytes(buf))
        }
    }

    /// 2026-09-29: A step wrote one drafter row, for sequence key `position - 1`; the catch-up
    /// path reads `last_pair_key` to find missing rows.
    pub(in crate::layers::mtp_head) fn finish_row(state: &mut MtpProposerState, position: usize) {
        state.seq_len += 1;
        state.last_pair_key = Some(position.saturating_sub(1));
    }
}
