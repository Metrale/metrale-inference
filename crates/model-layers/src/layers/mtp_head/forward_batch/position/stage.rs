// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The host steps of an n-row draft position, shared by the legacy forward
//! (`forward_batch_position`) and the circuit program's (`forward_batch_position_circuit`): the
//! embeddings, the attention metadata, the readback of the ids and confidences, and the
//! per-row state advance. Split from `position.rs`.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - The metadata is packed and uploaded in `mtp_meta::MtpBatchMetaLayout` at `propose_meta`;
//!   the returned host buffer must outlive the upload (the caller holds it until its readback,
//!   which synchronizes the stream).

use super::*;

impl MtpHead {
    /// 2026-09-30: Copy the n tokens' embedding rows into `ssm_qkvz`, row i at `i * hidden`.
    pub(super) fn stage_batch_embeds(
        &self,
        tokens: &[u32],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let row = ctx.config.hidden_size * 2;
        let embeds = ctx.buffers.ssm_qkvz();
        for (i, &t) in tokens.iter().enumerate() {
            let src = self.embed_tokens.weight.offset(t as usize * row);
            ctx.gpu
                .copy_d2d_async(src, embeds.offset(i * row), row, stream)?;
        }
        Ok(())
    }

    /// 2026-09-30: Give each row's next drafter KV slot a block, then pack and upload the n
    /// rows' positions, slots, sequence lengths and block tables to `propose_meta`. Returns the
    /// host buffer (to hold until the readback) and the block-table width.
    pub(super) fn stage_batch_meta(
        &self,
        states: &mut [&mut MtpProposerState],
        positions: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Vec<u8>, u32)> {
        let n = states.len();
        ensure!(positions.len() == n, "propose_batch: length mismatch");
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let mut slots = Vec::with_capacity(n);
        let mut seq_lens = Vec::with_capacity(n);
        let mut pos_u32 = Vec::with_capacity(n);
        for i in 0..n {
            let state = &mut *states[i];
            let blocks_needed = (state.seq_len / bs) + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            let block_idx = state.block_table[state.seq_len / bs];
            slots.push((block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64));
            seq_lens.push((state.seq_len + 1) as i32);
            pos_u32.push(positions[i] as u32);
        }
        drop(kv_cache);
        let max_blocks = states
            .iter()
            .map(|s| s.block_table.len())
            .max()
            .unwrap_or(1);
        let tables: Vec<&[u32]> = states.iter().map(|s| s.block_table.as_slice()).collect();
        // 2026-09-29: The whole `propose_meta` allocation: `PROPOSE_META_SEQS` strides.
        let (meta_buf, _) = crate::layers::mtp_meta::pack_mtp_attn_meta_batch(
            &pos_u32,
            &slots,
            &seq_lens,
            &tables,
            max_blocks,
            crate::layers::mtp_head::batch_caps::PROPOSE_META_SEQS * self.propose_meta_stride,
        )?;
        ctx.gpu
            .copy_h2d_async(&meta_buf, self.propose_meta, stream)?;
        Ok((meta_buf, u32::try_from(max_blocks)?))
    }

    /// 2026-09-30: One `copy_d2h` of the n ids at scratch's start and, with `want_lp`, the
    /// log-probabilities up to `LP_SCRATCH_OFF + n * 4`. Without them each `out_lp` row is 0.0
    /// (log 1: no row is reported as uncertain).
    pub(super) fn read_batch_ids(
        ctx: &ForwardContext,
        want_lp: bool,
        out_ids: &mut [u32],
        out_lp: Option<&mut [f32]>,
    ) -> Result<()> {
        let n = out_ids.len();
        let d2h_len = if want_lp {
            LP_SCRATCH_OFF + n * 4
        } else {
            n * 4
        };
        let mut buf = vec![0u8; d2h_len];
        ctx.gpu.copy_d2h(ctx.buffers.scratch(), &mut buf)?;
        let word = |o: usize| [buf[o], buf[o + 1], buf[o + 2], buf[o + 3]];
        for (i, id) in out_ids.iter_mut().enumerate() {
            *id = u32::from_le_bytes(word(i * 4));
        }
        if let Some(lp) = out_lp {
            for (i, slot) in lp.iter_mut().enumerate().take(n) {
                *slot = if want_lp {
                    f32::from_le_bytes(word(LP_SCRATCH_OFF + i * 4))
                } else {
                    0.0
                };
            }
        }
        Ok(())
    }

    /// 2026-09-30: `forward_one`'s tail, per row: one more drafter row, and the pair key.
    pub(super) fn finish_batch_rows(states: &mut [&mut MtpProposerState], positions: &[usize]) {
        for (i, state) in states.iter_mut().enumerate() {
            state.seq_len += 1;
            state.last_pair_key = Some(positions[i].saturating_sub(1));
        }
    }
}
