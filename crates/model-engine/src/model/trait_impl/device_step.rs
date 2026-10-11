// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: `METRALE_GLM_MTP_DEVICE_STEP=1`, part 1: the K=2 MTP step's draft stays on the
//! device.
//!
//! Rank 0's propose leaves its one draft in the drafter's device cell instead of reading it back
//! (`DraftProposer::propose_device`), so the host queues the next verify straight behind the
//! propose instead of waiting for it. The next verify's row 1 then reads its token from that cell
//! on every rank (the announced token is [`DEVICE_DRAFT`]), and the draft reaches the host with
//! the verify's own readback (`verify_device_row`).
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - BI: every rank verifies the same id it would have verified after a readback. The cell holds
//!   the device exchange's global id, which is identical on every rank; the host verdict is
//!   unchanged and judges the draft read back with the verify's argmax.
//! - A draft left on the device is consumed by the next verify of the same sequence, or read back
//!   first (`read_device_draft`): it is never verified as the sentinel id.
//! - Workers propose as before (with their own readback); only rank 0 skips it.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;
use metrale_model_layers::speculative::{DEVICE_DRAFT, DraftProposer, ProposerState};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// 2026-10-11: The device step's page-locked upload slots: the uploads of a device-step K=2
/// verify and its announcement are staged here and copied with `copy_h2d_async_retained`, so the
/// host never waits for the propose in flight to drain (a pageable copy may, and the blocking
/// broadcast does). Byte offsets of each slot; the block table takes the rest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum Stage {
    Block = 0,
    TokenIds = 64,
    Positions = 128,
    Slots = 192,
    SeqLens = 256,
    BlockTable = 512,
}

/// 2026-10-11: Bytes of the staging region for `block_table_bytes` of block table.
pub(in crate::model) fn staging_bytes(block_table_bytes: usize) -> usize {
    (Stage::BlockTable as usize + block_table_bytes).next_multiple_of(4096)
}

/// 2026-10-11: The slot `stage` of `len` bytes fits below the next slot (the block table: below
/// `total`).
pub(in crate::model) fn stage_fits(stage: Stage, len: usize, total: usize) -> bool {
    let end = match stage {
        Stage::Block => Stage::TokenIds as usize,
        Stage::TokenIds => Stage::Positions as usize,
        Stage::Positions => Stage::Slots as usize,
        Stage::Slots => Stage::SeqLens as usize,
        Stage::SeqLens => Stage::BlockTable as usize,
        Stage::BlockTable => total,
    };
    stage as usize + len <= end
}

/// 2026-10-11: The process's staging region `(address, bytes)`: one model per process (one
/// rank), allocated on first use for the largest block table this model can stage, held for the
/// process's life. A device-step upload rewrites its slot only after the host read the previous
/// verify back, which every copy staged for that verify preceded on the stream, so a slot is
/// never rewritten under a pending copy.
static STAGING: std::sync::OnceLock<(usize, usize)> = std::sync::OnceLock::new();

/// 2026-10-11: The lever. Read once; every rank must agree (`comm_setup`).
pub(crate) fn device_step_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_DEVICE_STEP").as_deref() == Ok("1"))
}

/// 2026-10-11: Whether a propose of `num_drafts` drafts may leave its draft on the device:
/// requested (`device`), the lever, one draft, and no grammar mask (a masked draft is chosen on
/// the host).
pub(super) fn wants_device_draft(
    device: bool,
    lever: bool,
    num_drafts: usize,
    grammar: bool,
) -> bool {
    device && lever && num_drafts == 1 && !grammar
}

impl TransformerModel {
    /// 2026-10-11: The end of every MTP propose (`run_mtp_propose_inner`): the accepted drafts'
    /// rows (`METRALE_MTP_ACCEPTED_ROWS`, `accepted_rows.rs`), then the propose, which leaves its
    /// draft on the device and returns `[DEVICE_DRAFT]` when [`wants_device_draft`] holds and the
    /// proposer can (`propose_device`), and otherwise proposes and reads back as before.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::model) fn finish_mtp_propose(
        &self,
        proposer: &dyn DraftProposer,
        token: u32,
        position: usize,
        num_drafts: usize,
        grammar_bitmask: Option<&[i32]>,
        device: bool,
        tokens: &[u32],
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        if self.levers.mtp_accepted_rows {
            self.feed_accepted_rows(proposer, tokens, position, state, ctx, stream);
        }
        if wants_device_draft(
            device,
            device_step_enabled(),
            num_drafts,
            grammar_bitmask.is_some(),
        ) && proposer.propose_device(
            token,
            self.mtp_hidden_save,
            position,
            state,
            ctx,
            stream,
        )? {
            return Ok(vec![DEVICE_DRAFT]);
        }
        proposer.propose(
            token,
            self.mtp_hidden_save,
            position,
            num_drafts,
            state,
            ctx,
            stream,
            None,
            grammar_bitmask,
            self.dflash_hidden_save,
        )
    }

    /// 2026-10-11: Upload `bytes` to `dst` on `stream`: staged in [`Stage`] slot `stage` and
    /// copied without a host wait when `device` (a device-step verify), else a plain async copy.
    pub(in crate::model) fn k2_upload(
        &self,
        device: bool,
        stage: Stage,
        bytes: &[u8],
        dst: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if !device {
            return self.gpu.copy_h2d_async(bytes, dst, stream);
        }
        let &(base, total) = match STAGING.get() {
            Some(s) => s,
            None => {
                let total = staging_bytes(2 * self.max_blocks_per_seq as usize * 4);
                let p = self.gpu.alloc_host_pinned(total)? as usize;
                STAGING.get_or_init(|| (p, total))
            }
        };
        anyhow::ensure!(
            stage_fits(stage, bytes.len(), total),
            "device-step upload of {} bytes overflows slot {stage:?}",
            bytes.len()
        );
        // 2026-10-11: SAFETY: `base..base + total` is the page-locked region allocated above for
        // this process and never freed; the slot bounds were checked, and no copy staged from
        // this slot is pending (see `STAGING`). Only the scheduler thread runs model calls.
        let staged = unsafe {
            let dst_host = (base as *mut u8).add(stage as usize);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst_host, bytes.len());
            std::slice::from_raw_parts(dst_host as *const u8, bytes.len())
        };
        self.gpu.copy_h2d_async_retained(staged, dst, stream)
    }

    /// 2026-10-11: Rank 0's announcement of a device-step K=2 verify: the block staged and
    /// broadcast without waiting, as the fed decode announces (`decode_fed_ep::send_fed_step`),
    /// so the host goes on to queue the verify behind the propose in flight.
    pub(in crate::model) fn ep_announce_device_step(
        &self,
        block: &[u8; metrale_model_layers::ep_wire::CMD_BLOCK_BYTES],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        let comm = self
            .comm
            .as_ref()
            .context("multi-rank protocol without a communicator")?;
        let stream = self.gpu.default_stream();
        self.k2_upload(true, Stage::Block, block, self.ep_cmd_buf, stream)?;
        comm.broadcast_enqueue(self.ep_cmd_buf.0, block.len(), 0)
    }

    /// 2026-10-11: The device cell holding `seq`'s draft left by `propose_device`.
    fn device_draft_cell(&self, seq: &mut SequenceState) -> Result<DevicePtr> {
        let proposer = self
            .proposer
            .as_ref()
            .context("a device draft without a proposer")?;
        let state = seq
            .proposer_state
            .as_mut()
            .context("a device draft without proposer state")?;
        proposer
            .device_draft(state.as_mut())
            .context("the proposer keeps no device draft")
    }

    /// 2026-10-11: Read `seq`'s device draft back, for a step that verifies it with a host id
    /// (`ModelDraft::read_device_draft`).
    pub(in crate::model) fn read_device_draft_dispatch(
        &self,
        seq: &mut SequenceState,
    ) -> Result<u32> {
        let cell = self.device_draft_cell(seq)?;
        let mut b = [0u8; 4];
        self.gpu.synchronize(self.gpu.default_stream())?;
        self.gpu.copy_d2h(cell, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    /// 2026-10-11: Row 1 of a K=2 verify whose draft is on the device: copy the draft id into
    /// `token_ids` row 1 (after the host ids were uploaded there) and gather its embedding into
    /// `row1`, the bytes `embed` copies for a host id (`batched_embed` + `scale_embeddings`, as
    /// the fed decode does). Every rank does it from its own drafter's cell.
    pub(in crate::model) fn verify_device_row(
        &self,
        seq: &mut SequenceState,
        row1: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            self.ngram_embed.is_none() && self.batched_embed_kernel.0 != 0,
            "a device draft needs a plain embedding gather"
        );
        let cell = self.device_draft_cell(seq)?;
        let ids = self.buffers.token_ids();
        self.gpu.copy_d2d_async(cell, ids.offset(4), 4, stream)?;
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            ids.offset(4),
            self.embed_tokens.weight,
            row1,
            1,
            self.config.hidden_size as u32,
            stream,
        )?;
        self.scale_embeddings(row1, 1, stream)
    }

    /// 2026-10-11: The id row 1 verified, read back after the verify's own readback: the draft
    /// the host verdict judges and `seq.tokens` records.
    pub(in crate::model) fn verified_device_draft(&self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.gpu
            .copy_d2h(self.buffers.token_ids().offset(4), &mut b)?;
        Ok(u32::from_le_bytes(b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: Every slot holds what the K=2 verify stages into it (the 24-byte block, two
    /// token ids, two u32 positions, two i64 slots, two i32 lengths) and no more, and the block
    /// table fills the region exactly as sized.
    #[test]
    fn each_staging_slot_holds_its_upload_and_no_more() {
        let total = staging_bytes(2 * 16_384 * 4);
        assert_eq!(total % 4096, 0);
        for (stage, len) in [
            (Stage::Block, metrale_model_layers::ep_wire::CMD_BLOCK_BYTES),
            (Stage::TokenIds, 8),
            (Stage::Positions, 8),
            (Stage::Slots, 16),
            (Stage::SeqLens, 8),
            (Stage::BlockTable, 2 * 16_384 * 4),
        ] {
            assert!(stage_fits(stage, len, total), "{stage:?}");
        }
        assert!(!stage_fits(Stage::Slots, 72, total), "into the next slot");
        assert!(
            !stage_fits(Stage::BlockTable, total, total),
            "past the region"
        );
    }

    /// 2026-10-11: Only a requested, lever-on, one-draft, grammarless propose leaves its draft on
    /// the device; a chain, a masked draft, the lever off or a caller that did not ask (the
    /// workers, every other step) read back as before.
    #[test]
    fn only_a_lone_unmasked_requested_draft_stays_on_the_device() {
        assert!(wants_device_draft(true, true, 1, false));
        assert!(!wants_device_draft(false, true, 1, false), "not requested");
        assert!(!wants_device_draft(true, false, 1, false), "lever off");
        assert!(!wants_device_draft(true, true, 2, false), "a chain");
        assert!(!wants_device_draft(true, true, 0, false), "no draft");
        assert!(!wants_device_draft(true, true, 1, true), "a grammar mask");
    }
}
