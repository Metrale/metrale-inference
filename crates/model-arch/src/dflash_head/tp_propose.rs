// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel drafter's propose (`tp.rs`): one collective step that rank
//! 0 starts from `propose_drafts` (after the model sent `EP_CMD_DFLASH_PROPOSE`) and the
//! other ranks run from that command (`propose_tp_follower`).
//!
//! Collectives, in order, on every rank: one all-gather of `[status, header]` blocks (rank 0's
//! block carries the [`Header`]); the header's overflow positions, broadcast when it has any;
//! per `ctx_window` chunk of the context rows to add, the broadcast of rank 0's `fc` rows; in
//! the block forward, the all-reduce after o_proj and after down of each layer and one
//! all-gather of the top-16 lists.
//!
//! The exchange costs one host round trip where a status and a header broadcast cost two, and
//! it carries every rank's readiness, so a rank that cannot propose (no state for the slot, no
//! block table, a bad header on rank 0) stops every rank at the same point.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - Every rank takes part in the exchange on every path, ready or not, and returns an error on
//!   every rank when any rank is not ready, before any later collective.
//! - The other ranks read no context state of their own: the header carries the block, the
//!   attention's context length and the new rows' positions, and the rows' `fc` comes from
//!   rank 0; each rank writes its K/V heads at those context rows through its own block table.
//! - After the exchange a rank-local failure (a failed launch) is not agreed: it surfaces as an
//!   error on that rank and the others wait in their next collective.

use anyhow::{Context, Result};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::ProposerState;

use super::dflash2::{Topk16Merge, topk16_parts};
use super::tp::{HEADER_WORDS, Header};
use super::{BlockDiffusionDraftHead, DflashProposerState};

/// 2026-10-10: The drafter cache's block size (`from_weights.rs`).
const BLOCK_SIZE: u32 = 16;
/// 2026-10-10: Words of one rank's exchange block: its status, then the header words.
pub(super) const EXCHANGE_WORDS: usize = 1 + HEADER_WORDS;
const STATUS_READY: u32 = 0x5244_5931;
const STATUS_FAILED: u32 = 0x4641_494c;

/// 2026-10-10: One rank's exchange block: its status and, on rank 0 when ready, the header.
pub(super) fn exchange_block(ready: bool, header: Option<&Header>) -> [u32; EXCHANGE_WORDS] {
    let mut w = [0u32; EXCHANGE_WORDS];
    w[0] = if ready { STATUS_READY } else { STATUS_FAILED };
    if let Some(h) = header {
        w[1..].copy_from_slice(&h.encode());
    }
    w
}

/// 2026-10-10: Every rank's gathered blocks (rank-major) resolved the same way on every rank:
/// an error naming the ranks that are not ready (or sent a malformed status), else rank 0's
/// header and how many of its positions come in the overflow broadcast.
pub(super) fn resolve_exchange(blocks: &[u32], world: usize) -> Result<(Header, usize)> {
    anyhow::ensure!(
        blocks.len() == world * EXCHANGE_WORDS,
        "DFlash TP exchange: {} words for {world} ranks",
        blocks.len()
    );
    let failed: Vec<usize> = (0..world)
        .filter(|&r| blocks[r * EXCHANGE_WORDS] != STATUS_READY)
        .collect();
    anyhow::ensure!(
        failed.is_empty(),
        "DFlash TP propose cancelled on every rank: rank(s) {failed:?} not ready"
    );
    let mut words = [0u32; HEADER_WORDS];
    words.copy_from_slice(&blocks[1..EXCHANGE_WORDS]);
    let header = Header::decode(&words)?.context("DFlash TP exchange: rank 0 sent no header")?;
    Ok((header, Header::overflow_len(&words)))
}

/// 2026-10-10: Refuse a header whose sizes would address past this rank's buffers.
pub(super) fn validate_tp_header(
    header: &Header,
    total_positions: usize,
    gamma: usize,
    max_seq_len: usize,
) -> Result<()> {
    anyhow::ensure!(
        header.width >= 2 && header.width <= gamma.max(2),
        "DFlash TP header: block width {} outside 2..={}",
        header.width,
        gamma.max(2)
    );
    anyhow::ensure!(
        header.ctx_count <= max_seq_len,
        "DFlash TP header: context count {} exceeds max_seq_len {max_seq_len}",
        header.ctx_count
    );
    let ship_end = header
        .ship_start
        .checked_add(total_positions)
        .context("DFlash TP header: context position range overflow")?;
    anyhow::ensure!(
        ship_end <= max_seq_len,
        "DFlash TP header: context rows end at {ship_end}, max_seq_len is {max_seq_len}"
    );
    Ok(())
}

impl BlockDiffusionDraftHead {
    /// 2026-10-10: The communicator of a tensor-parallel forward; an error without one (the
    /// model gives it to a proposer whose `needs_comm` is true).
    fn tp_comm<'a>(&self, ctx: &ForwardContext<'a>) -> Result<&'a dyn CommBackend> {
        ctx.comm
            .context("DFlash TP: the drafter forward has no communicator")
    }

    /// 2026-10-10: Sum `bytes` of BF16 at `ptr` over the ranks, in place; nothing without
    /// tensor parallelism. It is enqueued on the communicator's stream, the model's default
    /// stream the drafter runs on (`stream` must be it), with no event and no host wait, so a
    /// captured block graph holds it as the target's decode graphs hold theirs.
    pub(super) fn tp_all_reduce(
        &self,
        ctx: &ForwardContext,
        ptr: DevicePtr,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if self.tp.is_none() {
            return Ok(());
        }
        anyhow::ensure!(
            stream == ctx.gpu.default_stream(),
            "DFlash TP: the drafter forward runs on stream {stream:#x}, not the default stream \
             its collectives use"
        );
        self.tp_comm(ctx)?.all_reduce(ptr.0, bytes)
    }

    /// 2026-10-10: The top-16 of `rows` rows over the whole vocab from this rank's slice of
    /// it: the part kernel at this rank's first vocab row, an unpadded merge into `tp_send`
    /// (`[vals rows*16 | idx rows*16]`), an all-gather into `tp_recv` (rank-major) and the
    /// padded merge of every rank's lists into `sel_vals` / `sel_idx`. The lists hold global
    /// indices and the kernels' tie order is global, so the result is the one-block kernel's
    /// on the whole row (`dflash2.cu`); no logits are overwritten.
    pub(super) fn tp_topk16(&self, ctx: &ForwardContext, rows: u32, stream: u64) -> Result<()> {
        let tp = self
            .tp
            .context("DFlash TP top-16 without tensor parallelism")?;
        let gpu = ctx.gpu;
        let n = self.vocab_size as u32;
        let base = tp.vocab_base as u32;
        let parts = topk16_parts(self.vocab_size) as u32;
        let list = rows as usize * 16 * 4;
        self.launch_topk16_part(gpu, self.scratch.logits, rows, n, base, stream)?;
        self.launch_topk16_merge(
            gpu,
            None,
            (self.scratch.sel_cand_vals, self.scratch.sel_cand_idx),
            (self.scratch.tp_send, self.scratch.tp_send.offset(list)),
            rows,
            Topk16Merge {
                n,
                parts,
                base,
                part_stride: 16,
                row_stride: parts * 16,
                pad: false,
            },
            stream,
        )?;
        self.tp_comm(ctx)?
            .all_gather(self.scratch.tp_send.0, self.scratch.tp_recv.0, 2 * list)?;
        self.launch_topk16_merge(
            gpu,
            None,
            (self.scratch.tp_recv, self.scratch.tp_recv.offset(list)),
            (self.scratch.sel_vals, self.scratch.sel_idx),
            rows,
            Topk16Merge {
                n,
                parts: tp.world as u32,
                base,
                part_stride: rows * 32,
                row_stride: 16,
                pad: true,
            },
            stream,
        )
    }

    /// 2026-10-10: Rank 0's tensor-parallel propose of `num_drafts` drafts after `last_token`
    /// at `position`, from `propose_drafts` (block width set): its local setup (the decode
    /// append, the block table, the header), the exchange, the new context rows' K/V on every
    /// rank, the block forward.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn propose_tp_root(
        &self,
        last_token: u32,
        position: usize,
        num_drafts: usize,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
        target_hidden_stack: Option<DevicePtr>,
    ) -> Result<Vec<u32>> {
        let local = (|| -> Result<(DevicePtr, Header)> {
            let dstate = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .context("DFlash TP root: invalid proposer state")?;
            self.append_decode_ctx(
                dstate,
                last_token,
                position,
                ctx,
                stream,
                target_hidden_stack,
            )?;
            let bt = self.ensure_block_table(dstate, ctx)?;
            anyhow::ensure!(
                dstate.ctx_positions.len() == dstate.ctx_len,
                "DFlash TP root: {} ctx positions for {} ctx rows",
                dstate.ctx_positions.len(),
                dstate.ctx_len
            );
            let committed = if self.levers.full_precompute {
                0
            } else {
                dstate.ctx_committed.min(dstate.ctx_len)
            };
            let header = Header {
                width: self.block_g(),
                anchor: last_token,
                position,
                ctx_count: if self.levers.option_b_no_ctx {
                    0
                } else {
                    dstate.ctx_len
                },
                ship_start: committed,
                positions: dstate.ctx_positions[committed..dstate.ctx_len].to_vec(),
            };
            validate_tp_header(
                &header,
                header.positions.len(),
                self.gamma,
                self.max_seq_len,
            )?;
            Ok((bt, header))
        })();
        let (bt, header) = match local {
            Ok((bt, header)) => {
                self.tp_exchange(ctx, true, Some(&header))?;
                (bt, header)
            }
            Err(e) => {
                // 2026-10-10: The exchange then fails on every rank; this rank's own error
                // is the one to report.
                let _ = self.tp_exchange(ctx, false, None);
                return Err(e.context("DFlash TP root setup failed; propose cancelled"));
            }
        };
        let dstate = state
            .as_any_mut()
            .downcast_mut::<DflashProposerState>()
            .context("DFlash TP root: invalid proposer state")?;
        self.tp_ctx_rows(&header, Some(dstate.ctx_hidden_acc), bt, ctx, stream)?;
        dstate.ctx_committed = dstate.ctx_len;
        dstate.ctx_count_drafter = dstate.ctx_len;
        let drafts = self.forward_block(
            last_token,
            position,
            ctx,
            stream,
            None,
            Some((bt, header.ctx_count as u32)),
            None,
        )?;
        Ok(self.finish_drafts(drafts, last_token, position, num_drafts, dstate))
    }

    /// 2026-10-10: A non-root rank's half of a propose on `state`, the command's slot (`None`
    /// when this rank has none, which it reports in the exchange): its block table, the
    /// exchange, this rank's K/V of the new context rows, the block forward. Its drafts are
    /// discarded.
    pub(super) fn propose_tp_follower(
        &self,
        state: Option<&mut dyn ProposerState>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let local = (|| -> Result<DevicePtr> {
            let dstate = state
                .context("DFlash TP follower: no proposer state for the slot")?
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .context("DFlash TP follower: invalid proposer state")?;
            self.ensure_block_table(dstate, ctx)
        })();
        let header = self.tp_exchange(ctx, local.is_ok(), None);
        let (bt, header) = match (local, header) {
            (Ok(bt), Ok(header)) => (bt, header),
            (Err(e), _) => {
                return Err(e.context("DFlash TP follower setup failed; propose cancelled"));
            }
            (Ok(_), Err(e)) => return Err(e),
        };
        self.block_gamma
            .store(header.width, std::sync::atomic::Ordering::Relaxed);
        self.tp_ctx_rows(&header, None, bt, ctx, stream)?;
        self.forward_block(
            header.anchor,
            header.position,
            ctx,
            stream,
            None,
            Some((bt, header.ctx_count as u32)),
            None,
        )?;
        Ok(())
    }

    /// 2026-10-10: The exchange: all-gather every rank's `[status, header]` block (rank 0's
    /// header when `header` is given), resolve it the same way on every rank, then broadcast
    /// the header's overflow positions when it has any. Every rank's header is rank 0's.
    fn tp_exchange(
        &self,
        ctx: &ForwardContext,
        ready: bool,
        header: Option<&Header>,
    ) -> Result<Header> {
        let comm = self.tp_comm(ctx)?;
        let world = comm.world_size();
        let mine: Vec<u8> = exchange_block(ready, header)
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        ctx.gpu.copy_h2d(&mine, self.scratch.tp_send)?;
        comm.all_gather(self.scratch.tp_send.0, self.scratch.tp_recv.0, mine.len())?;
        // 2026-10-10: The gather is enqueued on the default stream; the host reads its result.
        ctx.gpu.synchronize(ctx.gpu.default_stream())?;
        let mut raw = vec![0u8; world * mine.len()];
        ctx.gpu.copy_d2h(self.scratch.tp_recv, &mut raw)?;
        let blocks: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let (mut got, overflow) = resolve_exchange(&blocks, world)?;
        validate_tp_header(
            &got,
            got.positions.len() + overflow,
            self.gamma,
            self.max_seq_len,
        )?;
        if overflow > 0 {
            let at = self.scratch.tp_header;
            if let Some(h) = header {
                let more: Vec<u8> = h.overflow().iter().flat_map(|p| p.to_le_bytes()).collect();
                ctx.gpu.copy_h2d(&more, at)?;
            }
            comm.broadcast(at.0, overflow * 4, 0)?;
            let mut raw = vec![0u8; overflow * 4];
            ctx.gpu.copy_d2h(at, &mut raw)?;
            got.positions.extend(
                raw.chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
        }
        Ok(got)
    }

    /// 2026-10-10: The header's new context rows on every rank, in `ctx_window` chunks: rank 0
    /// (`taps` = its context accumulator) runs `fc` on the chunk's tap rows and broadcasts the
    /// result; every rank then writes its K/V heads for the chunk at the chunk's context rows
    /// through block table `bt`.
    fn tp_ctx_rows(
        &self,
        header: &Header,
        taps: Option<DevicePtr>,
        bt: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let comm = self.tp_comm(ctx)?;
        let total = header.positions.len();
        anyhow::ensure!(
            self.ctx_window > 0 || total == 0,
            "DFlash TP: {total} context rows to add with ctx_window = 0"
        );
        let slot_bytes = self.target_layer_ids.len() * self.target_hidden_size * 2;
        let mut done = 0;
        while done < total {
            let count = (total - done).min(self.ctx_window);
            let start = header.ship_start + done;
            metrale_model_layers::layers::ops::fill_slots_from_block_table(
                ctx.gpu,
                self.kernels.fill_slots,
                self.scratch.slot_mapping_dev,
                bt,
                start as u32,
                count as u32,
                BLOCK_SIZE,
                stream,
            )?;
            if let Some(acc) = taps {
                self.ctx_fc_rows(acc.offset(start * slot_bytes), count, ctx, stream, false)?;
            }
            comm.broadcast(self.scratch.fc_proj.0, count * self.hidden_size * 2, 0)?;
            self.ctx_kv_from_fc(
                count,
                &header.positions[done..done + count],
                self.scratch.slot_mapping_dev,
                ctx,
                stream,
                true,
                false,
            )?;
            done += count;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "tp_propose_tests.rs"]
mod tests;
