// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The tensor-parallel drafter's propose (`tp.rs`): one collective step that rank
//! 0 starts from `propose_drafts` (after the model sent `EP_CMD_DFLASH_PROPOSE`) and the
//! other ranks run from that command (`propose_tp_follower`).
//!
//! Collectives, in order, on every rank: a setup-status all-gather; a header-staging status
//! all-gather; the fixed [`Header`] broadcast; a decode-status all-gather; the optional overflow
//! positions broadcast; an overflow-read status all-gather; per `ctx_window` chunk, the
//! broadcast of rank 0's `fc` rows and status exchanges around local K/V writes; in the block
//! forward, the all-reduce after o_proj and after down of each layer and one all-gather of the
//! top-16 lists.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants:
//! - After the command, every local setup failure is reported by the phase's common status
//!   all-gather before ranks enter the next protocol phase; context-chunk failures are agreed
//!   after the aligned payload broadcast and before K/V consumption, the next chunk or forward.
//! - The other ranks read no context state of their own: the header carries the block, the
//!   attention's context length and the new rows' positions, and the rows' `fc` comes from
//!   rank 0; each rank writes its K/V heads at those context rows through its own block table.
//! - A rank-local failure inside `forward_block` itself is not agreed before that block's next
//!   TP all-reduce; layer-forward error coordination remains outside this propose-level fix.

use anyhow::{Context, Result};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::speculative::ProposerState;

use super::dflash2::{Topk16Merge, topk16_parts};
use super::tp::{HEADER_INLINE, HEADER_WORDS, Header};
use super::{BlockDiffusionDraftHead, DflashProposerState};

/// 2026-10-10: The drafter cache's block size (`from_weights.rs`).
const BLOCK_SIZE: u32 = 16;

/// Resolve one rank's local preflight result against the rank-ordered statuses returned by an
/// all-gather. Kept separate from device I/O so the root/worker failure behavior is testable.
fn resolve_preflight<T>(local: Result<T>, statuses: &[u32], rank: usize, phase: &str) -> Result<T> {
    anyhow::ensure!(
        rank < statuses.len(),
        "DFlash TP {phase}: invalid rank {rank}"
    );
    anyhow::ensure!(
        statuses.iter().all(|&s| s <= 1),
        "DFlash TP {phase}: invalid preflight status"
    );
    anyhow::ensure!(
        statuses[rank] == u32::from(local.is_err()),
        "DFlash TP {phase}: gathered local status disagrees with local result"
    );
    if let Some(failed_rank) = statuses.iter().position(|&s| s != 0) {
        return match local {
            Err(e) => Err(e).with_context(|| format!("DFlash TP {phase}: local setup failed")),
            Ok(_) => anyhow::bail!(
                "DFlash TP {phase}: rank {failed_rank} failed local setup; propose cancelled"
            ),
        };
    }
    local
}

/// Always execute the status exchange before resolving a local error. This ordering is the
/// safety property: no rank returns from setup while another advances to a different collective.
pub(super) fn agree_preflight<T>(
    local: Result<T>,
    rank: usize,
    phase: &str,
    gather: impl FnOnce(u32) -> Result<Vec<u32>>,
) -> Result<T> {
    let statuses = gather(u32::from(local.is_err()))?;
    resolve_preflight(local, &statuses, rank, phase)
}

/// Run both independent pre-broadcast operations, retaining the first local error. In
/// particular, an unsuccessful slot fill must not prevent rank 0 from producing the payload
/// that all ranks are about to broadcast.
fn run_context_prepare(
    fill_slots: impl FnOnce() -> Result<()>,
    project_fc: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let slots = fill_slots();
    let projection = project_fc();
    match slots {
        Err(error) => Err(error),
        Ok(()) => projection,
    }
}

/// Keep every rank on the same per-chunk collective path. Payload broadcast is attempted even
/// when local preparation failed; agreement then prevents any rank from consuming that payload
/// or entering the next chunk. A broadcast/collective error propagates immediately, so no later
/// collective is attempted on a failed communicator.
fn run_context_chunk(
    prepare: impl FnOnce() -> Result<()>,
    broadcast: impl FnOnce() -> Result<()>,
    mut agree: impl FnMut(Result<()>, &str) -> Result<()>,
    write_kv: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let prepared = prepare();
    broadcast()?;
    agree(prepared, "context payload")?;

    let written = write_kv();
    agree(written, "context K/V write")
}

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
        total_positions <= max_seq_len,
        "DFlash TP header: {total_positions} context positions exceed max_seq_len {max_seq_len}"
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
        "DFlash TP header: context position range ends at {ship_end}, max_seq_len is {max_seq_len}"
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

    /// 2026-10-10: Exchange one status word on every rank even when local setup failed. The
    /// caller must invoke this at the same protocol boundary on every rank.
    pub(super) fn tp_agree_preflight<T>(
        &self,
        ctx: &ForwardContext,
        local: Result<T>,
        phase: &str,
    ) -> Result<T> {
        let comm = self.tp_comm(ctx)?;
        let tp = self
            .tp
            .context("DFlash TP preflight without tensor parallelism")?;
        agree_preflight(local, tp.rank, phase, |status| {
            ctx.gpu
                .copy_h2d(&status.to_le_bytes(), self.scratch.tp_send)?;
            comm.all_gather(self.scratch.tp_send.0, self.scratch.tp_recv.0, 4)?;
            let mut gathered = vec![0u8; comm.world_size() * 4];
            ctx.gpu.copy_d2h(self.scratch.tp_recv, &mut gathered)?;
            Ok(gathered
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        })
    }

    /// 2026-10-10: Sum `bytes` of BF16 at `ptr` over the ranks, in place, on the forward's
    /// stream (capturable); nothing without tensor parallelism.
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
        self.tp_comm(ctx)?.all_reduce_async(ptr.0, bytes, stream)
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
    /// at `position`, from `propose_drafts` (block width set, decode append done): the block
    /// table, the header, the new context rows' K/V on every rank, the block forward.
    pub(super) fn propose_tp_root(
        &self,
        last_token: u32,
        position: usize,
        num_drafts: usize,
        dstate: &mut DflashProposerState,
        bt: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<u32>> {
        let comm = self.tp_comm(ctx)?;
        let header = (|| -> Result<Option<Header>> {
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
            let ctx_count = if self.levers.option_b_no_ctx {
                0
            } else {
                dstate.ctx_len
            };
            Ok(Some(Header {
                width: self.block_g(),
                anchor: last_token,
                position,
                ctx_count,
                ship_start: committed,
                positions: dstate.ctx_positions[committed..dstate.ctx_len].to_vec(),
            }))
        })();
        let Some(header) = self.tp_exchange_header(ctx, comm, header)? else {
            anyhow::bail!("DFlash TP root: unexpected abort header");
        };
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

    /// 2026-10-10: A non-root rank's half of a propose on `state` (the command's slot): read
    /// the header (an abort returns at once), set the block width, write this rank's K/V of
    /// the new context rows, run the block forward. Its drafts are discarded.
    pub(super) fn propose_tp_follower(
        &self,
        state: &mut dyn ProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let comm = self.tp_comm(ctx)?;
        let local = (|| -> Result<DevicePtr> {
            let dstate = state
                .as_any_mut()
                .downcast_mut::<DflashProposerState>()
                .context("DFlash TP follower: invalid proposer state")?;
            self.ensure_block_table(dstate, ctx)
        })();
        let bt = self.tp_agree_preflight(ctx, local, "worker setup")?;
        let Some(header) = self.tp_exchange_header(ctx, comm, Ok(None))? else {
            return Ok(());
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

    /// 2026-10-10: Every rank calls this with `Some(header)` on rank 0 and `None` elsewhere.
    /// The root stages both header buffers before the stage-status gather. After the fixed
    /// broadcast, all ranks agree that decode/validation succeeded before any optional overflow
    /// broadcast; a second agreement follows overflow readback before context rows can begin.
    fn tp_exchange_header(
        &self,
        ctx: &ForwardContext,
        comm: &dyn CommBackend,
        root_header: Result<Option<Header>>,
    ) -> Result<Option<Header>> {
        let tp = self
            .tp
            .context("DFlash TP header exchange without tensor parallelism")?;
        let buf = self.scratch.tp_header;
        let staged = if tp.rank == 0 {
            root_header.and_then(|header| {
                let header = header.context("DFlash TP root did not provide its header")?;
                self.validate_header(&header, header.positions.len())?;
                let words = header.encode();
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                ctx.gpu.copy_h2d(&bytes, buf)?;
                if !header.overflow().is_empty() {
                    let more: Vec<u8> = header
                        .overflow()
                        .iter()
                        .flat_map(|p| p.to_le_bytes())
                        .collect();
                    ctx.gpu.copy_h2d(&more, buf.offset(HEADER_WORDS * 4))?;
                }
                Ok(Some(header))
            })
        } else {
            Ok(None)
        };
        let staged = self.tp_agree_preflight(ctx, staged, "header staging")?;

        comm.broadcast(buf.0, HEADER_WORDS * 4, 0)?;
        let decoded = (|| -> Result<(Option<Header>, usize)> {
            let mut raw = vec![0u8; HEADER_WORDS * 4];
            ctx.gpu.copy_d2h(buf, &mut raw)?;
            let mut words = [0u32; HEADER_WORDS];
            for (w, c) in words.iter_mut().zip(raw.chunks_exact(4)) {
                *w = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
            let overflow_len = Header::overflow_len(&words);
            let Some(header) = Header::decode(&words)? else {
                anyhow::ensure!(
                    overflow_len == 0,
                    "DFlash TP abort header has overflow rows"
                );
                return Ok((None, 0));
            };
            let total = words[6] as usize;
            self.validate_header(&header, total)?;
            let full = if tp.rank == 0 {
                let full = staged
                    .as_ref()
                    .context("DFlash TP root lost its staged header")?;
                anyhow::ensure!(
                    header.width == full.width
                        && header.anchor == full.anchor
                        && header.position == full.position
                        && header.ctx_count == full.ctx_count
                        && header.ship_start == full.ship_start
                        && header.positions
                            == full.positions[..full.positions.len().min(HEADER_INLINE)],
                    "DFlash TP staged header did not match its encoded words"
                );
                full.clone()
            } else {
                header
            };
            Ok((Some(full), overflow_len))
        })();
        let (header, overflow_len) = self.tp_agree_preflight(ctx, decoded, "header decode")?;
        let Some(mut header) = header else {
            return Ok(None);
        };

        let overflow = (|| -> Result<Header> {
            if overflow_len > 0 {
                let at = buf.offset(HEADER_WORDS * 4);
                comm.broadcast(at.0, overflow_len * 4, 0)?;
                if tp.rank != 0 {
                    let mut raw = vec![0u8; overflow_len * 4];
                    ctx.gpu.copy_d2h(at, &mut raw)?;
                    header.positions.extend(
                        raw.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])),
                    );
                }
            }
            anyhow::ensure!(
                header.positions.len() == overflow_len + header.positions.len().min(HEADER_INLINE),
                "DFlash TP header position count does not match its overflow"
            );
            Ok(header)
        })();
        self.tp_agree_preflight(ctx, overflow, "header overflow")
            .map(Some)
    }

    /// Validate fields used to size or address local buffers before their values reach a later
    /// collective or device operation. `total_positions` comes from the fixed words, before
    /// overflow storage is read.
    fn validate_header(&self, header: &Header, total_positions: usize) -> Result<()> {
        validate_tp_header(header, total_positions, self.gamma, self.max_seq_len)
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
            run_context_chunk(
                || {
                    run_context_prepare(
                        || {
                            metrale_model_layers::layers::ops::fill_slots_from_block_table(
                                ctx.gpu,
                                self.kernels.fill_slots,
                                self.scratch.slot_mapping_dev,
                                bt,
                                start as u32,
                                count as u32,
                                BLOCK_SIZE,
                                stream,
                            )
                            .context("DFlash TP context slot fill failed")
                        },
                        || {
                            if let Some(acc) = taps {
                                self.ctx_fc_rows(
                                    acc.offset(start * slot_bytes),
                                    count,
                                    ctx,
                                    stream,
                                    false,
                                )
                                .context("DFlash TP context fc projection failed")
                            } else {
                                Ok(())
                            }
                        },
                    )
                },
                || comm.broadcast(self.scratch.fc_proj.0, count * self.hidden_size * 2, 0),
                |local, phase| self.tp_agree_preflight(ctx, local, phase),
                || {
                    self.ctx_kv_from_fc(
                        count,
                        &header.positions[done..done + count],
                        self.scratch.slot_mapping_dev,
                        ctx,
                        stream,
                        true,
                        false,
                    )
                    .context("DFlash TP context K/V write failed")
                },
            )?;
            done += count;
        }
        Ok(())
    }
}

#[cfg(test)]
mod context_chunk_tests {
    use std::cell::RefCell;
    use std::collections::VecDeque;

    use anyhow::{Result, anyhow};

    use super::{agree_preflight, run_context_chunk, run_context_prepare};

    /// GPU-free scripted stand-in for the per-chunk collectives. Status vectors are ordered by
    /// rank, just like `CommBackend::all_gather` results.
    struct ScriptedCollectives {
        rank: usize,
        statuses: VecDeque<Vec<u32>>,
        events: Vec<&'static str>,
    }

    impl ScriptedCollectives {
        fn new(rank: usize, statuses: impl IntoIterator<Item = Vec<u32>>) -> Self {
            Self {
                rank,
                statuses: statuses.into_iter().collect(),
                events: Vec::new(),
            }
        }

        fn agree(&mut self, local: Result<()>, phase: &str) -> Result<()> {
            self.events.push(match phase {
                "context payload" => "status:payload",
                "context K/V write" => "status:kv",
                _ => "status:unexpected",
            });
            let statuses = self
                .statuses
                .pop_front()
                .ok_or_else(|| anyhow!("script ran out of status exchanges"))?;
            agree_preflight(local, self.rank, phase, move |_| Ok(statuses))
        }
    }

    #[test]
    fn context_chunk_success_orders_broadcast_and_both_statuses() {
        let script = RefCell::new(ScriptedCollectives::new(
            1,
            [vec![0, 0], vec![0, 0]],
        ));
        let result = run_context_chunk(
            || {
                script.borrow_mut().events.push("prepare");
                Ok(())
            },
            || {
                script.borrow_mut().events.push("broadcast");
                Ok(())
            },
            |local, phase| script.borrow_mut().agree(local, phase),
            || {
                script.borrow_mut().events.push("write-kv");
                Ok(())
            },
        );

        assert!(result.is_ok());
        assert_eq!(
            script.borrow().events.as_slice(),
            &["prepare", "broadcast", "status:payload", "write-kv", "status:kv"]
        );
        assert!(script.borrow().statuses.is_empty());
    }

    #[test]
    fn pre_broadcast_error_still_broadcasts_then_stops_before_kv() {
        let script = RefCell::new(ScriptedCollectives::new(0, [vec![1, 0]]));
        let result = run_context_chunk(
            || {
                run_context_prepare(
                    || {
                        script.borrow_mut().events.push("prepare");
                        Ok(())
                    },
                    || {
                        script.borrow_mut().events.push("fc-projection");
                        Err(anyhow!("fc projection failed"))
                    },
                )
            },
            || {
                script.borrow_mut().events.push("broadcast");
                Ok(())
            },
            |local, phase| script.borrow_mut().agree(local, phase),
            || {
                script.borrow_mut().events.push("write-kv");
                Ok(())
            },
        );

        assert!(format!("{:#}", result.unwrap_err()).contains("fc projection failed"));
        assert_eq!(
            script.borrow().events.as_slice(),
            &["prepare", "fc-projection", "broadcast", "status:payload"]
        );
        assert!(script.borrow().statuses.is_empty());
    }

    #[test]
    fn post_broadcast_kv_error_is_agreed_before_returning() {
        let script = RefCell::new(ScriptedCollectives::new(
            1,
            [vec![0, 0], vec![0, 1]],
        ));
        let result = run_context_chunk(
            || {
                script.borrow_mut().events.push("prepare");
                Ok(())
            },
            || {
                script.borrow_mut().events.push("broadcast");
                Ok(())
            },
            |local, phase| script.borrow_mut().agree(local, phase),
            || {
                script.borrow_mut().events.push("write-kv");
                Err(anyhow!("K/V write failed"))
            },
        );

        assert!(format!("{:#}", result.unwrap_err()).contains("K/V write failed"));
        assert_eq!(
            script.borrow().events.as_slice(),
            &["prepare", "broadcast", "status:payload", "write-kv", "status:kv"]
        );
        assert!(script.borrow().statuses.is_empty());
    }

    #[test]
    fn broadcast_transport_error_does_not_attempt_status_or_another_collective() {
        let script = RefCell::new(ScriptedCollectives::new(
            1,
            std::iter::empty::<Vec<u32>>(),
        ));
        let result = run_context_chunk(
            || {
                script.borrow_mut().events.push("prepare");
                Ok(())
            },
            || {
                script.borrow_mut().events.push("broadcast");
                Err(anyhow!("broadcast transport failed"))
            },
            |local, phase| script.borrow_mut().agree(local, phase),
            || {
                script.borrow_mut().events.push("write-kv");
                Ok(())
            },
        );

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("broadcast transport failed"));
        assert_eq!(script.borrow().events.as_slice(), &["prepare", "broadcast"]);
        assert!(script.borrow().statuses.is_empty());
    }

    #[test]
    fn status_transport_error_does_not_attempt_kv_or_another_collective() {
        let script = RefCell::new(ScriptedCollectives::new(
            1,
            std::iter::empty::<Vec<u32>>(),
        ));
        let result = run_context_chunk(
            || {
                script.borrow_mut().events.push("prepare");
                Ok(())
            },
            || {
                script.borrow_mut().events.push("broadcast");
                Ok(())
            },
            |local, phase| script.borrow_mut().agree(local, phase),
            || {
                script.borrow_mut().events.push("write-kv");
                Ok(())
            },
        );

        assert!(result
            .unwrap_err()
            .to_string()
            .contains("script ran out of status exchanges"));
        assert_eq!(
            script.borrow().events.as_slice(),
            &["prepare", "broadcast", "status:payload"]
        );
    }

    #[test]
    fn slot_fill_error_does_not_skip_independent_fc_projection() {
        let events = RefCell::new(Vec::new());
        let result = run_context_prepare(
            || {
                events.borrow_mut().push("slot-fill");
                Err(anyhow!("slot fill failed"))
            },
            || {
                events.borrow_mut().push("fc-projection");
                Ok(())
            },
        );

        assert!(result.unwrap_err().to_string().contains("slot fill failed"));
        assert_eq!(events.borrow().as_slice(), &["slot-fill", "fc-projection"]);
    }
}
