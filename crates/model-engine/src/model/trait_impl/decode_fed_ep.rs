// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The multi-rank fed decode step (`METRALE_EP_DEVICE_FEED=1`): rank 0 runs a
//! single-row decode step ahead of its host, fed from the previous step's device argmax, and
//! the token reaches the worker ranks device to device.
//!
//! Rank 0 (`ep_send_fed_step`, after `feed_resolve` wrote its feed ids) submits the step's
//! announcement on its compute stream without waiting: on the v3 wire one block
//! `[slot, EP_CMD_DECODE_FED, token]` with the token copied in on the device, otherwise three
//! broadcasts (slot word, [`EP_CMD_DECODE_FED`], then the 4-byte token from its feed ids). A
//! worker reads the preamble with `ep_recv_seq_and_cmd`, moves the token into its feed ids on
//! the device, and runs the same fed decode (`ep_worker_decode_fed`). Neither side reads the
//! token on the host before the step's graph is queued.
//!
//! Owner: model-engine.
//! Invariants:
//! - Rank 0 and every worker issue the same broadcasts, from root 0, in the same order for a
//!   fed step: the block (v3), or slot word, command word, token (v2).
//! - Rank 0 never waits on these broadcasts; its stream orders them after the step in flight
//!   and before the fed step's graph.

use anyhow::{Context, Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::ep_wire::{self, EpWire};

use super::super::types::TransformerModel;
use super::feed::DecodeInput;
use crate::traits::SequenceState;

/// 2026-10-10: Worker command: run a fed single-row decode in the preamble's slot; the token
/// follows as a 4-byte broadcast from rank 0's feed ids.
pub(in crate::model) const EP_CMD_DECODE_FED: u32 = 0xFFFF_FFE1;

// 2026-10-10: Far above any vocabulary, so a worker never decodes it as a token id.
const _: () = assert!(EP_CMD_DECODE_FED > 1 << 24);

/// 2026-10-10: The lever. `1` turns the multi-rank fed decode on; read once per process.
const ENV: &str = "METRALE_EP_DEVICE_FEED";

fn lever_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var(ENV).as_deref() == Ok("1"))
}

/// 2026-10-10: Rank 0's half of a fed step, submitted on `stream` without a host wait. On the
/// v3 wire (`one_block`) it is one broadcast of the block `[slot, EP_CMD_DECODE_FED, token]`,
/// the token copied into the block on the device; otherwise the slot word, the command word,
/// then the token in `feed_ids`. Host words are staged with pageable async copies, which the
/// driver stages before returning, so each is ordered after the broadcast before it has read
/// `cmd_buf`.
pub(in crate::model) fn send_fed_step(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    cmd_buf: DevicePtr,
    feed_ids: DevicePtr,
    stream: u64,
    slot: u32,
    one_block: bool,
) -> Result<()> {
    ensure!(comm.rank() == 0, "a fed step is announced by rank 0 only");
    if one_block {
        let block = ep_wire::host_block(slot, EP_CMD_DECODE_FED);
        gpu.copy_h2d_async(&block[..ep_wire::CMD_BLOCK_ARG], cmd_buf, stream)?;
        gpu.copy_d2d_async(feed_ids, cmd_buf.offset(ep_wire::CMD_BLOCK_ARG), 4, stream)?;
        return comm.broadcast_enqueue(cmd_buf.0, ep_wire::CMD_BLOCK_BYTES, 0);
    }
    for word in [slot, EP_CMD_DECODE_FED] {
        gpu.copy_h2d_async(&word.to_le_bytes(), cmd_buf, stream)?;
        comm.broadcast_enqueue(cmd_buf.0, 4, 0)?;
    }
    comm.broadcast_enqueue(feed_ids.0, 4, 0)
}

/// 2026-10-10: A worker's token for the fed step into its `feed_ids`, after
/// `ep_recv_seq_and_cmd` returned `EP_CMD_DECODE_FED`: on the v3 wire a device copy out of the
/// received block in `cmd_buf`, otherwise the token's own broadcast.
pub(in crate::model) fn recv_fed_token(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    cmd_buf: DevicePtr,
    feed_ids: DevicePtr,
    stream: u64,
    one_block: bool,
) -> Result<()> {
    ensure!(comm.rank() != 0, "rank 0 sends the fed token");
    if one_block {
        return gpu.copy_d2d_async(cmd_buf.offset(ep_wire::CMD_BLOCK_ARG), feed_ids, 4, stream);
    }
    comm.broadcast_enqueue(feed_ids.0, 4, 0)
}

impl TransformerModel {
    /// 2026-10-10: Whether the multi-rank fed decode is on: the lever, a live multi-rank
    /// protocol, and the v2 wire (the slot word routes the step on the workers).
    pub(super) fn ep_device_feed_active(&self) -> bool {
        lever_on() && self.multi_rank_protocol_active() && self.ep_protocol_v2
    }

    /// 2026-10-10: Rank 0, inside a fed launch: announce the step for `seq` and send the token
    /// that `feed_resolve` just wrote. Stamps the routes `decode()` stamps, so the fed step keys
    /// and runs the graph the host-token step would.
    pub(super) fn ep_send_fed_step(&self, seq: &SequenceState) -> Result<()> {
        let comm = self
            .comm
            .as_ref()
            .context("multi-rank fed decode without a communicator")?;
        self.stamp_overlay_route(seq.adapter_slot);
        self.stamp_decode_moe_single(seq.adapter_slot);
        send_fed_step(
            comm.as_ref(),
            self.gpu.as_ref(),
            self.ep_cmd_buf,
            self.feed_ids,
            self.gpu.default_stream(),
            seq.slot_idx as u32,
            EpWire::get().one_block(),
        )
    }

    /// 2026-10-10: Worker, on [`EP_CMD_DECODE_FED`]: receive the token, run the fed decode rank
    /// 0 runs, then book it. The token is read back once the step has finished: this host has
    /// nothing to do before its next command, whose first word cannot arrive before this step
    /// completes on the same stream, and waiting here starts that word's poll inside its yield
    /// window.
    pub(in crate::model) fn ep_worker_decode_fed(&self, seq: &mut SequenceState) -> Result<()> {
        let comm = self
            .comm
            .as_ref()
            .context("worker fed decode without a communicator")?;
        ensure!(
            self.feed_ids != DevicePtr::NULL && self.batched_embed_kernel.0 != 0,
            "worker fed decode: this build has no device token feed"
        );
        recv_fed_token(
            comm.as_ref(),
            self.gpu.as_ref(),
            self.ep_cmd_buf,
            self.feed_ids,
            self.gpu.default_stream(),
            EpWire::get().one_block(),
        )?;
        self.gdn_carry_flush_pending()?;
        self.stamp_overlay_route(seq.adapter_slot);
        self.stamp_decode_moe_single(seq.adapter_slot);
        self.decode_dispatch_with(DecodeInput::Fed, seq, self.gpu.default_stream())?;
        let mut id = [0u8; 4];
        self.gpu.copy_d2h(self.feed_ids, &mut id)?;
        seq.tokens.push(u32::from_le_bytes(id));
        seq.seq_len += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_pair_comm::run_pair;

    /// 2026-10-10: On each wire, rank 0's announcement and a worker's receive pair broadcast for
    /// broadcast, all rooted at rank 0 (three on v2, one on v3), and the worker ends with rank
    /// 0's slot, the fed command and rank 0's token in its feed ids. Fails if either side drops,
    /// adds or reorders a broadcast (a mismatch hangs a real serve), or if the token is taken
    /// from the wrong buffer or offset.
    #[test]
    fn a_fed_step_pairs_slot_command_and_token_across_ranks() {
        const SLOT: u32 = 5;
        const TOKEN: u32 = 151_329;
        for (one_block, broadcasts) in [(false, 3), (true, 1)] {
            let [(_, ops0), (r1, ops1)] = run_pair(move |rank, gpu, comm| {
                let cmd_buf = gpu.alloc(ep_wire::CMD_BLOCK_BYTES).unwrap();
                let feed_ids = gpu.alloc(4).unwrap();
                let mut w = [0u8; 4];
                if rank == 0 {
                    gpu.copy_h2d(&TOKEN.to_le_bytes(), feed_ids).unwrap();
                    send_fed_step(comm, gpu, cmd_buf, feed_ids, 0, SLOT, one_block).unwrap();
                    return Vec::new();
                }
                let (slot, cmd) = if one_block {
                    let mut block = [0u8; ep_wire::CMD_BLOCK_BYTES];
                    comm.recv_command_block(cmd_buf.0, block.len(), 0).unwrap();
                    gpu.copy_d2h(cmd_buf, &mut block).unwrap();
                    ep_wire::block_preamble(&block)
                } else {
                    comm.recv_command_u32(cmd_buf.0, 0).unwrap();
                    gpu.copy_d2h(cmd_buf, &mut w).unwrap();
                    let slot = u32::from_le_bytes(w);
                    comm.broadcast(cmd_buf.0, 4, 0).unwrap();
                    gpu.copy_d2h(cmd_buf, &mut w).unwrap();
                    (slot, u32::from_le_bytes(w))
                };
                recv_fed_token(comm, gpu, cmd_buf, feed_ids, 0, one_block).unwrap();
                gpu.copy_d2h(feed_ids, &mut w).unwrap();
                vec![slot, cmd, u32::from_le_bytes(w)]
            });
            assert_eq!(
                r1,
                vec![SLOT, EP_CMD_DECODE_FED, TOKEN],
                "one_block={one_block}"
            );
            assert_eq!(ops0.len(), broadcasts, "one_block={one_block}");
            assert_eq!(ops0, ops1);
            assert!(ops0.iter().all(|&(_, root)| root == 0));
        }
    }

    /// 2026-10-10: The command word is outside every token id and every other command, so a
    /// worker never runs it as a host-token decode or as another command.
    #[test]
    fn the_fed_command_word_collides_with_no_other_command() {
        let others = [
            0xFFFF_FFE0,
            0xFFFF_FFF0,
            0xFFFF_FFF1,
            0xFFFF_FFF2,
            0xFFFF_FFF3,
            0xFFFF_FFF4,
            metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
            metrale_model_layers::speculative::EP_CMD_VERIFY_KGAMMA,
            metrale_model_layers::speculative::EP_CMD_VERIFY_BATCH,
            super::super::prefill_b::spans_wire::EP_CMD_PREFILL_SPANS,
            super::super::decode_checkpoint::EP_CMD_DECODE_CKPT,
            0xFFFF_FFFF,
        ];
        assert!(!others.contains(&EP_CMD_DECODE_FED));
    }
}
