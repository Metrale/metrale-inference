// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The multi-rank fed decode step (`METRALE_EP_DEVICE_FEED=1`): rank 0 runs a
//! single-row decode step ahead of its host, fed from the previous step's device argmax, and
//! the token reaches the worker ranks device to device.
//!
//! Rank 0 (`ep_send_fed_step`, after `feed_resolve` wrote its feed ids) submits three
//! broadcasts on its compute stream without waiting for any: the slot word, the
//! [`EP_CMD_DECODE_FED`] command word, then the 4-byte token from its feed ids. The words
//! travel through the command buffer like every other command, so a worker reads them with
//! `ep_recv_seq_and_cmd`; the token lands in the worker's feed ids, and the worker runs the
//! same fed decode (`ep_worker_decode_fed`). Neither side reads the token on the host before
//! the step's graph is queued.
//!
//! Owner: model-engine.
//! Invariants:
//! - Rank 0 and every worker issue the same broadcasts, from root 0, in the same order for a
//!   fed step: slot word, command word, token.
//! - Rank 0 never waits on these broadcasts; its stream orders them after the step in flight
//!   and before the fed step's graph.

use anyhow::{Context, Result, ensure};
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

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

/// 2026-10-10: Rank 0's half of a fed step: the slot and command words, then the token in
/// `feed_ids`, each submitted on `stream` without a host wait. The words are staged through
/// `cmd_buf` with pageable async copies, which the driver stages before returning, so the
/// second copy is ordered after the first word's broadcast has read the buffer.
pub(in crate::model) fn send_fed_step(
    comm: &dyn CommBackend,
    gpu: &dyn GpuBackend,
    cmd_buf: DevicePtr,
    feed_ids: DevicePtr,
    stream: u64,
    slot: u32,
) -> Result<()> {
    ensure!(comm.rank() == 0, "a fed step is announced by rank 0 only");
    for word in [slot, EP_CMD_DECODE_FED] {
        gpu.copy_h2d_async(&word.to_le_bytes(), cmd_buf, stream)?;
        comm.broadcast_enqueue(cmd_buf.0, 4, 0)?;
    }
    comm.broadcast_enqueue(feed_ids.0, 4, 0)
}

/// 2026-10-10: A worker's receive of the fed step's token into its `feed_ids`, after
/// `ep_recv_seq_and_cmd` returned `EP_CMD_DECODE_FED`.
pub(in crate::model) fn recv_fed_token(comm: &dyn CommBackend, feed_ids: DevicePtr) -> Result<()> {
    ensure!(comm.rank() != 0, "rank 0 sends the fed token");
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
        recv_fed_token(comm.as_ref(), self.feed_ids)?;
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

    /// 2026-10-10: Rank 0's announcement and a worker's receive pair broadcast for broadcast,
    /// all rooted at rank 0, and the worker ends with rank 0's slot, the fed command and rank
    /// 0's token. Fails if either side drops, adds or reorders a broadcast (a mismatch hangs a
    /// real serve), or if the token is taken from the wrong buffer.
    #[test]
    fn a_fed_step_pairs_slot_command_and_token_across_ranks() {
        const SLOT: u32 = 5;
        const TOKEN: u32 = 151_329;
        let [(r0, ops0), (r1, ops1)] = run_pair(|rank, gpu, comm| {
            let cmd_buf = gpu.alloc(4).unwrap();
            let feed_ids = gpu.alloc(4).unwrap();
            if rank == 0 {
                gpu.copy_h2d(&TOKEN.to_le_bytes(), feed_ids).unwrap();
                send_fed_step(comm, gpu, cmd_buf, feed_ids, 0, SLOT).unwrap();
                Vec::new()
            } else {
                let mut words = Vec::new();
                let mut w = [0u8; 4];
                comm.recv_command_u32(cmd_buf.0, 0).unwrap();
                gpu.copy_d2h(cmd_buf, &mut w).unwrap();
                words.push(u32::from_le_bytes(w));
                comm.broadcast(cmd_buf.0, 4, 0).unwrap();
                gpu.copy_d2h(cmd_buf, &mut w).unwrap();
                words.push(u32::from_le_bytes(w));
                recv_fed_token(comm, feed_ids).unwrap();
                gpu.copy_d2h(feed_ids, &mut w).unwrap();
                words.push(u32::from_le_bytes(w));
                words
            }
        });
        assert!(r0.is_empty());
        assert_eq!(r1, vec![SLOT, EP_CMD_DECODE_FED, TOKEN]);
        assert_eq!(ops0.len(), 3);
        assert_eq!(ops0, ops1);
        assert!(ops0.iter().all(|&(_, root)| root == 0));
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
