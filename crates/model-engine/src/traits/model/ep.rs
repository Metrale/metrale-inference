// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelEp`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use crate::traits::SequenceState;
use anyhow::Result;

// 2026-09-26: Only the `ep_worker_step` doc link names it.
#[cfg(doc)]
use super::EpCommandFailed;
// 2026-10-11: Only the `k2_fused_wire` doc link names it.
#[cfg(doc)]
use super::ModelDraft;

/// 2026-09-26: The multi-rank (EP) command protocol.
pub trait ModelEp {
    /// 2026-09-25: EP worker step, on a rank above 0: receive a `(seq_id, cmd)` from rank 0 and
    /// run the command in slot `seq_id`. Returns `Ok(false)` when the worker should shut down.
    ///
    /// An `Err` carrying [`EpCommandFailed`] means the command ran and failed, and the worker
    /// stays up; any other `Err` means the receive failed, and the worker exits.
    ///
    /// `slots` is sized to `--max-batch-size`; a `seq_id >= slots.len()` fails. Default:
    /// `Ok(true)`.
    fn ep_worker_step(&self, _slots: &mut [Option<SequenceState>]) -> Result<bool> {
        Ok(true)
    }

    /// 2026-09-25: Whether the model runs the multi-rank command protocol. The scheduler then
    /// turns off its single-GPU fused paths, the mixed forward among them. Default `false`.
    fn is_ep(&self) -> bool {
        false
    }

    /// 2026-10-09: `Some(rows)` when `prefill_batch_chunk` can run on this multi-rank serve,
    /// `rows` being the prompt rows one batched prefill step should carry; the scheduler batches
    /// prefills under the multi-rank protocol only then (and with its lever on). Default `None`.
    fn ep_prefill_batch_rows(&self) -> Option<usize> {
        None
    }

    /// 2026-09-25: Send a command word to every worker rank. Default: `Ok(())`.
    fn ep_broadcast_cmd(&self, _cmd: u32) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Send a `(seq_id, cmd)` pair to every worker rank, as the first broadcast of a
    /// command; its follow-up broadcasts use [`Self::ep_broadcast_cmd`]. When
    /// [`Self::ep_protocol_v2`] is false, `seq_id` is not sent. Default: `Ok(())`.
    fn ep_broadcast_cmd_for_seq(&self, _seq_id: u32, _cmd: u32) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Whether the EP protocol carries a `seq_id` with each command
    /// (`METRALE_EP_PROTOCOL=v2`). Default `false`.
    fn ep_protocol_v2(&self) -> bool {
        false
    }

    /// 2026-09-25: Broadcast a token array to every worker rank in one broadcast. Default: an
    /// empty `Vec`.
    fn ep_broadcast_tokens(&self, _tokens: &[u32]) -> Result<Vec<u32>> {
        Ok(Vec::new())
    }

    /// 2026-10-11: Whether a K=2 MTP step runs the fused wire (`speculative::k2_wire`): one
    /// announcement block, one verdict that also starts the workers' propose, and a propose
    /// that sends nothing ([`ModelDraft::run_mtp_propose_announced`]). Default `false`.
    fn k2_fused_wire(&self) -> bool {
        false
    }

    /// 2026-10-11: Rank 0, fused wire: announce the 2-row verify of `tokens` in slot `seq_id`.
    /// `device_verdict` (`METRALE_GLM_MTP_DEVICE_STEP`, an identity-pick request): every rank
    /// takes the verdict from its own device verdict and proposes at once, and rank 0's verdict
    /// follows the propose ([`Self::ep_send_k2_verdict_after_propose`]). Default: an error, since
    /// only a model whose [`Self::k2_fused_wire`] is true is asked.
    fn ep_announce_k2_fused(
        &self,
        _seq_id: u32,
        _tokens: [u32; 2],
        _device_verdict: bool,
    ) -> Result<()> {
        anyhow::bail!("this model has no fused K=2 wire")
    }

    /// 2026-10-11: Rank 0, a device-verdict K=2 step: send the host verdict after the propose,
    /// without waiting, for the workers' cross-check. Default: an error, as above.
    fn ep_send_k2_verdict_after_propose(
        &self,
        _verdict: metrale_model_layers::speculative::K2Verdict,
    ) -> Result<()> {
        anyhow::bail!("this model has no fused K=2 wire")
    }

    /// 2026-10-11: Rank 0, fused wire: send the verdict of the verify just announced. Default:
    /// an error, as above.
    fn ep_send_k2_verdict(
        &self,
        _verdict: metrale_model_layers::speculative::K2Verdict,
    ) -> Result<()> {
        anyhow::bail!("this model has no fused K=2 wire")
    }
}
