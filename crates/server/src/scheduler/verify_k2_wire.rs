// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The fused wire's side of a K=2 step (`verify_k2_step`): the verdict send, the
//! pairing propose and the propose after a verdict. Moved out of `verify_k2_step.rs` unchanged
//! (file-size cap).
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-11: The fused wire's verdict: accept bit, the token the propose starts from, the
/// draft count (0 when the emits finished the sequence, so no rank proposes) and the propose
/// position (`seq_len`, after a reject's pop). Returns the draft count sent, or `None`, after
/// failing the sequence, when the send fails.
pub(super) fn send_fused_verdict(
    model: &dyn Model,
    a: &mut ActiveSeq,
    accepted: bool,
    next_token: u32,
    num_drafts: usize,
) -> Option<usize> {
    let drafts = if a.finished {
        0
    } else {
        crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts)
    };
    let verdict = metrale_model_layers::speculative::K2Verdict {
        accepted,
        next_token,
        num_drafts: drafts as u32,
        position: a.seq.seq_len as u32,
    };
    if let Err(e) = model.ep_send_k2_verdict(verdict) {
        tracing::error!("EP send fused verify_k2 verdict: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP send fused verify_k2 verdict: {e:#}"));
        return None;
    }
    Some(drafts)
}

/// 2026-10-11: Rank 0's side of a propose a fused verdict announced (`drafts > 0`) when the step
/// returns before its own propose: the same drafter forward, so the workers' collectives have a
/// partner; its drafts are discarded. Nothing when no propose was announced.
pub(super) fn pair_fused_propose(model: &dyn Model, token: u32, a: &mut ActiveSeq, drafts: usize) {
    if drafts == 0 {
        return;
    }
    let position = a.seq.seq_len;
    if let Err(e) = model.run_mtp_propose_announced(token, position, drafts, &mut a.seq, None) {
        tracing::error!("fused verify_k2: pairing propose: {e:#}");
    }
}

/// 2026-10-11: The propose after a K=2 verdict. `fused` is the draft count a fused verdict sent:
/// the workers were already told, so it sends nothing; `None` (the unfused wire) lets
/// `run_mtp_propose_multi` announce it.
pub(super) fn propose_k2(
    model: &dyn Model,
    fused: Option<usize>,
    token: u32,
    a: &mut ActiveSeq,
    num_drafts: usize,
    mask: Option<&[i32]>,
) -> anyhow::Result<Vec<u32>> {
    let position = a.seq.seq_len;
    match fused {
        // 2026-10-11: The lone draft may stay on the device (`METRALE_GLM_MTP_DEVICE_STEP`).
        Some(n) => model.run_mtp_propose_announced_device(token, position, n, &mut a.seq, mask),
        None => {
            let n = crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts);
            model.run_mtp_propose_multi(token, position, n, &mut a.seq, 0, mask)
        }
    }
}
