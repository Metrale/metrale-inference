// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: Rank 0's side of a device-verdict K=2 step (`METRALE_GLM_MTP_DEVICE_STEP` with
//! `FLAG_K2_DEVICE_VERDICT`): every worker takes the verdict from its own device verdict (the
//! argmax rows and the draft) and commits and proposes right after its verify, without waiting
//! for this host. Rank 0 does the same moves from its host verdict, proposes one draft always
//! (a finished sequence still pairs the workers' propose), then sends its verdict, which every
//! worker cross-checks (`speculative::device_verdict_agrees`).
//!
//! Owner: scheduler.
//! Invariants:
//! - Armed only for a request whose host picks are its argmaxes ([`identity_picks`]); the host
//!   verdict is then the device verdict. A step where they still differ fails the sequence on
//!   every rank, loudly: the ranks' states differ from that step on.
//! - The collectives keep one order on every rank: verify, propose, verdict broadcast.

use super::*;

/// 2026-10-11: Whether a request's K=2 picks are its argmaxes, so the device verdict is the host
/// verdict: greedy, no grammar, no log-probabilities, no penalty or bias that can move the
/// argmax, past its `min_tokens`, and outside thinking and the tool-call pin, where the chat
/// pipeline's masks can act (`verify_pipeline_helper/fast_masked.rs`).
pub(super) fn identity_picks(
    a: &ActiveSeq,
    ctx: &crate::scheduler::logit_processors::LogitsContext,
) -> bool {
    let greedy = a.temperature == 0.0 || ctx.sampling.force_temp_zero;
    let penalties = crate::scheduler::fast_greedy::classify_penalties(
        &crate::scheduler::sample_step::penalty_params_for(
            a,
            crate::scheduler::sample_step::PositionKind::Verify,
            0.0,
            None,
            Vec::new(),
            ctx.watchdog.min_reasoning_floor,
        ),
    );
    identity_rule(IdentityInputs {
        greedy,
        grammar: a.grammar_state.is_some(),
        logprobs: a.top_logprobs.is_some(),
        neutral_penalties: penalties == crate::scheduler::fast_greedy::PenaltyGate::Neutral,
        below_min_tokens: a.output_tokens.len() < a.min_tokens,
        thinking: a.inside_thinking || a.think_just_ended || a.force_end_thinking,
        tool_pin: a.require_tool_call,
    })
}

/// 2026-10-11: The inputs of [`identity_picks`], so its rule is tested without a sequence.
pub(super) struct IdentityInputs {
    pub greedy: bool,
    pub grammar: bool,
    pub logprobs: bool,
    pub neutral_penalties: bool,
    pub below_min_tokens: bool,
    pub thinking: bool,
    pub tool_pin: bool,
}

pub(super) fn identity_rule(i: IdentityInputs) -> bool {
    i.greedy
        && !i.grammar
        && !i.logprobs
        && i.neutral_penalties
        && !i.below_min_tokens
        && !i.thinking
        && !i.tool_pin
}

/// 2026-10-11: The rest of a device-verdict K=2 step after the verify and the host picks
/// (`step_verify_k2`): emits, commit, MTP input row, the one-draft propose, the verdict broadcast
/// and the cross-check of the host picks against the argmax verdict the workers acted on.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_device_verdict(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    draft0: u32,
    picks: [u32; 2],
    argmax: [u32; 2],
    verify_lps: Vec<crate::api::TokenLogprobs>,
    t_step: std::time::Instant,
) {
    use crate::scheduler::mtp_timing::Phase;
    let [v0, v1] = picks;
    let accepted = draft0 == v0;
    let next = if accepted { v1 } else { v0 };
    let dev = metrale_model_layers::speculative::k2_device_verdict(argmax, draft0);
    let agree = accepted == (dev[0] == 1) && next == dev[1];
    let mut failed: Option<String> = None;

    if accepted {
        emit_token(a, draft0, verify_lps.first().cloned(), sched);
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned(), sched);
        }
    } else {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        let na = super::prompt_lookup_step::drafter_accepted(a, 0);
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        emit_token(a, v0, verify_lps.first().cloned(), sched);
    }
    let t_commit = sched.io.clock.now();
    if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1 + usize::from(accepted), 2) {
        failed = Some(format!("commit_accepted_prefix (device verdict): {e:#}"));
    }
    sched.io.tel.mark(Phase::Commit, t_commit);
    if let Err(e) = model.save_hidden_for_mtp(usize::from(accepted), 0) {
        failed.get_or_insert(format!("save_hidden_for_mtp (device verdict): {e:#}"));
    }
    if accepted {
        let na = super::prompt_lookup_step::drafter_accepted(a, 1);
        if let Err(e) = model.trim_proposer_state(&mut a.seq, na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
    }
    // 2026-10-11: Every worker proposes one draft from its own verdict, so rank 0 always does,
    // finished or not, before the verdict broadcast: the collectives keep one order.
    let position = a.seq.seq_len;
    let t_propose = sched.io.clock.now();
    let drafts = model.run_mtp_propose_announced_device(next, position, 1, &mut a.seq, None);
    sched.io.tel.mark(Phase::Propose, t_propose);
    let verdict = metrale_model_layers::speculative::K2Verdict {
        accepted,
        next_token: next,
        num_drafts: 1,
        position: position as u32,
    };
    if let Err(e) = model.ep_send_k2_verdict_after_propose(verdict) {
        failed.get_or_insert(format!("EP send device-verdict K=2 verdict: {e:#}"));
    }
    if !agree {
        tracing::error!(
            "device-verdict K=2 step: host picks {picks:?} differ from the argmax verdict \
             {dev:?} the workers acted on (draft {draft0}); the sequence fails"
        );
        failed.get_or_insert("device-verdict K=2 step: host picks differ from the argmax".into());
    }
    match (failed, drafts) {
        (Some(msg), _) => super::lifecycle::fail_sequence(a, msg),
        (None, Err(e)) => tracing::error!("run_mtp_propose_announced_device: {e:#}"),
        (None, Ok(d)) => {
            if !a.finished {
                a.last_token = next;
                if !d.is_empty() {
                    a.pending_drafts = d;
                }
            }
        }
    }
    super::verify_k2_step::k2_record_outcome(sched, accepted, a.seq.seq_len);
    let t_marconi = sched.io.clock.now();
    model.decode_marconi_checkpoint(&mut a.seq);
    sched.io.tel.mark(Phase::MarconiCkpt, t_marconi);
    sched.io.tel.step_done(t_step, a.seq.seq_len);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain() -> IdentityInputs {
        IdentityInputs {
            greedy: true,
            grammar: false,
            logprobs: false,
            neutral_penalties: true,
            below_min_tokens: false,
            thinking: false,
            tool_pin: false,
        }
    }

    /// 2026-10-11: A plain greedy request is armed; each condition that can make a host pick
    /// differ from the argmax disarms it on its own.
    #[test]
    fn each_pick_changing_condition_disarms_the_device_verdict() {
        assert!(identity_rule(plain()));
        type Disarm = fn(&mut IdentityInputs);
        let cases: [(&str, Disarm); 7] = [
            ("sampled", |i| i.greedy = false),
            ("grammar", |i| i.grammar = true),
            ("logprobs", |i| i.logprobs = true),
            ("penalties", |i| i.neutral_penalties = false),
            ("min_tokens", |i| i.below_min_tokens = true),
            ("thinking", |i| i.thinking = true),
            ("tool pin", |i| i.tool_pin = true),
        ];
        for (name, f) in cases {
            let mut i = plain();
            f(&mut i);
            assert!(!identity_rule(i), "{name}");
        }
    }
}
