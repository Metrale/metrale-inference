// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: single-sequence K=2 verify: one 2-row forward judges one
//! draft, and the K=2 accept/reject counters.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

// 2026-09-25: the counters are fields of the run's `SpecStats`
// (`sched.io.tel.stats()`). Every `K2_SUMMARY_PERIOD` recorded steps they
// are logged and reset.
const K2_SUMMARY_PERIOD: u64 = 100;

#[inline]
pub(super) fn k2_record_outcome(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    accepted: bool,
    seq_len: usize,
) {
    let counter = if accepted {
        &sched.io.tel.stats().k2_accepts
    } else {
        &sched.io.tel.stats().k2_rejects
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = sched
        .io
        .tel
        .stats()
        .k2_accepts
        .load(std::sync::atomic::Ordering::Relaxed)
        + sched
            .io
            .tel
            .stats()
            .k2_rejects
            .load(std::sync::atomic::Ordering::Relaxed);
    if total >= K2_SUMMARY_PERIOD {
        let accepts = sched
            .io
            .tel
            .stats()
            .k2_accepts
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let rejects = sched
            .io
            .tel
            .stats()
            .k2_rejects
            .swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (accepts + rejects).max(1);
        let pct = 100.0 * (accepts as f64) / (total as f64);
        // 2026-09-25: below `DRIFT_THRESHOLD_PCT` the summary is logged as a
        // warning instead. It is only a log line; nothing acts on it.
        const DRIFT_THRESHOLD_PCT: f64 = 30.0;
        if pct < DRIFT_THRESHOLD_PCT && total >= K2_SUMMARY_PERIOD {
            tracing::warn!(
                "K2 drift gauge: accept rate {pct:.1}% < {DRIFT_THRESHOLD_PCT}% over last {total} steps (seq_len={seq_len}). Model logits likely in 'confidently wrong' attractor."
            );
        } else {
            tracing::info!(
                "K2 summary: {accepts} accept / {rejects} reject in last {total} steps ({pct:.1}% accept) seq_len={seq_len}"
            );
        }
    }
}

/// 2026-09-25: verify `[last_token, drafts[0]]` and accept or reject the
/// draft; `drafts` must hold at least 1. Picks go through
/// `verify_pick_all_with_pipeline` unless this is a DFlash serve with
/// `dflash_masked_verify` off, which judges on the raw argmax.
pub fn step_verify_k2(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    use crate::scheduler::mtp_timing::Phase;
    let t_step = sched.io.clock.now();
    let t_sync = sched.io.clock.now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        super::lifecycle::fail_sequence(a, format!("sync_secondary: {e:#}"));
        return;
    }
    sched.io.tel.mark(Phase::SyncSecondary, t_sync);
    let sync_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_sync)
        .as_micros();

    // 2026-09-25: EP: command 0xFFFFFFF2 makes the worker ranks run
    // `decode_verify_graphed` (the model's EP worker loop), then the 2
    // tokens follow.
    // 2026-10-11: Under the fused wire (`Model::k2_fused_wire`, a DFlash serve never) one block
    // carries the command and both tokens, and the verdict below also starts the workers'
    // propose (`speculative::k2_wire`).
    let t_ep = sched.io.clock.now();
    let tokens_k2 = [a.last_token, drafts[0]];
    let fused = !dflash_verify_raw_argmax && model.k2_fused_wire();
    // 2026-10-11: `METRALE_GLM_MTP_DEVICE_STEP`: a draft on the device of an identity-pick
    // request lets every worker act on its own device verdict (`verify_k2_device_step`).
    // 2026-10-21: And the screened head where the picks need only argmaxes (`screen_picks`).
    let flags = super::screen_picks::k2_flags(fused, drafts[0], a, verify_ctx);
    let device_verdict = flags.device_verdict;
    if fused {
        if let Err(e) = model.ep_announce_k2_fused(a.seq.slot_idx as u32, tokens_k2, flags) {
            tracing::error!("EP announce fused verify_k2: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP announce fused verify_k2: {e:#}"));
            return;
        }
    } else {
        if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
            tracing::error!("EP broadcast verify_k2 cmd: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k2 cmd: {e:#}"));
            return;
        }
        for &t in &tokens_k2 {
            if let Err(e) = model.ep_broadcast_cmd(t) {
                tracing::error!("EP broadcast verify_k2 token: {e:#}");
                super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k2 token: {e:#}"));
                return;
            }
        }
    }

    sched.io.tel.mark(Phase::EpBroadcast, t_ep);
    let ep_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_ep)
        .as_micros();

    let t_verify = sched.io.clock.now();
    // 2026-09-25: the fused forward is for DFlash on a single rank. Under EP
    // the workers run `decode_verify_graphed` for the command above, so the
    // master runs it too and the collectives stay matched.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // 2026-09-25: one M=2 forward; it captures the DFlash hidden of
        // every row (`Model::decode_and_verify_fused`).
        match model.decode_and_verify_fused(&tokens_k2, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k2): {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_and_verify_fused (k2): {e:#}"));
                return;
            }
        }
    } else {
        match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed: {e:#}");
                super::lifecycle::fail_sequence(a, format!("decode_verify_graphed: {e:#}"));
                return;
            }
        }
    };
    sched.io.tel.mark(Phase::VerifyForward, t_verify);
    let verify_us = sched
        .io
        .clock
        .now()
        .saturating_duration_since(t_verify)
        .as_micros();
    a.last_token_time = sched.io.clock.now();
    let (v0_argmax, v1_argmax) = (result_vec[0], result_vec[1]);
    // 2026-10-11: `METRALE_GLM_MTP_DEVICE_STEP`: a draft verified from the device reaches the host
    // as the row-1 token the verify recorded.
    // (A sentinel left unresolved could never equal `v0`, so it would only be rejected.)
    let draft0 = if drafts[0] == metrale_model_layers::speculative::DEVICE_DRAFT {
        a.seq.tokens.last().copied().unwrap_or(drafts[0])
    } else {
        drafts[0]
    };

    let (v0, v1) = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
        // 2026-09-25: DFlash without masked verify: judge on the raw argmax,
        // with no masks or penalties.
        (v0_argmax, v1_argmax)
    } else {
        // 2026-09-25: a position with no pick falls back to its GPU argmax.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax],
            a,
            verify_ctx,
            0,
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
        )
    };
    let accepted = draft0 == v0;

    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1], top_logprobs, 0)
    } else {
        Vec::new()
    };

    // 2026-09-25: `METRALE_MTP_CATCHUP`: ring the target's hidden for verify
    // row `t` in `0..=num_accepted` under label `L+t+1` plus
    // `mtp_refeed_shift()`, as `verify_k3_step` does (the label convention
    // is explained there). `L = seq_len - 2`: the verify forward added 2 and
    // the reject branch below has not popped yet.
    if metrale_model_layers::speculative::mtp_catchup_enabled() {
        let base = a.seq.seq_len.saturating_sub(2);
        let num_accepted = usize::from(accepted);
        let shift = metrale_model_layers::speculative::mtp_refeed_shift();
        for t in 0..=num_accepted {
            let label = ((base + t + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(t, label) {
                tracing::debug!("save_hidden_for_catchup(K=2, t={t}): {e:#} — degrading");
                break;
            }
        }
    }

    // 2026-09-25: EP: the workers block on this result after their verify,
    // so it is sent before any emit below can finish the sequence.
    // 2026-10-11: The fused verdict is sent after the emits instead (`send_fused_verdict`): an
    // emit only sets flags, and the verdict must know whether the sequence finished.
    if !fused && let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast verify_k2 result: {e:#}");
        super::lifecycle::fail_sequence(a, format!("EP broadcast verify_k2 result: {e:#}"));
        return;
    }

    // 2026-10-09: On a DFlash serve the drafter context gets the verified rows `0..=accepted`
    // (the anchor, and the draft when it was accepted) at their own positions, on accept and on
    // reject, as after a K=γ verify. The verify forward ran 2 rows past the pre-verify length.
    if dflash_verify_raw_argmax {
        let pre_verify_len = a.seq.seq_len.saturating_sub(2);
        commit_verified_ctx(
            model,
            &mut a.seq,
            usize::from(accepted),
            pre_verify_len,
            sched,
        );
    }

    sched
        .io
        .tel
        .count_spec_verify("2", if accepted { "accept" } else { "reject" });
    sched.io.tel.spec_verified(1, usize::from(accepted));
    // 2026-10-11: Width-1 accept telemetry, as `verify_k4_step` records: the per-width
    // `MTP accept n= k_drafts=` line and the rung's p1 see every serial arm, not only K=4.
    sched.accept.record(
        &sched.rung,
        sched.levers.mtp_accept_debug,
        1,
        1,
        accepted,
        usize::from(accepted),
    );
    if device_verdict {
        super::verify_k2_device_step::finish_device_verdict(
            model,
            a,
            sched,
            draft0,
            [v0, v1],
            [v0_argmax, v1_argmax],
            verify_lps,
            t_step,
        );
        return;
    }

    if accepted {
        emit_token(a, draft0, verify_lps.first().cloned(), sched);
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned(), sched);
        }
        // 2026-10-11: Under the fused wire the workers propose `fused_drafts` drafts after this
        // verdict, so every return below with drafts outstanding first runs rank 0's side
        // (`pair_fused_propose`), or the workers would wait in the drafter's collectives.
        let fused_drafts = if fused {
            match send_fused_verdict(model, a, true, v1, num_drafts) {
                Some(n) => n,
                None => return,
            }
        } else {
            0
        };
        if a.finished {
            return;
        }
        a.last_token = v1;

        // 2026-09-25: commit both rows; `commit_accepted_prefix` treats a
        // full accept as a no-op.
        let t_commit = sched.io.clock.now();
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 2) {
            // 2026-09-25: the SSM state cannot be trusted after a failed
            // commit, so the sequence fails.
            tracing::error!("commit_accepted_prefix (accept): {e:#}");
            super::lifecycle::fail_sequence(a, format!("commit_accepted_prefix (accept): {e:#}"));
            pair_fused_propose(model, v1, a, fused_drafts);
            return;
        }
        sched.io.tel.mark(Phase::Commit, t_commit);

        // 2026-09-25: `dflash_eagle_fix` (on unless
        // `METRALE_DFLASH_EAGLE_FIX=0`): before the propose, append verify
        // row 0 at N and row 1 at N+1 to the DFlash context, so the drafter
        // conditions on row 1, the hidden that produced the bonus token. It
        // also sets `skip_next_decode_append` so row 0 is not appended
        // twice. A no-op for models without a DFlash drafter.
        // 2026-10-09: A DFlash serve committed its rows after the verdict above.
        let eagle_fix = sched.levers.dflash_eagle_fix && !dflash_verify_raw_argmax;
        if eagle_fix && let Err(e) = model.dflash_eagle_accept_append(&mut a.seq) {
            tracing::error!("dflash_eagle_accept_append: {e:#}");
        }
        let t_save = sched.io.clock.now();
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            pair_fused_propose(model, v1, a, fused_drafts);
            return;
        }
        sched.io.tel.mark(Phase::SaveHidden, t_save);
        let t_trim = sched.io.clock.now();
        let drafter_na = drafter_accepted(a, 1);
        if let Err(e) = model.trim_proposer_state(&mut a.seq, drafter_na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        sched.io.tel.mark(Phase::TrimProposer, t_trim);
        let t_mask = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        sched.io.tel.mark(Phase::ProposeMask, t_mask);
        let t_propose = sched.io.clock.now();
        match propose_k2(
            model,
            fused.then_some(fused_drafts),
            v1,
            a,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        sched.io.tel.mark(Phase::Propose, t_propose);
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        // 2026-09-25: debug level: this fires on every verify step; the
        // periodic line from `k2_record_outcome` is the summary.
        tracing::debug!(
            "K2 ACCEPT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k2_record_outcome(sched, true, a.seq.seq_len);
        // 2026-09-25: after the commit the live SSM state is canonical, so
        // the model may save a Marconi checkpoint (it does so only at its
        // checkpoint-interval boundaries).
        let t_marconi = sched.io.clock.now();
        model.decode_marconi_checkpoint(&mut a.seq);
        sched.io.tel.mark(Phase::MarconiCkpt, t_marconi);
        sched.io.tel.step_done(t_step, a.seq.seq_len);
    } else {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        let t_trim = sched.io.clock.now();
        let drafter_na = drafter_accepted(a, 0);
        if let Err(e) = model.trim_proposer_state(&mut a.seq, drafter_na, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        sched.io.tel.mark(Phase::TrimProposer, t_trim);

        // 2026-10-11: The fused verdict goes out before the commit below. The commit's SSM
        // rollback is a 49 MB device copy on GB10 (TP=2), and no kernel starts while it runs, so
        // a verdict sent after it reached the workers ~0.43 ms late on every reject (spec1011b).
        emit_token(a, v0, verify_lps.first().cloned(), sched);
        let fused_drafts = if fused {
            match send_fused_verdict(model, a, false, v0, num_drafts) {
                Some(n) => n,
                None => return,
            }
        } else {
            0
        };

        // 2026-09-25: reject: commit row 0 only (the state after
        // `last_token`); the draft row is discarded.
        let t_commit = sched.io.clock.now();
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 2) {
            tracing::error!("commit_accepted_prefix (reject): {e:#}");
            super::lifecycle::fail_sequence(a, format!("commit_accepted_prefix (reject): {e:#}"));
            // 2026-10-11: The workers were told to propose; pair their drafter collectives.
            pair_fused_propose(model, v0, a, fused_drafts);
            return;
        }
        sched.io.tel.mark(Phase::Commit, t_commit);
        if a.finished {
            return;
        }
        a.last_token = v0;

        let t_save = sched.io.clock.now();
        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            pair_fused_propose(model, v0, a, fused_drafts);
            return;
        }
        sched.io.tel.mark(Phase::SaveHidden, t_save);
        let t_mask = sched.io.clock.now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        sched.io.tel.mark(Phase::ProposeMask, t_mask);
        let t_propose = sched.io.clock.now();
        match propose_k2(
            model,
            fused.then_some(fused_drafts),
            v0,
            a,
            num_drafts,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        sched.io.tel.mark(Phase::Propose, t_propose);
        let propose_us = sched
            .io
            .clock
            .now()
            .saturating_duration_since(t_propose)
            .as_micros();
        let new_draft = a.pending_drafts.first().copied().unwrap_or(0);
        // 2026-09-25: debug level, as on the accept path.
        tracing::debug!(
            "K2 REJECT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
            a.seq.seq_len,
            a.last_token,
            draft0,
            v0,
            new_draft,
        );
        k2_record_outcome(sched, false, a.seq.seq_len);
        // 2026-09-25: Marconi checkpoint, as on the accept path.
        let t_marconi = sched.io.clock.now();
        model.decode_marconi_checkpoint(&mut a.seq);
        sched.io.tel.mark(Phase::MarconiCkpt, t_marconi);
        sched.io.tel.step_done(t_step, a.seq.seq_len);
    }
}

/// 2026-10-11: The fused wire's verdict: accept bit, the token the propose starts from, the
/// draft count (0 when the emits finished the sequence, so no rank proposes) and the propose
/// position (`seq_len`, after a reject's pop). Returns the draft count sent, or `None`, after
/// failing the sequence, when the send fails.
fn send_fused_verdict(
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
fn pair_fused_propose(model: &dyn Model, token: u32, a: &mut ActiveSeq, drafts: usize) {
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
fn propose_k2(
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
