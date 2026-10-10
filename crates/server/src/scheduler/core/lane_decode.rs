// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The decode lane: one step of n-gram, self-speculative, MTP or DFlash
//! (through the speculation controller, `spec_host`) or plain decode, the plain step
//! pipelined when the router allows it. Skipped on a tick whose prefill lane already ran a
//! mixed step.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

impl SchedulerCore {
    pub(super) fn decode_lane(&mut self) -> LaneVerdict {
        let pipelined = self.pipelining_configured();
        let commit_tokens = self.commit_tokens();
        let Self {
            ctx: sched,
            active,
            prefilling,
            swapped,
            preempted,
            spec_host,
            dflash_depth_pinned,
            ngram_proposer,
            pipeline,
            ..
        } = self;
        let prefill_stream = self.prefill_stream;
        let prefill_event = self.prefill_event;
        let think_end_token = self.think_end_token;
        let think_start_token = self.think_start_token;
        let code_fence_token = self.code_fence_token;
        let tool_call_start_token = self.tool_call_start_token;
        let tool_call_end_token = self.tool_call_end_token;
        let spec_slot_cap = self.spec_slot_cap;
        let use_mtp = self.use_mtp;
        let use_ngram_speculative = self.use_ngram_speculative;
        let use_self_speculative = self.use_self_speculative;
        let num_drafts = self.num_drafts;
        let dflash_verify_raw_argmax = self.dflash_verify_raw_argmax;
        let adaptive_sampling = self.adaptive_sampling;
        if !self.did_mixed_step {
            // 2026-09-25: Order this decode on the default stream after the work queued
            // on the prefill stream (a device-side event wait).
            if !prefilling.is_empty() {
                let _ = sched
                    .io
                    .dev
                    .model()
                    .record_event(prefill_event, prefill_stream);
                let _ = sched
                    .io
                    .dev
                    .model()
                    .stream_wait_event(sched.io.dev.model().default_stream(), prefill_event);
            }

            // 2026-09-25: The LogitsContext the speculative steps below pass to the
            // logits processors: special-token ids, masks and sampling levers.
            let verify_ctx = crate::scheduler::logit_processors::LogitsContext {
                watchdog: sched.watchdog,
                scratch: &sched.scratch,
                tel: &*sched.io.tel,
                clock: &*sched.io.clock,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
                verify_pos: 0,
                boundary_mask: sched.masks.boundary.clone(),
                mid_word_mask: sched.masks.mid_word.clone(),
                sampling: sched.levers.sampling(),
            };
            // 2026-09-25: `METRALE_DFLASH_RESUME_GUARD` (tokens, default 0): a sequence
            // is not speculated until it has emitted this many tokens after
            // `</think>` (see `spec_dispatch_eligible`).
            let dflash_resume_guard = sched.levers.dflash_resume_guard;
            // 2026-09-25: `METRALE_DFLASH_SPEC_THINK`: without it, no sequence inside
            // `<think>` is speculated, for MTP and DFlash alike
            // (`spec_dispatch_eligible`).
            let dflash_spec_think = sched.levers.dflash_spec_think;
            // 2026-09-25: Every speculative branch also requires each active slot to be
            // below `spec_slot_cap` (see `SchedulerCore::new`); otherwise the batch
            // takes the plain-decode branch.
            let spec_slots_covered = active.iter().all(|a| a.seq.slot_idx < spec_slot_cap);
            // 2026-09-25: MTP also needs `active.len() <= mtp_max_seqs`.
            // `note_width_regime` only records this decision for reporting.
            let spec_width_ok = active.len() <= sched.levers.mtp_max_seqs;
            if use_mtp {
                sched.rung.note_width_regime(
                    active.len(),
                    spec_width_ok,
                    sched.levers.mtp_max_seqs,
                );
            }
            if use_ngram_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                if let Some(proposer) = ngram_proposer {
                    step_ngram(sched.io.dev.model(), active, sched, proposer, &verify_ctx);
                }
            } else if use_self_speculative
                && active.len() == 1
                && spec_slots_covered
                && active[0].grammar_state.is_none()
            {
                // 2026-09-25: The draft count is clamped to the slot's verify capacity,
                // as `step_mtp` does.
                let nd = metrale_speculative::spec_capacity::clamp_drafts_to_slot_capacity(
                    num_drafts,
                    active
                        .iter()
                        .map(|a| sched.io.dev.model().mtp_slot_draft_capacity(a.seq.slot_idx)),
                );
                step_self_spec(sched.io.dev.model(), active, sched, nd, &verify_ctx);
            } else if use_mtp
                && spec_width_ok
                && spec_slots_covered
                && (
                    // 2026-09-25: Every active sequence must be eligible, not only
                    // `active[0]`.
                    active.iter().all(|a| {
                        metrale_speculative::spec_eligibility::spec_dispatch_eligible(
                            a.inside_thinking,
                            a.post_think_emitted,
                            a.output_tokens.len() as u32,
                            a.suppress_tool_call,
                            a.disable_mtp,
                            dflash_spec_think,
                            dflash_resume_guard,
                            dflash_verify_raw_argmax,
                        )
                    })
                )
            {
                // 2026-10-10: The speculation controller (`spec_host`) chooses each step
                // between plain decode and the speculative depth(s), from per-sequence
                // acceptance and measured step costs. Without it (`--mtp-gate force` or
                // `METRALE_MTP_GATE_FORCE`) every eligible step speculates.
                // 2026-09-25: Spec-entry pin (`METRALE_SPEC_ENTRY_PIN`): while any active
                // sequence has emitted fewer than that many tokens after `</think>`, run the
                // verify step even where the controller would plain-decode. DFlash pin
                // (`SchedLevers::dflash_gate_pin_c2`, whose doc gives the reason): with
                // raw-argmax DFlash verify and at most 2 active sequences, always verify.
                let min_post_think_emitted = active
                    .iter()
                    .map(|a| a.post_think_emitted)
                    .min()
                    .unwrap_or(u32::MAX);
                let pinned = (dflash_verify_raw_argmax
                    && active.len() <= 2
                    && sched.levers.dflash_gate_pin_c2)
                    || metrale_speculative::spec_eligibility::entry_pin_forces_verify(
                        min_post_think_emitted,
                        sched.levers.spec_entry_pin_tokens,
                    );
                // 2026-10-10: `k` is the controller's depth (0 = plain decode); without the
                // controller the step speculates at `num_drafts` as before (`None`).
                let mut decision = None;
                let mut k: Option<usize> = None;
                if let Some(h) = spec_host.as_mut() {
                    let spec_k = crate::scheduler::mtp_step::step_depth(
                        sched.io.dev.model(),
                        active,
                        sched,
                        num_drafts,
                        dflash_verify_raw_argmax,
                    );
                    let allowed = crate::scheduler::spec_host::allowed(
                        dflash_verify_raw_argmax,
                        *dflash_depth_pinned,
                        spec_k,
                    );
                    let states: Vec<_> = active.iter().map(|a| &a.spec_ctl).collect();
                    let d = h.decide(&states, &allowed);
                    k = Some(if d.k == 0 && pinned { spec_k } else { d.k });
                    decision = Some(d);
                }
                if decision.is_some_and(|d| d.probe) {
                    for a in active.iter_mut() {
                        a.mtp_acct.note_regime_reprobe();
                    }
                }
                // 2026-09-25: On a switch to plain decode, drop pending drafts and order the
                // secondary stream before the plain step. Switching back needs nothing: the
                // next speculative step bootstraps from empty drafts.
                if k == Some(0) && decision.is_some_and(|d| d.entered_plain) {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                        a.pending_draft_conf.clear();
                    }
                    if let Err(e) = sched.io.dev.apply(io::Effect::SyncSecondary) {
                        tracing::error!("controller→decode sync_secondary: {e}");
                    }
                }
                let t0 = sched.io.clock.now();
                if k == Some(0) {
                    step_decode_only(
                        active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                        sched,
                        sched.io.spill.as_deref(),
                        swapped,
                        preempted,
                    );
                    let ms = sched
                        .io
                        .clock
                        .now()
                        .saturating_duration_since(t0)
                        .as_secs_f64()
                        * 1e3;
                    if let Some(h) = spec_host.as_mut() {
                        h.settle(active.iter_mut().map(|a| &mut a.spec_ctl), 0);
                        h.observe_plain(active.iter_mut().map(|a| &mut a.spec_ctl));
                        h.observe_step(active.len(), 0, ms, None, active.len());
                    }
                    for a in active.iter_mut() {
                        a.mtp_acct.record_serial();
                    }
                    // 2026-09-25: Single-sequence batches only (the ring has one label
                    // space): copy row 0's hidden into the MTP catch-up ring at label
                    // `seq_len`, which this step has already advanced past its input token.
                    // A no-op when the model allocated no ring.
                    if active.len() == 1
                        && let Err(e) = sched.io.dev.apply(io::Effect::SaveHiddenCatchup {
                            row: 0,
                            pos: active[0].seq.seq_len,
                        })
                    {
                        tracing::warn!("save_hidden_for_catchup: {e}");
                    }
                } else {
                    // 2026-10-10: Each sequence's drafts before the step, and its length, give
                    // its verify outcome: `emitted - 1` of the drafts it held were accepted (a
                    // bootstrap step holds none and observes nothing).
                    let before: Vec<(usize, usize)> = active
                        .iter()
                        .map(|a| (a.seq.seq_len, a.pending_drafts.len()))
                        .collect();
                    step_mtp(
                        sched.io.dev.model(),
                        active,
                        sched,
                        match k {
                            Some(k) if dflash_verify_raw_argmax => k,
                            _ => num_drafts,
                        },
                        &verify_ctx,
                        dflash_verify_raw_argmax,
                    );
                    let ms = sched
                        .io
                        .clock
                        .now()
                        .saturating_duration_since(t0)
                        .as_secs_f64()
                        * 1e3;
                    let mut emitted_all = 0;
                    for (a, &(len, drafts)) in active.iter_mut().zip(before.iter()) {
                        let emitted = a.seq.seq_len.saturating_sub(len);
                        emitted_all += emitted;
                        a.mtp_acct.record_verify_emitted(emitted);
                        if let Some(h) = spec_host.as_mut() {
                            h.observe_stream(&mut a.spec_ctl, drafts, emitted.saturating_sub(1));
                        }
                    }
                    if let (Some(h), Some(k)) = (spec_host.as_mut(), k) {
                        h.settle(active.iter_mut().map(|a| &mut a.spec_ctl), k);
                        h.observe_step(active.len(), k, ms, None, emitted_all);
                    }
                }
            } else {
                // 2026-09-25: Plain decode. With MTP on, drop any pending drafts first.
                if use_mtp {
                    for a in active.iter_mut() {
                        a.pending_drafts.clear();
                        a.pending_draft_conf.clear();
                    }
                    // 2026-09-25: Verify commits may still run on the secondary stream:
                    // make the default stream wait for it (a device-side event wait).
                    if let Err(e) = sched.io.dev.apply(io::Effect::SyncSecondary) {
                        tracing::error!("mtp→decode sync_secondary: {e}");
                    }
                }
                if pipelined {
                    pipeline::step_decode_pipelined(
                        active,
                        pipeline,
                        commit_tokens,
                        sched,
                        sched.io.spill.as_deref(),
                        swapped,
                        preempted,
                    );
                } else {
                    step_decode_only(
                        active,
                        think_end_token,
                        think_start_token,
                        code_fence_token,
                        tool_call_start_token,
                        tool_call_end_token,
                        adaptive_sampling,
                        sched,
                        sched.io.spill.as_deref(),
                        swapped,
                        preempted,
                    );
                }
                for a in active.iter_mut() {
                    a.mtp_acct.record_serial();
                }
            }
        }

        LaneVerdict::Proceed
    }
}
