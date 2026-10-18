// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Old reader vs new resolver, per scheduler lever. The readers that
//! `SchedLevers::from_env` used are frozen as [`Old`] rules; each lever is resolved
//! through `metrale_settings` + `SchedLevers::from_settings` over the same value and the
//! outcomes compared (`crate::lever_equivalence`). Every difference must appear in
//! [`CHANGES`], which is the record of the behaviour this migration changed on purpose.
//!
//! Owner: scheduler.
//! Invariants: [`CHANGES`] lists exactly the (rule, value) pairs whose outcome changed.

use std::borrow::Cow;

use super::resolved;
use crate::lever_equivalence::{Old, Out, Row, assert_old_readers_hold, num};
use crate::scheduler::levers::SchedLevers;

fn levers() -> Vec<Row<SchedLevers>> {
    use Old::*;
    vec![
        ("METRALE_DISABLE_FAST_GREEDY", OnUnless, |l| {
            Out::B(l.fast_greedy_grammar)
        }),
        ("METRALE_DISABLE_FAST_MASKED", OnUnless, |l| {
            Out::B(l.fast_masked)
        }),
        ("METRALE_NO_FAST_GREEDY_CHAT", OnUnless, |l| {
            Out::B(l.fast_greedy_chat)
        }),
        ("METRALE_NO_MTP_MINP", OnUnless, |l| Out::B(l.mtp_minp)),
        ("METRALE_NO_MTP_VERIFY_SAMPLE", OnUnless, |l| {
            Out::B(l.mtp_verify_sample)
        }),
        ("METRALE_NO_THINKENDED_GPU_ARGMAX", OnUnless, |l| {
            Out::B(l.think_ended_gpu_argmax)
        }),
        ("METRALE_FORCE_TEMP_ZERO", OptIn, |l| {
            Out::B(l.force_temp_zero)
        }),
        ("METRALE_DFLASH_STEP_TIMING", OptIn, |l| {
            Out::B(l.dflash_step_timing)
        }),
        ("METRALE_DFLASH_ADAPTIVE", OptIn, |l| {
            Out::B(l.dflash_adaptive)
        }),
        ("METRALE_DFLASH_SERIAL_APPEND", OptIn, |l| {
            Out::B(l.dflash_serial_append)
        }),
        ("METRALE_DFLASH_SPEC_THINK", OptIn, |l| {
            Out::B(l.dflash_spec_think)
        }),
        ("METRALE_MTP_TIMING", OptIn, |l| Out::B(l.mtp_timing)),
        ("METRALE_MTP_GATE_FORCE", OptIn, |l| {
            Out::B(l.mtp_gate_force)
        }),
        ("METRALE_EP_PREFILL_BATCH", OptIn, |l| {
            Out::B(l.ep_prefill_batch)
        }),
        ("METRALE_TOOL_RESPONSE_STOP", DefaultOn, |l| {
            Out::B(l.tool_response_stop)
        }),
        ("METRALE_TOOL_EOS_ESCAPE", DefaultOn, |l| {
            Out::B(l.tool_eos_escape)
        }),
        ("METRALE_GRAMMAR_BUDGET_CLOSE", DefaultOn, |l| {
            Out::B(l.grammar_budget_close)
        }),
        ("METRALE_DFLASH_EAGLE_FIX", OnUnlessZero, |l| {
            Out::B(l.dflash_eagle_fix)
        }),
        ("METRALE_DFLASH_MASKED_VERIFY", OnUnlessZero, |l| {
            Out::B(l.dflash_masked_verify)
        }),
        ("METRALE_DFLASH_SEAM_SERIAL", OnUnlessZero, |l| {
            Out::B(l.dflash_seam_serial)
        }),
        ("METRALE_DFLASH_UNIFIED_CTX", OnUnlessZero, |l| {
            Out::B(l.dflash_unified_ctx)
        }),
        ("METRALE_DFLASH_GATE_PIN_C2", OnUnlessZero, |l| {
            Out::B(l.dflash_gate_pin_c2)
        }),
        ("METRALE_DFLASH_BATCH_VERIFY", OnUnlessZero, |l| {
            Out::B(l.dflash_batch_verify)
        }),
        ("METRALE_PARALLEL_SAMPLE", OnUnlessZero, |l| {
            Out::B(l.parallel_sample)
        }),
        ("METRALE_VISION_TIMING", Present, |l| {
            Out::B(l.vision_timing)
        }),
        ("METRALE_DECODE_TIMING", Present, |l| {
            Out::B(l.decode_timing)
        }),
        ("METRALE_ADADEC_DIAGNOSTIC", Present, |l| {
            Out::B(l.adadec_diagnostic)
        }),
        ("METRALE_MTP_ACCEPT_FOLD_AT_16", Present, |l| {
            Out::B(l.mtp_accept_fold_at_16)
        }),
        ("METRALE_NO_MTP_BATCH_BOOTSTRAP", Absent, |l| {
            Out::B(l.mtp_batch_bootstrap)
        }),
        ("METRALE_NO_MTP_BOOT_ARGMAX", Absent, |l| {
            Out::B(l.mtp_boot_argmax)
        }),
        ("METRALE_NO_MTP_BATCH_VERIFY", Absent, |l| {
            Out::B(l.mtp_batch_verify)
        }),
        ("METRALE_NO_MTP_BATCH_PROPOSE", Absent, |l| {
            Out::B(l.mtp_batch_propose)
        }),
        ("METRALE_NO_MTP_DCUT", Absent, |l| Out::B(l.dcut_enabled)),
        ("METRALE_HOLO_ALWAYS_MIXED", OptInWord, |l| {
            Out::B(l.holo_always_mixed)
        }),
        ("METRALE_VISION_CODISPATCH", OptInWord, |l| {
            Out::B(l.vision_codispatch)
        }),
        ("METRALE_BISECT_Q12_DISABLE", OptInLowercase, |l| {
            Out::B(l.bisect_q12_disable)
        }),
        ("METRALE_BISECT_NO_MIX", OptInLowercase, |l| {
            Out::B(l.bisect_no_mix)
        }),
        ("METRALE_BEAM_CODISPATCH", Beam, |l| {
            Out::B(l.beam_codispatch)
        }),
        ("METRALE_DISABLE_WATCHDOGS", DisableWatchdogs, |l| {
            Out::B(l.disable_watchdogs)
        }),
        ("METRALE_DISABLE_FORCED_TOKEN", ForcedToken, |l| {
            Out::B(l.forced_token_fastpath)
        }),
        ("METRALE_DFLASH_ADAPTIVE_MIN", NumF32(2.0), |l| {
            num(l.dflash_adaptive_min)
        }),
        ("METRALE_DFLASH_ADAPTIVE_REPROBE", NumUint(256), |l| {
            num(l.dflash_adaptive_reprobe)
        }),
        ("METRALE_DFLASH_RESUME_GUARD", NumUint(0), |l| {
            num(l.dflash_resume_guard)
        }),
        ("METRALE_PREFILL_CODISPATCH_WINDOW_MS", NumUint(100), |l| {
            num(l.codispatch_window_ms)
        }),
        ("METRALE_PREFILL_CODISPATCH_SETTLE_MS", NumUint(10), |l| {
            num(l.codispatch_settle_ms)
        }),
        ("METRALE_MIXED_SLICE_TOKENS", NumUint(0), |l| {
            num(l.mixed_slice_tokens)
        }),
        ("METRALE_MTP_DCUT_MAX_SEQS", NumUint(8), |l| {
            num(l.dcut_width_cap)
        }),
        (
            "METRALE_MTP_DCUT_RATIO",
            DcutRatio(crate::scheduler::mtp_dcut::snap_ratio),
            |l| num(l.dcut_ratio),
        ),
    ]
}

/// 2026-10-17: Every (rule class, value) whose outcome changed, with the new outcome.
/// Reviewed line by line; three kinds, all intended: the word spellings (`true`/`yes`/`off`/
/// `no`, any case) now mean what they say, values are trimmed, and an empty, unparsable or
/// out-of-range value stops the load instead of falling back to the default.
const CHANGES: &[(&str, &str, Out)] = &[
    ("on_unless", "true", Out::B(false)),
    ("on_unless", "TRUE", Out::B(false)),
    ("on_unless", "yes", Out::B(false)),
    ("on_unless", "", Out::Err),
    ("on_unless", " 1 ", Out::B(false)),
    ("on_unless", "garbage", Out::Err),
    ("on_unless", "7", Out::Err),
    ("on_unless", "-1", Out::Err),
    ("opt_in", "true", Out::B(true)),
    ("opt_in", "TRUE", Out::B(true)),
    ("opt_in", "yes", Out::B(true)),
    ("opt_in", "", Out::Err),
    ("opt_in", " 1 ", Out::B(true)),
    ("opt_in", "garbage", Out::Err),
    ("opt_in", "7", Out::Err),
    ("opt_in", "-1", Out::Err),
    ("default_on", "off", Out::B(false)),
    ("default_on", "no", Out::B(false)),
    ("default_on", "", Out::Err),
    ("default_on", "garbage", Out::Err),
    ("default_on", "7", Out::Err),
    ("default_on", "-1", Out::Err),
    ("on_unless_zero", "false", Out::B(false)),
    ("on_unless_zero", "off", Out::B(false)),
    ("on_unless_zero", "no", Out::B(false)),
    ("on_unless_zero", "", Out::Err),
    ("on_unless_zero", "garbage", Out::Err),
    ("on_unless_zero", "7", Out::Err),
    ("on_unless_zero", "-1", Out::Err),
    ("opt_in_word", "yes", Out::B(true)),
    ("opt_in_word", "", Out::Err),
    ("opt_in_word", " 1 ", Out::B(true)),
    ("opt_in_word", "garbage", Out::Err),
    ("opt_in_word", "7", Out::Err),
    ("opt_in_word", "-1", Out::Err),
    ("opt_in_lowercase", "yes", Out::B(true)),
    ("opt_in_lowercase", "", Out::Err),
    ("opt_in_lowercase", " 1 ", Out::B(true)),
    ("opt_in_lowercase", "garbage", Out::Err),
    ("opt_in_lowercase", "7", Out::Err),
    ("opt_in_lowercase", "-1", Out::Err),
    ("beam", "off", Out::B(false)),
    ("beam", "no", Out::B(false)),
    ("beam", "", Out::Err),
    ("beam", "garbage", Out::Err),
    ("beam", "7", Out::Err),
    ("beam", "-1", Out::Err),
    ("disable_watchdogs", "yes", Out::B(true)),
    ("disable_watchdogs", "", Out::Err),
    ("disable_watchdogs", "garbage", Out::Err),
    ("disable_watchdogs", "7", Out::Err),
    ("disable_watchdogs", "-1", Out::Err),
    ("forced_token", "yes", Out::B(false)),
    ("forced_token", "", Out::Err),
    ("forced_token", "garbage", Out::Err),
    ("forced_token", "7", Out::Err),
    ("forced_token", "-1", Out::Err),
    ("f32", "true", Out::Err),
    ("f32", "TRUE", Out::Err),
    ("f32", "yes", Out::Err),
    ("f32", "false", Out::Err),
    ("f32", "off", Out::Err),
    ("f32", "no", Out::Err),
    ("f32", "", Out::Err),
    ("f32", " 1 ", Out::N(Cow::Borrowed("1"))),
    ("f32", "garbage", Out::Err),
    ("uint", "true", Out::Err),
    ("uint", "TRUE", Out::Err),
    ("uint", "yes", Out::Err),
    ("uint", "false", Out::Err),
    ("uint", "off", Out::Err),
    ("uint", "no", Out::Err),
    ("uint", "", Out::Err),
    ("uint", " 1 ", Out::N(Cow::Borrowed("1"))),
    ("uint", "garbage", Out::Err),
    ("uint", "-1", Out::Err),
    ("dcut_ratio", "true", Out::Err),
    ("dcut_ratio", "TRUE", Out::Err),
    ("dcut_ratio", "yes", Out::Err),
    ("dcut_ratio", "false", Out::Err),
    ("dcut_ratio", "off", Out::Err),
    ("dcut_ratio", "no", Out::Err),
    ("dcut_ratio", "", Out::Err),
    ("dcut_ratio", " 1 ", Out::N(Cow::Borrowed("1"))),
    ("dcut_ratio", "garbage", Out::Err),
];

#[test]
fn every_lever_resolves_as_its_old_reader_did_except_the_recorded_changes() {
    assert_old_readers_hold(&levers(), |env| resolved(env, &[]), CHANGES);
}
