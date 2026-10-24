// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: [`ChatLevers::resolve`] against the request-path readers it replaced. Each
//! lever is resolved over `MapEnv` as `met serve` resolves a load; no process environment is
//! read or written.
//!
//! Owner: server (chat API) tests.
//! Invariants: every outcome that differs from the old reader is listed in [`CHANGES`].

use std::borrow::Cow;

use super::*;
use crate::lever_equivalence::{
    Old, Out, Row, assert_old_readers_hold, default_settings, num, settings_from,
};
use crate::tokenizer::ToolJsonStyle;

/// 2026-10-17: What one load resolves for the chat path: its levers and the template's
/// `tojson` style.
type Resolved = (ChatLevers, ToolJsonStyle);

fn resolved(env: &[(&str, &str)]) -> anyhow::Result<Resolved> {
    let settings = settings_from(env)?;
    Ok((
        ChatLevers::resolve(&settings, false, false)?,
        ToolJsonStyle::from_settings(&settings),
    ))
}

#[test]
fn off_is_the_default() {
    assert_eq!(ChatLevers::default(), ChatLevers::OFF);
}

#[test]
fn off_is_what_an_empty_environment_resolves_to() {
    let (levers, style) = resolved(&[]).expect("defaults resolve");
    assert_eq!(levers, ChatLevers::OFF);
    assert_eq!(style, ToolJsonStyle::Compact);
}

#[test]
fn the_model_behavior_reaches_the_renderer() {
    // 2026-09-26: `resolve` takes `tscg` and `disable_cwd_hint_injection` from its caller
    // (MODEL.toml `[behavior]`), not from the settings.
    let settings = default_settings();
    let with = |tscg, cwd| ChatLevers::resolve(&settings, tscg, cwd).expect("resolves");
    assert!(with(true, false).prompt.tscg);
    assert!(!with(false, false).prompt.tscg);
    assert!(with(false, true).disable_cwd_hint_injection);
    assert!(!with(true, false).disable_cwd_hint_injection);
}

#[test]
fn leak_opener_zero_never_cancels_and_is_not_the_default() {
    let at = |v| resolved(&[("METRALE_INTHINK_TOOL_LEAK_OPENERS", v)]).map(|r| r.0);
    assert_eq!(at("0").expect("0 parses").in_think_leak_openers, 0);
    assert_eq!(at("3").expect("3 parses").in_think_leak_openers, 3);
    // 2026-10-17: Junk stops the load; it used to fall back to 1, never to 0.
    assert!(at("many").is_err());
}

fn rows() -> Vec<Row<Resolved>> {
    vec![
        ("METRALE_BASH_WANDER_WATCHDOG", Old::OptIn, |r| {
            Out::B(r.0.bash_wander)
        }),
        ("METRALE_CHAT_PHASE_TIMING", Old::OptIn, |r| {
            Out::B(r.0.phase_timing)
        }),
        (
            "METRALE_INTHINK_TOOL_LEAK_OPENERS",
            Old::UintTrimmed(1),
            |r| num(r.0.in_think_leak_openers),
        ),
        ("METRALE_LOG_TOOL_RAW", Old::OptIn, |r| {
            Out::B(r.0.log_tool_raw)
        }),
        ("METRALE_LOOP_NO_SUPPRESS", Old::OnUnless, |r| {
            Out::B(r.0.loop_suppress)
        }),
        ("METRALE_STRIP_REASONING_HISTORY", Old::OptInWord, |r| {
            Out::B(r.0.strip_reasoning_history)
        }),
        ("METRALE_NO_ERROR_DEDUP", Old::OnUnless, |r| {
            Out::B(r.0.error_dedup)
        }),
        ("METRALE_FORCE_TEMP_ZERO", Old::OptInWord, |r| {
            Out::B(r.0.force_temp_zero)
        }),
        ("METRALE_SIMHASH_LOOP", Old::OnUnlessZero, |r| {
            Out::B(r.0.simhash_loop)
        }),
        ("METRALE_DISABLE_WATCHDOGS", Old::DisableWatchdogs, |r| {
            Out::B(r.0.disable_watchdogs)
        }),
        (
            "METRALE_MAX_TOOL_CALLS_PER_RESPONSE",
            Old::NumUint(12),
            |r| num(r.0.max_tool_calls_per_response),
        ),
        ("METRALE_THINKING_DECODE_TRACE", Old::Present, |r| {
            Out::B(r.0.thinking_decode_trace)
        }),
        ("METRALE_DISABLE_REFUSAL_DETECTION", Old::OnUnless, |r| {
            Out::B(r.0.refusal_detection)
        }),
        ("METRALE_BUFFER_TOOL_ARGS", Old::OptInOneOrTrue, |r| {
            Out::B(r.0.buffer_tool_args)
        }),
        ("METRALE_TOOLCALL_DEBUG", Old::OptIn, |r| {
            Out::B(r.0.toolcall_debug)
        }),
        ("METRALE_OFFICIAL_TOOL_PROMPT", Old::OptIn, |r| {
            Out::B(r.0.prompt.official_tool_prompt)
        }),
        ("METRALE_USE_HF_REF_JSON_DUMPS", Old::OptIn, |r| {
            Out::B(r.1 == ToolJsonStyle::HfSpaced)
        }),
    ]
}

/// 2026-10-17: Every (old rule, value) whose outcome changed with the move to resolved
/// settings, and the new outcome.
/// Grouped by old rule; every group is the switch or integer grammar of the settings crate:
/// junk and empty values stop the load, and switch words parse in any case, trimmed.
const CHANGES: &[(&str, &str, Out)] = &[
    ("opt_in", "true", Out::B(true)),
    ("opt_in", "TRUE", Out::B(true)),
    ("opt_in", "yes", Out::B(true)),
    ("opt_in", "", Out::Err),
    ("opt_in", " 1 ", Out::B(true)),
    ("opt_in", "garbage", Out::Err),
    ("opt_in", "7", Out::Err),
    ("opt_in", "-1", Out::Err),
    ("uint_trimmed", "true", Out::Err),
    ("uint_trimmed", "TRUE", Out::Err),
    ("uint_trimmed", "yes", Out::Err),
    ("uint_trimmed", "false", Out::Err),
    ("uint_trimmed", "off", Out::Err),
    ("uint_trimmed", "no", Out::Err),
    ("uint_trimmed", "", Out::Err),
    ("uint_trimmed", "garbage", Out::Err),
    ("uint_trimmed", "-1", Out::Err),
    ("on_unless", "true", Out::B(false)),
    ("on_unless", "TRUE", Out::B(false)),
    ("on_unless", "yes", Out::B(false)),
    ("on_unless", "", Out::Err),
    ("on_unless", " 1 ", Out::B(false)),
    ("on_unless", "garbage", Out::Err),
    ("on_unless", "7", Out::Err),
    ("on_unless", "-1", Out::Err),
    ("opt_in_word", "yes", Out::B(true)),
    ("opt_in_word", "", Out::Err),
    ("opt_in_word", " 1 ", Out::B(true)),
    ("opt_in_word", "garbage", Out::Err),
    ("opt_in_word", "7", Out::Err),
    ("opt_in_word", "-1", Out::Err),
    ("on_unless_zero", "false", Out::B(false)),
    ("on_unless_zero", "off", Out::B(false)),
    ("on_unless_zero", "no", Out::B(false)),
    ("on_unless_zero", "", Out::Err),
    ("on_unless_zero", "garbage", Out::Err),
    ("on_unless_zero", "7", Out::Err),
    ("on_unless_zero", "-1", Out::Err),
    ("disable_watchdogs", "yes", Out::B(true)),
    ("disable_watchdogs", "", Out::Err),
    ("disable_watchdogs", "garbage", Out::Err),
    ("disable_watchdogs", "7", Out::Err),
    ("disable_watchdogs", "-1", Out::Err),
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
    ("opt_in_one_or_true", "TRUE", Out::B(true)),
    ("opt_in_one_or_true", "yes", Out::B(true)),
    ("opt_in_one_or_true", "", Out::Err),
    ("opt_in_one_or_true", " 1 ", Out::B(true)),
    ("opt_in_one_or_true", "garbage", Out::Err),
    ("opt_in_one_or_true", "7", Out::Err),
    ("opt_in_one_or_true", "-1", Out::Err),
];

#[test]
fn resolved_levers_match_the_old_readers_except_the_recorded_changes() {
    assert_old_readers_hold(&rows(), resolved, CHANGES);
}
