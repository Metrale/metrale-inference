// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Loop and spinning detection over a request's history. It reads
//! the messages and decides whether this turn hard-masks `<tool_call>` and how
//! far `build_sampling` biases against it.
//!
//! Owner: server (chat API).
//! Invariants:
//! - With tools inactive, both outputs are zero/false.
//! - `suppress_tool_call` is false whenever `METRALE_LOOP_NO_SUPPRESS=1`.

use crate::ir::{Message, Role};

pub(super) struct LoopDetectOut {
    /// 2026-09-26: Hard-mask `<tool_call>` for this turn. Set by spinning, or
    /// by a Suppress verdict whose repeated calls neither all failed nor
    /// returned changing results. When set, `build_sampling` adds no
    /// `<tool_call>` bias.
    pub(super) suppress_tool_call: bool,
    /// 2026-09-26: Run length in assistant turns from a Suppress or Hint
    /// verdict, or from `detect_exact_failing_repeat` when that is longer;
    /// 0 otherwise. `build_sampling` maps it to the `<tool_call>` logit bias.
    pub(super) tool_call_repeat_count: usize,
}

/// 2026-10-17: With `suppress` off (`ChatLevers::loop_suppress`), verdicts are still
/// detected, logged and counted, but never set `suppress_tool_call`.
pub(super) fn check_loops(
    messages: &[Message],
    tools_active: bool,
    suppress: bool,
) -> LoopDetectOut {
    let mut suppress_tool_call = false;
    let mut tool_call_repeat_count: usize = 0;

    if !tools_active {
        return LoopDetectOut {
            suppress_tool_call,
            tool_call_repeat_count,
        };
    }

    // 2026-09-26: Arguments are re-serialized from the IR's JSON, so the
    // client's original whitespace does not affect similarity.
    let signatures: Vec<crate::loop_detector::Signature> = messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::Assistant)
        .map(|m| {
            let text = m.text();
            let owned: Vec<(String, String)> = m
                .tool_calls
                .iter()
                .map(|tc| (tc.name.clone(), tc.arguments.to_string()))
                .collect();
            let calls: Vec<(&str, &str)> = owned
                .iter()
                .map(|(n, a)| (n.as_str(), a.as_str()))
                .collect();
            crate::loop_detector::Signature::build(&text, calls)
        })
        .take(8)
        .collect();
    let verdict = crate::loop_detector::detect(&signatures);

    // 2026-09-26: Pair each assistant turn with the tool results after it,
    // which `Signature` never sees, for the failing-call fast path and the
    // Suppress gates below. Built oldest-first, then reversed to newest-first.
    // Tuple: (call unit, saw a result, every result error-shaped, result text).
    let mut outcomes_fwd: Vec<(Option<String>, bool, bool, String)> = Vec::new();
    for m in messages.iter() {
        match m.role {
            Role::Assistant => {
                let unit = if m.tool_calls.is_empty() {
                    None
                } else {
                    let mut s = String::new();
                    for tc in &m.tool_calls {
                        if !s.is_empty() {
                            s.push('\u{1e}');
                        }
                        s.push_str(&tc.name);
                        s.push('\u{1f}');
                        s.push_str(&tc.arguments.to_string());
                    }
                    Some(s)
                };
                outcomes_fwd.push((unit, false, true, String::new()));
            }
            Role::Tool => {
                if let Some(last) = outcomes_fwd.last_mut() {
                    let text = m.text();
                    last.1 = true;
                    last.2 &= crate::hint_injector::looks_like_error(&text);
                    // 2026-09-26: Up to 2000 chars per result, for
                    // `recent_results_progressing`.
                    let take = text.chars().take(2000);
                    last.3.extend(take);
                    last.3.push('\n');
                }
            }
            _ => {}
        }
    }
    let call_outcomes: Vec<crate::loop_detector::CallOutcome> = outcomes_fwd
        .into_iter()
        .rev()
        .take(8)
        .map(
            |(unit, saw_result, all_err, result_text)| crate::loop_detector::CallOutcome {
                call_unit: unit,
                failing: saw_result && all_err,
                result_unit: if saw_result { Some(result_text) } else { None },
            },
        )
        .collect();
    let exact_failing_run = crate::loop_detector::detect_exact_failing_repeat(&call_outcomes);

    // 2026-09-26: Spinning: the latest 5 or more assistant turns (counting
    // stops at 8) are text-only and under 500 bytes each. 2026-10-19: Length
    // alone suppressed healthy sessions — 5 short acks ("Done.") ratcheted
    // `recent_short` (the suppressed `<tool_call>` forces another short text turn),
    // and `ccx-tp3` saw 13 consecutive firings in one minute. The short run
    // must also be loop-like: the same `Signature`/jaccard channels
    // `detect` uses must score the run at or above `MODERATE_SIMILARITY`.
    let mut recent_short: usize = 0;
    for m in messages.iter().rev() {
        if m.role != Role::Assistant {
            continue;
        }
        let tool_args_len: usize = m
            .tool_calls
            .iter()
            .map(|tc| tc.arguments.to_string().len())
            .sum();
        // 2026-09-26: A turn with any tool call ends the count. Repeated tool
        // calls are `loop_detector::detect`'s job; spinning counts only
        // text-only turns.
        let made_tool_call = !m.tool_calls.is_empty();
        let is_substantial = made_tool_call || m.text().len() >= 500 || tool_args_len >= 100;
        if is_substantial {
            break;
        }
        recent_short += 1;
        if recent_short >= 8 {
            break;
        }
    }
    let spinning = recent_short >= 5 && short_run_is_loop_like(messages, recent_short);

    match &verdict {
        crate::loop_detector::LoopState::Suppress {
            score,
            run_length,
            channel,
        } => {
            // 2026-09-26: No hard mask when every repeated call failed:
            // `<tool_call>` is the model's way out of that loop.
            let failing_repeat =
                crate::loop_detector::recent_calls_all_failing(&call_outcomes, *run_length);
            // 2026-09-26: No hard mask either while the results still differ
            // from round to round; only a loop whose results also repeat
            // keeps it.
            let progressing = crate::loop_detector::recent_results_progressing(
                &call_outcomes,
                (*run_length).max(2),
            );
            if progressing && !failing_repeat {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS on PROGRESSING cycle: results differ                      round-to-round; <tool_call> hard-mask SKIPPED (soft bias decay only)"
                );
            }
            if failing_repeat {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS on FAILING repeated call: <tool_call> hard-mask \
                     SKIPPED (escape action stays available); soft bias decay only"
                );
            } else {
                tracing::warn!(
                    score = *score,
                    run_length = *run_length,
                    channel = channel.name(),
                    "Loop detector → SUPPRESS: hard-mask <tool_call> for one turn"
                );
            }
            suppress_tool_call = !failing_repeat && !progressing && suppress;
            tool_call_repeat_count = *run_length;
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["suppress", channel.name(), if spinning { "1" } else { "0" }])
                .inc();
        }
        crate::loop_detector::LoopState::Hint {
            score,
            run_length,
            channel,
        } => {
            // 2026-09-26: A Hint only sets `tool_call_repeat_count`, which
            // `build_sampling` turns into the `<tool_call>` bias.
            tracing::info!(
                score = *score,
                run_length = *run_length,
                channel = channel.name(),
                "Loop detector → HINT: soft <tool_call> bias decay via tool_call_repeat_count \
                 (no hard-mask, nothing injected)"
            );
            tool_call_repeat_count = *run_length;
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["hint", channel.name(), if spinning { "1" } else { "0" }])
                .inc();
        }
        crate::loop_detector::LoopState::None => {
            crate::metrics::LOOP_DETECTOR_VERDICTS
                .with_label_values(&["none", "n/a", if spinning { "1" } else { "0" }])
                .inc();
        }
    }
    // 2026-09-26: Byte-identical failing calls too short for
    // `loop_detector`'s `MIN_CHANNEL_TOKENS` are invisible to `detect`. This
    // path raises `tool_call_repeat_count` for them and never sets the hard
    // mask.
    if let Some(run) = exact_failing_run
        && run > tool_call_repeat_count
    {
        tracing::warn!(
            run_length = run,
            "Loop detector → FAILING-REPEAT fast path: byte-identical failing tool calls; \
             soft <tool_call> bias decay only (no hard-mask — escape action stays available)"
        );
        tool_call_repeat_count = run;
        crate::metrics::LOOP_DETECTOR_VERDICTS
            .with_label_values(&["failing_repeat", "tools", if spinning { "1" } else { "0" }])
            .inc();
    }

    if spinning {
        tracing::warn!(
            recent_short,
            "Spinning detection fired — suppressing <tool_call>"
        );
        suppress_tool_call = suppress;
    }

    LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    }
}

#[cfg(test)]
#[path = "loop_detect_tests.rs"]
mod tests;

/// 2026-10-19: True when the newest `run` assistant turns are loop-like: the
/// shortest of them carries at least `MIN_CHANNEL_TOKENS` text tokens (shorter
/// acks cannot be compared and must not suppress) and each adjacent pair in
/// the run scores at or above `MODERATE_SIMILARITY` under the same tokenizer
/// and threshold as `detect` — one notion of "repeating", with no channel
/// floor because short acks never reach shingle size.
fn short_run_is_loop_like(messages: &[Message], run: usize) -> bool {
    let mut texts = messages
        .iter()
        .rev()
        .filter(|m| m.role == Role::Assistant)
        .take(run)
        .map(|m| m.text());
    let mut prev = match texts.next() {
        Some(t) => t,
        None => return false,
    };
    for text in texts {
        if !crate::loop_detector::Signature::raw_texts_similar(&prev, &text) {
            return false;
        }
        prev = text;
    }
    true
}
