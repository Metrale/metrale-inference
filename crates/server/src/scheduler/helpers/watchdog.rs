// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-model watchdog tunables (`WatchdogParams`) and their
//! env/CLI resolution.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: Per-model tunables for the decode-time watchdogs. A served
/// run builds them with [`WatchdogParams::from_behavior`] from the model's
/// MODEL.toml `[behavior]` table; `Default` returns
/// [`DEFAULT_WATCHDOG_PARAMS`]. The two differ for an unset table:
/// `confidence_run_length` is 30 from `ModelBehavior` and
/// `CONFIDENCE_RUN_LIMIT` (60) from `Default`.
#[derive(Debug, Clone, Copy)]
pub struct WatchdogParams {
    /// 2026-09-25: Thinking-loop watchdog: how many end-anchored copies of
    /// a period trip a forced `</think>`.
    pub think_loop_min_repeats: usize,
    /// 2026-09-25: Passed to [`super::detectors::detect_token_loop`], which
    /// ignores it, so this field has no effect.
    pub think_loop_scan_window: usize,
    /// 2026-09-25: Whether `F2ConfidenceEarlyStop` may arm a forced
    /// `</think>`.
    pub confidence_early_stop: bool,
    /// 2026-09-25: Consecutive confident thinking tokens after which
    /// `F2ConfidenceEarlyStop` arms a forced `</think>`.
    pub confidence_run_length: u32,
    /// 2026-09-25: Fuzzy-repetition tolerance divisor: a `pattern_len`
    /// window tolerates `max(pattern_len / div, 1)` mismatches.
    pub fuzzy_repeat_tolerance_div: usize,
    /// 2026-09-25: Cap on free-text tokens since the last `<tool_call>`
    /// open on a tool request. `u32::MAX` means the guard is disabled (see
    /// [`resolve_max_inter_tool_prose`]).
    pub max_inter_tool_prose: u32,
    /// 2026-09-25: Cap on content tokens per generation for a sequence
    /// with a grammar attached.
    pub max_post_think_content_tokens: u32,
    /// 2026-09-25: Per-model gate for the thinking-loop watchdog.
    pub enable_think_loop_watchdog: bool,
    /// 2026-09-25: Thinking tokens during which `</think>` gets a negative
    /// logit bias (`[behavior].min_reasoning_floor_tokens`); 0 disables it.
    pub min_reasoning_floor: u32,
    /// 2026-09-25: Allow a firing watchdog to roll back to the last
    /// boundary and re-steer; false makes
    /// [`crate::scheduler::rollback::rollback_to_boundary`] decline.
    pub rollback_resteer: bool,
    /// 2026-09-25: Operator override for the content-loop detector's
    /// repeat threshold (`--content-loop-min-repeats` /
    /// `METRALE_CONTENT_LOOP_MIN_REPEATS`). `None` = the built-in
    /// [`CONTENT_LOOP_MIN_REPEATS`]. A per-request `repetition_detection`
    /// object outranks it (see [`WatchdogParams::content_loop_params`]).
    pub content_loop_min_repeats: Option<u32>,
    /// 2026-10-14: MODEL.toml `[behavior].thinking_steering`. False leaves the
    /// thinking block to the model, as vLLM does: no confidence or loop
    /// watchdog arms `</think>`, no `</think>` floor, mid-word or post-close
    /// mask, a stray `</think>` is kept as a content token, and a `<think>`
    /// re-opens reasoning. A client budget still forces `</think>`, at once
    /// (vLLM's thinking-budget processor). `max_tokens`, `max_seq_len`, the
    /// request timeout and cancellation bound the block.
    pub thinking_steering: bool,
}

/// 2026-09-25: What `WatchdogParams::default()` returns. Served runs use
/// [`WatchdogParams::from_behavior`] instead.
const DEFAULT_WATCHDOG_PARAMS: WatchdogParams = WatchdogParams {
    think_loop_min_repeats: THINK_LOOP_MIN_REPEATS,
    think_loop_scan_window: THINK_LOOP_SCAN_WINDOW,
    confidence_early_stop: true,
    confidence_run_length: crate::scheduler::confidence::CONFIDENCE_RUN_LIMIT,
    fuzzy_repeat_tolerance_div: 12,
    max_inter_tool_prose: MAX_INTER_TOOL_PROSE,
    max_post_think_content_tokens: MAX_POST_THINK_CONTENT_TOKENS,
    rollback_resteer: true,
    enable_think_loop_watchdog: true,
    content_loop_min_repeats: None,
    min_reasoning_floor: 16,
    thinking_steering: true,
};

impl Default for WatchdogParams {
    fn default() -> Self {
        DEFAULT_WATCHDOG_PARAMS
    }
}

impl WatchdogParams {
    /// 2026-09-25: The default fuzzy-repeat tolerance divisor, exposed so
    /// the `repetition` tests can name it.
    pub const DEFAULT_FUZZY_TOLERANCE_DIV: usize =
        DEFAULT_WATCHDOG_PARAMS.fuzzy_repeat_tolerance_div;

    /// 2026-09-25: Resolve this model's watchdog tunables from its
    /// MODEL.toml `[behavior]` table, then the overrides that outrank it.
    ///
    /// `max_inter_tool_prose` and `content_loop_min_repeats` are the operator overrides as the
    /// serve's settings resolved them (`--max-inter-tool-prose` /
    /// `METRALE_MAX_INTER_TOOL_PROSE`, `--content-loop-min-repeats` /
    /// `METRALE_CONTENT_LOOP_MIN_REPEATS`, flag first). `None` keeps MODEL.toml's budget and
    /// the built-in [`CONTENT_LOOP_MIN_REPEATS`].
    pub fn from_behavior(
        b: &metrale_kernel_set::ModelBehavior,
        max_inter_tool_prose: Option<u32>,
        content_loop_min_repeats: Option<u32>,
    ) -> Self {
        let steering = b.thinking_steering;
        Self {
            min_reasoning_floor: if steering {
                b.min_reasoning_floor_tokens
            } else {
                0
            },
            think_loop_min_repeats: b.think_loop_min_repeats as usize,
            think_loop_scan_window: b.think_loop_scan_window as usize,
            confidence_early_stop: b.confidence_early_stop && steering,
            confidence_run_length: b.confidence_run_length,
            fuzzy_repeat_tolerance_div: b.fuzzy_repeat_tolerance_div as usize,
            max_inter_tool_prose: resolve_max_inter_tool_prose(
                b.max_inter_tool_prose,
                max_inter_tool_prose,
            ),
            max_post_think_content_tokens: b.max_post_think_content_tokens,
            rollback_resteer: b.rollback_resteer,
            enable_think_loop_watchdog: b.enable_think_loop_watchdog && steering,
            content_loop_min_repeats,
            thinking_steering: steering,
        }
    }

    /// 2026-09-25: The content-loop detector params in force for one
    /// sequence: the request's own `repetition_detection` object outranks
    /// the operator override; `None` = the built-in constants. The operator
    /// override sets only the repeat threshold; its periods are the built-in
    /// range.
    pub fn content_loop_params(
        &self,
        request: Option<crate::api::inference_types::RepetitionDetectionParams>,
    ) -> Option<crate::api::inference_types::RepetitionDetectionParams> {
        request.or_else(|| {
            self.content_loop_min_repeats.map(|n| {
                crate::api::inference_types::RepetitionDetectionParams {
                    min_pattern_size: CONTENT_LOOP_PERIOD_MIN as u32,
                    max_pattern_size: CONTENT_LOOP_PERIOD_MAX as u32,
                    min_count: n,
                }
            })
        })
    }
}

/// 2026-09-25: Resolve the effective inter-tool prose budget.
///
/// The operator override (`--max-inter-tool-prose`, else `METRALE_MAX_INTER_TOOL_PROSE`, as the
/// serve's settings resolved it) wins over MODEL.toml `[behavior].max_inter_tool_prose`, which
/// already holds the shared default (`metrale_kernel_set::DEFAULT_MAX_INTER_TOOL_PROSE`) for an
/// unset key.
///
/// 0 means "guard disabled" and maps to `u32::MAX`: the check sites fire
/// on `prose_tokens_since_last_tool > max`, so a literal 0 would end a tool
/// request at its first free-text token.
pub fn resolve_max_inter_tool_prose(toml: u32, operator: Option<u32>) -> u32 {
    let v = operator.unwrap_or(toml);
    if v == 0 { u32::MAX } else { v }
}

/// 2026-09-25: The default inter-tool prose budget. With
/// `tool_choice="auto"` the tool grammar compiles with
/// `stop_after_first=false`, so it does not end the turn after a tool call;
/// this budget bounds the free text in between. It aliases
/// `metrale_kernel_set::DEFAULT_MAX_INTER_TOOL_PROSE`, the value the build-time
/// `[behavior]` parse uses for an unset key, so `Default` and an unset
/// MODEL.toml key agree.
pub const MAX_INTER_TOOL_PROSE: u32 = metrale_kernel_set::DEFAULT_MAX_INTER_TOOL_PROSE;

/// 2026-09-25: `Default`'s cap on content tokens per generation for a
/// sequence with a grammar attached (`grammar_state.is_some()`); the check
/// ignores `inside_tool_body`. The build-time parse of an unset
/// `[behavior].max_post_think_content_tokens` also gives 100_000; the
/// qwen3.6-35b-a3b MODEL.tomls set 8192.
pub const MAX_POST_THINK_CONTENT_TOKENS: u32 = 100_000;
