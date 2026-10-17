// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`ChatLevers`]: the chat request path's per-model and
//! per-deployment switches, resolved once when [`crate::AppState`] is built
//! and read from `state.chat`.
//!
//! Owner: server (chat API).
//! Invariants:
//! - 2026-10-17: A request reads its levers only from `state.chat`; no request-path code reads
//!   the environment (ROTI). The values come from the load's resolved settings.

use metrale_settings::InferenceServerSettings;

use crate::lever_value::{text_int, text_switch};
use crate::tool_parser::PromptLevers;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatLevers {
    /// 2026-09-26: Prompt-rendering decisions, handed to
    /// `ToolCallParser::system_prompt` and read by the chat template step.
    pub prompt: PromptLevers,
    /// 2026-09-26: `METRALE_BASH_WANDER_WATCHDOG`: append the steering hint
    /// from `hint_injector::bash_wander_hint` (enough tool calls in the
    /// conversation, none of them productive) to the latest tool result.
    pub bash_wander: bool,
    /// 2026-09-26: `METRALE_CHAT_PHASE_TIMING`: log `CHAT_PHASE` timing lines
    /// from `chat_completions_inner` and `prepare_chat_prompt`.
    pub phase_timing: bool,
    /// 2026-09-26: MODEL.toml `[behavior] disable_cwd_hint_injection`: do not
    /// append the `<environment>working_directory: …</environment>` hint to
    /// the system message of a tools-active request.
    pub disable_cwd_hint_injection: bool,
    /// 2026-09-26: `METRALE_INTHINK_TOOL_LEAK_OPENERS=N`: a streaming request
    /// is cancelled once its reasoning has shown `N` tool-call openers
    /// (`api/chat_stream/handle_token.rs`). Default 1; 0 never cancels.
    pub in_think_leak_openers: u32,
    /// 2026-10-17: `METRALE_LOG_TOOL_RAW`: log raw generated text
    /// (`chat_blocking_choice::log_raw_generation`).
    pub log_tool_raw: bool,
    /// 2026-10-17: Off under `METRALE_LOOP_NO_SUPPRESS`: loop verdicts are still detected,
    /// logged and counted, but never suppress the tool call (`loop_detect::check_loops`).
    pub loop_suppress: bool,
    /// 2026-10-17: `METRALE_STRIP_REASONING_HISTORY`: drop the reasoning of every earlier turn.
    pub strip_reasoning_history: bool,
    /// 2026-10-17: Off under `METRALE_NO_ERROR_DEDUP`: duplicate tool errors are not masked.
    pub error_dedup: bool,
    /// 2026-10-17: `METRALE_FORCE_TEMP_ZERO`: greedy decoding whatever the request, preset or
    /// MODEL.toml floors say.
    pub force_temp_zero: bool,
    /// 2026-10-17: `METRALE_SIMHASH_LOOP` (on by default): the SimHash semantic-loop guard.
    pub simhash_loop: bool,
    /// 2026-10-17: `METRALE_DISABLE_WATCHDOGS`: the streamed token loop watchdog is off too.
    pub disable_watchdogs: bool,
    /// 2026-10-17: `METRALE_MAX_TOOL_CALLS_PER_RESPONSE` (default
    /// [`MAX_TOOL_CALLS_PER_RESPONSE`]): a streamed response stops after this many calls.
    pub max_tool_calls_per_response: usize,
    /// 2026-10-17: `METRALE_THINKING_DECODE_TRACE` (any value): the thinking-phase emit trace
    /// (`chat_stream::strip::log_decode_trace`).
    pub thinking_decode_trace: bool,
    /// 2026-10-17: Off under `METRALE_DISABLE_REFUSAL_DETECTION`: content that opens with a
    /// refusal is not moved to `refusal` (`crate::refusal::detect`).
    pub refusal_detection: bool,
    /// 2026-10-17: `METRALE_BUFFER_TOOL_ARGS`: a streamed tool call's arguments arrive in one
    /// `ToolCallDelta` at its close instead of live fragments.
    pub buffer_tool_args: bool,
    /// 2026-10-17: `METRALE_TOOLCALL_DEBUG`: log the argument shape of a write-family call
    /// whose path is empty (`tool_parser::assess_tool_call`).
    pub toolcall_debug: bool,
}

/// 2026-10-17: The streamed tool-call cap when `METRALE_MAX_TOOL_CALLS_PER_RESPONSE` is unset.
pub const MAX_TOOL_CALLS_PER_RESPONSE: usize = 12;

/// 2026-10-17: The in-think leak threshold when `METRALE_INTHINK_TOOL_LEAK_OPENERS` is unset.
const IN_THINK_LEAK_OPENERS: u32 = 1;

impl Default for ChatLevers {
    /// 2026-09-26: [`ChatLevers::OFF`]. Not derived: a derived `Default` would
    /// set `in_think_leak_openers` to 0, which never cancels.
    fn default() -> Self {
        Self::OFF
    }
}

impl ChatLevers {
    /// 2026-09-26: What `resolve` gives with no lever set and no `[behavior]` flag. Request-path
    /// tests build against it.
    pub const OFF: Self = Self {
        prompt: PromptLevers::OFF,
        bash_wander: false,
        phase_timing: false,
        disable_cwd_hint_injection: false,
        in_think_leak_openers: IN_THINK_LEAK_OPENERS,
        log_tool_raw: false,
        loop_suppress: true,
        strip_reasoning_history: false,
        error_dedup: true,
        force_temp_zero: false,
        simhash_loop: true,
        disable_watchdogs: false,
        max_tool_calls_per_response: MAX_TOOL_CALLS_PER_RESPONSE,
        thinking_decode_trace: false,
        refusal_detection: true,
        buffer_tool_args: false,
        toolcall_debug: false,
    };

    /// 2026-10-17: Resolve from the load's settings and this model's `[behavior]` values.
    /// Called when `AppState` is built (`serve_load.rs`).
    ///
    /// # Errors
    /// A `Text`-declared lever whose value does not parse as its reader's type.
    pub fn resolve(
        settings: &InferenceServerSettings,
        tscg: bool,
        disable_cwd_hint_injection: bool,
    ) -> anyhow::Result<Self> {
        let sched = &settings.scheduler;
        let sampling = &settings.sampling;
        let telemetry = &settings.telemetry;
        Ok(Self {
            prompt: PromptLevers::new(tscg, sched.official_tool_prompt),
            bash_wander: sched.bash_wander_watchdog,
            phase_timing: telemetry.chat_phase_timing,
            disable_cwd_hint_injection,
            in_think_leak_openers: text_int(
                "METRALE_INTHINK_TOOL_LEAK_OPENERS",
                sampling.inthink_tool_leak_openers.as_deref(),
            )?
            .unwrap_or(IN_THINK_LEAK_OPENERS),
            log_tool_raw: telemetry.log_tool_raw,
            loop_suppress: !sched.loop_no_suppress,
            strip_reasoning_history: settings.runtime.strip_reasoning_history,
            error_dedup: !settings.kernels.no_error_dedup,
            force_temp_zero: sampling.force_temp_zero,
            simhash_loop: sched.simhash_loop,
            disable_watchdogs: sched.disable_watchdogs,
            max_tool_calls_per_response: text_int(
                "METRALE_MAX_TOOL_CALLS_PER_RESPONSE",
                sched.max_tool_calls_per_response.as_deref(),
            )?
            .unwrap_or(MAX_TOOL_CALLS_PER_RESPONSE),
            thinking_decode_trace: telemetry.thinking_decode_trace,
            refusal_detection: !sampling.disable_refusal_detection,
            buffer_tool_args: text_switch(
                "METRALE_BUFFER_TOOL_ARGS",
                sched.buffer_tool_args.as_deref(),
            )? == Some(true),
            toolcall_debug: telemetry.toolcall_debug,
        })
    }
}

#[cfg(test)]
#[path = "levers_tests.rs"]
mod tests;
