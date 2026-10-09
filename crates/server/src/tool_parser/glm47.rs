// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The GLM-4.7 tool-call format with fail-closed checking, for
//! GLM-5.3 (`glm5_next`). The wire format is Poolside v1's
//! (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`),
//! so the prompt, history rendering, grammar and leak markers are
//! `PoolsideV1Parser`'s. What differs is what happens to a parsed call
//! ([`CallPolicy::FailClosed`]):
//!
//! - values stay strings until the tool's schema types them (`args.rs`), so a
//!   string parameter holding `42` or `{"a":1}` is delivered as that string;
//! - a call is delivered only when its name resolves to an offered tool and
//!   every key is in that tool's schema; anything else is delivered as a
//!   refusal with a reason, never dropped and never turned into content
//!   (`judge.rs`);
//! - a call that holds the call the model meant (opened inside reasoning, or
//!   abandoned for another call) is refused with the meant call spelled out
//!   (`recover.rs`);
//! - while streaming, arguments are held until the call closes, the header is
//!   sent as soon as the name is an offered tool, and an empty argument chunk
//!   goes out every `KEEPALIVE` while a long call is written
//!   (`streaming_glm47.rs`, `api/chat_stream`).
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - Every method that shapes the prompt or the grammar delegates to
//!   `PoolsideV1Parser`, so the two formats render and constrain identically.

use std::time::Duration;

use super::*;

mod args;
mod blocking;
mod judge;
mod recover;

pub(super) use args::ARG_KEY;
pub use blocking::parse_glm47_answer;
pub(super) use blocking::{TOOL_CALL, TOOL_CALL_END};
pub(super) use judge::split_body;
pub use judge::{REFUSAL_KEY, Verdict, judge, resolve_name};

/// 2026-10-08: Interval between keep-alive chunks while a call's arguments are
/// held. Agent clients drop a stream that sends nothing for a few minutes, and
/// a long call (a large file write) can take that long to write.
pub const KEEPALIVE: Duration = Duration::from_secs(5);

impl Verdict {
    /// 2026-10-08: The call to deliver, under a new call id.
    pub fn into_tool_call(self) -> ToolCall {
        let (name, arguments) = match self {
            Verdict::Accept { name, arguments }
            | Verdict::Refuse {
                name, arguments, ..
            } => (name, arguments),
        };
        ToolCall {
            id: next_tool_call_id(),
            call_type: "function".into(),
            function: FunctionCall { name, arguments },
        }
    }
}

/// 2026-10-08: GLM-4.7 calls, checked fail-closed (module doc).
pub struct Glm47Parser;

impl ToolCallParser for Glm47Parser {
    fn name(&self) -> &str {
        "glm47"
    }

    fn system_prompt(
        &self,
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        levers: &PromptLevers,
    ) -> String {
        PoolsideV1Parser.system_prompt(tools, tool_choice, levers)
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        PoolsideV1Parser.format_tool_calls(calls)
    }

    fn leak_markers(&self) -> LeakMarkers {
        PoolsideV1Parser.leak_markers()
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        PoolsideV1Parser.compile_tool_grammar(engine, tools, use_triggers)
    }

    fn has_tool_grammar(&self) -> bool {
        PoolsideV1Parser.has_tool_grammar()
    }

    fn param_value_close_delim(&self) -> Option<&'static str> {
        PoolsideV1Parser.param_value_close_delim()
    }

    fn promotes_bare_call_names(&self) -> bool {
        PoolsideV1Parser.promotes_bare_call_names()
    }

    fn call_policy(&self) -> CallPolicy {
        CallPolicy::FailClosed {
            keepalive: KEEPALIVE,
        }
    }
}
