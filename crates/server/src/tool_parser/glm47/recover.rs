// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Recovery of the call the model meant from a call that holds
//! another one. Two shapes:
//!
//! - Nested: the model opened `<tool_call>` while still reasoning, kept
//!   reasoning, then wrote the real call after `</think>`. The first opener
//!   ended the reasoning (`reasoning_parser`'s GLM policy), so the first call's
//!   last argument swallows the rest of the reasoning, the `</think>` and the
//!   whole real call.
//! - Abandoned: the model left a call unfinished and started another inside an
//!   argument value (`...<arg_value># comment\nweb_search<arg_key>limit...`),
//!   so both read as one call.
//!
//! The recovered call is never delivered: its text sits inside another call's
//! arguments, where it can as well be file content (an agent writing parser
//! tests writes exactly these strings). The outer call is refused instead,
//! with a reason that spells out the recovered call, so the model can issue it
//! on its own in one step.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - A recovery is reported only when the recovered call would itself be
//!   accepted (offered name, schema keys).

use super::super::ToolDefinition;
use super::args::{ARG_KEY, ARG_VALUE, ARG_VALUE_END, convert_args};
use super::judge::{is_name_char, reject_reason, typed_arguments};

const THINK_END: &str = "</think>";
const TOOL_CALL: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";

/// 2026-10-08: Which shape `recover_intended_call` found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shape {
    Nested,
    Abandoned,
}

/// 2026-10-08: The refusal reason for the call `outer`, when its argument text
/// `raw_args` holds a valid call (nested or abandoned); `None` otherwise.
pub(super) fn recover_intended_call(
    outer: &str,
    raw_args: &str,
    tools: &[ToolDefinition],
) -> Option<String> {
    let (shape, name, inner_args) =
        find_nested(raw_args).or_else(|| find_abandoned(raw_args, tools))?;
    let inner_args = inner_args.strip_suffix(TOOL_CALL_END).unwrap_or(inner_args);
    if reject_reason(name, &convert_args(inner_args), tools).is_some() {
        return None;
    }
    let arguments = typed_arguments(inner_args, tools, name);
    let subject = if outer.is_empty() {
        "this call".to_string()
    } else {
        format!("this {outer} call")
    };
    let how = match shape {
        Shape::Nested => {
            "was opened inside your reasoning, so its arguments took in the rest of the \
             reasoning and the call you wrote after it"
        }
        Shape::Abandoned => {
            "was left unfinished: another call was started inside one of its argument values"
        }
    };
    tracing::warn!(
        outer,
        recovered = name,
        ?shape,
        "glm47 tool call holds another call; refusing it with the recovered call as a hint"
    );
    Some(format!(
        "Refused, nothing was run: {subject} {how}. The call you meant appears to be \
         {name} with arguments {arguments}; issue that call by itself. (To write tool-call \
         tags literally inside an argument, split them, e.g. '</th' + 'ink>'.)"
    ))
}

/// 2026-10-08: The last `</think>`, optional whitespace, `<tool_call>`,
/// optional whitespace, a name, optional whitespace and `<arg_key>` in
/// `raw`: the nested call, with its argument text from that `<arg_key>` on.
fn find_nested(raw: &str) -> Option<(Shape, &str, &str)> {
    let mut found = None;
    let mut from = 0;
    while let Some(rel) = raw[from..].find(THINK_END) {
        let at = from + rel;
        from = at + THINK_END.len();
        if let Some(hit) = nested_call_at(raw, from) {
            found = Some(hit);
        }
    }
    found.map(|(name, args)| (Shape::Nested, name, args))
}

/// 2026-10-08: The nested call whose `<tool_call>` follows `pos` after
/// optional whitespace: its name and its argument text.
fn nested_call_at(raw: &str, pos: usize) -> Option<(&str, &str)> {
    let rest = raw[pos..]
        .trim_start()
        .strip_prefix(TOOL_CALL)?
        .trim_start();
    let name_len = rest.find(|c: char| !is_name_char(c)).unwrap_or(rest.len());
    let name = &rest[..name_len];
    if !super::judge::is_name_shaped(name) {
        return None;
    }
    let args = rest[name_len..].trim_start();
    args.starts_with(ARG_KEY).then_some((name, args))
}

/// 2026-10-08: The earliest offered tool name written directly before an
/// `<arg_key>` inside an argument value of `raw` (not at its start, and not as
/// the tail of a longer name): the abandoned call, with its argument text from
/// that `<arg_key>` on. Each tool contributes its first such position.
fn find_abandoned<'r>(raw: &'r str, tools: &[ToolDefinition]) -> Option<(Shape, &'r str, &'r str)> {
    let mut best: Option<(usize, &'r str)> = None;
    for tool in tools {
        let name = tool.function.name.as_str();
        if name.is_empty() {
            continue;
        }
        let needle = format!("{name}{ARG_KEY}");
        let mut from = 0;
        while let Some(rel) = raw[from..].find(&needle) {
            let at = from + rel;
            from = at + 1;
            let starts_mid_name = raw[..at].chars().next_back().is_none_or(is_name_char);
            if starts_mid_name || !inside_value(raw, at) {
                continue;
            }
            if best.is_none_or(|(b, _)| at < b) {
                best = Some((at, &raw[at..at + name.len()]));
            }
            break;
        }
    }
    best.map(|(at, name)| (Shape::Abandoned, name, &raw[at + name.len()..]))
}

/// 2026-10-08: Whether `pos` is inside an argument value: the last
/// `<arg_value>` before it is later than the last `</arg_value>` before it.
fn inside_value(raw: &str, pos: usize) -> bool {
    let before = &raw[..pos];
    match (before.rfind(ARG_VALUE), before.rfind(ARG_VALUE_END)) {
        (Some(open), Some(close)) => open > close,
        (Some(_), None) => true,
        (None, _) => false,
    }
}
