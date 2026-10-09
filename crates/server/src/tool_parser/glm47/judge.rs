// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The fail-closed verdict on one complete GLM-4.7 call body: the
//! call is delivered only when its name resolves to a tool the request offered
//! and every argument key is one that tool's schema declares. Any other call
//! is delivered as a refusal: the same slot, under the best name available,
//! with a single argument (`REFUSAL_KEY`) that says what was wrong. A refusal
//! is a retryable error for an agent loop, where a dropped call or a call
//! turned into text ends the loop as a finished turn.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - `judge` returns `Refuse` for every call it does not return as `Accept`;
//!   no call body is dropped.
//! - A refusal's arguments hold exactly one key, `REFUSAL_KEY`, which no tool
//!   schema declares, so a client that validates arguments refuses to run it.

use serde_json::{Map, Value};

use super::super::ToolDefinition;
use super::args::{ARG_KEY, coerce_to_schema, convert_args, tool_properties};
use super::recover::recover_intended_call;

/// 2026-10-08: The one argument key of a refused call.
pub const REFUSAL_KEY: &str = "_rejected_by_server";

/// 2026-10-08: The name given to a refusal whose emitted name is not
/// name-shaped and resolves to no offered tool.
const UNKNOWN_TOOL: &str = "unknown_tool";

/// 2026-10-08: Longest tool name or argument key accepted, in bytes.
const MAX_NAME_LEN: usize = 64;

/// 2026-10-08: The outcome for one call body.
#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    /// 2026-10-08: Deliver this call; `arguments` is the JSON object, already
    /// converted to the schema's types.
    Accept { name: String, arguments: String },
    /// 2026-10-08: Deliver this refusal; `arguments` is
    /// `{REFUSAL_KEY: reason}`.
    Refuse {
        name: String,
        arguments: String,
        reason: String,
    },
}

impl Verdict {
    pub fn name(&self) -> &str {
        match self {
            Self::Accept { name, .. } | Self::Refuse { name, .. } => name,
        }
    }

    pub fn arguments(&self) -> &str {
        match self {
            Self::Accept { arguments, .. } | Self::Refuse { arguments, .. } => arguments,
        }
    }
}

/// 2026-10-08: Shaped like a tool name or argument key: a letter or `_`, then
/// letters, digits, `_`, `.` or `-`, at most `MAX_NAME_LEN` bytes in all.
pub(super) fn is_name_shaped(s: &str) -> bool {
    let mut chars = s.chars();
    s.len() <= MAX_NAME_LEN
        && chars
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
        && chars.all(is_name_char)
}

pub(super) fn is_name_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-')
}

/// 2026-10-08: Split a call body into the emitted name (trimmed text before
/// the first `<arg_key>`) and the argument text from that `<arg_key>` on.
pub(crate) fn split_body(body: &str) -> (&str, &str) {
    let name_end = body.find(ARG_KEY).unwrap_or(body.len());
    (body[..name_end].trim(), &body[name_end..])
}

/// 2026-10-08: The offered tool `emitted` means: itself when offered, else the
/// longest offered name it starts with, provided the character after that
/// prefix cannot continue a name. So `bash</arg_key>` resolves to `bash`, and
/// `bash1635` resolves to nothing.
pub fn resolve_name(emitted: &str, tools: &[ToolDefinition]) -> Option<String> {
    let offered = || tools.iter().map(|t| t.function.name.as_str());
    if offered().any(|n| n == emitted) {
        return Some(emitted.to_string());
    }
    let best = offered()
        .filter(|n| !n.is_empty() && emitted.starts_with(n))
        .max_by_key(|n| n.len())?;
    let continues_name = emitted[best.len()..]
        .chars()
        .next()
        .is_some_and(is_name_char);
    (!continues_name).then(|| best.to_string())
}

/// 2026-10-08: Why the call `name(args)` cannot be delivered, or `None`.
pub(super) fn reject_reason(
    name: &str,
    args: &Map<String, Value>,
    tools: &[ToolDefinition],
) -> Option<String> {
    if !is_name_shaped(name) {
        return Some(format!("tool name {name:?} is not a valid tool name"));
    }
    if !tools.iter().any(|t| t.function.name == name) {
        return Some(format!(
            "tool name {name:?} is not one of the tools offered in this request"
        ));
    }
    // 2026-10-08: A tool without declared properties cannot say a key is wrong,
    // so only the key's shape is checked for it.
    let properties = tool_properties(tools, name).filter(|p| !p.is_empty());
    for key in args.keys() {
        if !is_name_shaped(key) {
            return Some(format!("argument key {key:?} is not a valid argument name"));
        }
        if let Some(properties) = properties
            && !properties.contains_key(key)
        {
            let mut valid: Vec<&str> = properties.keys().map(String::as_str).collect();
            valid.sort_unstable();
            return Some(format!(
                "argument {key:?} is not a parameter of {name}; its parameters are: {}",
                valid.join(", ")
            ));
        }
    }
    None
}

/// 2026-10-08: The arguments of `name` from raw argument text, converted to the
/// schema's types, as a JSON object string.
pub(super) fn typed_arguments(raw_args: &str, tools: &[ToolDefinition], name: &str) -> String {
    let mut args = convert_args(raw_args);
    coerce_to_schema(&mut args, tools, name);
    Value::Object(args).to_string()
}

/// 2026-10-08: The verdict on one complete call body (the text between
/// `<tool_call>` and `</tool_call>`, or to the end of output for a call the
/// output cut short).
///
/// First, a call that holds the call the model meant (`recover_intended_call`)
/// is refused with that call spelled out. Otherwise the call is accepted when
/// its name resolves (`resolve_name`) and `reject_reason` finds nothing, and
/// refused with that reason when it does.
pub fn judge(body: &str, tools: &[ToolDefinition]) -> Verdict {
    let (emitted, raw_args) = split_body(body);
    if let Some(reason) = recover_intended_call(emitted, raw_args, tools) {
        return refuse(emitted, tools, reason);
    }
    let resolved = resolve_name(emitted, tools);
    let name = resolved.as_deref().unwrap_or(emitted);
    match reject_reason(name, &convert_args(raw_args), tools) {
        None => Verdict::Accept {
            name: name.to_string(),
            arguments: typed_arguments(raw_args, tools, name),
        },
        Some(reason) => refuse(emitted, tools, reason),
    }
}

/// 2026-10-08: A refusal for the call emitted as `emitted`: under the offered
/// tool it resolves to, else under `emitted` when that is name-shaped, else
/// under `UNKNOWN_TOOL`. Naming the real tool lets the client match the call
/// to the tool and reject it on its arguments.
fn refuse(emitted: &str, tools: &[ToolDefinition], reason: String) -> Verdict {
    let name = resolve_name(emitted, tools).unwrap_or_else(|| {
        if is_name_shaped(emitted) {
            emitted.to_string()
        } else {
            UNKNOWN_TOOL.to_string()
        }
    });
    tracing::warn!(tool = %name, "glm47 tool call refused: {reason}");
    let mut args = Map::new();
    args.insert(REFUSAL_KEY.to_string(), Value::String(reason.clone()));
    Verdict::Refuse {
        name,
        arguments: Value::Object(args).to_string(),
        reason,
    }
}
