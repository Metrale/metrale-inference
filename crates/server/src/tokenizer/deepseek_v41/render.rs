// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Rendering of preprocessed DeepSeek-V4.1 messages: role tokens, the numeric
//! effort prefix, tool schemas and DSML tool calls with the V4.1 spaced tag names, and the
//! assistant header that follows a turn.
//!
//! Owner: server (tokenizer).
//! Invariants: tool names are validated (no `::` inside a bare name or a namespace, no
//! conflicting namespaces) wherever they appear, as in the publisher.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value};

use super::super::deepseek_v4::{
    ASSISTANT, BOS, DSML, EOS, LATEST_REMINDER, THINK_END, THINK_START, USER, python_json,
};
use super::{ReasoningBudget, SYSTEM, last_user_index, role, truthy};

const CALLS: &str = " calls";
const INVOKE: &str = " invoke";
const PARAMETER: &str = " parameter";

const TOOL_INSTRUCTIONS: &str = r##"## Tools

You have access to a set of tools to help answer the user's question. You can invoke tools by writing a "<{dsml}{calls}>" block like the following:

<{dsml}{calls}>
<{dsml}{invoke} name="$TOOL_NAME">
<{dsml}{parameter} name="$PARAMETER_NAME" string="true|false">$PARAMETER_VALUE</{dsml}{parameter}>
...
</{dsml}{invoke}>
<{dsml}{invoke} name="$TOOL_NAME2">
...
</{dsml}{invoke}>
</{dsml}{calls}>

String parameters should be specified as is and set `string="true"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string="false"`.

If thinking_mode is enabled (triggered by {think_start}), you MUST output your complete reasoning inside {think_start}...{think_end} BEFORE any tool calls or final response.

Otherwise, output directly after {think_end} with tool calls or final response.

### Available Tool Schemas

{schemas}

You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.
"##;

const TASK_TOKENS: [(&str, &str); 6] = [
    ("action", "<｜action｜>"),
    ("query", "<｜query｜>"),
    ("authority", "<｜authority｜>"),
    ("domain", "<｜domain｜>"),
    ("title", "<｜title｜>"),
    ("read_url", "<｜read_url｜>"),
];

struct Turn<'a> {
    messages: &'a [Value],
    thinking: bool,
    drop_thinking: bool,
    budget: ReasoningBudget,
    last_user: Option<usize>,
}

impl Turn<'_> {
    /// 2026-10-18: The publisher compares indices with a last-user index of -1 when there
    /// is no user turn; `None` keeps that meaning (every index is after it).
    fn after_last_user(&self, index: usize) -> bool {
        self.last_user.is_none_or(|last| index > last)
    }

    fn at_or_after_last_user(&self, index: usize) -> bool {
        self.last_user.is_none_or(|last| index >= last)
    }
}

pub(super) fn render_messages(
    messages: &[Value],
    thinking: bool,
    drop_thinking: bool,
    budget: ReasoningBudget,
) -> Result<String> {
    let turn = Turn {
        messages,
        thinking,
        drop_thinking,
        budget,
        last_user: last_user_index(messages),
    };
    let mut prompt = String::from(BOS);
    for index in 0..messages.len() {
        render_message(&turn, index, &mut prompt)?;
    }
    Ok(prompt)
}

fn render_message(turn: &Turn<'_>, index: usize, out: &mut String) -> Result<()> {
    let message = &turn.messages[index];
    let message_role = role(message);
    let tools = match message.get("tools") {
        Some(tools) if truthy(tools) => Some(tools_from_openai_format(tools)?),
        _ => None,
    };
    let tool_calls = match message.get("tool_calls") {
        Some(calls) if truthy(calls) => Some(tool_calls_from_openai_format(calls)?),
        _ => None,
    };

    let effort = (index == 0 && turn.thinking).then(|| turn.budget.prompt());
    if index == 0 && (effort.is_some() || message_role == Some("system")) {
        out.push_str(SYSTEM);
    }
    out.push_str(effort.as_deref().unwrap_or_default());

    match message_role {
        Some("system") => {
            if index > 0 {
                out.push_str(SYSTEM);
            }
            out.push_str(&or_empty(message.get("content"))?);
            if let Some(tools) = tools {
                out.push_str("\n\n");
                out.push_str(&render_tools(&tools)?);
            }
            if let Some(format) = message.get("response_format").filter(|f| truthy(f)) {
                out.push_str("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                out.push_str(&python_json(format)?);
            }
        }
        Some("user") => {
            out.push_str(USER);
            match message.get("content_blocks").and_then(Value::as_array) {
                Some(blocks) if !blocks.is_empty() => {
                    let parts = blocks
                        .iter()
                        .map(render_block)
                        .collect::<Result<Vec<_>>>()?;
                    out.push_str(&parts.join("\n\n"));
                }
                _ => out.push_str(&or_empty(message.get("content"))?),
            }
        }
        Some("latest_reminder") => {
            out.push_str(LATEST_REMINDER);
            out.push_str(&python_str(message.get("content"))?);
        }
        Some("tool") => bail!("DeepSeek-V4.1 tool messages must be merged before rendering"),
        Some("assistant") => render_assistant(turn, index, tool_calls.as_deref(), out)?,
        other => bail!("unsupported DeepSeek-V4.1 message role {other:?}"),
    }

    let next = turn.messages.get(index + 1);
    if next.is_some() && !matches!(next.and_then(role), Some("assistant" | "latest_reminder")) {
        return Ok(());
    }

    match message.get("task").filter(|t| !t.is_null()) {
        Some(task) => {
            let (name, token) = task
                .as_str()
                .and_then(|task| TASK_TOKENS.iter().find(|(name, _)| *name == task))
                .with_context(|| format!("invalid DeepSeek-V4.1 task {task}"))?;
            if *name == "action" {
                out.push_str(ASSISTANT);
                out.push_str(if turn.thinking {
                    THINK_START
                } else {
                    THINK_END
                });
            }
            out.push_str(token);
        }
        None if message_role == Some("user") || (message_role == Some("system") && index > 0) => {
            out.push_str(ASSISTANT);
            let open = turn.thinking && (!turn.drop_thinking || turn.at_or_after_last_user(index));
            out.push_str(if open { THINK_START } else { THINK_END });
        }
        None => {}
    }
    Ok(())
}

fn render_assistant(
    turn: &Turn<'_>,
    index: usize,
    tool_calls: Option<&[ToolCall]>,
    out: &mut String,
) -> Result<()> {
    let message = &turn.messages[index];
    let previous_has_task = index > 0
        && turn.messages[index - 1]
            .get("task")
            .is_some_and(|t| !t.is_null());
    if turn.thinking && !previous_has_task && (!turn.drop_thinking || turn.after_last_user(index)) {
        out.push_str(&or_empty(message.get("reasoning_content"))?);
        out.push_str(THINK_END);
    }
    out.push_str(&or_empty(message.get("content"))?);
    if let Some(calls) = tool_calls {
        let invokes = calls
            .iter()
            .map(|call| {
                Ok(format!(
                    "<{DSML}{INVOKE} name=\"{}\">\n{}\n</{DSML}{INVOKE}>",
                    call.qualified_name(),
                    dsml_arguments(&call.arguments)?
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        out.push_str(&format!(
            "\n\n<{DSML}{CALLS}>\n{}\n</{DSML}{CALLS}>",
            invokes.join("\n")
        ));
    }
    if !message.get("wo_eos").is_some_and(truthy) {
        out.push_str(EOS);
    }
    Ok(())
}

fn render_block(block: &Value) -> Result<String> {
    match block.get("type") {
        Some(Value::String(kind)) if kind == "text" => block_text(block),
        Some(Value::String(kind)) if kind == "tool_result" => {
            let content = match block.get("content") {
                Some(Value::Array(parts)) => parts
                    .iter()
                    .map(|part| match part.get("type") {
                        Some(Value::String(kind)) if kind == "text" => block_text(part),
                        kind => Ok(format!("[Unsupported {}]", python_str(kind)?)),
                    })
                    .collect::<Result<Vec<_>>>()?
                    .join("\n\n"),
                None => String::new(),
                content => python_str(content)?,
            };
            Ok(format!("<tool_result>{content}</tool_result>"))
        }
        kind => Ok(format!("[Unsupported {}]", python_str(kind)?)),
    }
}

fn block_text(block: &Value) -> Result<String> {
    match block.get("text") {
        None => Ok(String::new()),
        Some(Value::String(text)) => Ok(text.clone()),
        Some(_) => bail!("DeepSeek-V4.1 text block must hold a string"),
    }
}

/// 2026-10-18: Python's `value or ""` for a scalar message field.
fn or_empty(value: Option<&Value>) -> Result<String> {
    match value {
        Some(value) if truthy(value) => python_str(Some(value)),
        _ => Ok(String::new()),
    }
}

/// 2026-10-18: Python's `str(value)` for the scalars the publisher formats into the
/// prompt unchanged. A list or object has no faithful rendering and is rejected.
fn python_str(value: Option<&Value>) -> Result<String> {
    Ok(match value {
        None | Some(Value::Null) => "None".to_string(),
        Some(Value::String(s)) => s.clone(),
        Some(Value::Bool(true)) => "True".to_string(),
        Some(Value::Bool(false)) => "False".to_string(),
        Some(number @ Value::Number(_)) => python_json(number)?,
        Some(_) => bail!("DeepSeek-V4.1 cannot render a list or object as message text"),
    })
}

/// 2026-10-18: Split `ns::name`, checking it against an explicit namespace.
fn split_tool_name(name: &str, namespace: Option<&str>) -> Result<(Option<String>, String)> {
    let (namespace, name) = match name.split_once("::") {
        Some((prefix, bare)) => {
            if namespace.is_some_and(|ns| ns != prefix) {
                bail!(
                    "Conflicting tool namespaces: {} != {prefix}",
                    namespace.unwrap_or_default()
                );
            }
            (Some(prefix), bare)
        }
        None => (namespace, name),
    };
    if name.contains("::") {
        bail!("Tool name must not contain '::': {name}");
    }
    if let Some(ns) = namespace
        && ns.contains("::")
    {
        bail!("Tool namespace must not contain '::': {ns}");
    }
    Ok((namespace.map(str::to_string), name.to_string()))
}

fn qualify(namespace: Option<&str>, name: &str) -> String {
    match namespace {
        Some(ns) => format!("{ns}::{name}"),
        None => name.to_string(),
    }
}

/// 2026-10-18: The tool schemas the prompt shows: each tool's `function`, with a namespace
/// (on the tool or the function; a string, or an object with `name` and an optional
/// `description`) folded into a qualified name and a description prefix.
fn tools_from_openai_format(tools: &Value) -> Result<Vec<Map<String, Value>>> {
    let tools = tools
        .as_array()
        .context("DeepSeek-V4.1 tools must be an array")?;
    tools
        .iter()
        .map(|tool| {
            let mut function = tool
                .get("function")
                .and_then(Value::as_object)
                .cloned()
                .context("DeepSeek-V4.1 tools must use OpenAI function format")?;
            if let Some(namespace) = tool.get("namespace").filter(|ns| !ns.is_null()) {
                function.insert("namespace".into(), namespace.clone());
            }
            let namespace = function
                .get("namespace")
                .filter(|ns| !ns.is_null())
                .cloned();
            let namespace_name = match &namespace {
                None => None,
                Some(Value::String(ns)) => Some(ns.as_str()),
                Some(Value::Object(ns)) => Some(
                    ns.get("name")
                        .and_then(Value::as_str)
                        .context("DeepSeek-V4.1 tool namespace object needs a string name")?,
                ),
                Some(_) => bail!("DeepSeek-V4.1 tool namespace must be a string or an object"),
            };
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("DeepSeek-V4.1 tool function is missing name")?;
            let (namespace_name, name) = split_tool_name(name, namespace_name)?;
            function.insert(
                "name".into(),
                Value::String(qualify(namespace_name.as_deref(), &name)),
            );
            function.shift_remove("namespace");
            if let Some(Value::Object(ns)) = &namespace
                && let Some(prefix) = ns.get("description").filter(|d| truthy(d))
            {
                let prefix = prefix
                    .as_str()
                    .context("DeepSeek-V4.1 tool namespace description must be a string")?;
                let description = or_empty(function.get("description"))?;
                function.insert(
                    "description".into(),
                    Value::String(format!("{prefix}\n{description}")),
                );
            }
            Ok(function)
        })
        .collect()
}

fn render_tools(tools: &[Map<String, Value>]) -> Result<String> {
    let schemas = tools
        .iter()
        .map(|tool| python_json(&Value::Object(tool.clone())))
        .collect::<Result<Vec<_>>>()?
        .join("\n");
    Ok(TOOL_INSTRUCTIONS
        .replace("{dsml}", DSML)
        .replace("{calls}", CALLS)
        .replace("{invoke}", INVOKE)
        .replace("{parameter}", PARAMETER)
        .replace("{think_start}", THINK_START)
        .replace("{think_end}", THINK_END)
        .replace("{schemas}", &schemas))
}

struct ToolCall {
    namespace: Option<String>,
    name: String,
    arguments: Value,
}

impl ToolCall {
    fn qualified_name(&self) -> String {
        qualify(self.namespace.as_deref(), &self.name)
    }
}

fn tool_calls_from_openai_format(calls: &Value) -> Result<Vec<ToolCall>> {
    let calls = calls
        .as_array()
        .context("DeepSeek-V4.1 tool_calls must be an array")?;
    calls
        .iter()
        .map(|call| {
            let function = call
                .get("function")
                .context("DeepSeek-V4.1 tool call is missing function")?;
            let namespace = [call.get("namespace"), function.get("namespace")]
                .into_iter()
                .flatten()
                .find(|ns| truthy(ns))
                .map(|ns| {
                    ns.as_str()
                        .context("DeepSeek-V4.1 tool call namespace must be a string")
                })
                .transpose()?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("DeepSeek-V4.1 tool call is missing function.name")?;
            let (namespace, name) = split_tool_name(name, namespace)?;
            let arguments = function
                .get("arguments")
                .cloned()
                .context("DeepSeek-V4.1 tool call is missing function.arguments")?;
            Ok(ToolCall {
                namespace,
                name,
                arguments,
            })
        })
        .collect()
}

/// 2026-10-18: DSML parameters for a call's arguments. A JSON string is decoded up to twice
/// (to tolerate double encoding); arguments that still are not an object are passed as a
/// single `arguments` parameter.
fn dsml_arguments(arguments: &Value) -> Result<String> {
    let mut decoded = arguments.clone();
    for _ in 0..2 {
        let Value::String(raw) = &decoded else {
            break;
        };
        match serde_json::from_str::<Value>(raw) {
            Ok(value) => decoded = value,
            Err(_) => break,
        }
    }
    let object = match decoded {
        Value::Object(object) => object,
        _ => Map::from_iter([("arguments".to_string(), arguments.clone())]),
    };
    let params = object
        .iter()
        .map(|(key, value)| {
            let (is_string, rendered) = match value {
                Value::String(s) => ("true", s.clone()),
                other => ("false", python_json(other)?),
            };
            Ok(format!(
                "<{DSML}{PARAMETER} name=\"{key}\" string=\"{is_string}\">{rendered}</{DSML}{PARAMETER}>"
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(params.join("\n"))
}
