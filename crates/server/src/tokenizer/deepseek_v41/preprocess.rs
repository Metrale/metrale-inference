// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Message preprocessing for the DeepSeek-V4.1 encoder, in the publisher's
//! order: request-level tools go on the leading system message, user-supplied image
//! placeholders are rejected, tool messages become `tool_result` blocks in the user turn,
//! image blocks become placeholder text, and (thinking mode without tools) completed
//! turns lose their reasoning.
//!
//! Owner: server (tokenizer).
//! Invariants: an image block without a source is an error, as in the publisher.

use anyhow::{Context, Result, bail};
use serde_json::{Map, Value, json};

use super::{IMAGE_PLACEHOLDER, last_user_index, role, truthy};

/// 2026-10-18: The publisher's fixtures put case-level tools on `messages[0]`, which only
/// renders them when that message is a system message; a conversation that opens with
/// another role gets an empty leading system message to carry them.
pub(super) fn attach_top_level_tools(
    messages: &mut Vec<Value>,
    tools: Option<&[Value]>,
) -> Result<()> {
    let Some(tools) = tools.filter(|t| !t.is_empty()) else {
        return Ok(());
    };
    if messages.is_empty() || messages.iter().any(|m| m.get("tools").is_some_and(truthy)) {
        return Ok(());
    }
    let tools = Value::Array(tools.to_vec());
    if role(&messages[0]) == Some("system") {
        messages[0]
            .as_object_mut()
            .context("DeepSeek-V4.1 message must be an object")?
            .insert("tools".into(), tools);
    } else {
        messages.insert(0, json!({"role": "system", "content": "", "tools": tools}));
    }
    Ok(())
}

pub(super) fn reject_image_placeholder(message: &Value) -> Result<()> {
    for field in ["content", "reasoning_content"] {
        if message
            .get(field)
            .and_then(Value::as_str)
            .is_some_and(|text| text.contains(IMAGE_PLACEHOLDER))
        {
            bail!(
                "DeepSeek-V4.1 message {field} contains the image placeholder token; \
                 images must be image content blocks"
            );
        }
    }
    Ok(())
}

pub(super) fn merge_tool_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut merged: Vec<Value> = Vec::with_capacity(messages.len());
    for message in messages {
        match role(message) {
            Some("tool") => {
                let content = match message.get("content") {
                    Some(Value::String(_)) => message.get("content"),
                    _ if message.get("content_blocks").is_some_and(truthy) => {
                        message.get("content_blocks")
                    }
                    other => other,
                };
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id").cloned().unwrap_or_else(|| json!("")),
                    "content": content.cloned().unwrap_or_else(|| json!("")),
                });
                if let Some(blocks) = merged.last_mut().and_then(user_blocks_mut) {
                    blocks.push(block);
                } else {
                    merged.push(json!({"role": "user", "content_blocks": [block]}));
                }
            }
            Some("user") => {
                let blocks = match message.get("content_blocks") {
                    Some(Value::Null) | None => match message.get("content") {
                        Some(Value::Array(parts)) => parts.clone(),
                        content => {
                            vec![
                                json!({"type": "text", "text": content.cloned().unwrap_or_else(|| json!(""))}),
                            ]
                        }
                    },
                    Some(Value::Array(blocks)) => blocks.clone(),
                    Some(_) => bail!("DeepSeek-V4.1 content_blocks must be an array"),
                };
                let merges = merged
                    .last()
                    .is_some_and(|m| m.get("task").is_none_or(Value::is_null));
                match merged.last_mut().and_then(user_blocks_mut) {
                    Some(previous) if merges => previous.extend(blocks),
                    _ => {
                        let mut message = message.clone();
                        message
                            .as_object_mut()
                            .context("DeepSeek-V4.1 user message must be an object")?
                            .insert("content_blocks".into(), Value::Array(blocks));
                        merged.push(message);
                    }
                }
            }
            _ => merged.push(message.clone()),
        }
    }
    Ok(merged)
}

fn user_blocks_mut(message: &mut Value) -> Option<&mut Vec<Value>> {
    (role(message) == Some("user"))
        .then(|| message.get_mut("content_blocks")?.as_array_mut())
        .flatten()
}

/// 2026-10-18: Replace every image block (top level, or nested in a tool result) with a
/// placeholder text block. A message whose content was a block list gets the `\n\n`-join
/// of its text blocks as its content.
pub(super) fn replace_image_blocks(messages: Vec<Value>) -> Result<Vec<Value>> {
    messages
        .into_iter()
        .map(|mut message| {
            let object = message
                .as_object_mut()
                .context("DeepSeek-V4.1 message must be an object")?;
            if matches!(object.get("content"), Some(Value::Array(_)))
                && !object.contains_key("content_blocks")
            {
                let content = object.shift_remove("content").expect("checked content");
                object.insert("content_blocks".into(), content);
            }
            let Some(Value::Array(blocks)) = object.get_mut("content_blocks") else {
                return Ok(message);
            };
            if blocks.is_empty() {
                return Ok(message);
            }
            replace_in_blocks(blocks)?;
            if !matches!(object.get("content"), Some(Value::String(_))) {
                let texts = text_blocks(object.get("content_blocks"));
                object.insert("content".into(), Value::String(texts.join("\n\n")));
            }
            Ok(message)
        })
        .collect()
}

fn replace_in_blocks(blocks: &mut [Value]) -> Result<()> {
    for block in blocks {
        let Some(object) = block.as_object_mut() else {
            continue;
        };
        match object.get("type").and_then(Value::as_str) {
            Some("image" | "image_url") => {
                require_image_source(object)?;
                *block = json!({"type": "text", "text": IMAGE_PLACEHOLDER});
            }
            Some("tool_result") => {
                if let Some(Value::Array(nested)) = object.get_mut("content") {
                    replace_in_blocks(nested)?;
                }
            }
            Some("text") => {
                if object
                    .get("text")
                    .and_then(Value::as_str)
                    .is_some_and(|text| text.contains(IMAGE_PLACEHOLDER))
                {
                    bail!(
                        "DeepSeek-V4.1 text block contains the image placeholder token; \
                         images must be separate content blocks"
                    );
                }
            }
            _ => {}
        }
    }
    Ok(())
}

fn require_image_source(block: &Map<String, Value>) -> Result<()> {
    let has_source = if block.get("type").and_then(Value::as_str) == Some("image_url") {
        match block.get("image_url") {
            Some(Value::String(url)) => !url.is_empty(),
            Some(Value::Object(image_url)) => image_url.get("url").is_some_and(truthy),
            _ => false,
        }
    } else {
        ["source", "url", "data"]
            .iter()
            .any(|key| block.get(*key).is_some_and(truthy))
    };
    if !has_source {
        bail!("DeepSeek-V4.1 image block does not contain a valid source");
    }
    Ok(())
}

fn text_blocks(blocks: Option<&Value>) -> Vec<String> {
    blocks
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .map(|block| {
            block
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string()
        })
        .collect()
}

/// 2026-10-18: Keep turn-starting roles and everything from the last turn on; earlier
/// assistant messages lose `reasoning_content` and any other earlier role is dropped.
pub(super) fn drop_thinking_messages(messages: &[Value]) -> Vec<Value> {
    let last_user = last_user_index(messages);
    let mut out = Vec::with_capacity(messages.len());
    for (index, message) in messages.iter().enumerate() {
        let current_turn = last_user.is_none_or(|last| index >= last);
        if current_turn
            || matches!(
                role(message),
                Some("user" | "system" | "tool" | "latest_reminder" | "direct_search_results")
            )
        {
            out.push(message.clone());
        } else if role(message) == Some("assistant") {
            let mut message = message.clone();
            if let Some(object) = message.as_object_mut() {
                object.shift_remove("reasoning_content");
            }
            out.push(message);
        }
    }
    out
}
