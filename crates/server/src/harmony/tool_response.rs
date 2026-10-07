// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: One exact declared function handoff or final text; never execute tools.
use super::{Ending, stream::ByteTokenizer, tool_schema::ToolSchema};

pub struct ToolCall {
    pub name: String,
    pub arguments: serde_json::Value,
}
pub struct Response {
    pub content: Option<String>,
    pub tool_call: Option<ToolCall>,
    pub reasoning_tokens: u32,
}

pub fn response(
    tokenizer: &ByteTokenizer,
    prompt: &[u32],
    output: &[u32],
    tools: &[ToolSchema],
) -> Result<Response, &'static str> {
    let mut stream =
        tokenizer.assistant_stream_with_tools(prompt, tools.iter().map(ToolSchema::recipient))?;
    let mut result = None;
    for id in output {
        if let Some(message) = stream.push(*id)? {
            match (message.channel.as_deref(), message.ending) {
                (Some("analysis"), Ending::Message) if message.recipient.is_none() => {}
                (None | Some("final"), Ending::Turn) if message.recipient.is_none() => {
                    result = Some(Response {
                        content: Some(message.body),
                        tool_call: None,
                        reasoning_tokens: 0,
                    });
                }
                (Some("commentary"), Ending::Tool)
                    if matches!(message.content_type.as_deref(), None | Some("json")) =>
                {
                    let tool = tools
                        .iter()
                        .find(|tool| {
                            Some(tool.recipient().as_str()) == message.recipient.as_deref()
                        })
                        .ok_or("undeclared tool recipient")?;
                    let arguments = super::strict_json::parse(&message.body)?;
                    tool.validate(&arguments)?;
                    result = Some(Response {
                        content: None,
                        tool_call: Some(ToolCall {
                            name: tool.name.clone(),
                            arguments,
                        }),
                        reasoning_tokens: 0,
                    });
                }
                _ => return Err("unsupported Harmony channel or tool ending"),
            }
        }
    }
    stream.finish()?;
    let mut response = result.ok_or("missing final answer or tool handoff")?;
    response.reasoning_tokens = stream.reasoning_tokens();
    Ok(response)
}
