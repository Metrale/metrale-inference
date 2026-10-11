// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Incremental final-channel text; analysis and framing never leave this adapter.
use super::{
    Ending,
    stream::{ByteTokenizer, Stream},
};
use std::sync::Arc;

pub struct TextStream {
    stream: Stream<'static>,
    emitted: usize,
    completed: bool,
    failed: bool,
    tools: Vec<super::tool_schema::ToolSchema>,
    tool_call: Option<super::tool_response::ToolCall>,
}
impl TextStream {
    pub fn new(tokenizer: Arc<ByteTokenizer>, prompt: &[u32]) -> Result<Self, &'static str> {
        Self::with_tools(tokenizer, prompt, vec![])
    }
    // 2026-10-07: Tool arguments are withheld until the completed handoff validates.
    pub fn with_tools(
        tokenizer: Arc<ByteTokenizer>,
        prompt: &[u32],
        tools: Vec<super::tool_schema::ToolSchema>,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            stream: Stream::shared_with_tools(
                tokenizer,
                prompt,
                tools.iter().map(super::tool_schema::ToolSchema::recipient),
            )?,
            tools,
            tool_call: None,
            emitted: 0,
            completed: false,
            failed: false,
        })
    }
    pub fn push(&mut self, id: u32) -> Result<String, &'static str> {
        if self.failed || self.completed {
            self.failed = true;
            return Err("Harmony stream already finished");
        }
        let result = self.advance(id);
        if result.is_err() {
            self.failed = true;
        }
        result
    }
    fn advance(&mut self, id: u32) -> Result<String, &'static str> {
        if let Some(message) = self.stream.push(id)? {
            return match (message.channel.as_deref(), message.ending) {
                (Some("analysis"), Ending::Message) if self.emitted == 0 => Ok(String::new()),
                (None | Some("final"), Ending::Turn) if message.recipient.is_none() => {
                    let tail = message
                        .body
                        .get(self.emitted..)
                        .ok_or("invalid visible byte offset")?
                        .to_owned();
                    self.completed = true;
                    Ok(tail)
                }
                (Some("commentary"), Ending::Tool)
                    if self.emitted == 0
                        && matches!(message.content_type.as_deref(), None | Some("json")) =>
                {
                    self.tool_call =
                        Some(super::tool_response::validated_call(&message, &self.tools)?);
                    self.completed = true;
                    Ok(String::new())
                }
                _ => Err("unsupported Harmony response channel or ending"),
            };
        }
        if let Some(body) = self.stream.visible_body() {
            let delta = body
                .get(self.emitted..)
                .ok_or("invalid visible byte offset")?
                .to_owned();
            self.emitted = body.len();
            Ok(delta)
        } else {
            Ok(String::new())
        }
    }
    pub fn take_tool_call(&mut self) -> Option<super::tool_response::ToolCall> {
        self.tool_call.take()
    }
    pub fn reasoning_tokens(&self) -> u32 {
        self.stream.reasoning_tokens()
    }

    pub fn finish(&self) -> Result<(), &'static str> {
        if self.failed || !self.completed {
            return Err("incomplete Harmony final turn");
        }
        self.stream.finish()
    }
}
