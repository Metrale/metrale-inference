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
}
impl TextStream {
    pub fn new(tokenizer: Arc<ByteTokenizer>, prompt: &[u32]) -> Result<Self, &'static str> {
        Ok(Self {
            stream: Stream::shared(tokenizer, prompt)?,
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
    pub fn finish(&self) -> Result<(), &'static str> {
        if self.failed || !self.completed {
            return Err("incomplete Harmony final turn");
        }
        self.stream.finish()
    }
}
