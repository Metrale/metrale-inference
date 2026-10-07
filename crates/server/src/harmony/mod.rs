// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Token-aware framing for GPT-OSS assistant generation.
//!
//! Owner: server protocol. Blocking and streaming text responses use the strict token-aware adapters.
//! The tokenizer must supply actual special-token events, not match text spellings.
//! Message boundaries are not EOS. Blocking tool handoffs require exact declared schemas.
//! Streaming tools and analysis-channel handoffs remain unsupported.
//! API reasoning usage counts generated analysis-body IDs, excluding protocol overhead;
//! scheduler thought budgets and provider billing conventions are separate contracts.
//! Callers seed the exact unfinished assistant header from the rendered prompt and
//! carry Finish/Handoff before the scheduler discards an EOS token.

pub mod adapter;
pub mod api;
pub mod stream;
pub(crate) mod strict_json;
pub mod text_stream;
pub mod tool_response;
pub mod tool_schema;
#[cfg(test)]
mod tool_tests;

#[cfg(test)]
mod stream_tests;

use std::collections::BTreeSet;

/// 2026-10-06: Token classes supplied by the checkpoint-specific tokenizer adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token<'a> {
    Start,
    Channel,
    Constrain,
    Separator,
    End,
    Finish,
    Handoff,
    Text(&'a str),
}

/// 2026-10-06: A message boundary, completed assistant turn, or transfer to a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ending {
    Message,
    Turn,
    Tool,
}

/// 2026-10-06: A complete frame; analysis must never be flattened into visible content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub channel: Option<String>,
    pub recipient: Option<String>,
    pub content_type: Option<String>,
    pub body: String,
    pub ending: Ending,
}

#[derive(Debug, PartialEq, Eq)]
enum Phase {
    AwaitStart,
    Header,
    Body,
    Done,
    Failed,
}

/// 2026-10-06: Bounded incremental assistant decoder. Errors poison the stream.
#[derive(Debug)]
pub struct Decoder {
    phase: Phase,
    head: String,
    meta: Option<String>,
    constraint: Option<String>,
    channel: Option<String>,
    recipient: Option<String>,
    content_type: Option<String>,
    body: String,
    tools: BTreeSet<String>,
    header_limit: usize,
    body_limit: usize,
}

impl Decoder {
    /// 2026-10-06: Tools are exact allowed recipients (for example `functions.lookup`).
    pub fn new(
        tools: impl IntoIterator<Item = String>,
        header_limit: usize,
        body_limit: usize,
    ) -> Self {
        Self {
            phase: Phase::AwaitStart,
            head: String::new(),
            meta: None,
            constraint: None,
            channel: None,
            recipient: None,
            content_type: None,
            body: String::new(),
            tools: tools.into_iter().collect(),
            header_limit,
            body_limit,
        }
    }

    /// 2026-10-06: Seed an unfinished header already present in the prompt, without delimiters.
    /// For a channel-bearing prefix, supply subsequent Channel/Text events explicitly.
    pub fn seed_assistant_header(&mut self, prefix: &str) -> Result<(), &'static str> {
        self.push(Token::Start)?;
        self.push(Token::Text(prefix))?;
        Ok(())
    }

    /// 2026-10-06: Emit a frame once, only when its terminal event arrives.
    pub fn push(&mut self, token: Token<'_>) -> Result<Option<Message>, &'static str> {
        let result = self.advance(token);
        if result.is_err() {
            self.phase = Phase::Failed;
        }
        result
    }

    /// 2026-10-06: End-of-stream succeeds only after turn completion or valid handoff.
    pub fn finish(&self) -> Result<(), &'static str> {
        if self.phase == Phase::Done {
            Ok(())
        } else {
            Err("incomplete or invalid Harmony generation")
        }
    }

    fn advance(&mut self, token: Token<'_>) -> Result<Option<Message>, &'static str> {
        match (&self.phase, token) {
            (Phase::AwaitStart, Token::Start) => {
                self.head.clear();
                self.meta = None;
                self.constraint = None;
                self.channel = None;
                self.recipient = None;
                self.content_type = None;
                self.body.clear();
                self.phase = Phase::Header;
            }
            (Phase::Header, Token::Text(s)) => {
                let size = self
                    .head
                    .len()
                    .checked_add(self.meta.as_ref().map_or(0, String::len))
                    .and_then(|n| n.checked_add(self.constraint.as_ref().map_or(0, String::len)))
                    .and_then(|n| n.checked_add(s.len()))
                    .ok_or("header size overflow")?;
                if size > self.header_limit {
                    return Err("Harmony header exceeds configured limit");
                }
                if let Some(constraint) = &mut self.constraint {
                    constraint.push_str(s);
                } else {
                    match self.meta.as_mut() {
                        Some(meta) => meta.push_str(s),
                        None => self.head.push_str(s),
                    }
                }
            }
            (Phase::Header, Token::Channel) if self.meta.is_none() && self.constraint.is_none() => {
                self.meta = Some(String::new())
            }
            // 2026-10-07: Checkpoint format metadata is distinct from channel text.
            (Phase::Header, Token::Constrain) if self.constraint.is_none() => {
                self.constraint = Some(String::new());
            }
            (Phase::Header, Token::Separator) => {
                self.parse_header()?;
                self.phase = Phase::Body;
            }
            (Phase::Body, Token::Text(s)) => {
                if self
                    .body
                    .len()
                    .checked_add(s.len())
                    .ok_or("body size overflow")?
                    > self.body_limit
                {
                    return Err("Harmony body exceeds configured limit");
                }
                self.body.push_str(s);
            }
            (Phase::Body, Token::End | Token::Finish | Token::Handoff) => {
                let ending = match token {
                    Token::End => Ending::Message,
                    Token::Finish => Ending::Turn,
                    _ => Ending::Tool,
                };
                if ending == Ending::Tool && self.recipient.is_none() {
                    return Err("tool handoff requires recipient");
                }
                if self.recipient.is_some() && ending != Ending::Tool {
                    return Err("tool recipient requires handoff");
                }
                self.phase = if ending == Ending::Message {
                    Phase::AwaitStart
                } else {
                    Phase::Done
                };
                return Ok(Some(Message {
                    channel: self.channel.take(),
                    recipient: self.recipient.take(),
                    content_type: self.content_type.take(),
                    body: std::mem::take(&mut self.body),
                    ending,
                }));
            }
            _ => return Err("unexpected Harmony token or generation already terminated"),
        }
        Ok(None)
    }

    fn parse_header(&mut self) -> Result<(), &'static str> {
        let mut head = self.head.split_ascii_whitespace();
        if head.next() != Some("assistant") {
            return Err("expected assistant generation");
        }
        let mut fields: Vec<&str> = head.collect();
        if let Some(meta) = &self.meta {
            let mut words = meta.split_ascii_whitespace();
            let channel = words.next().ok_or("empty Harmony channel")?;
            if !matches!(channel, "analysis" | "final" | "commentary") {
                return Err("unsupported Harmony channel");
            }
            self.channel = Some(channel.into());
            for word in words {
                if word == "json" {
                    if self.content_type.is_some() {
                        return Err("duplicate Harmony content type");
                    }
                    self.content_type = Some(word.into());
                } else {
                    fields.push(word);
                }
            }
        }
        for field in fields {
            let recipient = field
                .strip_prefix("to=")
                .ok_or("unsupported Harmony header field")?;
            if self.recipient.is_some() {
                return Err("duplicate Harmony recipient");
            }
            if !self.tools.contains(recipient) {
                return Err("undeclared Harmony tool recipient");
            }
            self.recipient = Some(recipient.into());
        }
        if let Some(constraint) = &self.constraint {
            if constraint.trim() != "json" || self.content_type.is_some() {
                return Err("unsupported or duplicate Harmony constrained format");
            }
            self.content_type = Some("json".into());
        }
        if self.content_type.is_some() && self.recipient.is_none() {
            return Err("Harmony JSON content type requires tool recipient");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod adapter_tests;
