// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Strict byte-preserving streaming for the checkpoint's ByteLevel BPE.
//!
//! tokenizers DecodeStream is lossy and has no flush API: incomplete trailing bytes
//! and a genuine U+FFFD cannot be distinguished through its string interface.
//! Decode bytes strictly here, then pass only complete Unicode to Harmony framing.
use std::collections::HashMap;

use super::adapter::{TokenClass, TokenMap};
use super::{Decoder, Ending, Message, Token};

/// 2026-10-06: Immutable tokenizer data, shared across request-local streams.
pub struct ByteTokenizer {
    map: TokenMap,
    bytes: HashMap<u32, Vec<u8>>,
}

impl ByteTokenizer {
    /// 2026-10-07: Only turn completion and handoff stop generation; End is a message boundary.
    pub fn stop_ids(&self) -> Vec<u32> {
        self.map.terminal_ids()
    }

    /// 2026-10-07: Seed from actual prompt tokens, never guessed delimiter spellings.
    pub fn assistant_stream(&self, prompt: &[u32]) -> Result<Stream<'_>, &'static str> {
        self.assistant_stream_with_tools(prompt, [])
    }

    pub fn assistant_stream_with_tools(
        &self,
        prompt: &[u32],
        tools: impl IntoIterator<Item = String>,
    ) -> Result<Stream<'_>, &'static str> {
        let start = prompt
            .iter()
            .rposition(|id| self.map.classify(*id) == Ok(TokenClass::Framing(Token::Start)))
            .ok_or("missing assistant prompt header")?;
        let mut stream = Stream::new(self, Decoder::new(tools, 1024, 1_048_576));
        for id in &prompt[start..] {
            if stream.push(*id)?.is_some() {
                return Err("completed assistant prompt header");
            }
        }
        if stream.decoder.phase != super::Phase::Header || !stream.pending.is_empty() {
            return Err("prompt must end with unfinished assistant header");
        }
        Ok(stream)
    }

    pub fn from_tokenizer_json(json: &str) -> Result<Self, &'static str> {
        let map = TokenMap::from_tokenizer_json(json)?;
        let data: serde_json::Value =
            serde_json::from_str(json).map_err(|_| "invalid tokenizer JSON")?;
        if data["model"]["type"] != "BPE" || data["decoder"]["type"] != "ByteLevel" {
            return Err("Harmony byte stream requires BPE with ByteLevel decoder");
        }
        // 2026-10-07: ByteLevel's bijection maps printable Latin-1 directly, all remaining
        // bytes to successive codepoints starting at U+0100 (not token IDs).
        let mut alphabet = HashMap::new();
        let mut extra = 256;
        for byte in 0..=255u8 {
            let code = if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                u32::from(byte)
            } else {
                let code = extra;
                extra += 1;
                code
            };
            alphabet.insert(
                char::from_u32(code).ok_or("invalid ByteLevel alphabet")?,
                byte,
            );
        }
        let mut bytes = HashMap::new();
        for (name, id) in data["model"]["vocab"]
            .as_object()
            .ok_or("missing vocabulary")?
        {
            let id = id
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("invalid token ID")?;
            if map.classify(id) == Ok(TokenClass::Ordinary) {
                let value = name
                    .chars()
                    .map(|c| {
                        alphabet
                            .get(&c)
                            .copied()
                            .ok_or("invalid ByteLevel vocabulary character")
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                if value.is_empty() {
                    return Err("empty ordinary token");
                }
                bytes.insert(id, value);
            }
        }
        // 2026-10-07: Added ordinary tokens also pass through the ByteLevel decoder.
        for entry in data["added_tokens"]
            .as_array()
            .ok_or("missing added tokens")?
        {
            let id = entry["id"]
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or("invalid added token ID")?;
            if map.classify(id) == Ok(TokenClass::Ordinary) {
                let text = entry["content"]
                    .as_str()
                    .ok_or("missing added token text")?;
                if text.is_empty() {
                    return Err("empty ordinary added token");
                }
                let value = text
                    .chars()
                    .map(|c| alphabet.get(&c).copied())
                    .collect::<Option<Vec<_>>>()
                    .unwrap_or_else(|| text.as_bytes().to_vec());
                bytes.insert(id, value);
            }
        }
        Ok(Self { map, bytes })
    }
}

enum TokenizerRef<'a> {
    Borrowed(&'a ByteTokenizer),
    Shared(std::sync::Arc<ByteTokenizer>),
}
impl std::ops::Deref for TokenizerRef<'_> {
    type Target = ByteTokenizer;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(t) => t,
            Self::Shared(t) => t,
        }
    }
}

/// 2026-10-06: Request-local state; incomplete UTF-8 may never cross a delimiter.
pub struct Stream<'a> {
    tokenizer: TokenizerRef<'a>,
    decoder: Decoder,
    pending: Vec<u8>,
    failed: bool,
    terminated: bool,
    reasoning_tokens: u32,
}

impl<'a> Stream<'a> {
    /// 2026-10-07: Request streams own shared metadata without copying the vocabulary.
    pub fn shared(
        tokenizer: std::sync::Arc<ByteTokenizer>,
        prompt: &[u32],
    ) -> Result<Stream<'static>, &'static str> {
        let seeded = tokenizer.assistant_stream(prompt)?;
        let decoder = seeded.decoder;
        Ok(Stream {
            tokenizer: TokenizerRef::Shared(tokenizer),
            decoder,
            pending: vec![],
            failed: false,
            terminated: false,
            reasoning_tokens: 0,
        })
    }

    /// 2026-10-07: Only validated final headers permit incremental visible UTF-8.
    pub fn visible_body(&self) -> Option<&str> {
        (self.decoder.phase == super::Phase::Body
            && self.decoder.recipient.is_none()
            && matches!(self.decoder.channel.as_deref(), None | Some("final")))
        .then_some(self.decoder.body.as_str())
    }

    pub fn new(tokenizer: &'a ByteTokenizer, decoder: Decoder) -> Self {
        Self {
            tokenizer: TokenizerRef::Borrowed(tokenizer),
            decoder,
            pending: Vec::new(),
            failed: false,
            terminated: false,
            reasoning_tokens: 0,
        }
    }

    pub fn push(&mut self, id: u32) -> Result<Option<Message>, &'static str> {
        if self.failed || self.terminated {
            self.failed = true;
            return Err("Harmony byte stream failed or already terminated");
        }
        // 2026-10-07: Count generated ordinary token IDs inside analysis body only.
        // Excludes channel headers/delimiters; this is not a billing-provider convention.
        let analysis = self.decoder.phase == super::Phase::Body
            && self.decoder.channel.as_deref() == Some("analysis")
            && self.tokenizer.map.classify(id) == Ok(TokenClass::Ordinary);
        let result = self.advance(id).and_then(|message| {
            if analysis {
                self.reasoning_tokens = self
                    .reasoning_tokens
                    .checked_add(1)
                    .ok_or("analysis token count overflow")?;
            }
            Ok(message)
        });
        self.failed = result.is_err();
        result
    }

    fn advance(&mut self, id: u32) -> Result<Option<Message>, &'static str> {
        match self.tokenizer.map.classify(id)? {
            TokenClass::Framing(token) => {
                if !self.pending.is_empty() {
                    return Err("incomplete UTF-8 at Harmony delimiter");
                }
                let message = self.decoder.push(token)?;
                self.terminated = message
                    .as_ref()
                    .is_some_and(|m| m.ending != Ending::Message);
                Ok(message)
            }
            TokenClass::Ordinary => {
                self.pending.extend_from_slice(
                    self.tokenizer
                        .bytes
                        .get(&id)
                        .ok_or("ordinary token has no bytes")?,
                );
                let end = match std::str::from_utf8(&self.pending) {
                    Ok(text) => text.len(),
                    Err(error) if error.error_len().is_none() => error.valid_up_to(),
                    Err(_) => return Err("invalid UTF-8 in Harmony generation"),
                };
                {
                    let text = std::str::from_utf8(&self.pending[..end])
                        .map_err(|_| "invalid UTF-8 prefix")?;
                    self.decoder.push(Token::Text(text))?;
                    self.pending.drain(..end);
                }
                Ok(None)
            }
        }
    }

    pub fn reasoning_tokens(&self) -> u32 {
        self.reasoning_tokens
    }

    pub fn finish(&self) -> Result<(), &'static str> {
        if self.failed || !self.pending.is_empty() {
            return Err("incomplete or invalid Harmony byte stream");
        }
        self.decoder.finish()
    }
}
