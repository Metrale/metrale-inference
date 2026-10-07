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

/// 2026-10-06: Request-local state; incomplete UTF-8 may never cross a delimiter.
pub struct Stream<'a> {
    tokenizer: &'a ByteTokenizer,
    decoder: Decoder,
    pending: Vec<u8>,
    failed: bool,
    terminated: bool,
}

impl<'a> Stream<'a> {
    pub fn new(tokenizer: &'a ByteTokenizer, decoder: Decoder) -> Self {
        Self {
            tokenizer,
            decoder,
            pending: Vec::new(),
            failed: false,
            terminated: false,
        }
    }

    pub fn push(&mut self, id: u32) -> Result<Option<Message>, &'static str> {
        if self.failed || self.terminated {
            self.failed = true;
            return Err("Harmony byte stream failed or already terminated");
        }
        let result = self.advance(id);
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

    pub fn finish(&self) -> Result<(), &'static str> {
        if self.failed || !self.pending.is_empty() {
            return Err("incomplete or invalid Harmony byte stream");
        }
        self.decoder.finish()
    }
}
