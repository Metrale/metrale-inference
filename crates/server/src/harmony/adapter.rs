// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Resolve checkpoint token identities before ordinary text decoding.
//!
//! No I/O: the caller supplies the exact tokenizer.json used by the model. Resolve
//! each generated ID before any skip-special-tokens decode or scheduler EOS filter.
//! Accumulate ordinary IDs with a byte-safe tokenizer decoder and flush text before
//! framing events. Unknown/padding/reserved IDs are errors, never successful EOS.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use super::Token;

/// 2026-10-06: Ordinary IDs need incremental text decoding; framing IDs must bypass it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenClass {
    Ordinary,
    Framing(Token<'static>),
}

/// 2026-10-06: Immutable, metadata-derived ID classification, not decoded-text matching.
#[derive(Debug)]
pub struct TokenMap {
    classes: HashMap<u32, Option<TokenClass>>,
}

impl TokenMap {
    pub(super) fn terminal_ids(&self) -> Vec<u32> {
        let mut ids: Vec<_> = self
            .classes
            .iter()
            .filter_map(|(&id, class)| {
                matches!(
                    class,
                    Some(TokenClass::Framing(Token::Finish | Token::Handoff))
                )
                .then_some(id)
            })
            .collect();
        ids.sort_unstable();
        ids
    }

    /// 2026-10-06: Require all seven checkpoint-native framing tokens, declared special.
    /// Full tokenizer metadata is required in production, not just added_tokens.
    /// Token spellings belong to this checkpoint dialect; IDs come only from metadata.
    pub fn from_tokenizer_json(json: &str) -> Result<Self, &'static str> {
        let data: Value = serde_json::from_str(json).map_err(|_| "invalid tokenizer JSON")?;
        let vocab = data["model"]["vocab"]
            .as_object()
            .ok_or("tokenizer model vocabulary is required")?;
        let added = data["added_tokens"]
            .as_array()
            .ok_or("tokenizer added-token metadata is required")?;
        let mut names = HashMap::new();
        let mut ids = HashMap::new();
        let mut classes = HashMap::new();
        for (name, value) in vocab {
            let id = token_id(value)?;
            if names.insert(id, name.as_str()).is_some() {
                return Err("duplicate vocabulary token ID");
            }
            ids.insert(name.as_str(), id);
            classes.insert(id, Some(TokenClass::Ordinary));
        }
        let mut seen_added_ids = HashSet::new();
        let mut seen_added_names = HashSet::new();
        let mut framing = HashSet::new();
        for entry in added {
            let id = token_id(&entry["id"])?;
            let name = entry["content"].as_str().ok_or("missing token content")?;
            if !seen_added_ids.insert(id) || !seen_added_names.insert(name) {
                return Err("duplicate added token");
            }
            if names.get(&id).is_some_and(|existing| *existing != name)
                || ids.get(name).is_some_and(|existing| *existing != id)
            {
                return Err("added token conflicts with vocabulary");
            }
            names.insert(id, name);
            ids.insert(name, id);
            let special = entry["special"].as_bool().ok_or("missing special flag")?;
            let kind = delimiter(name);
            if kind.is_some() {
                if !special {
                    return Err("Harmony framing token must be special");
                }
                for flag in ["normalized", "single_word", "lstrip", "rstrip"] {
                    if entry[flag].as_bool() != Some(false) {
                        return Err("Harmony framing token has incompatible matching flags");
                    }
                }
                framing.insert(name);
            }
            classes.insert(
                id,
                if special {
                    kind.map(TokenClass::Framing)
                } else {
                    Some(TokenClass::Ordinary)
                },
            );
        }
        if framing.len() != 7 {
            return Err("checkpoint lacks required Harmony framing tokens");
        }
        Ok(Self { classes })
    }

    /// 2026-10-06: An ID absent from tokenizer vocabulary is not a text token, even
    /// when it falls inside the model's larger, padded output-head vocabulary.
    pub fn classify(&self, id: u32) -> Result<TokenClass, &'static str> {
        match self.classes.get(&id) {
            Some(Some(class)) => Ok(*class),
            Some(None) => Err("unsupported Harmony special token"),
            None => Err("generated ID is absent from checkpoint tokenizer"),
        }
    }
}

fn token_id(value: &Value) -> Result<u32, &'static str> {
    value
        .as_u64()
        .and_then(|id| u32::try_from(id).ok())
        .ok_or("token ID must be an unsigned 32-bit integer")
}

fn delimiter(name: &str) -> Option<Token<'static>> {
    match name {
        "<|start|>" => Some(Token::Start),
        "<|constrain|>" => Some(Token::Constrain),
        "<|channel|>" => Some(Token::Channel),
        "<|message|>" => Some(Token::Separator),
        "<|end|>" => Some(Token::End),
        "<|return|>" => Some(Token::Finish),
        "<|call|>" => Some(Token::Handoff),
        _ => None,
    }
}
