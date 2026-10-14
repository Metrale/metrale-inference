// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The reasoning field of a chat message or stream delta, written
//! under the name and presence rule of the model's
//! [`ReasoningWire`](crate::reasoning_parser::ReasoningWire). It is flattened
//! into its parent, so it adds one key or none.
//!
//! Owner: server (OpenAI adapter).
//! Invariants:
//! - At most one reasoning key is written; the text is never mirrored under a
//!   second name.

use serde::ser::{Serialize, SerializeMap, Serializer};

use crate::reasoning_parser::ReasoningWire;

/// 2026-10-14: One message's or delta's reasoning (module doc).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WireReasoning {
    /// No reasoning key.
    Omitted,
    /// The key `wire` names, with `text` or null.
    Field {
        wire: ReasoningWire,
        text: Option<String>,
    },
}

impl WireReasoning {
    /// 2026-10-14: A blocking message's reasoning: under `Vllm` the key is
    /// always there, null without reasoning; under `ReasoningContent` it is
    /// left out without reasoning.
    pub fn message(wire: ReasoningWire, text: Option<String>) -> Self {
        match (wire, text) {
            (ReasoningWire::Vllm, text) => Self::Field { wire, text },
            (ReasoningWire::ReasoningContent, Some(text)) => Self::Field {
                wire,
                text: Some(text),
            },
            (ReasoningWire::ReasoningContent, None) => Self::Omitted,
        }
    }

    /// 2026-10-14: A stream delta's reasoning text, under `wire`'s key.
    pub fn delta(wire: ReasoningWire, text: String) -> Self {
        Self::Field {
            wire,
            text: Some(text),
        }
    }

    /// 2026-10-14: The reasoning text, whatever key carries it.
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Omitted => None,
            Self::Field { text, .. } => text.as_deref(),
        }
    }
}

impl Serialize for WireReasoning {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(None)?;
        if let Self::Field { wire, text } = self {
            let key = match wire {
                ReasoningWire::ReasoningContent => "reasoning_content",
                ReasoningWire::Vllm => "reasoning",
            };
            map.serialize_entry(key, text)?;
        }
        map.end()
    }
}
