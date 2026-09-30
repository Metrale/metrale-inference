// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The model buffers a circuit's declared outputs bind to (M5, LIFECYCLE-DESIGN.md
//! section 3.4): a block's `outputs` name each edge read outside the program and the buffer it
//! lands in, so the executor places external edges by declaration, never by guessing from the
//! producing op.
//!
//! Owner: metrale-circuit.
//! Invariants: the stream edges are the model's hidden buffer by construction and are not
//! listed here.

/// 2026-09-30: A buffer the model owns that a program's output lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModelBuffer {
    /// 2026-09-30: The vocabulary logits the sampler (or the host) reads.
    Logits,
    /// 2026-09-30: The sampled tokens.
    Tokens,
    /// 2026-09-30: The MTP draft head's token embedding, which the draft runner fills.
    DraftEmbed,
}

impl ModelBuffer {
    const NAMES: [(ModelBuffer, &'static str); 3] = [
        (ModelBuffer::Logits, "logits"),
        (ModelBuffer::Tokens, "tokens"),
        (ModelBuffer::DraftEmbed, "draft_embed"),
    ];

    /// 2026-09-30: The spelling in the circuit TOML (`outputs = [{ edge, buffer }]`).
    pub fn parse(s: &str) -> Option<Self> {
        Self::NAMES.iter().find(|(_, n)| *n == s).map(|(b, _)| *b)
    }

    /// 2026-09-30: The spelling.
    pub fn name(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(b, _)| *b == self)
            .map(|(_, n)| *n)
            .unwrap_or("?")
    }
}
