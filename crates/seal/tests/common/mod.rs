// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Shared test helpers: hex decoding and a counting `Entropy` fake (the port, not
//! a crypto mock: it yields a fixed byte stream and counts calls).

#![allow(dead_code)]

use std::cell::Cell;

use metrale_abstract_io::{Entropy, Result};

/// 2026-10-22: Decode hex (either case); malformed input is a test bug.
pub fn hex(text: &str) -> Vec<u8> {
    let text: String = text.chars().filter(|c| !c.is_whitespace()).collect();
    assert!(text.len().is_multiple_of(2), "odd-length hex");
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).expect("hex digit"))
        .collect()
}

pub fn hex_array<const N: usize>(text: &str) -> [u8; N] {
    hex(text).try_into().expect("hex length")
}

/// 2026-10-22: Byte `i` of the stream is `(start + i) mod 256`; `calls` counts `fill`s.
pub struct CountingEntropy {
    pub start: u8,
    pub calls: Cell<u32>,
    pub drawn: Cell<usize>,
}

impl CountingEntropy {
    pub fn new(start: u8) -> Self {
        Self {
            start,
            calls: Cell::new(0),
            drawn: Cell::new(0),
        }
    }
}

impl Entropy for CountingEntropy {
    fn fill(&self, buf: &mut [u8]) -> Result<()> {
        let base = self.drawn.get();
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self.start.wrapping_add(((base + i) % 256) as u8);
        }
        self.calls.set(self.calls.get() + 1);
        self.drawn.set(base + buf.len());
        Ok(())
    }
}
