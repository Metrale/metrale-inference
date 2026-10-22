// SPDX-License-Identifier: MIT OR Apache-2.0

//! A carry-less binary arithmetic coder over 32-bit range bounds with 12-bit probabilities
//! (the lpaq coder). The decoder reads zeros past the end of its input, so a truncated stream
//! yields wrong bytes but never an out-of-bounds read or an unbounded loop: the block loop
//! around it runs exactly the declared number of bytes.

use crate::cast::ix;
use alloc::vec::Vec;

/// One binary decision, shared by both directions so every model is written once: the encoder
/// codes `bit` and returns it, the decoder ignores `bit` and returns what it reads.
pub trait BitCoder {
    /// Code one bit under `p1` = P(1) in 12 bits.
    fn code(&mut self, bit: u32, p1: i32) -> u32;
}

impl BitCoder for Encoder {
    #[inline(always)]
    fn code(&mut self, bit: u32, p1: i32) -> u32 {
        self.encode(bit, p1);
        bit
    }
}

impl BitCoder for Decoder<'_> {
    #[inline(always)]
    fn code(&mut self, _bit: u32, p1: i32) -> u32 {
        self.decode(p1)
    }
}

/// Encodes bits into a byte vector.
pub struct Encoder {
    x1: u32,
    x2: u32,
    out: Vec<u8>,
}

impl Encoder {
    /// An encoder writing into an empty vector.
    pub fn new() -> Self {
        Self {
            x1: 0,
            x2: 0xffff_ffff,
            out: Vec::new(),
        }
    }

    /// An encoder that reserves `cap` output bytes up front.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            x1: 0,
            x2: 0xffff_ffff,
            out: Vec::with_capacity(cap),
        }
    }

    /// Code `bit` with `p1` = P(bit = 1) in 12 bits, clamped into `[1, 4095]`.
    #[inline]
    pub fn encode(&mut self, bit: u32, p1: i32) {
        let p = u32::try_from(p1.clamp(1, 4095)).unwrap_or(2048);
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p + (((range & 0xfff) * p) >> 12);
        if bit != 0 {
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.out.push(self.x2.to_be_bytes()[0]);
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
        }
    }

    /// Flush the final bytes and return the coded stream.
    pub fn finish(mut self) -> Vec<u8> {
        // Four bytes of x1 pin the final interval on any input length.
        self.out.extend_from_slice(&self.x1.to_be_bytes());
        self.out
    }
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Decodes bits from a byte slice.
pub struct Decoder<'a> {
    x1: u32,
    x2: u32,
    x: u32,
    data: &'a [u8],
    pos: u32,
}

impl<'a> Decoder<'a> {
    /// A decoder over `data`; it primes itself with the first four bytes.
    pub fn new(data: &'a [u8]) -> Self {
        let mut d = Self {
            x1: 0,
            x2: 0xffff_ffff,
            x: 0,
            data,
            pos: 0,
        };
        for _ in 0..4 {
            d.x = (d.x << 8) | d.next_byte();
        }
        d
    }

    #[inline]
    fn next_byte(&mut self) -> u32 {
        let b = self.data.get(ix(self.pos)).copied().unwrap_or(0);
        self.pos = self.pos.saturating_add(1);
        u32::from(b)
    }

    /// Decode one bit given `p1` = P(bit = 1) in 12 bits.
    #[inline]
    pub fn decode(&mut self, p1: i32) -> u32 {
        let p = u32::try_from(p1.clamp(1, 4095)).unwrap_or(2048);
        let range = self.x2 - self.x1;
        let xmid = self.x1 + (range >> 12) * p + (((range & 0xfff) * p) >> 12);
        let bit = if self.x <= xmid {
            self.x2 = xmid;
            1
        } else {
            self.x1 = xmid + 1;
            0
        };
        while (self.x1 ^ self.x2) & 0xff00_0000 == 0 {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) | 255;
            self.x = (self.x << 8) | self.next_byte();
        }
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_skewed_and_uniform_bits() {
        let mut e = Encoder::new();
        let mut bits = Vec::new();
        let mut s = 0x1234_5678u32;
        for i in 0..20_000u32 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let p = if i % 3 == 0 { 4000 } else { 2048 };
            let bit = if i % 3 == 0 {
                u32::from(s % 50 != 0)
            } else {
                s & 1
            };
            e.encode(bit, p);
            bits.push((bit, p));
        }
        let out = e.finish();
        let mut d = Decoder::new(&out);
        for (bit, p) in bits {
            assert_eq!(d.decode(p), bit);
        }
    }

    #[test]
    fn truncated_input_still_terminates() {
        let data = [0xffu8; 2];
        let mut d = Decoder::new(&data);
        for _ in 0..1000 {
            let _ = d.decode(2048);
        }
    }
}
