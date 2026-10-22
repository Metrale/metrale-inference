// SPDX-License-Identifier: MIT OR Apache-2.0

//! The PTX transform and its number model.
//!
//! PTX bodies repeat almost exactly except for the numbers in register and label names
//! (`%r123`, `%f4`, `$L__BB6_219`): sequential allocation makes every unrolled line differ from
//! its twin by a few digits, which breaks byte matches and dilutes every context. The encoder
//! replaces each canonical digit run inside a `%`/`$` identifier with the marker byte
//! [`MARKER`]; the model sees that skeleton (long exact repeats), and the number is coded right
//! after the marker by [`NumberModel`] as a delta against the previous number of the same
//! identifier class, predicted from the skeleton line it sits in. Raw `0x01`/`0x02` bytes are
//! escaped with [`ESCAPE`], so the transform is exact for arbitrary input.

use crate::cast::{ix, lo32};
use crate::coder::BitCoder;
use crate::slots::{COUNTER_INIT, Rates, predict};
use alloc::vec::Vec;

/// Replaces an extracted digit run in the skeleton.
pub const MARKER: u8 = 0x01;
/// Precedes a literal `0x01` or `0x02` byte in the skeleton.
pub const ESCAPE: u8 = 0x02;
/// Longest digit run extracted (longer runs stay literal; `u64` holds 19 digits).
const MAX_DIGITS: usize = 18;
/// How far back an identifier may start (longer identifiers keep their digits).
const IDENT_SCAN: usize = 96;

#[inline(always)]
fn is_ident(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// What the encoder emits next for the skeleton.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    /// One literal input byte (the engine escapes it if needed).
    Literal(u8),
    /// A digit run of `digits` bytes with value `value`, replaced by the marker.
    Number {
        /// Length of the run in the input.
        digits: usize,
        /// Its decimal value.
        value: u64,
    },
}

/// Decide the event at `pos` of `raw`. A digit run becomes a number when it is maximal,
/// canonical (no leading zero unless it is `0`), at most [`MAX_DIGITS`] long and preceded,
/// through identifier bytes, by a `%` or `$` that starts the identifier.
pub fn next_event(raw: &[u8], pos: usize) -> Event {
    let b = raw[pos];
    if !b.is_ascii_digit() {
        return Event::Literal(b);
    }
    if pos > 0 && raw[pos - 1].is_ascii_digit() {
        return Event::Literal(b);
    }
    let mut i = pos;
    let mut steps = 0usize;
    loop {
        if i == 0 || steps >= IDENT_SCAN {
            return Event::Literal(b);
        }
        let prev = raw[i - 1];
        if prev == b'%' || prev == b'$' {
            if i >= 2 && is_ident(raw[i - 2]) {
                return Event::Literal(b);
            }
            break;
        }
        if !is_ident(prev) {
            return Event::Literal(b);
        }
        i -= 1;
        steps += 1;
    }
    let mut end = pos;
    while end < raw.len() && raw[end].is_ascii_digit() && end - pos < MAX_DIGITS + 1 {
        end += 1;
    }
    let digits = end - pos;
    if digits > MAX_DIGITS || (digits > 1 && b == b'0') {
        return Event::Literal(b);
    }
    let mut value = 0u64;
    for &d in &raw[pos..end] {
        value = value * 10 + u64::from(d - b'0');
    }
    Event::Number { digits, value }
}

/// Append the decimal digits of `v` to `out` (bounded: at most 20 digits).
pub fn push_decimal(out: &mut Vec<u8>, v: u64) {
    let mut buf = [0u8; 20];
    let mut n = 0usize;
    let mut x = v;
    loop {
        buf[n] = b'0' + u8::try_from(x % 10).unwrap_or(0);
        n += 1;
        x /= 10;
        if x == 0 || n == 20 {
            break;
        }
    }
    while n > 0 {
        n -= 1;
        out.push(buf[n]);
    }
}

/// Reinterpret the bits of a signed delta (no numeric cast).
#[inline(always)]
fn as_u64(d: i64) -> u64 {
    u64::from_ne_bytes(d.to_ne_bytes())
}

/// Reinterpret the bits of an unsigned word as a signed delta.
#[inline(always)]
fn as_i64(u: u64) -> i64 {
    i64::from_ne_bytes(u.to_ne_bytes())
}

#[inline(always)]
fn zigzag(d: i64) -> u64 {
    as_u64((d << 1) ^ (d >> 63))
}

#[inline(always)]
fn unzigzag(z: u64) -> i64 {
    as_i64(z >> 1) ^ -as_i64(z & 1)
}

#[inline(always)]
fn h32(a: u32, b: u32) -> u32 {
    lo32((u64::from(a) ^ (u64::from(b) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 29)
}

/// A keyed memory cell: a 32-bit key guards a value.
#[derive(Clone, Copy)]
struct Cell {
    key: u32,
    value: u64,
}

struct Memory {
    cells: Vec<Cell>,
    mask: u32,
}

impl Memory {
    fn new(log2: u32) -> Self {
        Self {
            cells: alloc::vec![Cell { key: 0, value: 0 }; ix(1u32 << log2)],
            mask: (1u32 << log2) - 1,
        }
    }

    #[inline]
    fn get(&self, key: u32) -> Option<u64> {
        let c = self.cells[ix(key & self.mask)];
        (c.key == key | 1).then_some(c.value)
    }

    #[inline]
    fn set(&mut self, key: u32, value: u64) {
        self.cells[ix(key & self.mask)] = Cell { key: key | 1, value };
    }
}

/// Codes one number per marker: two "same delta as last time" flags under skeleton contexts,
/// then an explicit zigzag delta (bit length, then mantissa) when both miss.
pub struct NumberModel {
    last: Memory,
    by_line: Memory,
    by_class: Memory,
    p_line: Vec<u32>,
    p_class: Vec<u32>,
    p_len: Vec<u32>,
    p_mant: Vec<u32>,
    rates: Rates,
}

impl NumberModel {
    /// A fresh model.
    pub fn new() -> Self {
        Self {
            last: Memory::new(14),
            by_line: Memory::new(18),
            by_class: Memory::new(14),
            p_line: alloc::vec![COUNTER_INIT; 1 << 16],
            p_class: alloc::vec![COUNTER_INIT; 1 << 12],
            p_len: alloc::vec![COUNTER_INIT; 256 * 128],
            p_mant: alloc::vec![COUNTER_INIT; 65 * 64 * 8],
            rates: Rates::new(),
        }
    }

    /// Bytes the model occupies.
    pub fn bytes() -> u64 {
        ((1u64 << 14) + (1 << 18) + (1 << 14)) * 16 + ((1u64 << 16) + (1 << 12) + 256 * 128 + 65 * 64 * 8) * 4
    }

    #[inline]
    fn flag(&mut self, io: &mut impl BitCoder, table: usize, at: u32, bit: Option<bool>) -> bool {
        let want = u32::from(bit.unwrap_or(false));
        let t = if table == 0 { &mut self.p_line } else { &mut self.p_class };
        let i = ix(at) % t.len();
        let got = io.code(want, predict(t[i]));
        self.rates.update(&mut t[i], got, 1023);
        got == 1
    }

    /// Code (encode when `value` is `Some`, else decode) the number at a marker whose identifier
    /// class hashes to `class` on skeleton line context `line`. Returns the value.
    pub fn code(&mut self, io: &mut impl BitCoder, class: u32, line: u32, value: Option<u64>) -> u64 {
        let last = self.last.get(class).unwrap_or(0);
        let delta = value.map(|v| as_i64(v.wrapping_sub(last)));
        let line_key = h32(class, line);
        let guess_line = self.by_line.get(line_key);
        let guess_class = self.by_class.get(class);
        let mut result: Option<i64> = None;
        if let Some(g) = guess_line {
            let hit = self.flag(io, 0, line_key >> 2, delta.map(|d| as_u64(d) == g));
            if hit {
                result = Some(as_i64(g));
            }
        }
        if result.is_none() {
            if let Some(g) = guess_class {
                if guess_line != Some(g) {
                    let hit = self.flag(io, 1, class >> 4, delta.map(|d| as_u64(d) == g));
                    if hit {
                        result = Some(as_i64(g));
                    }
                }
            }
        }
        let d = match result {
            Some(d) => d,
            None => self.code_explicit(io, class, delta),
        };
        let v = last.wrapping_add(as_u64(d));
        self.last.set(class, v);
        self.by_line.set(line_key, as_u64(d));
        self.by_class.set(class, as_u64(d));
        v
    }

    fn code_explicit(&mut self, io: &mut impl BitCoder, class: u32, delta: Option<i64>) -> i64 {
        let z = delta.map(zigzag);
        let nbits = z.map_or(0, |z| 64 - z.leading_zeros());
        let cls = (class >> 8) & 0xff;
        let mut node = 1u32;
        for k in (0..7u32).rev() {
            let want = (nbits >> k) & 1;
            let i = ix(cls * 128 + node);
            let got = io.code(want, predict(self.p_len[i]));
            self.rates.update(&mut self.p_len[i], got, 255);
            node = (node << 1) | got;
        }
        let n = (node & 127).min(64);
        let mut out: u64 = if n > 0 { 1 } else { 0 };
        if n > 1 {
            for k in (0..n - 1).rev() {
                let want = z.map_or(0, |z| u32::try_from((z >> k) & 1).unwrap_or(0));
                let top = if n - 2 - k < 3 { u32::try_from(out & 7).unwrap_or(0) } else { 0 };
                let i = ix((n * 64 + k.min(63)) * 8 + top);
                let got = io.code(want, predict(self.p_mant[i]));
                self.rates.update(&mut self.p_mant[i], got, 255);
                out = (out << 1) | u64::from(got);
            }
        }
        unzigzag(out)
    }
}

impl Default for NumberModel {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn events_follow_the_identifier_rules() {
        let s = b"\tadd.s32 \t%r12, %r007, $L__BB6_2, 0f3F800000, x9;";
        let at = |needle: &[u8]| s.windows(needle.len()).position(|w| w == needle).unwrap();
        assert_eq!(next_event(s, at(b"12,")), Event::Number { digits: 2, value: 12 });
        assert_eq!(next_event(s, at(b"007")), Event::Literal(b'0'));
        assert_eq!(next_event(s, at(b"6_2")), Event::Number { digits: 1, value: 6 });
        assert_eq!(next_event(s, at(b"2,") + 0), Event::Number { digits: 1, value: 2 });
        assert_eq!(next_event(s, at(b"0f3F")), Event::Literal(b'0'));
        assert_eq!(next_event(s, at(b"9;")), Event::Literal(b'9'));
    }

    #[test]
    fn zigzag_round_trips() {
        for d in [0i64, 1, -1, 5, -5, i64::MAX, i64::MIN, 123_456_789] {
            assert_eq!(unzigzag(zigzag(d)), d);
        }
    }

    #[test]
    fn decimal_rendering() {
        let mut v = Vec::new();
        push_decimal(&mut v, 0);
        push_decimal(&mut v, 1234);
        push_decimal(&mut v, u64::MAX);
        assert_eq!(v, b"0123418446744073709551615");
    }
}
