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

use crate::cast::ix;
use crate::coder::BitCoder;
use crate::engine::code_byte;
use crate::format::Profile;
use crate::predictor::Predictor;
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
        self.cells[ix(key & self.mask)] = Cell {
            key: key | 1,
            value,
        };
    }
}

/// Memory parameter of the inner delta-stream predictor.
const INNER_MEM_LOG2: u8 = 16;

/// Codes one number per marker as a zigzag varint delta against the previous number of the
/// same identifier class, through an inner generic context-mixing predictor whose history is
/// the delta byte stream itself, each delta preceded (free of charge, since both sides know
/// it) by a byte naming the class. Unrolled code yields repeating delta sequences, which the
/// inner orders and match model capture.
pub struct NumberModel {
    last: Memory,
    pred: Predictor,
    hist: Vec<u8>,
}

impl NumberModel {
    /// A fresh model.
    pub fn new() -> Self {
        Self {
            last: Memory::new(14),
            pred: Predictor::new(Profile::Generic, INNER_MEM_LOG2),
            hist: Vec::new(),
        }
    }

    /// Bytes the model occupies beside its delta history (≤ 11 bytes per number).
    pub fn bytes() -> u64 {
        (1u64 << 14) * 16 + crate::predictor::memory_bytes(Profile::Generic, INNER_MEM_LOG2)
    }

    /// Code (encode when `value` is `Some`, else decode) the number at a marker whose identifier
    /// class hashes to `class` on skeleton line context `line`. Returns the value.
    pub fn code(
        &mut self,
        io: &mut impl BitCoder,
        class: u32,
        line: u32,
        value: Option<u64>,
    ) -> u64 {
        let last = self.last.get(class).unwrap_or(0);
        let tag = (class ^ (class >> 8) ^ (class >> 16) ^ (class >> 24)) & 0xff;
        self.hist.push(u8::try_from(tag).unwrap_or(0));
        self.pred.byte_done(&self.hist);
        let lt = (line ^ (line >> 8) ^ (line >> 16) ^ (line >> 24)) & 0xff;
        self.hist.push(u8::try_from(lt).unwrap_or(0));
        self.pred.byte_done(&self.hist);
        let mut z = value.map(|v| zigzag(as_i64(v.wrapping_sub(last))));
        let mut out = 0u64;
        let mut shift = 0u32;
        for _ in 0..10 {
            let want = z.map(|z| {
                let low = u8::try_from(z & 0x7f).unwrap_or(0);
                if z >= 0x80 { low | 0x80 } else { low }
            });
            let got = code_byte(&mut self.pred, io, want, &mut self.hist);
            z = z.map(|z| z >> 7);
            out |= u64::from(got & 0x7f) << shift;
            shift += 7;
            if got & 0x80 == 0 {
                break;
            }
        }
        let v = last.wrapping_add(as_u64(unzigzag(out)));
        self.last.set(class, v);
        v
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
        assert_eq!(
            next_event(s, at(b"12,")),
            Event::Number {
                digits: 2,
                value: 12
            }
        );
        assert_eq!(next_event(s, at(b"007")), Event::Literal(b'0'));
        assert_eq!(
            next_event(s, at(b"6_2")),
            Event::Number {
                digits: 1,
                value: 6
            }
        );
        assert_eq!(
            next_event(s, at(b"_2,") + 1),
            Event::Number {
                digits: 1,
                value: 2
            }
        );
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
