// SPDX-License-Identifier: MIT OR Apache-2.0

//! The long-range match model. A hash of the last `min_len` bytes finds the most recent
//! earlier occurrence; while the following bytes keep agreeing, the byte that followed it is
//! predicted bit by bit with a confidence learned per match length. This is what captures
//! duplication across kernels in one solid stream: the window is the whole block.

use crate::cast::{ix, lo32};
use crate::mixer::Mixer;
use crate::slots::{COUNTER_INIT, Rates, predict};
use crate::tables::Stretch;
use alloc::vec::Vec;

/// Longest match length tracked (also the confidence saturation point).
pub const MAX_LEN: u32 = 65_535;
/// Bytes compared backwards when a fresh candidate is verified.
const VERIFY: u32 = 48;

/// Match state over a byte history.
pub struct MatchModel {
    ht: Vec<u32>,
    mask: u32,
    min_len: u32,
    ptr: u32,
    len: u32,
    expected: u32,
    sm: Vec<u32>,
    sm_ctx: u32,
    active: bool,
}

impl MatchModel {
    /// A model with `2^log2_entries` hash slots whose candidates need `min_len` equal bytes.
    pub fn new(log2_entries: u8, min_len: u32) -> Self {
        let bits = u32::from(log2_entries.min(28));
        Self {
            ht: alloc::vec![0u32; ix(1u32 << bits)],
            mask: (1u32 << bits) - 1,
            min_len: min_len.clamp(4, 8),
            ptr: 0,
            len: 0,
            expected: 0,
            sm: alloc::vec![COUNTER_INIT; 64 * 2],
            sm_ctx: 0,
            active: false,
        }
    }

    /// Bytes a model of `2^log2_entries` slots occupies.
    pub fn bytes(log2_entries: u8) -> u64 {
        (1u64 << u32::from(log2_entries.min(28))) * 4 + 512
    }

    /// The current match length (0 when none).
    #[inline(always)]
    pub fn len(&self) -> u32 {
        self.len
    }

    /// The byte the match predicts next, if a match is live.
    #[inline(always)]
    pub fn expected(&self) -> Option<u8> {
        if self.len > 0 {
            u8::try_from(self.expected & 0xff).ok()
        } else {
            None
        }
    }

    /// Length bucket in `0..16`, log-spaced.
    #[inline(always)]
    pub fn len_bucket(len: u32) -> u32 {
        match len {
            0 => 0,
            1..=7 => len,
            8..=11 => 8,
            12..=15 => 9,
            16..=23 => 10,
            24..=31 => 11,
            32..=47 => 12,
            48..=63 => 13,
            64..=127 => 14,
            _ => 15,
        }
    }

    #[inline]
    fn hash_at(&self, hist: &[u8], end: usize) -> u32 {
        let start = end.saturating_sub(ix(self.min_len));
        let mut h: u64 = 0x9E37_79B9_7F4A_7C15;
        for &b in &hist[start..end] {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
        lo32(h ^ (h >> 29))
    }

    /// Advance after `hist` gained one byte (`hist.len()` is the new position).
    pub fn update_byte(&mut self, hist: &[u8]) {
        let pos = hist.len();
        let Some(&last) = hist.last() else {
            return;
        };
        if self.len > 0 && self.expected & 0xff == u32::from(last) {
            self.len = (self.len + 1).min(MAX_LEN);
            self.ptr = self.ptr.wrapping_add(1);
        } else {
            self.len = 0;
        }
        if pos < ix(self.min_len) {
            return;
        }
        let h = self.hash_at(hist, pos) & self.mask;
        if self.len == 0 {
            let cand = self.ht[ix(h)];
            let cand_u = ix(cand);
            if cand > 0 && cand_u < pos {
                let mut n = 0u32;
                while n < VERIFY {
                    let k = ix(n);
                    if k >= cand_u || hist[cand_u - 1 - k] != hist[pos - 1 - k] {
                        break;
                    }
                    n += 1;
                }
                if n >= self.min_len {
                    self.len = n;
                    self.ptr = cand;
                }
            }
        }
        let pos32 = u32::try_from(pos).unwrap_or(u32::MAX);
        self.ht[ix(h)] = pos32;
        if self.len > 0 {
            match hist.get(ix(self.ptr)) {
                Some(&e) => self.expected = u32::from(e),
                None => self.len = 0,
            }
        }
    }

    /// Add this model's two inputs for the bit at `bitpos` of a byte whose coded prefix is
    /// `c0` (leading 1 bit then the bits so far).
    #[inline]
    pub fn add_inputs(&mut self, c0: u32, bitpos: u32, mixer: &mut Mixer, stretch: &Stretch) {
        self.active = false;
        if self.len > 0 && (self.expected | 0x100) >> (8 - bitpos) == c0 {
            let bit = (self.expected >> (7 - bitpos)) & 1;
            self.sm_ctx = Self::len_bucket(self.len) * 2 + bit;
            self.active = true;
            let p = predict(self.sm[ix(self.sm_ctx)]);
            let st = stretch.stretch(p);
            mixer.add(st);
            let strength = i32::try_from(Self::len_bucket(self.len)).unwrap_or(0) * 64;
            mixer.add(if bit == 1 { strength } else { -strength });
        } else {
            mixer.add(0);
            mixer.add(0);
        }
    }

    /// Learn from the coded bit.
    #[inline]
    pub fn update_bit(&mut self, bit: u32, rates: &Rates) {
        if self.active {
            rates.update(&mut self.sm[ix(self.sm_ctx)], bit, 1023);
        }
    }

    /// Mixer context: 0 when idle, else the length bucket.
    #[inline(always)]
    pub fn mixer_context(&self) -> u32 {
        if self.len == 0 {
            0
        } else {
            Self::len_bucket(self.len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_repeat_and_predicts_its_continuation() {
        let text = b"the quick brown fox jumps over the lazy dog; the quick brown fox jumps";
        let mut m = MatchModel::new(10, 6);
        let mut hist = Vec::new();
        for &b in text.iter() {
            hist.push(b);
            m.update_byte(&hist);
        }
        assert!(m.len() >= 20, "len {}", m.len());
        assert_eq!(m.expected(), Some(b' '));
    }
}
