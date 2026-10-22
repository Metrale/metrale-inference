// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hashed context statistics. A context hash selects a 16-byte bucket holding a check byte and
//! 15 one-byte bit-history states, one per node of the binary tree over one nibble, so each byte
//! costs two bucket lookups per context. Buckets are probed three deep within one cache line and
//! the least-used one is recycled on a miss (the lpaq layout).
//!
//! A [`StateMap`] turns a state into a probability and learns that mapping online; its counters
//! pack a 22-bit probability and a 10-bit hit count so the step shrinks as evidence accumulates.

use crate::cast::{ix, ixi};
use crate::states::{STATES, StateTable};
use alloc::vec::Vec;

/// Bytes per bucket: one check byte and 15 states.
pub const BUCKET_BYTES: u32 = 16;
/// A fresh counter: probability one half, count zero.
pub const COUNTER_INIT: u32 = 1 << 31;

/// Per-count learning rates, `16384 / (2n + 3)`.
pub struct Rates {
    dt: [i32; 1024],
}

impl Rates {
    /// Build the table.
    pub fn new() -> Self {
        let mut dt = [0i32; 1024];
        let mut i = 0i32;
        while i < 1024 {
            dt[ixi(i)] = 16384 / (i + i + 3);
            i += 1;
        }
        Self { dt }
    }

    /// Move counter `v` toward `bit`, capping the count at `limit` (≤ 1023).
    #[inline(always)]
    pub fn update(&self, v: &mut u32, bit: u32, limit: u32) {
        let n = *v & 1023;
        let p = i32::try_from(*v >> 10).unwrap_or(0);
        if n < limit {
            *v = v.wrapping_add(1);
        } else {
            *v = (*v & !1023) | limit;
        }
        let target = i32::try_from(bit << 22).unwrap_or(0);
        let delta = ((target - p) >> 3).wrapping_mul(self.dt[ix(n)]) & !1023;
        *v = v.wrapping_add_signed(delta);
    }
}

impl Default for Rates {
    fn default() -> Self {
        Self::new()
    }
}

/// The 12-bit probability a counter holds.
#[inline(always)]
pub fn predict(v: u32) -> i32 {
    i32::try_from(v >> 20).unwrap_or(2048)
}

/// Learned state → probability map for one context.
pub struct StateMap {
    t: Vec<u32>,
}

impl StateMap {
    /// Initialise from the counts behind each state.
    pub fn new(st: &StateTable) -> Self {
        let mut t = alloc::vec![COUNTER_INIT; STATES];
        for (s, slot) in t.iter_mut().enumerate() {
            let (n0, n1) = st.counts(u8::try_from(s).unwrap_or(0));
            let p22 = ((u64::from(n1) * 2 + 1) << 22) / (u64::from(n0 + n1) * 2 + 2);
            *slot = u32::try_from(p22).unwrap_or(1 << 21) << 10;
        }
        Self { t }
    }

    /// P(1) in 12 bits for state `s`.
    #[inline(always)]
    pub fn predict(&self, s: u8) -> i32 {
        predict(self.t[ix(u32::from(s))])
    }

    /// Learn `bit` under state `s`.
    #[inline(always)]
    pub fn update(&mut self, s: u8, bit: u32, rates: &Rates) {
        rates.update(&mut self.t[ix(u32::from(s))], bit, 1023);
    }
}

/// One hashed table of nibble buckets.
pub struct Slots {
    t: Vec<u8>,
    mask: u32,
}

impl Slots {
    /// A table of `2^log2_buckets` buckets (`log2_buckets` ≤ 27).
    pub fn new(log2_buckets: u8) -> Self {
        let bits = u32::from(log2_buckets.min(27));
        let buckets = 1u32 << bits;
        Self {
            t: alloc::vec![0u8; ix(buckets * BUCKET_BYTES)],
            mask: buckets - 1,
        }
    }

    /// Bytes a table of `2^log2_buckets` buckets occupies.
    pub fn bytes(log2_buckets: u8) -> u64 {
        (1u64 << u32::from(log2_buckets.min(27))) * u64::from(BUCKET_BYTES)
    }

    /// The byte index of the bucket for hash `h`, allocating or recycling on a miss.
    #[inline]
    pub fn find(&mut self, h: u32, st: &StateTable) -> u32 {
        let check = h.to_be_bytes()[0];
        let i0 = h & self.mask;
        let mut best = i0;
        let mut best_n = u32::MAX;
        for k in 0..3u32 {
            let i = i0 ^ k;
            let base = i * BUCKET_BYTES;
            if self.t[ix(base)] == check {
                return base;
            }
            let n = st.total(self.t[ix(base + 1)]);
            if n < best_n {
                best_n = n;
                best = i;
            }
        }
        let base = best * BUCKET_BYTES;
        let b = ix(base);
        self.t[b..b + 16].fill(0);
        self.t[b] = check;
        base
    }

    /// Read the state at byte index `i`.
    #[inline(always)]
    pub fn get(&self, i: u32) -> u8 {
        self.t[ix(i)]
    }

    /// Write the state at byte index `i`.
    #[inline(always)]
    pub fn set(&mut self, i: u32, s: u8) {
        self.t[ix(i)] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_learn_toward_the_bit() {
        let r = Rates::new();
        let mut v = COUNTER_INIT;
        for _ in 0..50 {
            r.update(&mut v, 1, 255);
        }
        assert!(predict(v) > 3900, "p = {}", predict(v));
        for _ in 0..50 {
            r.update(&mut v, 0, 255);
        }
        assert!(predict(v) < 600, "p = {}", predict(v));
    }

    #[test]
    fn find_returns_the_same_bucket_for_the_same_hash() {
        let st = StateTable::new();
        let mut s = Slots::new(8);
        let a = s.find(0xdead_beef, &st);
        s.set(a + 3, 7);
        assert_eq!(s.find(0xdead_beef, &st), a);
        assert_eq!(s.get(a + 3), 7);
        let b = s.find(0x1234_5678, &st);
        assert_ne!(a, b);
    }

    #[test]
    fn statemap_starts_from_the_counts() {
        let st = StateTable::new();
        let sm = StateMap::new(&st);
        let mut s = 0u8;
        for _ in 0..10 {
            s = st.next(s, 1);
        }
        assert!(sm.predict(s) > 3500);
        assert!((sm.predict(0) - 2048).abs() < 8);
    }
}
