// SPDX-License-Identifier: MIT OR Apache-2.0

//! Hashed context statistics. A context hash selects a 64-byte bucket holding a 16-bit check
//! and 15 adaptive bit counters, one per node of the binary tree over one nibble, so each byte
//! costs two bucket lookups per context. Buckets are probed three deep and the least-used one
//! is recycled on a miss (the lpaq layout).
//!
//! A counter packs a 22-bit probability and a 10-bit hit count; the update step shrinks as the
//! count grows (`1/(n+1.5)`), so young contexts adapt fast and old ones average.

use crate::cast::{ix, ixi};
use alloc::vec::Vec;

/// Words per bucket: one check word and 15 counters.
pub const BUCKET_WORDS: u32 = 16;
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

/// One hashed table of nibble buckets.
pub struct Slots {
    t: Vec<u32>,
    mask: u32,
}

impl Slots {
    /// A table of `2^log2_buckets` buckets (`log2_buckets` ≤ 26).
    pub fn new(log2_buckets: u8) -> Self {
        let bits = u32::from(log2_buckets.min(26));
        let buckets = 1u32 << bits;
        Self {
            t: alloc::vec![COUNTER_INIT; ix(buckets * BUCKET_WORDS)],
            mask: buckets - 1,
        }
    }

    /// Bytes a table of `2^log2_buckets` buckets occupies.
    pub fn bytes(log2_buckets: u8) -> u64 {
        (1u64 << u32::from(log2_buckets.min(26))) * u64::from(BUCKET_WORDS) * 4
    }

    /// The word index of the bucket for hash `h`, allocating or recycling on a miss.
    #[inline]
    pub fn find(&mut self, h: u32) -> u32 {
        let check = h >> 16;
        let i0 = h & self.mask;
        let mut best = i0;
        let mut best_n = u32::MAX;
        for k in 0..3u32 {
            let i = i0 ^ k;
            let base = i * BUCKET_WORDS;
            let w0 = self.t[ix(base)];
            if w0 & 0xffff == check {
                return base;
            }
            let n = self.t[ix(base + 1)] & 1023;
            if n < best_n {
                best_n = n;
                best = i;
            }
        }
        let base = best * BUCKET_WORDS;
        self.t[ix(base)] = check;
        for j in 1..BUCKET_WORDS {
            self.t[ix(base + j)] = COUNTER_INIT;
        }
        base
    }

    /// Read the counter at word index `i`.
    #[inline(always)]
    pub fn get(&self, i: u32) -> u32 {
        self.t[ix(i)]
    }

    /// Update the counter at word index `i`.
    #[inline(always)]
    pub fn update(&mut self, i: u32, bit: u32, rates: &Rates, limit: u32) {
        rates.update(&mut self.t[ix(i)], bit, limit);
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
        let mut s = Slots::new(8);
        let a = s.find(0xdead_beef);
        s.update(a + 3, 1, &Rates::new(), 255);
        assert_eq!(s.find(0xdead_beef), a);
        assert_ne!(s.get(a + 3), COUNTER_INIT);
        let b = s.find(0x1234_5678);
        assert_ne!(a, b);
    }
}
