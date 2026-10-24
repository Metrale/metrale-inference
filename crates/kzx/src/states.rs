// SPDX-License-Identifier: MIT OR Apache-2.0

//! Bit histories as one-byte states. A state is a pair of bounded counts `(n0, n1)` of the bits
//! seen in a context; when one count grows the other is discounted, so the state follows a
//! changing source (nonstationary) and also says whether a context has been deterministic.
//! The table is generated once, identically on every target, from the rules below.

use crate::cast::ix;

/// Largest count a state carries.
const MAX_COUNT: u32 = 40;
/// Number of states (reachable pairs, see [`allowed`]).
pub const STATES: usize = 256;

/// Whether `(a, b)` with `a >= b` is a representable pair: the smaller count caps the larger.
const fn allowed(n0: u32, n1: u32) -> bool {
    let (a, b) = if n0 >= n1 { (n0, n1) } else { (n1, n0) };
    match b {
        0 | 1 => a <= MAX_COUNT,
        2 => a <= 20,
        3 => a <= 12,
        4 => a <= 8,
        5 => a <= 6,
        6 => a <= 6,
        _ => false,
    }
}

/// The state table: next state per bit, and the counts behind each state.
pub struct StateTable {
    next: [[u8; 2]; STATES],
    n0: [u8; STATES],
    n1: [u8; STATES],
}

impl StateTable {
    /// Generate the table.
    pub fn new() -> Self {
        // Enumerate allowed pairs in a fixed order; index 0 is (0, 0).
        let mut pairs = [(0u32, 0u32); STATES];
        let mut index = [[0u8; 41]; 41];
        let mut count = 0usize;
        let mut total = 0u32;
        while total <= MAX_COUNT * 2 && count < STATES {
            let mut n1 = 0u32;
            while n1 <= total && count < STATES {
                let n0 = total - n1;
                if n0 <= MAX_COUNT && n1 <= MAX_COUNT && allowed(n0, n1) {
                    pairs[count] = (n0, n1);
                    index[ix(n0)][ix(n1)] = u8::try_from(count).unwrap_or(0);
                    count += 1;
                }
                n1 += 1;
            }
            total += 1;
        }
        let mut next = [[0u8; 2]; STATES];
        let mut n0s = [0u8; STATES];
        let mut n1s = [0u8; STATES];
        for s in 0..count {
            let (n0, n1) = pairs[s];
            n0s[s] = u8::try_from(n0).unwrap_or(0);
            n1s[s] = u8::try_from(n1).unwrap_or(0);
            for (bit, slot) in next[s].iter_mut().enumerate() {
                let (mut same, mut other) = if bit == 1 { (n1, n0) } else { (n0, n1) };
                if same < MAX_COUNT {
                    same += 1;
                }
                if other > 3 {
                    other = 3 + (other - 3) / 3;
                }
                let (mut a, mut b) = if bit == 1 {
                    (other, same)
                } else {
                    (same, other)
                };
                let mut guard = 0;
                while !allowed(a, b) && guard < 64 {
                    if bit == 1 {
                        a = a.saturating_sub(1);
                    } else {
                        b = b.saturating_sub(1);
                    }
                    guard += 1;
                }
                *slot = index[ix(a)][ix(b)];
            }
        }
        // Unused slots (if any) loop to state 0.
        Self {
            next,
            n0: n0s,
            n1: n1s,
        }
    }

    /// The state after seeing `bit` in state `s`.
    #[inline(always)]
    pub fn next(&self, s: u8, bit: u32) -> u8 {
        self.next[ix(u32::from(s))][ix(bit & 1)]
    }

    /// Counts `(n0, n1)` of state `s`.
    #[inline(always)]
    pub fn counts(&self, s: u8) -> (u32, u32) {
        let i = ix(u32::from(s));
        (u32::from(self.n0[i]), u32::from(self.n1[i]))
    }

    /// Total count of `s`, used as a replacement priority.
    #[inline(always)]
    pub fn total(&self, s: u8) -> u32 {
        let (a, b) = self.counts(s);
        a + b
    }
}

impl Default for StateTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn states_are_closed_and_counts_move_with_the_bits() {
        let t = StateTable::new();
        let mut s = 0u8;
        for _ in 0..100 {
            s = t.next(s, 1);
        }
        let (n0, n1) = t.counts(s);
        assert_eq!(n0, 0);
        assert_eq!(n1, MAX_COUNT);
        s = t.next(s, 0);
        let (n0, n1) = t.counts(s);
        assert_eq!(n0, 1);
        assert!(n1 < MAX_COUNT, "opposite count discounted: {n1}");
        for st in 0..=255u8 {
            for bit in 0..2 {
                let n = t.next(st, bit);
                let (a, b) = t.counts(n);
                assert!(allowed(a, b), "state {st} bit {bit} -> ({a},{b})");
            }
        }
    }
}
