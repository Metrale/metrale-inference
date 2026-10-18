// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The bitshift trellis and its Viterbi search (arXiv:2406.11235 — trellis (QTIP);
//! layout as in spiritbuun/buun-llama-cpp, MIT; credit spiritbuun).
//!
//! A state is the last `L` bits of the stream. Each step shifts `k` new bits in at the top:
//! `next = (prev >> k) | (symbol << (L - k))`, so a state's predecessors are the `2^k` states
//! `((s & low) << k) | p` with `low = 2^(L-k) - 1`. Every state with the same `s & low` (its
//! *group*) shares those predecessors. The value a state reconstructs is `codebook[state]`.
//!
//! Owner: kv-tq128.
//! Invariants (the kernels reproduce each exactly):
//! - The initial state is free: every state's cost starts at 0.
//! - `cost[s] = best[s & low] + d * d` with `d = x - codebook[s]`, no fused multiply-add.
//! - `best[g]` scans `p = 0, 1, ..` with a strict `<`: the lowest `p` wins a tie.
//! - The final state is the lowest index among the minimum costs.

use alloc::vec;
use alloc::vec::Vec;

use crate::Tq128Error;

/// 2026-10-18: A bitshift trellis with `k` bits per step and `L`-bit states.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Trellis {
    k: u32,
    l: u32,
}

/// 2026-10-18: A search result: the state after every step, and the path's total squared error.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub states: Vec<u16>,
    pub cost: f32,
}

impl Trellis {
    /// 2026-10-18: turbo3_tcq: 3 bits per step, 512 states.
    pub const TCQ3: Self = Self { k: 3, l: 9 };
    /// 2026-10-18: turbo2_tcq: 2 bits per step, 256 states.
    pub const TCQ2: Self = Self { k: 2, l: 8 };
    /// 2026-10-18: turbo1_tcq: 1 bit per step, 256 states.
    pub const TCQ1: Self = Self { k: 1, l: 8 };

    /// 2026-10-18: Any shape the 16-bit window read can carry: `1 <= k < L <= 9`.
    pub const fn new(k: u32, l: u32) -> Result<Self, Tq128Error> {
        if k == 0 || k >= l || l > 9 {
            return Err(Tq128Error::TrellisShape { k, l });
        }
        Ok(Self { k, l })
    }

    /// 2026-10-18: Bits per step.
    pub const fn k(self) -> u32 {
        self.k
    }

    /// 2026-10-18: Bits per state.
    pub const fn l(self) -> u32 {
        self.l
    }

    /// 2026-10-18: `2^L`.
    pub const fn states(self) -> usize {
        1 << self.l
    }

    /// 2026-10-18: `2^(L-k)`: groups of states sharing predecessors.
    pub const fn groups(self) -> usize {
        1 << (self.l - self.k)
    }

    /// 2026-10-18: `2^k`: predecessors (and successors) per state.
    pub const fn fan(self) -> usize {
        1 << self.k
    }

    /// 2026-10-18: Bits of a `steps`-symbol stream: the first state's low `L - k` bits, then
    /// `k` bits per step.
    pub const fn stream_bits(self, steps: usize) -> usize {
        (self.l - self.k) as usize + self.k as usize * steps
    }

    /// 2026-10-18: The state reached from `prev` by emitting `symbol`.
    pub const fn next(self, prev: usize, symbol: usize) -> usize {
        (prev >> self.k) | (symbol << (self.l - self.k))
    }

    /// 2026-10-18: Predecessor `p` of state `s`.
    pub const fn predecessor(self, s: usize, p: usize) -> usize {
        ((s & (self.groups() - 1)) << self.k) | p
    }

    /// 2026-10-18: The symbol that entered state `s`.
    pub const fn symbol(self, s: usize) -> usize {
        s >> (self.l - self.k)
    }
}

/// 2026-10-18: The minimum squared-error path of `x` through `trellis` over `codebook`, with
/// the tie rules in the module invariants. `x` and `codebook` must be finite (the encoder
/// checks); an empty `x` gives an empty path of cost 0.
pub fn viterbi(trellis: Trellis, codebook: &[f32], x: &[f32]) -> Result<Path, Tq128Error> {
    let n = trellis.states();
    if codebook.len() != n {
        return Err(Tq128Error::CodebookLength {
            expected: n,
            found: codebook.len(),
        });
    }
    if x.is_empty() {
        return Ok(Path {
            states: Vec::new(),
            cost: 0.0,
        });
    }
    let (g_count, fan) = (trellis.groups(), trellis.fan());
    let mut cost = vec![0.0_f32; n];
    let mut next = vec![0.0_f32; n];
    let mut best = vec![0.0_f32; g_count];
    // back[t * g_count + g]: the predecessor index p chosen for group g at step t (t >= 1).
    let mut back = vec![0_u8; x.len() * g_count];
    for (t, &xt) in x.iter().enumerate() {
        for (g, b) in best.iter_mut().enumerate() {
            let mut m = cost[g << trellis.k];
            let mut arg = 0_u8;
            for (p, pa) in (1..fan).zip(1_u8..) {
                let c = cost[(g << trellis.k) | p];
                if c < m {
                    m = c;
                    arg = pa;
                }
            }
            *b = m;
            back[t * g_count + g] = arg;
        }
        for (s, (c, &cb)) in next.iter_mut().zip(codebook).enumerate() {
            let d = xt - cb;
            let sq = d * d;
            *c = best[s & (g_count - 1)] + sq;
        }
        core::mem::swap(&mut cost, &mut next);
    }
    let mut end = 0;
    for (s, &c) in cost.iter().enumerate() {
        if c < cost[end] {
            end = s;
        }
    }
    let total = cost[end];
    let mut states = vec![0_u16; x.len()];
    let mut s = end;
    for t in (0..x.len()).rev() {
        states[t] = u16::try_from(s).unwrap_or(u16::MAX);
        let g = s & (g_count - 1);
        s = (g << trellis.k) | usize::from(back[t * g_count + g]);
    }
    Ok(Path {
        states,
        cost: total,
    })
}

/// 2026-10-18: The squared error of a state sequence, accumulated in the order Viterbi does
/// (`((0 + d0^2) + d1^2) + ..`): the brute-force oracle's cost function.
pub fn path_cost(codebook: &[f32], x: &[f32], states: &[u16]) -> f32 {
    let mut acc = 0.0_f32;
    for (&xt, &s) in x.iter().zip(states) {
        let d = xt - codebook[usize::from(s)];
        let sq = d * d;
        acc += sq;
    }
    acc
}

/// 2026-10-18: Whether `states` is a walk of `trellis` (each state a successor of the last).
pub fn is_walk(trellis: Trellis, states: &[u16]) -> bool {
    states.windows(2).all(|w| {
        let (a, b) = (usize::from(w[0]), usize::from(w[1]));
        (0..trellis.fan()).any(|p| trellis.predecessor(b, p) == a)
    })
}
