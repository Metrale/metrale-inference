// SPDX-License-Identifier: MIT OR Apache-2.0

//! Logistic mixing and secondary estimation. The [`Mixer`] combines stretched predictions with
//! a weight vector selected by a small context and trains it online by gradient descent on
//! coding cost; the [`Apm`] refines a probability by interpolating a learned curve indexed by a
//! context (the lpaq SSE stage).

use crate::cast::{ix, ixi};
use crate::tables::{Stretch, squash};
use alloc::vec::Vec;

/// The most inputs one mixer accepts.
pub const MAX_INPUTS: usize = 16;

/// A gated linear mixer in the logistic domain.
pub struct Mixer {
    n: usize,
    w: Vec<i32>,
    x: [i32; MAX_INPUTS],
    k: usize,
    base: usize,
    pr: i32,
    lr: i32,
}

impl Mixer {
    /// A mixer over `n` inputs with `contexts` weight sets, learning rate `lr`
    /// (about 2–12; higher adapts faster) and initial weight `w0` (16.16 fixed point).
    pub fn new(n: usize, contexts: u32, lr: i32, w0: i32) -> Self {
        let n = n.min(MAX_INPUTS);
        Self {
            n,
            w: alloc::vec![w0; ix(contexts) * n],
            x: [0; MAX_INPUTS],
            k: 0,
            base: 0,
            pr: 2048,
            lr,
        }
    }

    /// Append one stretched input (clamped to ±2047).
    #[inline(always)]
    pub fn add(&mut self, st: i32) {
        if self.k < self.n {
            self.x[self.k] = st.clamp(-2047, 2047);
            self.k += 1;
        }
    }

    /// Select the weight set for this bit.
    #[inline(always)]
    pub fn set_context(&mut self, c: u32) {
        self.base = ix(c) * self.n;
    }

    /// Mix the inputs added since the last update into a 12-bit probability.
    #[inline]
    pub fn mix(&mut self) -> i32 {
        let mut dot: i64 = 0;
        let w = &self.w[self.base..self.base + self.n];
        for (xi, wi) in self.x[..self.n].iter().zip(w) {
            dot += i64::from(*xi) * i64::from(*wi);
        }
        let d = i32::try_from(dot >> 16).unwrap_or(0).clamp(-2047, 2047);
        self.pr = squash(d);
        self.pr
    }

    /// Train toward `bit` and clear the inputs.
    #[inline]
    pub fn update(&mut self, bit: u32) {
        let target = i32::try_from(bit << 12).unwrap_or(0);
        let err = (target - self.pr) * self.lr;
        let w = &mut self.w[self.base..self.base + self.n];
        for (xi, wi) in self.x[..self.n].iter().zip(w.iter_mut()) {
            let delta = (xi.wrapping_mul(err) + 0x8000) >> 16;
            *wi = wi.saturating_add(delta);
        }
        self.k = 0;
    }
}

/// An adaptive probability map: 24 interpolation bins per context.
pub struct Apm {
    t: Vec<u16>,
    index: usize,
}

impl Apm {
    /// A map over `contexts` contexts, initialised to the identity curve.
    pub fn new(contexts: u32) -> Self {
        let n = ix(contexts) * 24;
        let mut t = alloc::vec![0u16; n];
        for (i, slot) in t.iter_mut().enumerate() {
            let j = i32::try_from(i % 24).unwrap_or(0);
            let p = squash((j - 12) * 178) * 16;
            *slot = u16::try_from(p.clamp(0, 65535)).unwrap_or(32768);
        }
        Self { t, index: 0 }
    }

    /// Refine 12-bit `pr` under context `cx`; remember the bin for [`Apm::update`].
    #[inline]
    pub fn refine(&mut self, pr: i32, cx: u32, stretch: &Stretch) -> i32 {
        let s = (stretch.stretch(pr) + 2048) * 23;
        let wt = s & 0xfff;
        let bin = ixi(s >> 12);
        let cxt = ix(cx) * 24 + bin.min(22);
        self.index = cxt + ixi(wt >> 11);
        let lo = i32::from(self.t[cxt]);
        let hi = i32::from(self.t[cxt + 1]);
        (lo * (4096 - wt) + hi * wt) >> 16
    }

    /// Move the remembered bin toward `bit` with rate `1/2^rate`.
    #[inline]
    pub fn update(&mut self, bit: u32, rate: u32) {
        let y = i32::try_from(bit).unwrap_or(0);
        let g = (y << 16) + (y << rate) - y - y;
        let cur = i32::from(self.t[self.index]);
        let next = cur + ((g - cur) >> rate);
        self.t[self.index] = u16::try_from(next.clamp(0, 65535)).unwrap_or(32768);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixer_learns_to_trust_the_informative_input() {
        let st = Stretch::new();
        let mut m = Mixer::new(3, 1, 6, 0);
        let mut s = 7u32;
        let mut late_cost = 0i64;
        for i in 0..4000 {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let bit = (s >> 16) & 1;
            m.set_context(0);
            m.add(st.stretch(if bit == 1 { 3800 } else { 300 }));
            m.add(st.stretch(2048 + i32::try_from((s >> 8) & 1023).unwrap_or(0) - 512));
            m.add(256);
            let p = m.mix();
            if i >= 3000 {
                let p1 = if bit == 1 { p } else { 4096 - p };
                late_cost += i64::from(4096 - p1);
            }
            m.update(bit);
        }
        assert!(late_cost / 1000 < 700, "late cost {late_cost}");
    }

    #[test]
    fn apm_moves_toward_observed_bits() {
        let st = Stretch::new();
        let mut a = Apm::new(1);
        let before = a.refine(2048, 0, &st);
        for _ in 0..200 {
            a.refine(2048, 0, &st);
            a.update(1, 6);
        }
        let after = a.refine(2048, 0, &st);
        assert!(after > before + 500, "{before} -> {after}");
    }
}
