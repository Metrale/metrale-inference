// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The measurement order. Every load window is flanked by idle windows (A-L-B), so
//! its static power is interpolated from both sides and a linear drift (warming, baseline creep)
//! cancels in the dynamic energy. With several arms (kernels or A/B variants) the load order
//! across repeats is ABBA (forward, then mirrored: 0 1 1 0 …), which cancels a linear drift in
//! the arm comparison as well, or a seeded random permutation per repeat (cancels drift in
//! expectation and any periodic structure); the seed is stored. Adjacent loads share the idle
//! window between them.
//!
//! Owner: metrale-stats.
//! Invariants: an order is a pure function of (arms, repeats, kind).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::rng::SplitMix64;

/// 2026-10-22: Idle or load (a single-arm step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Arm {
    Idle,
    Load,
}

/// 2026-10-22: One window of an order: idle, or a load of arm `i`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Idle,
    Load(u16),
}

/// 2026-10-22: How the loads are arranged across repeats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderKind {
    Abba,
    Random,
}

/// 2026-10-22: A full window order and how it was made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Order {
    pub seed: Option<u64>,
    pub kind: OrderKind,
    pub arms: Vec<Step>,
}

/// 2026-10-22: The two-arm ABBA pattern over idle and load for `pairs` pairs: I L L I …
#[must_use]
pub fn abba(pairs: usize) -> Vec<Arm> {
    (0..pairs)
        .flat_map(|_| [Arm::Idle, Arm::Load, Arm::Load, Arm::Idle])
        .collect()
}

impl Order {
    /// 2026-10-22: `repeats` visits of each of `arms` loads, ABBA-arranged, every load flanked.
    #[must_use]
    pub fn abba(arms: u16, repeats: usize) -> Self {
        let fwd: Vec<u16> = (0..arms).collect();
        let loads = (0..repeats).flat_map(|r| {
            let mut v = fwd.clone();
            if r % 2 == 1 {
                v.reverse();
            }
            v
        });
        Self {
            seed: None,
            kind: OrderKind::Abba,
            arms: flank(loads),
        }
    }

    /// 2026-10-22: As [`Order::abba`] but each repeat's arm order a seeded Fisher-Yates shuffle.
    #[must_use]
    pub fn random(arms: u16, repeats: usize, seed: u64) -> Self {
        let mut r = SplitMix64::new(seed);
        let loads: Vec<u16> = (0..repeats)
            .flat_map(|_| {
                let mut v: Vec<u16> = (0..arms).collect();
                for i in (1..v.len()).rev() {
                    v.swap(i, r.below(i + 1));
                }
                v
            })
            .collect();
        Self {
            seed: Some(seed),
            kind: OrderKind::Random,
            arms: flank(loads),
        }
    }

    /// 2026-10-22: The stored spelling of the kind: `abba` or `random:seed=<n>`.
    #[must_use]
    pub fn label(&self) -> String {
        match (self.kind, self.seed) {
            (OrderKind::Random, Some(s)) => format!("random:seed={s}"),
            (OrderKind::Random, None) => String::from("random"),
            (OrderKind::Abba, _) => String::from("abba"),
        }
    }

    /// 2026-10-22: The whole sequence, e.g. `I L0 I L1 I L1 I L0 I`.
    #[must_use]
    pub fn render(&self) -> String {
        let parts: Vec<String> = self
            .arms
            .iter()
            .map(|s| match s {
                Step::Idle => String::from("I"),
                Step::Load(i) => format!("L{i}"),
            })
            .collect();
        parts.join(" ")
    }
}

fn flank(loads: impl IntoIterator<Item = u16>) -> Vec<Step> {
    let mut out = Vec::from([Step::Idle]);
    for l in loads {
        out.push(Step::Load(l));
        out.push(Step::Idle);
    }
    out
}

#[cfg(test)]
#[path = "design_tests.rs"]
mod tests;
