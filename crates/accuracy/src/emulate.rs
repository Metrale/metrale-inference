// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: A conforming f32 emulation of a declared pipeline: the arithmetic a kernel that
//! obeys the contract performs, in one fixed bracketing no deeper than the declared depth. It
//! serves three purposes: the noise floor of a calibration (what any legitimate implementation
//! scores), the `accumulate:<fmt>` mutation arm (the same emulation with a narrower
//! accumulator), and the CPU runner the crate's own tests prove both arms with.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - [`reduce`] never brackets deeper than the depth it is given, so a conforming emulation
//!   stays inside the derived bound by construction (the bound tests check it).

use crate::elem::Elem;

/// 2026-10-09: The bracketings an emulation can use; the noise floor is the worst of them.
pub const VARIANTS: u32 = 3;

/// 2026-10-09: Sum `xs` rounding into `acc` after every add, with total depth at most `depth`
/// (when `depth` allows any bracketing). Variant 0: the longest sequential runs the depth
/// leaves, joined in a balanced tree; 1: the same over the reversed terms; 2: a balanced
/// pairwise tree.
pub fn reduce(xs: &[f64], acc: Elem, depth: u64, variant: u32) -> f64 {
    let rnd = |v: f64| acc.round_saturating(v).unwrap_or(f64::NAN);
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as u64;
    let tree = u64::from(64 - (n.max(1) - 1).leading_zeros());
    // 2026-10-09: The longest sequential run the depth leaves after the tree above it.
    let run = match variant % VARIANTS {
        2 => 1,
        _ => depth.saturating_sub(tree).max(1).min(n) as usize,
    };
    let ordered: Vec<f64> = if variant % VARIANTS == 1 {
        xs.iter().rev().copied().collect()
    } else {
        xs.to_vec()
    };
    let mut partials: Vec<f64> = ordered
        .chunks(run)
        .map(|c| c.iter().fold(0.0, |s, &x| rnd(s + x)))
        .collect();
    while partials.len() > 1 {
        partials = partials
            .chunks(2)
            .map(|p| if p.len() == 2 { rnd(p[0] + p[1]) } else { p[0] })
            .collect();
    }
    partials[0]
}

#[cfg(test)]
#[path = "emulate_tests.rs"]
mod tests;
