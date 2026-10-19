// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Tier-2 parity between two logit tensors of the same positions: the
//! reference's (the CPU path) and a candidate's (the GPU path). Per position it reports the
//! KL divergence of the candidate's distribution from the reference's and whether their
//! arg-max ids agree.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants:
//! - Softmax and KL are computed in f64 with the max subtracted; a row holding a non-finite
//!   logit is refused, never scored.
//! - Ties in the arg-max go to the lowest id, the same rule the CPU driver's greedy uses.

use metrale_error::{Error, ErrorCode, error};

/// 2026-10-19: One position's score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PositionParity {
    /// KL(reference || candidate) in nats.
    pub kl: f64,
    pub reference_top1: usize,
    pub candidate_top1: usize,
}

/// 2026-10-19: Every position's score and their summary.
#[derive(Clone, Debug, PartialEq)]
pub struct Parity {
    pub positions: Vec<PositionParity>,
}

impl Parity {
    pub fn mean_kl(&self) -> f64 {
        self.positions.iter().map(|p| p.kl).sum::<f64>() / self.positions.len().max(1) as f64
    }

    pub fn max_kl(&self) -> f64 {
        self.positions.iter().map(|p| p.kl).fold(0.0, f64::max)
    }

    pub fn top1_matches(&self) -> usize {
        self.positions
            .iter()
            .filter(|p| p.reference_top1 == p.candidate_top1)
            .count()
    }
}

/// 2026-10-19: The lowest id holding the row's largest logit.
pub fn top1(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best
}

fn log_softmax(row: &[f32]) -> Vec<f64> {
    let m = row
        .iter()
        .fold(f64::NEG_INFINITY, |a, &v| a.max(f64::from(v)));
    let lse = m + row
        .iter()
        .map(|&v| (f64::from(v) - m).exp())
        .sum::<f64>()
        .ln();
    row.iter().map(|&v| f64::from(v) - lse).collect()
}

/// 2026-10-19: Score `candidate` against `reference`, both `[positions, vocab]` row-major.
pub fn score(reference: &[f32], candidate: &[f32], vocab: usize) -> Result<Parity, Error> {
    let refuse = |why: String| error!(ErrorCode::InvalidArgument, "parity logits", why);
    if vocab == 0 || reference.len() != candidate.len() || !reference.len().is_multiple_of(vocab) {
        return Err(refuse(format!(
            "{} reference and {} candidate logits do not form rows of {vocab}",
            reference.len(),
            candidate.len()
        )));
    }
    let positions = reference
        .chunks_exact(vocab)
        .zip(candidate.chunks_exact(vocab))
        .enumerate()
        .map(|(p, (r, c))| {
            if let Some(i) = r.iter().chain(c).position(|v| !v.is_finite()) {
                return Err(refuse(format!(
                    "position {p}: non-finite logit at flat index {i}"
                )));
            }
            let (lr, lc) = (log_softmax(r), log_softmax(c));
            let kl = lr
                .iter()
                .zip(&lc)
                .map(|(&a, &b)| a.exp() * (a - b))
                .sum::<f64>()
                .max(0.0);
            Ok(PositionParity {
                kl,
                reference_top1: top1(r),
                candidate_top1: top1(c),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(Parity { positions })
}

#[cfg(test)]
#[path = "parity_tests.rs"]
mod tests;
