// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The model-level layer above the kernel contracts: a teacher-forced comparison of a
//! run's logits with a pinned reference on a fixed corpus (per-token KL, top-1 agreement, max
//! |Δlogit|), the composition of kernel bounds into per-layer budgets, and the attribution of a
//! failing model check to the first layer whose measured delta leaves its budget.
//!
//! The producer of the logits (an HTTP prompt-logprob harness, a hidden-state dump of one
//! forward) is the caller's: this module only judges, so every model check shares one
//! definition of each metric and one record.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A comparison names the corpus and the reference by SHA-256; logits of another corpus or
//!   another reference are refused, never compared.
//! - Log-softmax is computed in f64 with the max subtracted; KL uses the reference as `p`.
//! - A budget is built only from measured amplifications (`Amplification::measured_by` names
//!   the record); attribution without per-layer deltas says it is unavailable.

/// 2026-10-09: The pinned inputs of a model check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pins {
    /// 2026-10-09: SHA-256 of the corpus token ids (little-endian u32, in order).
    pub corpus_sha256: String,
    /// 2026-10-09: SHA-256 of the reference logits file (or of its declared digest source).
    pub reference_sha256: String,
}

/// 2026-10-09: Per-token metrics of one sequence position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenMetrics {
    /// 2026-10-09: KL(reference ‖ run) in nats.
    pub kl: f64,
    /// 2026-10-09: The two argmaxes agree.
    pub top1: bool,
    /// 2026-10-09: Largest |run − reference| over the vocabulary.
    pub max_dlogit: f64,
}

/// 2026-10-09: Summary of a teacher-forced comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Summary {
    /// 2026-10-09: Positions compared.
    pub tokens: usize,
    /// 2026-10-09: Mean KL.
    pub mean_kl: f64,
    /// 2026-10-09: Largest KL.
    pub max_kl: f64,
    /// 2026-10-09: Share of positions whose argmaxes agree.
    pub top1: f64,
    /// 2026-10-09: Largest |Δlogit| over every position.
    pub max_dlogit: f64,
}

/// 2026-10-09: A comparison that cannot be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ModelCheckError {
    /// 2026-10-09: The run was produced on another corpus or against another reference.
    #[error("pins differ: run {run:?}, reference {reference:?}")]
    Pins {
        /// 2026-10-09: The run's pins.
        run: String,
        /// 2026-10-09: The expected pins.
        reference: String,
    },
    /// 2026-10-09: Shapes differ, or there is nothing to compare.
    #[error("{0}")]
    Shape(String),
    /// 2026-10-09: A logit is not finite.
    #[error("non-finite logit at position {0}")]
    NonFinite(usize),
}

fn log_softmax(z: &[f64]) -> Vec<f64> {
    let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let lse = m + z.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
    z.iter().map(|v| v - lse).collect()
}

fn argmax(z: &[f64]) -> usize {
    // 2026-10-09: The first maximum, as greedy sampling picks it.
    z.iter()
        .enumerate()
        .fold(0, |b, (i, v)| if *v > z[b] { i } else { b })
}

/// 2026-10-09: Metrics of one position: `run` and `reference` are the logits over the vocabulary.
pub fn token(run: &[f64], reference: &[f64]) -> TokenMetrics {
    let (lr, lp) = (log_softmax(run), log_softmax(reference));
    let kl = lp
        .iter()
        .zip(&lr)
        .map(|(p, q)| p.exp() * (p - q))
        .sum::<f64>()
        .max(0.0);
    let max_dlogit = run
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    TokenMetrics {
        kl,
        top1: argmax(run) == argmax(reference),
        max_dlogit,
    }
}

/// 2026-10-09: Compare `run` with `reference`, both `[positions, vocab]` row-major, under the
/// same pins.
pub fn compare(
    run: &[f64],
    reference: &[f64],
    vocab: usize,
    run_pins: &Pins,
    ref_pins: &Pins,
) -> Result<(Vec<TokenMetrics>, Summary), ModelCheckError> {
    if run_pins != ref_pins {
        return Err(ModelCheckError::Pins {
            run: format!("{run_pins:?}"),
            reference: format!("{ref_pins:?}"),
        });
    }
    if vocab == 0 || run.is_empty() || run.len() != reference.len() || run.len() % vocab != 0 {
        return Err(ModelCheckError::Shape(format!(
            "{} run and {} reference logits for vocab {vocab}",
            run.len(),
            reference.len()
        )));
    }
    let mut per = Vec::with_capacity(run.len() / vocab);
    for (i, (r, p)) in run.chunks(vocab).zip(reference.chunks(vocab)).enumerate() {
        if r.iter().chain(p).any(|v| !v.is_finite()) {
            return Err(ModelCheckError::NonFinite(i));
        }
        per.push(token(r, p));
    }
    let n = per.len() as f64;
    let s = Summary {
        tokens: per.len(),
        mean_kl: per.iter().map(|t| t.kl).sum::<f64>() / n,
        max_kl: per.iter().map(|t| t.kl).fold(0.0, f64::max),
        top1: per.iter().filter(|t| t.top1).count() as f64 / n,
        max_dlogit: per.iter().map(|t| t.max_dlogit).fold(0.0, f64::max),
    };
    Ok((per, s))
}

/// 2026-10-09: How much a stage amplifies an input perturbation (relative, sup-norm), measured.
#[derive(Debug, Clone, PartialEq)]
pub struct Amplification {
    /// 2026-10-09: The factor.
    pub factor: f64,
    /// 2026-10-09: The record that measured it (never an assumed constant).
    pub measured_by: String,
}

/// 2026-10-09: One stage of a layer chain: the relative error its kernels may add (the
/// contracts' derived bounds, relative to the stage output) and how it amplifies what it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct Stage {
    /// 2026-10-09: Name (`l3.attn`, `l3.ffn`).
    pub name: String,
    /// 2026-10-09: The relative error the stage's kernels add.
    pub eps: f64,
    /// 2026-10-09: Its measured amplification.
    pub amplification: Amplification,
}

/// 2026-10-09: The composed budget after each stage: `d_i = L_i * d_{i-1} + eps_i`, so the last
/// is `sum_i (prod_{j>i} L_j) eps_i`.
pub fn compose(chain: &[Stage]) -> Vec<f64> {
    let mut d = 0.0;
    chain
        .iter()
        .map(|s| {
            d = s.amplification.factor * d + s.eps;
            d
        })
        .collect()
}

/// 2026-10-09: The attribution of a model-check failure.
#[derive(Debug, Clone, PartialEq)]
pub enum Attribution {
    /// 2026-10-09: The first stage whose measured delta exceeds its composed budget, with the
    /// delta and the budget.
    Stage {
        /// 2026-10-09: Stage name.
        name: String,
        /// 2026-10-09: Measured relative delta.
        delta: f64,
        /// 2026-10-09: Composed budget.
        budget: f64,
    },
    /// 2026-10-09: Every stage is inside its budget: the deviation is explained by the
    /// contracts (look at the head, the sampler or the reference).
    WithinBudgets,
    /// 2026-10-09: The producer gave no per-stage deltas.
    Unavailable,
}

/// 2026-10-09: Attribute a failure: `deltas[i]` is the measured relative delta of stage `i`'s
/// output (run vs reference hidden states), empty when the producer has none.
pub fn attribute(chain: &[Stage], deltas: &[f64]) -> Attribution {
    if deltas.is_empty() {
        return Attribution::Unavailable;
    }
    let budgets = compose(chain);
    for ((s, d), b) in chain.iter().zip(deltas).zip(budgets) {
        if *d > b || !d.is_finite() {
            return Attribution::Stage {
                name: s.name.clone(),
                delta: *d,
                budget: b,
            };
        }
    }
    Attribution::WithinBudgets
}

#[cfg(test)]
#[path = "model_check_tests.rs"]
mod tests;
