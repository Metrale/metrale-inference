// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The expert-union curve from recorded routes. A decode step at concurrency `C`
//! with `k` draft tokens verifies `C` sequences of `k + 1` consecutive rows each, and one
//! layer's MoE reads every routed expert those rows select; the curve is the mean count of
//! distinct experts per layer, next to the count uniform routing would give.
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants:
//! - Routes are `[layers][tokens][topk]` expert ids, each below `n_experts`.
//! - The `C` windows of one placement are spaced `tokens / C` apart; the mean runs over every
//!   layer and every placement offset, so the result is deterministic.

use metrale_error::{Error, ErrorCode, error};

/// 2026-10-19: Recorded routes of one prompt.
#[derive(Clone, Debug, PartialEq)]
pub struct Routes {
    layers: usize,
    tokens: usize,
    topk: usize,
    n_experts: usize,
    ids: Vec<u16>,
}

fn refuse(why: String) -> Error {
    error!(ErrorCode::InvalidArgument, "expert routes", why)
}

impl Routes {
    pub fn new(
        layers: usize,
        tokens: usize,
        topk: usize,
        n_experts: usize,
        ids: Vec<u16>,
    ) -> Result<Self, Error> {
        let want = layers * tokens * topk;
        if want == 0 || ids.len() != want {
            return Err(refuse(format!(
                "{} ids for {layers} layers x {tokens} tokens x {topk}",
                ids.len()
            )));
        }
        if let Some(&bad) = ids.iter().find(|&&e| usize::from(e) >= n_experts) {
            return Err(refuse(format!("expert id {bad} of {n_experts}")));
        }
        Ok(Self {
            layers,
            tokens,
            topk,
            n_experts,
            ids,
        })
    }

    /// 2026-10-19: Mean distinct experts one layer reads for `c` sequences of `k + 1` rows.
    pub fn union_mean(&self, c: usize, k: usize) -> Result<f64, Error> {
        let rows = k + 1;
        let stride = if c == 0 { 0 } else { self.tokens / c };
        if stride < rows {
            return Err(refuse(format!(
                "{} tokens hold no {c} windows of {rows}",
                self.tokens
            )));
        }
        let mut seen = vec![false; self.n_experts];
        let (mut sum, mut n) = (0usize, 0usize);
        for l in 0..self.layers {
            let layer = &self.ids[l * self.tokens * self.topk..(l + 1) * self.tokens * self.topk];
            for off in 0..=stride - rows {
                seen.fill(false);
                for w in 0..c {
                    let t0 = off + w * stride;
                    for &e in &layer[t0 * self.topk..(t0 + rows) * self.topk] {
                        seen[usize::from(e)] = true;
                    }
                }
                sum += seen.iter().filter(|&&s| s).count();
                n += 1;
            }
        }
        Ok(sum as f64 / n as f64)
    }
}

/// 2026-10-19: Expected distinct experts for `rows` rows each picking `topk` of `n_experts`
/// uniformly: `E (1 - (1 - K / E)^R)`.
pub fn uniform_union(n_experts: usize, topk: usize, rows: usize) -> f64 {
    let e = n_experts as f64;
    e * (1.0 - (1.0 - topk as f64 / e).powi(rows as i32))
}

#[cfg(test)]
#[path = "routes_tests.rs"]
mod tests;
