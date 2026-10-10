// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The one formula for what a chain of drafts is worth:
//! `E[tokens | K] = 1 + sum_{i=1..K} prod_{j=1..i} c_j`, where `c_j` is the probability that
//! draft `j` is accepted given drafts `1..j` were (a conditional rate). Every K decision in the
//! engine (the MTP rung, the measured planner, the per-stream DFlash count, the replay) derives
//! from these functions; none restates the product.
//!
//! Owner: speculative.
//! Invariants:
//! - [`chain_sum`] multiplies and adds in draft order, so a caller that summed the same terms
//!   in the same order before gets bit-identical results.
//! - A rate function is read only at positions `1..=k`.

/// 2026-10-10: Positions a rate array carries; deeper drafts share the last position's rate.
pub const MAX_POSITIONS: usize = 16;

/// 2026-10-10: `start + sum_{i=1..k} prod_{j=1..i} rate(j)`: the running products of the
/// conditional rates added to `start` in draft order. `start` 1 gives expected emitted tokens,
/// 0 expected accepted drafts.
pub fn chain_sum(rate: impl Fn(usize) -> f64, k: usize, start: f64) -> f64 {
    let mut run = 1.0;
    let mut total = start;
    for j in 1..=k {
        run *= rate(j);
        total += run;
    }
    total
}

/// 2026-10-10: The rate of 1-based position `j` in `c`; positions past the end take the last.
/// An empty slice has no acceptance (0).
pub fn at(c: &[f64], j: usize) -> f64 {
    match c.len() {
        0 => 0.0,
        n => c[(j - 1).min(n - 1)],
    }
}

/// 2026-10-10: Expected emitted tokens of a verify of `k` drafts at conditional rates `c`.
pub fn expected_tokens(c: &[f64], k: usize) -> f64 {
    chain_sum(|j| at(c, j), k, 1.0)
}

/// 2026-10-10: Expected accepted drafts of a verify of `k` drafts at conditional rates `c`.
pub fn expected_accepted(c: &[f64], k: usize) -> f64 {
    chain_sum(|j| at(c, j), k, 0.0)
}

/// 2026-10-10: The marginal expected-token gain of draft `j` (1-based): `prod_{i<=j} c_i`.
/// Non-increasing in `j` because every factor is at most 1.
pub fn marginal_gain(c: &[f64], j: usize) -> f64 {
    (1..=j).fold(1.0, |run, i| run * at(c, i))
}

/// 2026-10-10: Conditional rates from marginal ones (`P(at least j accepted)`), the form most
/// acceptance reports use: `c_j = m_j / m_{j-1}` with `m_0 = 1`; a position after a zero
/// marginal takes 0.
pub fn conditional_from_marginal(m: &[f64]) -> Vec<f64> {
    let mut prev = 1.0;
    m.iter()
        .map(|&x| {
            let c = if prev > 0.0 {
                (x / prev).clamp(0.0, 1.0)
            } else {
                0.0
            };
            prev = x;
            c
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expected_tokens_is_one_plus_the_survival_sum() {
        let c = [0.6, 0.5, 0.4];
        assert_eq!(expected_tokens(&c, 0), 1.0);
        assert!((expected_tokens(&c, 1) - 1.6).abs() < 1e-12);
        assert!((expected_tokens(&c, 2) - 1.9).abs() < 1e-12);
        assert!((expected_tokens(&c, 3) - 2.02).abs() < 1e-12);
        // 2026-10-10: Past the end the last rate repeats: 2.02 + 0.12 * 0.4.
        assert!((expected_tokens(&c, 4) - 2.068).abs() < 1e-12);
        assert_eq!(expected_accepted(&c, 0), 0.0);
        assert!((expected_accepted(&c, 3) - 1.02).abs() < 1e-12);
        assert_eq!(expected_tokens(&[], 3), 1.0);
    }

    /// 2026-10-10: Each draft's gain is the running product; the gains sum to E - 1 and never
    /// increase with depth.
    #[test]
    fn marginal_gains_are_the_running_products_and_never_increase() {
        let c = [0.9, 0.7, 1.0, 0.2];
        let mut last = 1.0;
        let mut sum = 0.0;
        for j in 1..=6 {
            let g = marginal_gain(&c, j);
            assert!(g <= last, "gain rose at {j}");
            last = g;
            sum += g;
        }
        assert!((sum - expected_accepted(&c, 6)).abs() < 1e-12);
        assert!((marginal_gain(&c, 2) - 0.63).abs() < 1e-12);
    }

    #[test]
    fn marginal_reports_convert_to_conditional_rates() {
        let c = conditional_from_marginal(&[0.6, 0.3, 0.0, 0.0]);
        assert!((c[0] - 0.6).abs() < 1e-12 && (c[1] - 0.5).abs() < 1e-12);
        assert_eq!(&c[2..], &[0.0, 0.0]);
        let m = [0.611, 0.312, 0.117];
        let c = conditional_from_marginal(&m);
        assert!((expected_tokens(&c, 3) - (1.0 + 0.611 + 0.312 + 0.117)).abs() < 1e-12);
    }
}
