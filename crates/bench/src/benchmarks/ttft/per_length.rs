// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: One median per requested prompt length, so a BENCH.toml entry can bound each
//! length on its own (`median_ms_<tokens>`), where `median_ms` pools every length into one
//! number. The GLM-5.3 TTFT gates bound 256, 1024 and 4096 tokens separately because each
//! length has its own vLLM reference.
//!
//! Owner: bench, ttft.
//! Invariants:
//! - Keyed by the REQUESTED length, so the metric name does not move with the tokenizer.
//! - A length requested twice pools its samples into one median; a length with no sample has
//!   no metric, so a bound on it fails as missing rather than passing.

use std::collections::BTreeMap;

use crate::benchmarks::stats;

/// 2026-10-15: `median_ms_<tokens>` for each requested length with at least one sample.
/// `samples[i]` holds the samples measured at `lengths[i]`.
pub(super) fn per_length_medians(lengths: &[usize], samples: &[Vec<f64>]) -> BTreeMap<String, f64> {
    let mut pooled: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for (tokens, s) in lengths.iter().zip(samples) {
        pooled.entry(*tokens).or_default().extend_from_slice(s);
    }
    pooled
        .into_iter()
        .filter_map(|(tokens, s)| stats::median(&s).map(|m| (format!("median_ms_{tokens}"), m)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::per_length_medians;

    /// 2026-10-15: A length requested twice is one median over both rows' samples. Control: a
    /// per-row insert would leave the second 256-token row's 5.0 as the 256 median.
    #[test]
    fn a_repeated_length_pools_its_rows() {
        let m = per_length_medians(&[256, 1024, 256], &[vec![1.0, 3.0], vec![10.0], vec![5.0]]);
        assert_eq!(m.get("median_ms_256"), Some(&3.0));
        assert_eq!(m.get("median_ms_1024"), Some(&10.0));
        assert_eq!(m.len(), 2);
    }

    /// 2026-10-15: A length with no sample has no metric, so a bound on it reads as missing.
    #[test]
    fn a_length_without_samples_has_no_metric() {
        let m = per_length_medians(&[256, 4096], &[vec![2.0], vec![]]);
        assert_eq!(m.get("median_ms_256"), Some(&2.0));
        assert!(!m.contains_key("median_ms_4096"), "{m:?}");
    }
}
