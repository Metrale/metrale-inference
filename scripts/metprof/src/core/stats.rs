// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: true medians (the mean of the two middle values for an even
//! count) and means. Callers guarantee a non-empty input; an empty one is a
//! refusal upstream, so these return `None` rather than invent a value.

pub fn median_i64(values: &[i64]) -> Option<f64> {
    let mut v = values.to_vec();
    v.sort_unstable();
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(v[n / 2] as f64),
        _ => Some((v[n / 2 - 1] as i128 + v[n / 2] as i128) as f64 / 2.0),
    }
}

/// The `p` quantile (0 <= p <= 1) of ascending `sorted` values, by linear
/// interpolation between the closest ranks at h = (n - 1) * p. At p = 0.5 it
/// is the median above. `None` for no values or a `p` outside [0, 1].
pub fn quantile_sorted(sorted: &[i64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let h = (sorted.len() - 1) as f64 * p;
    let (lo, hi) = (h.floor() as usize, h.ceil() as usize);
    let (a, b) = (sorted[lo] as f64, sorted[hi] as f64);
    Some(a + (h - lo as f64) * (b - a))
}

pub fn median_f64(values: &[f64]) -> Option<f64> {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let n = v.len();
    match n {
        0 => None,
        _ if n % 2 == 1 => Some(v[n / 2]),
        _ => Some((v[n / 2 - 1] + v[n / 2]) / 2.0),
    }
}

pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_count_is_the_mean_of_the_two_middles() {
        assert_eq!(median_i64(&[9, 1, 4, 2]), Some(3.0));
        assert_eq!(median_i64(&[5, 1, 3]), Some(3.0));
        assert_eq!(median_f64(&[0.5, 0.25]), Some(0.375));
        assert_eq!(median_i64(&[]), None);
        assert_eq!(mean(&[]), None);
    }

    #[test]
    fn quantiles_interpolate_between_ranks_and_agree_with_the_median() {
        let v = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
        assert_eq!(quantile_sorted(&v, 0.1), Some(20.0));
        assert_eq!(quantile_sorted(&v, 0.0), Some(10.0));
        assert_eq!(quantile_sorted(&v, 1.0), Some(110.0));
        let q = quantile_sorted(&[1, 2, 3, 10], 0.9);
        assert!((q.expect("q") - 7.9).abs() < 1e-12, "{q:?}");
        assert_eq!(quantile_sorted(&[7], 0.9), Some(7.0));
        for v in [
            &[1, 5][..],
            &[2, 3, 9],
            &[1, 4, 4, 100],
            &[3, 3, 8, 8, 9, 20],
        ] {
            assert_eq!(quantile_sorted(v, 0.5), median_i64(v));
        }
        assert_eq!(quantile_sorted(&[], 0.5), None);
        assert_eq!(quantile_sorted(&[1, 2], 1.5), None);
        assert_eq!(quantile_sorted(&[1, 2], -0.1), None);
    }

    #[test]
    fn large_values_do_not_overflow_the_middle_sum() {
        assert_eq!(median_i64(&[i64::MAX, i64::MAX]), Some(i64::MAX as f64));
    }
}
