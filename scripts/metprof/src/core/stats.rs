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
    fn large_values_do_not_overflow_the_middle_sum() {
        assert_eq!(median_i64(&[i64::MAX, i64::MAX]), Some(i64::MAX as f64));
    }
}
