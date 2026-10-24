// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Non-negative least squares for in-serve attribution (window power = static +
//! Σ occupancy · per-kernel power, every coefficient ≥ 0), by Lawson & Hanson's (1974) active-set
//! algorithm, with the collinearity checks that decide whether an attribution may be reported:
//! the 2-norm condition number of the column-equilibrated design (Belsley, Kuh & Welsch 1980:
//! above 30 is strong dependence) and variance inflation factors (above 10 is collinear). The
//! coefficients' intervals come from a moving-block bootstrap over windows.
//!
//! Owner: metrale-stats.
//! Invariants: refusal thresholds are the declared constants [`MAX_CONDITION`], [`MAX_VIF`].

use alloc::vec;
use alloc::vec::Vec;

use metrale_error::{ErrorCode, error};

use crate::Result;
use crate::bootstrap::{block_length, moving_block_indices};
use crate::interval::{CiMethod, Interval};
use crate::linalg::{jacobi_eigen, lstsq};
use crate::num::{f, quantile_sorted, sorted};
use crate::rng::SplitMix64;

/// 2026-10-22: Belsley's strong-dependence threshold for the scaled condition number.
pub const MAX_CONDITION: f64 = 30.0;
/// 2026-10-22: The conventional collinearity threshold for a variance inflation factor.
pub const MAX_VIF: f64 = 10.0;

/// 2026-10-22: A dense row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// 2026-10-22: `StatsInvalid` unless `data.len() == rows · cols` and every value is finite.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if data.len() != rows * cols || data.iter().any(|x| !x.is_finite()) {
            return Err(error!(
                ErrorCode::StatsInvalid,
                "matrix", "shape or values invalid"
            ));
        }
        Ok(Self { rows, cols, data })
    }

    #[must_use]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    fn rows_of(&self, idx: &[usize]) -> Matrix {
        let data = idx
            .iter()
            .flat_map(|&r| {
                self.data[r * self.cols..(r + 1) * self.cols]
                    .iter()
                    .copied()
            })
            .collect();
        Matrix {
            rows: idx.len(),
            cols: self.cols,
            data,
        }
    }
}

fn shape(a: &Matrix, b: &[f64]) -> Result<()> {
    if b.len() != a.rows || a.rows < a.cols || a.cols == 0 {
        return Err(error!(ErrorCode::StatsInsufficient, "nnls", a.cols, a.rows));
    }
    Ok(())
}

/// 2026-10-22: argmin ‖Ax − b‖ subject to x ≥ 0 (Lawson-Hanson).
pub fn nnls(a: &Matrix, b: &[f64]) -> Result<Vec<f64>> {
    shape(a, b)?;
    let (m, n) = (a.rows, a.cols);
    let mut x = vec![0.0; n];
    let mut passive: Vec<usize> = Vec::new();
    let tol = 1e-12 * a.data.iter().fold(0.0f64, |s, v| s.max(v.abs())).max(1.0);
    let grad = |x: &[f64]| -> Vec<f64> {
        let r: Vec<f64> = (0..m)
            .map(|i| b[i] - (0..n).map(|j| a.get(i, j) * x[j]).sum::<f64>())
            .collect();
        (0..n)
            .map(|j| (0..m).map(|i| a.get(i, j) * r[i]).sum())
            .collect()
    };
    for _ in 0..3 * n + 10 {
        let w = grad(&x);
        let Some(t) = (0..n)
            .filter(|j| !passive.contains(j) && w[*j] > tol)
            .max_by(|&i, &j| w[i].total_cmp(&w[j]))
        else {
            return Ok(x);
        };
        passive.push(t);
        loop {
            passive.sort_unstable();
            let Some(z) = lstsq(&a.data, m, n, &passive, b) else {
                passive.retain(|&j| j != t);
                return Ok(x);
            };
            if z.iter().all(|&v| v > 0.0) {
                for (k, &j) in passive.iter().enumerate() {
                    x[j] = z[k];
                }
                break;
            }
            let mut alpha = f64::INFINITY;
            for (k, &j) in passive.iter().enumerate() {
                if z[k] <= 0.0 {
                    alpha = alpha.min(x[j] / (x[j] - z[k]));
                }
            }
            for (k, &j) in passive.iter().enumerate() {
                x[j] += alpha * (z[k] - x[j]);
            }
            passive.retain(|&j| x[j] > tol);
            for (j, v) in x.iter_mut().enumerate() {
                if !passive.contains(&j) {
                    *v = 0.0;
                }
            }
            if passive.is_empty() {
                break;
            }
        }
    }
    Err(error!(ErrorCode::StatsNoConvergence, "nnls", 3 * n + 10))
}

fn gram(a: &Matrix, scale: &[f64]) -> Vec<f64> {
    let n = a.cols;
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            g[i * n + j] =
                (0..a.rows).map(|r| a.get(r, i) * a.get(r, j)).sum::<f64>() / (scale[i] * scale[j]);
        }
    }
    g
}

/// 2026-10-22: The 2-norm condition number of `a` with each column scaled to unit length
/// (∞ when a column is zero or the columns are dependent).
pub fn condition_number(a: &Matrix) -> Result<f64> {
    if a.rows < a.cols || a.cols == 0 {
        return Err(error!(
            ErrorCode::StatsInsufficient,
            "condition_number", a.cols, a.rows
        ));
    }
    let norms: Vec<f64> = (0..a.cols)
        .map(|j| libm::sqrt((0..a.rows).map(|r| a.get(r, j) * a.get(r, j)).sum()))
        .collect();
    if norms.contains(&0.0) {
        return Ok(f64::INFINITY);
    }
    let (eig, _) = jacobi_eigen(&gram(a, &norms), a.cols);
    let max = eig.iter().fold(f64::MIN, |s, &v| s.max(v));
    let min = eig.iter().fold(f64::MAX, |s, &v| s.min(v));
    Ok(if min <= max * 1e-15 {
        f64::INFINITY
    } else {
        libm::sqrt(max / min)
    })
}

/// 2026-10-22: Variance inflation factors: the diagonal of the inverse correlation matrix of the
/// centred columns. A constant column (the intercept) has VIF 1 by definition; ∞ marks a column
/// that is an exact combination of the others.
pub fn vif(a: &Matrix) -> Result<Vec<f64>> {
    if a.rows < a.cols + 1 || a.cols == 0 {
        return Err(error!(
            ErrorCode::StatsInsufficient,
            "vif",
            a.cols + 1,
            a.rows
        ));
    }
    let means: Vec<f64> = (0..a.cols)
        .map(|j| (0..a.rows).map(|r| a.get(r, j)).sum::<f64>() / f(a.rows))
        .collect();
    let var: Vec<usize> = (0..a.cols)
        .filter(|&j| {
            (0..a.rows).any(|r| (a.get(r, j) - means[j]).abs() > 1e-12 * means[j].abs().max(1e-300))
        })
        .collect();
    let k = var.len();
    let centred = Matrix {
        rows: a.rows,
        cols: k,
        data: (0..a.rows)
            .flat_map(|r| var.iter().map(move |&j| (r, j)))
            .map(|(r, j)| a.get(r, j) - means[j])
            .collect(),
    };
    let norms: Vec<f64> = (0..k)
        .map(|j| {
            libm::sqrt(
                (0..a.rows)
                    .map(|r| centred.get(r, j) * centred.get(r, j))
                    .sum(),
            )
        })
        .collect();
    let (eig, v) = jacobi_eigen(&gram(&centred, &norms), k);
    let max = eig.iter().fold(0.0f64, |s, &e| s.max(e));
    let mut out = vec![1.0; a.cols];
    for (c, &j) in var.iter().enumerate() {
        let mut d = 0.0;
        for (e, &lam) in eig.iter().enumerate() {
            if lam <= max * 1e-15 {
                d = f64::INFINITY;
                break;
            }
            d += v[c * k + e] * v[c * k + e] / lam;
        }
        out[j] = d;
    }
    Ok(out)
}

/// 2026-10-22: Percentile intervals of each NNLS coefficient over `reps` moving-block resamples of
/// the rows (block length [`block_length`] of the row count), seeded.
pub fn bootstrap_coefficients(
    a: &Matrix,
    b: &[f64],
    reps: usize,
    level: f64,
    seed: u64,
) -> Result<Vec<Interval>> {
    shape(a, b)?;
    let mut r = SplitMix64::new(seed);
    let block = block_length(a.rows);
    let mut draws: Vec<Vec<f64>> = vec![Vec::with_capacity(reps); a.cols];
    for _ in 0..reps {
        let idx = moving_block_indices(a.rows, block, &mut r);
        let bb: Vec<f64> = idx.iter().map(|&i| b[i]).collect();
        let Ok(x) = nnls(&a.rows_of(&idx), &bb) else {
            continue;
        };
        for (j, v) in x.into_iter().enumerate() {
            draws[j].push(v);
        }
    }
    if draws[0].len() < reps / 2 + 1 {
        return Err(error!(
            ErrorCode::StatsInsufficient,
            "bootstrap_coefficients",
            reps / 2 + 1,
            draws[0].len()
        ));
    }
    let alpha = (1.0 - level) / 2.0;
    Ok(draws
        .iter()
        .map(|d| {
            let s = sorted(d);
            Interval {
                lo: quantile_sorted(&s, alpha),
                hi: quantile_sorted(&s, 1.0 - alpha),
                level,
                method: CiMethod::PercentileBlockBootstrap,
            }
        })
        .collect())
}

#[cfg(test)]
#[path = "nnls_tests.rs"]
mod tests;
