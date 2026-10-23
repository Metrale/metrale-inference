// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Small dense linear algebra for the regression checks: the cyclic Jacobi
//! eigen-decomposition of a symmetric matrix and Householder-QR least squares.
//!
//! Owner: metrale-stats.
//! Invariants: matrices are row-major `n × n` (eigen) or `rows × cols` (least squares).

use alloc::vec;
use alloc::vec::Vec;

/// 2026-10-22: Eigenvalues and eigenvectors (columns of `v`, row-major n × n) of symmetric `a`.
pub(crate) fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _ in 0..100 {
        let off: f64 = (0..n)
            .flat_map(|i| (0..n).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| m[i * n + j] * m[i * n + j])
            .sum();
        let scale: f64 = m.iter().map(|x| x * x).sum::<f64>().max(1e-300);
        if off <= 1e-30 * scale {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let theta = (m[q * n + q] - m[p * n + p]) / (2.0 * apq);
                let sgn = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sgn / (theta.abs() + libm::sqrt(theta * theta + 1.0));
                let c = 1.0 / libm::sqrt(t * t + 1.0);
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (m[k * n + p], m[k * n + q]);
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (m[p * n + k], m[q * n + k]);
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| m[i * n + i]).collect(), v)
}

/// 2026-10-22: Least squares of `b` on the columns `cols` of the row-major `rows × ncols`
/// matrix `a` by Householder QR; `None` when those columns are rank deficient.
pub(crate) fn lstsq(
    a: &[f64],
    rows: usize,
    ncols: usize,
    cols: &[usize],
    b: &[f64],
) -> Option<Vec<f64>> {
    let k = cols.len();
    if k == 0 || rows < k {
        return None;
    }
    let mut r: Vec<f64> = (0..rows)
        .flat_map(|i| cols.iter().map(move |&j| (i, j)))
        .map(|(i, j)| a[i * ncols + j])
        .collect();
    let mut y = b.to_vec();
    let norm0: f64 = r.iter().map(|x| x * x).sum::<f64>().max(1e-300);
    for c in 0..k {
        let alpha2: f64 = (c..rows).map(|i| r[i * k + c] * r[i * k + c]).sum();
        let alpha = libm::sqrt(alpha2);
        if alpha2 <= 1e-24 * norm0 {
            return None;
        }
        let sign = if r[c * k + c] >= 0.0 { 1.0 } else { -1.0 };
        let mut u: Vec<f64> = (c..rows).map(|i| r[i * k + c]).collect();
        u[0] += sign * alpha;
        let unorm2: f64 = u.iter().map(|x| x * x).sum();
        for j in c..k {
            let d: f64 = (c..rows).map(|i| u[i - c] * r[i * k + j]).sum::<f64>() * 2.0 / unorm2;
            for i in c..rows {
                r[i * k + j] -= d * u[i - c];
            }
        }
        let d: f64 = (c..rows).map(|i| u[i - c] * y[i]).sum::<f64>() * 2.0 / unorm2;
        for i in c..rows {
            y[i] -= d * u[i - c];
        }
    }
    let mut x = vec![0.0; k];
    for c in (0..k).rev() {
        let s: f64 = (c + 1..k).map(|j| r[c * k + j] * x[j]).sum();
        x[c] = (y[c] - s) / r[c * k + c];
    }
    Some(x)
}
