// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: From mocks to the full model. A per-step quantity (decode step time, energy per
//! step, TTFT) is `fixed + sum_s units_s * per_unit_s`: per-layer kernels are identical between a
//! mock and its source, so only the count of each signature's units differs. With S signatures,
//! S + 1 mocks whose unit counts are affinely independent determine `fixed` and every
//! `per_unit_s`; the full model's value is then a prediction. More mocks are a least-squares fit.
//!
//! Throughput is not linear in layers, its reciprocal is: a `rate` metric (tok/s) is fitted as
//! seconds per token and inverted back; a `linear` metric (J/tok at a fixed concurrency, TTFT) is
//! fitted as it is.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - Points that do not determine every coefficient are refused, never regularised.

use crate::error::{MlError, Result};

/// 2026-10-03: How a metric scales with layer count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scaling {
    /// 2026-10-03: The metric itself is affine in units (J/tok, TTFT).
    Linear,
    /// 2026-10-03: Its reciprocal is (tok/s).
    Rate,
}

impl Scaling {
    /// 2026-10-03: Parse `linear` or `rate`.
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "linear" => Ok(Scaling::Linear),
            "rate" => Ok(Scaling::Rate),
            other => Err(MlError::Extrapolate(format!(
                "scaling `{other}`: expected `linear` or `rate`"
            ))),
        }
    }
}

/// 2026-10-03: One measured mock: its kept units per signature and the metric.
#[derive(Debug, Clone, PartialEq)]
pub struct Point {
    /// 2026-10-03: Units kept, per signature.
    pub units: Vec<f64>,
    /// 2026-10-03: The measured value.
    pub value: f64,
}

/// 2026-10-03: The fitted model and its prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct Estimate {
    /// 2026-10-03: The per-step cost no layer count changes (in the fitted space).
    pub fixed: f64,
    /// 2026-10-03: Cost per unit of each signature (in the fitted space).
    pub per_unit: Vec<f64>,
    /// 2026-10-03: The full model's predicted value (back in the metric's space).
    pub full: f64,
    /// 2026-10-03: Largest relative residual of the points (0 when exactly determined).
    pub max_residual: f64,
}

/// 2026-10-03: Fit `points` and predict the value at `full_units`.
pub fn extrapolate(points: &[Point], full_units: &[f64], scaling: Scaling) -> Result<Estimate> {
    let s = full_units.len();
    if points.len() < s + 1 {
        return Err(MlError::Extrapolate(format!(
            "{} point(s) for {s} signature(s): need at least {}",
            points.len(),
            s + 1
        )));
    }
    let to_fit = |v: f64| -> Result<f64> {
        match scaling {
            Scaling::Linear => Ok(v),
            Scaling::Rate if v > 0.0 => Ok(1.0 / v),
            Scaling::Rate => Err(MlError::Extrapolate(format!("rate {v} is not positive"))),
        }
    };
    let n = s + 1;
    let mut ata = vec![vec![0.0f64; n]; n];
    let mut atb = vec![0.0f64; n];
    let mut ys = Vec::with_capacity(points.len());
    for p in points {
        if p.units.len() != s {
            return Err(MlError::Extrapolate(format!(
                "a point has {} unit counts; the source has {s} signatures",
                p.units.len()
            )));
        }
        let row: Vec<f64> = std::iter::once(1.0)
            .chain(p.units.iter().copied())
            .collect();
        let y = to_fit(p.value)?;
        ys.push(y);
        for i in 0..n {
            atb[i] += row[i] * y;
            for j in 0..n {
                ata[i][j] += row[i] * row[j];
            }
        }
    }
    let coef = solve(ata, atb).ok_or_else(|| {
        MlError::Extrapolate(
            "the points' unit counts do not determine every signature (vary each signature's \
             count on its own)"
                .into(),
        )
    })?;
    let predict = |units: &[f64]| {
        coef[0]
            + units
                .iter()
                .zip(&coef[1..])
                .map(|(u, c)| u * c)
                .sum::<f64>()
    };
    let max_residual = points
        .iter()
        .zip(&ys)
        .map(|(p, y)| ((predict(&p.units) - y) / y).abs())
        .fold(0.0, f64::max);
    let full_fit = predict(full_units);
    let full = match scaling {
        Scaling::Linear => full_fit,
        Scaling::Rate => 1.0 / full_fit,
    };
    Ok(Estimate {
        fixed: coef[0],
        per_unit: coef[1..].to_vec(),
        full,
        max_residual,
    })
}

/// 2026-10-03: What extrapolation reads from a mock's resolved spec.
#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedUnits {
    /// 2026-10-03: The source identity: `config_sha256` and `index_digest`.
    pub source: (String, String),
    /// 2026-10-03: Each signature's digest, in order.
    pub signatures: Vec<String>,
    /// 2026-10-03: Units the mock keeps, per signature.
    pub kept: Vec<f64>,
    /// 2026-10-03: Units the source has, per signature.
    pub full: Vec<f64>,
}

/// 2026-10-03: Read a resolved spec (`mock.resolved.toml`).
pub fn units_of_resolved(text: &str) -> Result<ResolvedUnits> {
    let bad = |w: &str| MlError::Extrapolate(format!("resolved spec: {w}"));
    let v: toml::Value = toml::from_str(text).map_err(|e| bad(&e.to_string()))?;
    let s = |t: &str, k: &str| -> Result<String> {
        v.get(t)
            .and_then(|x| x.get(k))
            .and_then(toml::Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| bad(&format!("no {t}.{k}")))
    };
    let sigs = v
        .get("derived")
        .and_then(|d| d.get("signature"))
        .and_then(toml::Value::as_array)
        .ok_or_else(|| bad("no [[derived.signature]]"))?;
    let mut out = ResolvedUnits {
        source: (s("source", "config_sha256")?, s("source", "index_digest")?),
        signatures: Vec::new(),
        kept: Vec::new(),
        full: Vec::new(),
    };
    for g in sigs {
        let n = |k: &str| {
            g.get(k)
                .and_then(toml::Value::as_integer)
                .map(|i| i as f64)
                .ok_or_else(|| bad(&format!("a signature without {k}")))
        };
        out.kept.push(n("units_kept")?);
        out.full.push(n("units_full")?);
        out.signatures.push(
            g.get("digest")
                .and_then(toml::Value::as_str)
                .ok_or_else(|| bad("a signature without digest"))?
                .to_string(),
        );
    }
    Ok(out)
}

/// 2026-10-03: Points from mocks of one source: refuses mocks of different sources or
/// signature lists. Returns the points and the source's full unit counts.
pub fn points_of(mocks: &[(ResolvedUnits, f64)]) -> Result<(Vec<Point>, Vec<f64>)> {
    let first = mocks
        .first()
        .ok_or_else(|| MlError::Extrapolate("no mock measurements".into()))?;
    for (m, _) in mocks {
        if m.source != first.0.source || m.signatures != first.0.signatures {
            return Err(MlError::Extrapolate(
                "the mocks come from different checkpoints or signature lists".into(),
            ));
        }
    }
    let points = mocks
        .iter()
        .map(|(m, v)| Point {
            units: m.kept.clone(),
            value: *v,
        })
        .collect();
    Ok((points, first.0.full.clone()))
}

/// 2026-10-03: Gaussian elimination with partial pivoting; `None` when singular.
fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    let n = b.len();
    let scale = a
        .iter()
        .flatten()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(1.0);
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() <= 1e-9 * scale {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..n {
            let f = a[r][col] / a[col][col];
            for c in col..n {
                a[r][c] -= f * a[col][c];
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for r in (0..n).rev() {
        let s: f64 = (r + 1..n).map(|c| a[r][c] * x[c]).sum();
        x[r] = (b[r] - s) / a[r][r];
    }
    Some(x)
}

#[cfg(test)]
#[path = "extrapolate_tests.rs"]
mod extrapolate_tests;
