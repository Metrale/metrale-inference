// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Reproducing a real model's expert load in a mock's router *weights*, so any
//! engine that loads the mock sees it (the bias-channel construction, ml-utils DESIGN.md 4).
//!
//! 1. Hidden channel `c = hidden - 1` of the embedding holds a constant `K = sqrt(hidden) / 2`;
//!    every other embedding value is unit normal.
//! 2. Every residual writer of a main layer has row `c` zero, so the residual keeps `K` there.
//! 3. After the router's RMSNorm the channel reads `s = K / rms`, the other channels have total
//!    energy `(hidden - 1) / rms^2`. A router row `e` holding `b_e / s` at column `c` and normal
//!    values of std `sigma = rms / sqrt(hidden - 1)` elsewhere gives logits `b_e + N(0, 1)`, up
//!    to the norm weight's constant factor, which leaves the top-k order unchanged.
//! 4. `b` is fitted against the target load by a deterministic Monte-Carlo top-k over that
//!    unit noise.
//!
//! No kernel-visible shape or format changes: only values of the embedding, the routers and
//! one row of each residual writer.
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - The fit uses only `+ - * /` and comparisons (no logarithm), so the router bytes are the
//!   same on every platform.
//! - Not reproduced, and disclosed: token-to-token and temporal correlation of routing.

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::error::{MlError, Result};
use crate::rng::Stream;

/// 2026-10-03: An expert-load profile: per MoE layer (in layer order), how often each expert
/// was selected.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingProfile {
    /// 2026-10-03: The checkpoint the profile was recorded on.
    pub source: String,
    /// 2026-10-03: Experts per layer.
    pub experts: usize,
    /// 2026-10-03: Experts selected per token.
    pub top_k: usize,
    /// 2026-10-03: The prompts the counts were taken over, as the recorder named them.
    pub prompt_set: Option<String>,
    /// 2026-10-03: Selection counts, one row per MoE layer.
    pub layers: Vec<Vec<u64>>,
    /// 2026-10-03: sha256 of the profile text.
    pub digest: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ProfileFile {
    schema: u32,
    source: String,
    experts: usize,
    top_k: usize,
    #[serde(default)]
    prompt_set: Option<String>,
    layers: Vec<Vec<u64>>,
}

impl RoutingProfile {
    /// 2026-10-03: Parse a profile (`met serve --record-routing` writes this JSON).
    pub fn parse(text: &str) -> Result<Self> {
        use sha2::Digest;
        let f: ProfileFile =
            serde_json::from_str(text).map_err(|e| MlError::Routing(e.to_string()))?;
        if f.schema != 1 {
            return Err(MlError::Routing(format!(
                "schema {} (this build reads 1)",
                f.schema
            )));
        }
        if f.experts == 0 || f.top_k == 0 || f.top_k > f.experts {
            return Err(MlError::Routing(format!(
                "top_k {} of {} experts",
                f.top_k, f.experts
            )));
        }
        for (i, row) in f.layers.iter().enumerate() {
            if row.len() != f.experts || row.iter().sum::<u64>() == 0 {
                return Err(MlError::Routing(format!(
                    "layer {i}: {} counts summing to {} (want {} counts, not all zero)",
                    row.len(),
                    row.iter().sum::<u64>(),
                    f.experts
                )));
            }
        }
        Ok(RoutingProfile {
            source: f.source,
            experts: f.experts,
            top_k: f.top_k,
            prompt_set: f.prompt_set,
            layers: f.layers,
            digest: crate::index::hex(&sha2::Sha256::digest(text.as_bytes())),
        })
    }
}

/// 2026-10-03: The bias channel of a model of width `hidden`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasChannel {
    /// 2026-10-03: The channel.
    pub channel: u64,
    /// 2026-10-03: The embedding's constant in it.
    pub k: f32,
    /// 2026-10-03: Its value after RMS normalization.
    pub s: f32,
    /// 2026-10-03: Router std of the other columns for unit logit noise.
    pub sigma: f32,
}

/// 2026-10-03: The bias channel for `hidden` (> 1).
pub fn bias_channel(hidden: u64) -> BiasChannel {
    let h = hidden as f32;
    let k = h.sqrt() / 2.0;
    let rms = ((k * k + (h - 1.0)) / h).sqrt();
    BiasChannel {
        channel: hidden - 1,
        k,
        s: k / rms,
        sigma: rms / (h - 1.0).sqrt(),
    }
}

/// 2026-10-03: A fitted router bias.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit {
    /// 2026-10-03: Logit bias per expert (unit noise).
    pub bias: Vec<f32>,
    /// 2026-10-03: Total-variation distance between the target and the fitted load on the
    /// fit's own samples.
    pub tv: f64,
    /// 2026-10-03: Experts whose target share was raised to the floor.
    pub floored: usize,
}

/// 2026-10-03: Monte-Carlo samples the fit uses.
const FIT_SAMPLES: usize = 8192;
/// 2026-10-03: Fit iterations.
const FIT_ITERS: usize = 120;
/// 2026-10-03: The smallest target share, as a fraction of the uniform share: an expert never
/// selected in the profile still gets a reachable bias.
const FLOOR_OF_UNIFORM: f64 = 0.02;

/// 2026-10-03: `samples` rows of `experts` unit-noise values from `stream`.
pub fn noise(stream: Stream, samples: usize, experts: usize) -> Vec<f32> {
    (0..samples * experts)
        .map(|i| stream.normal12(i as u64))
        .collect()
}

/// 2026-10-03: Selection counts of the top `k` of `bias + row` over the rows of `noise`. Rows
/// are split over threads; the counts are integer sums, so the split does not change them.
pub fn sample_counts(bias: &[f32], k: usize, noise: &[f32]) -> Vec<u64> {
    let e = bias.len();
    let rows = noise.len() / e;
    let threads = std::thread::available_parallelism()
        .map_or(1, |t| t.get())
        .min(rows.max(1));
    let per = rows.div_ceil(threads).max(1);
    let parts: Vec<Vec<u64>> = std::thread::scope(|s| {
        let handles: Vec<_> = noise
            .chunks(per * e)
            .map(|chunk| {
                s.spawn(move || {
                    let mut counts = vec![0u64; e];
                    let mut row = vec![0.0f32; e];
                    for r in chunk.chunks(e) {
                        for (j, v) in row.iter_mut().enumerate() {
                            *v = bias[j] + r[j];
                        }
                        for j in top_k(&row, k) {
                            counts[j] += 1;
                        }
                    }
                    counts
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("a counting thread panicked"))
            .collect()
    });
    let mut counts = vec![0u64; e];
    for p in parts {
        for (c, x) in counts.iter_mut().zip(p) {
            *c += x;
        }
    }
    counts
}

/// 2026-10-03: The indices of the `k` largest values, largest first (equal values: the lower
/// index first).
pub fn top_k(row: &[f32], k: usize) -> Vec<usize> {
    let order = |a: &usize, b: &usize| row[*b].total_cmp(&row[*a]).then(a.cmp(b));
    let mut idx: Vec<usize> = (0..row.len()).collect();
    if k < idx.len() {
        idx.select_nth_unstable_by(k, order);
        idx.truncate(k);
    }
    idx.sort_by(order);
    idx
}

/// 2026-10-03: Total-variation distance between two count vectors, as shares.
pub fn total_variation(a: &[u64], b: &[u64]) -> f64 {
    let (sa, sb) = (a.iter().sum::<u64>() as f64, b.iter().sum::<u64>() as f64);
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 / sa - y as f64 / sb).abs())
        .sum::<f64>()
        / 2.0
}

/// 2026-10-03: Fit per-expert logit biases so that top-`k` of `bias + N(0, 1)` selects each
/// expert in proportion to `target`.
pub fn fit(target: &[u64], k: usize, stream: Stream) -> Result<Fit> {
    let e = target.len();
    if k == 0 || k >= e {
        return Err(MlError::Routing(format!("top_k {k} of {e} experts")));
    }
    let total = target.iter().sum::<u64>() as f64;
    let floor = FLOOR_OF_UNIFORM * k as f64 / e as f64;
    let mut want: Vec<f64> = target
        .iter()
        .map(|&c| c as f64 / total * k as f64)
        .collect();
    let floored = want.iter().filter(|&&p| p < floor).count();
    for p in &mut want {
        *p = p.max(floor);
    }
    let norm = want.iter().sum::<f64>() / k as f64;
    for p in &mut want {
        *p /= norm;
    }
    let noise = noise(stream, FIT_SAMPLES, e);
    let mut bias = vec![0.0f32; e];
    for _ in 0..FIT_ITERS {
        let got = sample_counts(&bias, k, &noise);
        for j in 0..e {
            let q = got[j] as f64 / FIT_SAMPLES as f64;
            let d = ((want[j] - q) / (0.5 * (want[j] + q))).clamp(-1.0, 1.0);
            bias[j] += (0.5 * d) as f32;
        }
        let mean = bias.iter().sum::<f32>() / e as f32;
        for b in &mut bias {
            *b -= mean;
        }
    }
    let got = sample_counts(&bias, k, &noise);
    let want_counts: Vec<u64> = want.iter().map(|p| (p * 1e9) as u64).collect();
    Ok(Fit {
        tv: total_variation(&want_counts, &got),
        bias,
        floored,
    })
}

/// 2026-10-04: Per source layer, the factor `met ml-utils calibrate-routing` measured: the router
/// bias column of that layer is multiplied by it, so the bias stands to the noise the mock's own
/// hidden states produce as the fit assumed (unit noise).
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingCalibration {
    /// 2026-10-04: Source layer -> gain.
    pub gains: BTreeMap<usize, f32>,
    /// 2026-10-04: sha256 of the file text.
    pub digest: String,
}

impl RoutingCalibration {
    /// 2026-10-04: The file text for `gains`.
    pub fn to_text(gains: &BTreeMap<usize, f32>) -> String {
        let g: serde_json::Map<String, serde_json::Value> = gains
            .iter()
            .map(|(l, v)| (l.to_string(), serde_json::json!(v)))
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({ "schema": 1, "gains": g }))
            .expect("a JSON value serializes")
            + "\n"
    }

    /// 2026-10-04: Parse a calibration file.
    pub fn parse(text: &str) -> Result<Self> {
        use sha2::Digest;
        let bad = |w: String| MlError::Routing(format!("calibration: {w}"));
        let v: serde_json::Value = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        if v["schema"] != 1 {
            return Err(bad(format!("schema {}", v["schema"])));
        }
        let mut gains = BTreeMap::new();
        for (k, g) in v["gains"]
            .as_object()
            .ok_or_else(|| bad("no gains".into()))?
        {
            let l: usize = k.parse().map_err(|_| bad(format!("layer `{k}`")))?;
            let g = g.as_f64().filter(|g| g.is_finite() && *g > 0.0);
            gains.insert(
                l,
                g.ok_or_else(|| bad(format!("layer {l}: a gain that is not > 0")))? as f32,
            );
        }
        Ok(RoutingCalibration {
            gains,
            digest: crate::index::hex(&sha2::Sha256::digest(text.as_bytes())),
        })
    }
}

/// 2026-10-04: The bias-to-noise ratio a mock layer actually ran at: the `lambda` for which
/// top-`k` of `lambda * bias + N(0, 1)` reproduces the `measured` expert load best (total
/// variation, golden-section search over log lambda in [1/64, 64]), and that distance. The layer's
/// calibration gain is `1 / lambda`. Floating-point transcendental functions are used here: the
/// gain is written to a file once and read as data, so the mock bytes stay platform-independent.
pub fn realized_ratio(bias: &[f32], k: usize, measured: &[u64], stream: Stream) -> (f32, f64) {
    let noise = noise(stream, FIT_SAMPLES, bias.len());
    let tv_at = |log_l: f64| {
        let l = log_l.exp() as f32;
        let scaled: Vec<f32> = bias.iter().map(|b| b * l).collect();
        total_variation(measured, &sample_counts(&scaled, k, &noise))
    };
    let (mut a, mut b) = (-(64f64.ln()), 64f64.ln());
    let phi = (5f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - phi * (b - a), a + phi * (b - a));
    let (mut fc, mut fd) = (tv_at(c), tv_at(d));
    for _ in 0..30 {
        if fc <= fd {
            b = d;
            d = c;
            fd = fc;
            c = b - phi * (b - a);
            fc = tv_at(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + phi * (b - a);
            fd = tv_at(d);
        }
    }
    let best = (a + b) / 2.0;
    (best.exp() as f32, tv_at(best))
}

#[cfg(test)]
#[path = "routing_tests.rs"]
mod routing_tests;
