// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The model-level check over an engine-neutral logprob dump: three legs taken over
//! the OpenAI-compatible API (`tf`: teacher-forced prompt logprobs over a fixed corpus; `dec1`,
//! `dec4`: greedy decode one and four requests at a time), each position carrying the chosen
//! token, its logprob and the top-k list. Two judgements:
//! - `exact`: every leg equals the pinned reference byte for byte (bit-identical levers);
//! - `numerics`: per leg, tie-aware top-1 agreement, a top-k KL(ref ‖ test) lower bound, the
//!   p99 and max of |Δlogprob| of the forced or shared token, and, for decode legs, the
//!   reference's top-1/top-2 margin at each prompt's first divergence.
//!
//! The dump producer (a script over HTTP) only records; every metric lives here, so every model
//! check computes them one way.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - The reference is pinned by the SHA-256 of its bytes and both dumps by the corpus digest;
//!   a mismatch is a refusal, never a re-baseline.
//! - Every threshold is the caller's (measured from good and bad arms); none is defaulted.
//! - A leg with no measured position fails.

use std::collections::BTreeMap;

use serde::Deserialize;
use sha2::{Digest, Sha256};

/// 2026-10-09: One sequence of a leg.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Seq {
    /// 2026-10-09: Token strings, in order.
    pub tokens: Vec<String>,
    /// 2026-10-09: The chosen (or forced) token's logprob per position (`None` where the API
    /// gives none, the first prompt token).
    pub lp: Vec<Option<f64>>,
    /// 2026-10-09: The top-k list per position (`None` where the API gives none).
    pub top: Vec<Option<BTreeMap<String, f64>>>,
}

/// 2026-10-09: A dump: the corpus digest and the three legs.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Dump {
    /// 2026-10-09: SHA-256 the producer computed over its corpus and prompts.
    pub corpus_sha256: String,
    /// 2026-10-09: Teacher-forced prompt logprobs.
    pub tf: Vec<Seq>,
    /// 2026-10-09: Greedy decode, one request at a time.
    pub dec1: Vec<Seq>,
    /// 2026-10-09: Greedy decode, four at a time.
    pub dec4: Vec<Seq>,
}

/// 2026-10-09: The legs, in report order.
pub const LEGS: [&str; 3] = ["tf", "dec1", "dec4"];

impl Dump {
    fn leg(&self, name: &str) -> &[Seq] {
        match name {
            "tf" => &self.tf,
            "dec1" => &self.dec1,
            _ => &self.dec4,
        }
    }
}

/// 2026-10-09: A comparison that cannot be made.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DumpError {
    /// 2026-10-09: The reference bytes are not the pinned ones.
    #[error("reference sha256 {got} != pinned {pinned}")]
    Reference {
        /// 2026-10-09: Measured.
        got: String,
        /// 2026-10-09: Pinned.
        pinned: String,
    },
    /// 2026-10-09: The two dumps come from different corpora.
    #[error("corpus sha256 differs: reference {reference}, test {test}")]
    Corpus {
        /// 2026-10-09: Reference's.
        reference: String,
        /// 2026-10-09: Test's.
        test: String,
    },
    /// 2026-10-09: A dump does not parse.
    #[error("{0}")]
    Parse(String),
}

/// 2026-10-09: Parse the reference (checking its pin) and the test dump.
pub fn load(reference: &[u8], pinned_sha256: &str, test: &[u8]) -> Result<(Dump, Dump), DumpError> {
    let got: String = Sha256::digest(reference)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if got != pinned_sha256 {
        return Err(DumpError::Reference {
            got,
            pinned: pinned_sha256.to_string(),
        });
    }
    let parse = |b: &[u8], what: &str| {
        serde_json::from_slice::<Dump>(b).map_err(|e| DumpError::Parse(format!("{what}: {e}")))
    };
    let (r, t) = (parse(reference, "reference")?, parse(test, "test")?);
    if r.corpus_sha256 != t.corpus_sha256 {
        return Err(DumpError::Corpus {
            reference: r.corpus_sha256,
            test: t.corpus_sha256,
        });
    }
    Ok((r, t))
}

/// 2026-10-09: Exact mode: per leg, the sequences that differ and the first position where a
/// token or logprob differs (`None` when they differ only in length or top-k lists).
pub fn exact(r: &Dump, t: &Dump) -> Vec<(String, usize, Option<usize>)> {
    let mut out = Vec::new();
    for leg in LEGS {
        let (rl, tl) = (r.leg(leg), t.leg(leg));
        for i in 0..rl.len().max(tl.len()) {
            match (rl.get(i), tl.get(i)) {
                (Some(a), Some(b)) if a == b => {}
                (Some(a), Some(b)) => {
                    let n = (0..a.tokens.len().min(b.tokens.len()))
                        .find(|&k| a.tokens[k] != b.tokens[k] || a.lp.get(k) != b.lp.get(k));
                    out.push((leg.to_string(), i, n));
                }
                _ => out.push((leg.to_string(), i, None)),
            }
        }
    }
    out
}

/// 2026-10-09: The tokens tied at the top logprob (BF16 logits tie often, and a server may
/// list a top-k in any order).
fn top1_set(top: &BTreeMap<String, f64>) -> Vec<&str> {
    let m = top.values().copied().fold(f64::NEG_INFINITY, f64::max);
    top.iter()
        .filter(|(_, v)| **v == m)
        .map(|(k, _)| k.as_str())
        .collect()
}

/// 2026-10-09: Per position: tie-aware top-1 agreement, the top-k KL(ref ‖ test) (a token
/// missing from the test's list takes the test's smallest listed logprob, so the KL is a lower
/// bound), and |Δlogprob| of the forced or shared token.
pub fn position(
    rtop: &BTreeMap<String, f64>,
    ttop: &BTreeMap<String, f64>,
    rlp: Option<f64>,
    tlp: Option<f64>,
) -> (bool, f64, f64) {
    let tset = top1_set(ttop);
    let agree = top1_set(rtop).iter().any(|t| tset.contains(t));
    let floor = ttop.values().copied().fold(f64::INFINITY, f64::min);
    let floor = if floor.is_finite() { floor } else { -1e9 };
    let kl: f64 = rtop
        .iter()
        .map(|(tok, l)| l.exp() * (l - ttop.get(tok).copied().unwrap_or(floor)))
        .sum();
    let d = match (rlp, tlp) {
        (Some(a), Some(b)) => (a - b).abs(),
        _ => 0.0,
    };
    (agree, kl.max(0.0), d)
}

/// 2026-10-09: The numerics of one leg.
#[derive(Debug, Clone, PartialEq)]
pub struct LegMetrics {
    /// 2026-10-09: Positions measured.
    pub positions: usize,
    /// 2026-10-09: Share of positions whose top-1 sets intersect.
    pub top1: f64,
    /// 2026-10-09: Mean top-k KL lower bound.
    pub kl_mean: f64,
    /// 2026-10-09: p99 of |Δlogprob| (nearest rank, as `sorted[floor(0.99 (n - 1))]`).
    pub dlp_p99: f64,
    /// 2026-10-09: Largest |Δlogprob|.
    pub dlp_max: f64,
    /// 2026-10-09: Decode sequences that diverged.
    pub diverged: usize,
    /// 2026-10-09: Divergences where the reference listed fewer than two tokens (no margin).
    pub unmeasured_divergences: usize,
    /// 2026-10-09: The largest reference top-1/top-2 margin at a divergence.
    pub max_margin_at_divergence: Option<f64>,
}

/// 2026-10-09: Numerics of `leg`: decode legs are compared up to each sequence's first token
/// divergence.
pub fn leg_metrics(r: &Dump, t: &Dump, leg: &str) -> LegMetrics {
    let decode = leg != "tf";
    let (mut agree, mut kls, mut n) = (0usize, 0.0, 0usize);
    let (mut ds, mut margins, mut unmeasured) = (Vec::new(), Vec::new(), 0usize);
    for (a, b) in r.leg(leg).iter().zip(t.leg(leg)) {
        for k in 0..a.tokens.len().min(b.tokens.len()) {
            if decode && a.tokens[k] != b.tokens[k] {
                let mut rt: Vec<f64> = a
                    .top
                    .get(k)
                    .and_then(|x| x.as_ref())
                    .map(|m| m.values().copied().collect())
                    .unwrap_or_default();
                rt.sort_by(|x, y| y.total_cmp(x));
                if rt.len() > 1 {
                    margins.push(rt[0] - rt[1])
                } else {
                    unmeasured += 1
                }
                break;
            }
            let (Some(Some(rt)), Some(Some(tt))) = (a.top.get(k), b.top.get(k)) else {
                continue;
            };
            if rt.is_empty() || tt.is_empty() {
                continue;
            }
            let (g, kl, d) = position(
                rt,
                tt,
                a.lp.get(k).copied().flatten(),
                b.lp.get(k).copied().flatten(),
            );
            agree += usize::from(g);
            kls += kl;
            ds.push(d);
            n += 1;
        }
    }
    ds.sort_by(f64::total_cmp);
    LegMetrics {
        positions: n,
        top1: agree as f64 / n.max(1) as f64,
        kl_mean: kls / n.max(1) as f64,
        dlp_p99: if ds.is_empty() {
            0.0
        } else {
            ds[(0.99 * (ds.len() - 1) as f64) as usize]
        },
        dlp_max: ds.last().copied().unwrap_or(0.0),
        diverged: margins.len() + unmeasured,
        unmeasured_divergences: unmeasured,
        max_margin_at_divergence: margins.iter().copied().reduce(f64::max),
    }
}

/// 2026-10-09: The numerics limits, each measured by the check's owner.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Limits {
    /// 2026-10-09: tf: least top-1 agreement.
    pub tf_min_top1: f64,
    /// 2026-10-09: tf: largest mean KL.
    pub tf_max_kl: f64,
    /// 2026-10-09: tf: largest p99 |Δlogprob|.
    pub tf_max_dlp_p99: f64,
    /// 2026-10-09: decode: largest mean KL.
    pub dec_max_kl: f64,
    /// 2026-10-09: decode: largest p99 |Δlogprob|.
    pub dec_max_dlp_p99: f64,
    /// 2026-10-09: decode: largest reference margin at a divergence.
    pub max_divergence_margin: f64,
    /// 2026-10-09: decode: most divergences without a measurable margin.
    pub max_unmeasured_divergences: f64,
}

/// 2026-10-09: The leg passes `l`.
pub fn judge(leg: &str, m: &LegMetrics, l: &Limits) -> bool {
    if m.positions == 0 {
        return false;
    }
    if leg == "tf" {
        m.top1 >= l.tf_min_top1 && m.kl_mean <= l.tf_max_kl && m.dlp_p99 <= l.tf_max_dlp_p99
    } else {
        m.kl_mean <= l.dec_max_kl
            && m.dlp_p99 <= l.dec_max_dlp_p99
            && m.unmeasured_divergences as f64 <= l.max_unmeasured_divergences
            && m.max_margin_at_divergence.unwrap_or(0.0) <= l.max_divergence_margin
    }
}

#[cfg(test)]
#[path = "model_logprobs_tests.rs"]
mod tests;
