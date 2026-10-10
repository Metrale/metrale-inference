// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The concurrency section of `model-bring-up-bench`, pure: the `--concs` grammar,
//! the per-rung rows read from the `concurrency-sweep` gate's own metrics, and GPU-rail energy
//! integrated from NVML counter series over each rung's measured window.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - tok/s, TTFT and TPOT are the sweep's metrics (`c{c}_aggregate_tok_s`, `c{c}_ttft_p50_ms`,
//!   `c{c}_tpot_p50_ms`); a rung the sweep did not publish (vacuous, errored or
//!   cache-uncontrolled) is `comparable: false`, never a number.
//! - Energy is `None` ("not measured") unless every listed host's series covers the rung's
//!   window; it is never 0 for a missing reading.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 2026-10-10: `--concs`'s default: the GLM campaign's ladder (owner, 2026-10-10: "up to and
/// including C=16").
pub(crate) const DEFAULT_CONCS_ARG: &str = "1,2,4,8,12,16";
/// 2026-10-10: The standard rungs a single `--concs N` takes up to N: the default ladder, then
/// the sweep gate's wide rungs.
pub(crate) const STANDARD_CONCS: &[usize] = &[1, 2, 4, 8, 12, 16, 32, 64, 128];
/// 2026-10-10: The sweep's `concurrencies` bound.
const MAX_CONC: usize = 256;

/// 2026-10-10: `--concs`: one standard rung N (every standard rung up to and including N), or
/// a comma-separated, strictly increasing list of rungs in 1..=256.
pub(crate) fn parse_concs(s: &str) -> Result<Vec<usize>> {
    let parts: Vec<&str> = s.split(',').map(str::trim).collect();
    let mut out = Vec::with_capacity(parts.len());
    for p in &parts {
        let c: usize = p
            .parse()
            .map_err(|_| anyhow::anyhow!("--concs {s}: {p:?} is not a whole number"))?;
        if c == 0 || c > MAX_CONC {
            bail!("--concs {s}: {c} is outside 1..={MAX_CONC}");
        }
        out.push(c);
    }
    if let [max] = out.as_slice() {
        if !STANDARD_CONCS.contains(max) {
            bail!(
                "--concs {max}: a single value is the largest standard rung to run, one of \
                 {STANDARD_CONCS:?}; list the rungs instead, e.g. --concs 1,4,{max}"
            );
        }
        return Ok(STANDARD_CONCS
            .iter()
            .copied()
            .filter(|c| c <= max)
            .collect());
    }
    if out.windows(2).any(|w| w[0] >= w[1]) {
        bail!("--concs {s}: list the rungs in strictly increasing order");
    }
    Ok(out)
}

/// 2026-10-10: The sweep instrument the section runs; recorded with it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Instrument {
    pub concs: Vec<usize>,
    pub isl: usize,
    pub osl: usize,
    pub prompt_mode: String,
    pub warmup: usize,
}

/// 2026-10-10: One rung of the ladder.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Rung {
    pub conc: usize,
    /// 2026-10-10: Whether the sweep published this rung (`CellRow::comparable`).
    pub comparable: bool,
    pub tok_s: Option<f64>,
    pub ttft_p50_ms: Option<f64>,
    pub tpot_p50_ms: Option<f64>,
    pub completion_tokens: Option<f64>,
    /// 2026-10-10: The measured batch's window, unix seconds.
    pub window: Option<(f64, f64)>,
    /// 2026-10-10: GPU-rail joules over the window, summed over the energy hosts.
    pub energy_j: Option<f64>,
    pub j_per_tok: Option<f64>,
}

/// 2026-10-10: The concurrency section of a record.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct Section {
    pub instrument: Instrument,
    /// 2026-10-10: The hosts whose NVML energy counters were read; empty = not measured.
    pub energy_hosts: Vec<String>,
    pub rungs: Vec<Rung>,
    pub status: String,
    pub run: Value,
}

/// 2026-10-10: A host's NVML cumulative-energy series: (unix seconds, millijoules).
pub(crate) type Series = Vec<(f64, f64)>;

fn at(series: &[(f64, f64)], t: f64) -> Option<f64> {
    let i = series.partition_point(|(ts, _)| *ts < t);
    if i == 0 || i >= series.len() {
        return None;
    }
    let ((t0, e0), (t1, e1)) = (series[i - 1], series[i]);
    if t1 <= t0 {
        return None;
    }
    Some(e0 + (e1 - e0) * (t - t0) / (t1 - t0))
}

/// 2026-10-10: Joules over `[start, end]`, summed over hosts, each counter linearly
/// interpolated at both ends. `None` when any host's series does not bracket the window.
pub(crate) fn energy_j(hosts: &[Series], start: f64, end: f64) -> Option<f64> {
    if hosts.is_empty() || end <= start {
        return None;
    }
    let mut total = 0.0;
    for s in hosts {
        total += (at(s, end)? - at(s, start)?) / 1000.0;
    }
    Some(total)
}

/// 2026-10-10: The rungs from the sweep's terminal metrics and, when given, the hosts' series.
pub(crate) fn rungs(concs: &[usize], m: &BTreeMap<String, f64>, energy: &[Series]) -> Vec<Rung> {
    concs
        .iter()
        .map(|&c| {
            let get = |k: &str| m.get(&format!("c{c}_{k}")).copied();
            let window = get("window_start_unix").zip(get("window_end_unix"));
            let completion_tokens = get("completion_tokens");
            let energy_j = window.and_then(|(s, e)| energy_j(energy, s, e));
            Rung {
                conc: c,
                comparable: get("aggregate_tok_s").is_some(),
                tok_s: get("aggregate_tok_s"),
                ttft_p50_ms: get("ttft_p50_ms"),
                tpot_p50_ms: get("tpot_p50_ms"),
                completion_tokens,
                window,
                energy_j,
                j_per_tok: energy_j
                    .zip(completion_tokens)
                    .filter(|(_, n)| *n > 0.0)
                    .map(|(j, n)| j / n),
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "bring_up_conc_tests.rs"]
mod tests;
