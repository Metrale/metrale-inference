// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The what-if analysis of one capture: how well the model reproduces the step, the
//! critical chain, per-site slack and exact upside (max-plus and simulated) beside the
//! headroom's Amdahl upside, and the rewrites asked for.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A site's upside is the re-evaluated step with the site's operations at its floor (their
//!   work scaled by `floor / measured`), never a sum of exclusive times; its subgradient is the
//!   share of its work on the critical chain.
//! - The DRAM sharing constant is calibrated on this capture's overlapped intervals
//!   ([`super::sim::calibrate_dram`]); nothing is fitted on a ledger outcome.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::capture::Capture;
use super::dag::build;
use super::graph::{EdgeKind, NodeKind, node_of};
use super::maxplus::{
    Latency, critical_chain, fit_latency, latest, longest, median, step_us, weights,
    work_in_context,
};
use super::scenario::{Rewrite, Scenario};
use super::sim::{DramCalibration, Resources, calibrate_dram, simulate, solo_work};
use crate::hardware::headroom::Headroom;

/// 2026-10-23: What the analysis needs besides the capture.
#[derive(Debug, Clone)]
pub struct Inputs<'a> {
    /// 2026-10-23: Per capture group, the headroom row `(run, site)` it is attributed to.
    pub group_site: &'a [Option<(String, String)>],
    /// 2026-10-23: The headroom of the same capture (floors, bytes, Amdahl upside).
    pub headroom: &'a Headroom,
    /// 2026-10-23: SMs and per-SM limits; its DRAM field is replaced by the calibration.
    pub resources: Resources,
    /// 2026-10-23: An unexplained graph-node start beyond this gets an inferred fork/join, ns.
    pub tau_ns: i64,
    /// 2026-10-23: A graph node launched this soon after a running node on another stream
    /// started, and finishing after it, is its programmatic dependent, ns.
    pub prog_ns: i64,
    /// 2026-10-23: Named rewrite sets to evaluate.
    pub rewrites: &'a [(String, Vec<Rewrite>)],
}

/// 2026-10-23: One site's critical-path view.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SiteCrit {
    /// 2026-10-23: `mode n=rows`.
    pub run: String,
    /// 2026-10-23: `block [locals]`.
    pub site: String,
    /// 2026-10-23: Work in context per step, microseconds.
    pub work_us: f64,
    /// 2026-10-23: Floor per step (headroom), microseconds.
    pub floor_us: f64,
    /// 2026-10-23: Work on the critical chain per step, microseconds.
    pub critical_us: f64,
    /// 2026-10-23: `critical / work`: the subgradient of the step in the site's time.
    pub subgradient: f64,
    /// 2026-10-23: Smallest and median slack of its operations that feed a step anchor,
    /// microseconds (`None`: none does, the slack is unbounded).
    pub slack_min_us: Option<f64>,
    /// 2026-10-23: Median slack (`None` as for `slack_min_us`).
    pub slack_median_us: Option<f64>,
    /// 2026-10-23: The headroom's Amdahl upside at the floor (exclusive time), microseconds.
    pub amdahl_us: Option<f64>,
    /// 2026-10-23: Max-plus step saved with the site at its floor, microseconds.
    pub exact_us: f64,
    /// 2026-10-23: Simulated (SM + DRAM contention) step saved, microseconds.
    pub sim_us: f64,
}

/// 2026-10-23: One evaluated rewrite set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RewriteResult {
    /// 2026-10-23: Its name.
    pub id: String,
    /// 2026-10-23: Places each rewrite changed.
    pub places: Vec<usize>,
    /// 2026-10-23: Max-plus step change, microseconds (+: slower).
    pub maxplus_delta_us: f64,
    /// 2026-10-23: Simulated step change, microseconds (+: slower).
    pub sim_delta_us: f64,
    /// 2026-10-23: Simulated step change with SM residency only, microseconds (+: slower).
    pub sim_sm_delta_us: f64,
    /// 2026-10-23: The five labels whose critical-chain time per step moved most (max-plus):
    /// `(label, before, after)`.
    pub shifts: Vec<(String, f64, f64)>,
}

/// 2026-10-23: One critical-chain item of a sample step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChainItem {
    /// 2026-10-23: Site, kernel or edge kind.
    pub what: String,
    /// 2026-10-23: Microseconds.
    pub us: f64,
}

/// 2026-10-23: The analysis.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WhatIf {
    /// 2026-10-23: `whatif/v1`.
    pub schema: String,
    /// 2026-10-23: The capture's fingerprint.
    pub source: String,
    /// 2026-10-23: Whole steps.
    pub steps: usize,
    /// 2026-10-23: KV tokens per sequence the capture attended (the headroom's statement).
    pub kv_tokens: f64,
    /// 2026-10-23: Measured mean step, microseconds.
    pub measured_step_us: f64,
    /// 2026-10-23: Every edge its measured gap (a consistency check: exact by construction).
    pub replay_step_us: f64,
    /// 2026-10-23: Max-plus with in-context work and one latency per kind.
    pub model_step_us: f64,
    /// 2026-10-23: The simulation with SM residency and DRAM sharing.
    pub sim_step_us: f64,
    /// 2026-10-23: The simulation with SM residency only (the shared bus unbounded).
    pub sim_sm_step_us: f64,
    /// 2026-10-23: `(kind, us, binding edges)`.
    pub latency: Vec<(String, f64, usize)>,
    /// 2026-10-23: Edges dropped for running backwards, per kind.
    pub dropped: Vec<(String, usize)>,
    /// 2026-10-23: The model's step with one edge kind removed (sabotage).
    pub without: Vec<(String, f64)>,
    /// 2026-10-23: The DRAM calibration.
    pub dram: DramCalibration,
    /// 2026-10-23: Critical-chain time per step by edge kind.
    pub critical_by_kind: Vec<(String, f64)>,
    /// 2026-10-23: The critical chain of the middle step, compressed.
    pub chain_step: Vec<ChainItem>,
    /// 2026-10-23: Per site, by exact upside.
    pub sites: Vec<SiteCrit>,
    /// 2026-10-23: Evaluated rewrite sets.
    pub rewrites: Vec<RewriteResult>,
}

struct Base {
    sc: Scenario,
    lat: Latency,
    res: Resources,
    row: Vec<Option<usize>>,
}

fn sim_step(sc: &Scenario, lat: &Latency, res: &Resources) -> f64 {
    let w = sc.sim_weights(lat);
    step_us(
        &sc.g,
        &simulate(&sc.g, &w, &sc.solo, &sc.bytes, &sc.cap, res),
    )
}

fn maxplus_step(sc: &Scenario, lat: &Latency) -> f64 {
    let w = sc.weights(lat);
    step_us(&sc.g, &longest(&sc.g, &w))
}

fn base(c: &Capture, inp: &Inputs<'_>) -> metrale_error::Result<(Base, DramCalibration, f64)> {
    let g = build(c, inp.tau_ns, inp.prog_ns)?;
    let lat = fit_latency(&g);
    let work = work_in_context(&g);
    let replay = step_us(&g, &longest(&g, &weights(&g, &work, None)));
    let rows: BTreeMap<(&str, &str), usize> = inp
        .headroom
        .sites
        .iter()
        .enumerate()
        .map(|(i, r)| ((r.run.as_str(), r.site.as_str()), i))
        .collect();
    let row: Vec<Option<usize>> = g
        .nodes
        .iter()
        .map(|n| {
            let (run, site) = inp.group_site.get(n.group? as usize)?.as_ref()?;
            rows.get(&(run.as_str(), site.as_str())).copied()
        })
        .collect();
    let mut sum = vec![0.0; inp.headroom.sites.len()];
    for (i, r) in row.iter().enumerate() {
        if let Some(r) = r
            && g.t[2 * i] >= 0.0
        {
            sum[*r] += work[i];
        }
    }
    let steps = c.steps() as f64;
    let bytes: Vec<f64> = (0..g.nodes.len())
        .map(|i| match row[i] {
            Some(r) if sum[r] > 0.0 && g.nodes[i].kind == NodeKind::Kernel => {
                inp.headroom.sites[r].bytes * steps * work[i] / sum[r]
            }
            _ => 0.0,
        })
        .collect();
    let peak = inp.resources.dram_peak_bytes_per_us;
    let mut clipped = 0;
    let bytes: Vec<f64> = bytes
        .iter()
        .zip(&work)
        .map(|(&b, &w)| {
            if b > peak * w {
                clipped += 1;
                peak * w
            } else {
                b
            }
        })
        .collect();
    let (dram, bytes) = calibrate_dram(&g, &work, &bytes, peak, clipped);
    let mut res = inp.resources;
    res.dram_bytes_per_us = dram.bytes_per_us;
    let solo = solo_work(&g, &work, &bytes, &res);
    let site = g
        .nodes
        .iter()
        .map(|n| {
            n.group
                .and_then(|k| c.groups.get(k as usize))
                .and_then(|x| x.1.clone())
        })
        .collect();
    let name = g
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| match n.group {
            Some(k) => c.groups[k as usize].0.clone(),
            None => c
                .api
                .get(i - c.ops.len())
                .map_or_else(String::new, |a| a.0.clone()),
        })
        .collect();
    let sc = Scenario {
        g,
        work0: work.clone(),
        work,
        solo,
        bytes,
        cap: Vec::new(),
        over: BTreeMap::new(),
        site,
        name,
    };
    Ok((Base { sc, lat, res, row }, dram, replay))
}

/// 2026-10-23: Critical-chain time per step by label (site, kernel or edge kind).
fn chain_by_label(b: &Base, sc: &Scenario, h: &Headroom, steps: f64) -> BTreeMap<String, f64> {
    let w = sc.weights(&b.lat);
    let at = longest(&sc.g, &w);
    let mut m = BTreeMap::new();
    for k in critical_chain(&sc.g, &w, &at) {
        let e = sc.g.edges[k as usize];
        let what = match e.kind {
            EdgeKind::Duration | EdgeKind::PdlWait if node_of(e.dst) < b.row.len() => {
                label(b, h, node_of(e.dst))
            }
            EdgeKind::Duration | EdgeKind::PdlWait => sc.name[node_of(e.dst)].clone(),
            k => format!("({})", k.name()),
        };
        *m.entry(what).or_insert(0.0) += w[k as usize] / steps;
    }
    m
}

fn label(b: &Base, h: &Headroom, node: usize) -> String {
    match b.row[node] {
        Some(r) => format!("{} · {}", h.sites[r].run, h.sites[r].site),
        None => b.sc.name[node].clone(),
    }
}

fn chain_of_step(b: &Base, h: &Headroom, w: &[f64], at: &[f64], chain: &[u32]) -> Vec<ChainItem> {
    let g = &b.sc.g;
    let mid = g.anchors.len() / 2;
    let (lo, hi) = (at[g.anchors[mid - 1] as usize], at[g.anchors[mid] as usize]);
    let mut items: Vec<ChainItem> = Vec::new();
    for &k in chain.iter().rev() {
        let e = g.edges[k as usize];
        let t = at[e.dst as usize];
        if t <= lo || t > hi {
            continue;
        }
        let what = match e.kind {
            EdgeKind::Duration | EdgeKind::PdlWait => label(b, h, node_of(e.dst)),
            k => format!("({})", k.name()),
        };
        let us = w[k as usize];
        match items.last_mut() {
            Some(x) if x.what == what => x.us += us,
            _ => items.push(ChainItem { what, us }),
        }
    }
    items
}

/// 2026-10-23: Analyze `c`.
pub fn analyze(c: &Capture, inp: &Inputs<'_>) -> metrale_error::Result<WhatIf> {
    let (b, dram, replay) = base(c, inp)?;
    let h = inp.headroom;
    let g = &b.sc.g;
    let steps = c.steps() as f64;
    let w = b.sc.weights(&b.lat);
    let at = longest(g, &w);
    let model = step_us(g, &at);
    let sim = sim_step(&b.sc, &b.lat, &b.res);
    let sm_only = Resources {
        dram_bytes_per_us: f64::INFINITY,
        ..b.res
    };
    let sim_sm = sim_step(&b.sc, &b.lat, &sm_only);
    let chain = critical_chain(g, &w, &at);
    let late = latest(g, &w, &at);
    let mut by_kind: BTreeMap<EdgeKind, f64> = BTreeMap::new();
    let mut crit = vec![0.0; h.sites.len()];
    for &k in &chain {
        let e = g.edges[k as usize];
        *by_kind.entry(e.kind).or_default() += w[k as usize] / steps;
        if matches!(e.kind, EdgeKind::Duration | EdgeKind::PdlWait)
            && let Some(r) = b.row[node_of(e.dst)]
        {
            crit[r] += w[k as usize] / steps;
        }
    }
    let without = [
        EdgeKind::Join,
        EdgeKind::Event,
        EdgeKind::PdlWait,
        EdgeKind::Sync,
    ]
    .iter()
    .map(|&kind| {
        let mut w2 = w.clone();
        for (k, e) in g.edges.iter().enumerate() {
            if e.kind == kind {
                w2[k] = f64::NEG_INFINITY;
            }
        }
        (kind.name().to_string(), step_us(g, &longest(g, &w2)))
    })
    .collect();
    let mut sites = Vec::new();
    for (r, row) in h.sites.iter().enumerate() {
        let members: Vec<usize> = (0..g.nodes.len())
            .filter(|&i| b.row[i] == Some(r) && g.t[2 * i] >= 0.0)
            .collect();
        if members.is_empty() {
            continue;
        }
        let work: f64 = members.iter().map(|&i| b.sc.work[i]).sum::<f64>() / steps;
        let factor = if work > row.floor_us && work > 0.0 {
            row.floor_us / work
        } else {
            1.0
        };
        let mut sc = b.sc.clone();
        for &i in &members {
            sc.work[i] *= factor;
            sc.solo[i] *= factor;
        }
        let mut slack: Vec<f64> = members
            .iter()
            .map(|&i| late[2 * i] - at[2 * i])
            .filter(|x| x.is_finite())
            .collect();
        let slack_min = slack.iter().copied().reduce(f64::min);
        sites.push(SiteCrit {
            run: row.run.clone(),
            site: row.site.clone(),
            work_us: work,
            floor_us: row.floor_us,
            critical_us: crit[r],
            subgradient: if work > 0.0 { crit[r] / work } else { 0.0 },
            slack_min_us: slack_min,
            slack_median_us: (!slack.is_empty()).then(|| median(&mut slack)),
            amdahl_us: row.upside_full.map(|u| u.saved_us),
            exact_us: model - maxplus_step(&sc, &b.lat),
            sim_us: sim - sim_step(&sc, &b.lat, &b.res),
        });
    }
    sites.sort_by(|a, b| {
        b.exact_us
            .total_cmp(&a.exact_us)
            .then_with(|| a.site.cmp(&b.site))
    });
    let mut rewrites = Vec::new();
    let before = if inp.rewrites.is_empty() {
        BTreeMap::new()
    } else {
        chain_by_label(&b, &b.sc, h, steps)
    };
    for (id, set) in inp.rewrites {
        let mut sc = b.sc.clone();
        let mut places = Vec::new();
        for rw in set {
            places.push(sc.apply(rw, &b.lat, &b.res)?);
        }
        let after = chain_by_label(&b, &sc, h, steps);
        let mut shifts: Vec<(String, f64, f64)> = before
            .keys()
            .chain(after.keys())
            .map(|k| {
                (
                    k.clone(),
                    before.get(k).copied().unwrap_or(0.0),
                    after.get(k).copied().unwrap_or(0.0),
                )
            })
            .collect();
        shifts.sort_by(|x, y| {
            (y.2 - y.1)
                .abs()
                .total_cmp(&(x.2 - x.1).abs())
                .then_with(|| x.0.cmp(&y.0))
        });
        shifts.dedup_by(|x, y| x.0 == y.0);
        shifts.truncate(5);
        rewrites.push(RewriteResult {
            id: id.clone(),
            shifts,
            places,
            maxplus_delta_us: maxplus_step(&sc, &b.lat) - model,
            sim_delta_us: sim_step(&sc, &b.lat, &b.res) - sim,
            sim_sm_delta_us: sim_step(&sc, &b.lat, &sm_only) - sim_sm,
        });
    }
    Ok(WhatIf {
        schema: "whatif/v1".into(),
        source: c.source.clone(),
        steps: c.steps(),
        kv_tokens: h.kv_tokens.value,
        measured_step_us: c.step_us(),
        replay_step_us: replay,
        model_step_us: model,
        sim_step_us: sim,
        sim_sm_step_us: sim_sm,
        latency: b
            .lat
            .us
            .iter()
            .map(|(k, v)| (k.name().to_string(), *v, b.lat.n[k]))
            .collect(),
        dropped: g
            .dropped
            .iter()
            .map(|(k, v)| (k.name().to_string(), *v))
            .collect(),
        without,
        dram,
        critical_by_kind: by_kind
            .into_iter()
            .map(|(k, v)| (k.name().to_string(), v))
            .collect(),
        chain_step: chain_of_step(&b, h, &w, &at, &chain),
        sites,
        rewrites,
    })
}
