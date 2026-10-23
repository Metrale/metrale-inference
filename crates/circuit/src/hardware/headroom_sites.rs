// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The launch sites of a step and their floors, and the attribution of a capture's
//! kernels to them. A launch site is one plan group template across its layers: the same nodes
//! (`kda [q,k,v]`), the same kernels, any rule (a rule split by layer kind that launches the same
//! kernels for the same nodes is one site).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - One cost model: every node's bytes, FLOPs and peak come from `venn::roofline` at the run's
//!   rows and the stated KV length. A single-launch group's floor drops the edges the plan fuses
//!   inside it (their round trip is the fusion's gain); a group of several kernels counts every
//!   edge, since its kernels hand intermediates through memory.
//! - Attribution is exact or refused: a kernel the exporter attributed to a site must be one of
//!   that site's kernels, of the same kernel family, of a family that implements one of the
//!   site's ops, or run at a placeholder site (no rule); each but the first is noted. An
//!   unattributed kernel matches by name
//!   when exactly one site launches it, and is otherwise reported outside the plan, never dropped.

use crate::prelude::*;

use alloc::collections::{BTreeMap, BTreeSet};

use super::HwError;
use super::headroom_trace::TraceKernel;
use super::plan::{Planned, Resolved};
use super::render::{group_kernels, group_site};
use crate::fuser::{EdgeState, section_of};
use crate::ir::Circuit;
use crate::rules::KernelId;
use crate::venn::Run;
use crate::venn::roofline::{edge_traffic, node_cost, peak_tflops};

/// 2026-10-22: One launch site of one run.
#[derive(Debug, Clone, PartialEq)]
pub struct LaunchSite {
    /// 2026-10-22: The run.
    pub run: Run,
    /// 2026-10-22: `block [locals]`.
    pub label: String,
    /// 2026-10-22: Its nodes as `block.local`.
    pub nodes: BTreeSet<String>,
    /// 2026-10-22: Its nodes' op base names (`linear`, `mla_attention`).
    pub ops: BTreeSet<String>,
    /// 2026-10-22: The rules that form it.
    pub rules: BTreeSet<String>,
    /// 2026-10-22: What one group launches (`group_kernels`).
    pub what: String,
    /// 2026-10-22: The kernels of one group.
    pub kernels: Vec<KernelId>,
    /// 2026-10-22: Groups (layers) per step.
    pub groups: u64,
    /// 2026-10-22: Kernel launches per step.
    pub launches: u64,
    /// 2026-10-22: DRAM bytes per step at the floor.
    pub bytes: f64,
    /// 2026-10-22: FLOPs per step.
    pub flops: f64,
    /// 2026-10-22: `bytes / bandwidth`, microseconds.
    pub mem_us: f64,
    /// 2026-10-22: Each node's FLOPs over its peak, summed, microseconds.
    pub compute_us: f64,
    /// 2026-10-22: The fused edges' round trip was dropped from `bytes`.
    pub fused: bool,
    /// 2026-10-22: Σ per-node `max(bytes / bandwidth, FLOPs / peak)` with every edge counted (the
    /// hardware report's estimate of these nodes), microseconds.
    pub node_sum_us: f64,
}

fn node_site(c: &Circuit, n: usize) -> String {
    format!("{}.{}", c.nodes[n].block, c.nodes[n].local)
}

/// 2026-10-22: The launch sites of `planned` (one run), each node costed at `kv_tokens` of
/// context when given, else at the class's assumption.
pub fn launch_sites(
    r: &Resolved,
    c: &Circuit,
    settings: &BTreeMap<String, String>,
    planned: &Planned,
    kv_tokens: Option<u64>,
) -> Result<Vec<LaunchSite>, HwError> {
    let plan = &planned.plan;
    let (mode, rows) = (plan.mode, plan.rows);
    let section = section_of(mode);
    let in_section: BTreeSet<usize> = c
        .blocks
        .iter()
        .filter(|b| b.section == section)
        .flat_map(|b| b.first..b.end)
        .collect();
    let mut sites: BTreeMap<(String, String), LaunchSite> = BTreeMap::new();
    for g in plan.groups.iter().filter(|g| in_section.contains(&g.nodes[0])) {
        let single = g.launch_count(rows) <= 1;
        let (mut bytes, mut flops, mut compute, mut node_sum) = (0.0, 0.0, 0.0, 0.0);
        for &n in &g.nodes {
            let node = &c.nodes[n];
            let mut rf = r.roofline_of(n);
            if let Some(kv) = kv_tokens {
                rf.context_tokens = kv;
            }
            let cost = node_cost(c, node, mode, rows, settings, &rf)?;
            node_sum += cost.time_us;
            bytes += cost.bytes;
            flops += cost.flops;
            compute += cost.flops / (peak_tflops(c, node, &rf) * 1e6);
            if single {
                for (e, b) in edge_traffic(c, node, rows)? {
                    if matches!(plan.edge_states[e], Some(EdgeState::Fused(_))) {
                        bytes -= b;
                    }
                }
            }
        }
        let label = group_site(c, g);
        let what = group_kernels(g);
        let site = sites
            .entry((label.clone(), what.clone()))
            .or_insert_with(|| LaunchSite {
                run: planned.run,
                label,
                nodes: g.nodes.iter().map(|&n| node_site(c, n)).collect(),
                ops: g
                    .nodes
                    .iter()
                    .map(|&n| {
                        let name = c.nodes[n].op.name();
                        name.split(':').next().unwrap_or_default().to_string()
                    })
                    .collect(),
                rules: BTreeSet::new(),
                what,
                kernels: g.kernels.clone(),
                groups: 0,
                launches: 0,
                bytes: 0.0,
                flops: 0.0,
                mem_us: 0.0,
                compute_us: 0.0,
                fused: single,
                node_sum_us: 0.0,
            });
        site.rules.insert(g.rule.clone());
        site.groups += 1;
        site.launches += g.launch_count(rows);
        site.bytes += bytes;
        site.flops += flops;
        site.compute_us += compute;
        site.node_sum_us += node_sum;
    }
    let bw = r.roofline.roofline.dram_gbps * 1e3;
    let mut out: Vec<LaunchSite> = sites.into_values().collect();
    for s in &mut out {
        s.mem_us = s.bytes / bw;
    }
    Ok(out)
}

/// 2026-10-22: Why a captured kernel is outside the plan.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outside {
    /// 2026-10-22: No launch site of the step launches a kernel of this name.
    Unknown,
    /// 2026-10-22: Several sites launch it and the capture did not say which launch is whose.
    Ambiguous(Vec<String>),
}

/// 2026-10-22: Where each captured kernel row went.
#[derive(Debug, Clone, PartialEq)]
pub struct Attribution {
    /// 2026-10-22: Per launch site (index into the sites): the rows it measured, and a note per
    /// row that runs another kernel of the same family than the plan names.
    pub measured: BTreeMap<usize, Vec<(TraceKernel, Option<String>)>>,
    /// 2026-10-22: Rows outside the plan.
    pub outside: Vec<(TraceKernel, Outside)>,
}

fn novel_what() -> String {
    format!("({})", super::plan::NOVEL_EMITTER)
}

fn family_of<'a>(r: &'a Resolved, func: &str) -> Vec<&'a str> {
    r.families
        .families
        .iter()
        .filter(|f| f.kernels.iter().any(|k| k.func == func))
        .map(|f| f.id.as_str())
        .collect()
}

/// 2026-10-22: Attribute `rows` to `sites`, or refuse with every mismatch listed.
pub fn attribute(
    r: &Resolved,
    sites: &[LaunchSite],
    rows: &[TraceKernel],
) -> Result<Attribution, Vec<String>> {
    let mut a = Attribution {
        measured: BTreeMap::new(),
        outside: Vec::new(),
    };
    let mut refused = Vec::new();
    let runs = |i: usize, k: &str| sites[i].kernels.iter().any(|x| x.func == k);
    for row in rows {
        let Some(site) = &row.site else {
            let hits: Vec<usize> = (0..sites.len()).filter(|&i| runs(i, &row.kernel)).collect();
            match hits.as_slice() {
                [i] => a.measured.entry(*i).or_default().push((row.clone(), row.note.clone())),
                [] => a.outside.push((row.clone(), Outside::Unknown)),
                many => {
                    let names = many.iter().map(|&i| sites[i].label.clone()).collect();
                    a.outside.push((row.clone(), Outside::Ambiguous(names)));
                }
            }
            continue;
        };
        let holding: Vec<usize> = (0..sites.len())
            .filter(|&i| sites[i].nodes.contains(site))
            .collect();
        if holding.is_empty() {
            refused.push(format!(
                "`{}` is attributed to site `{site}`, which no launch site of the step holds",
                row.kernel
            ));
            continue;
        }
        let exact: Vec<usize> = holding
            .iter()
            .copied()
            .filter(|&i| runs(i, &row.kernel))
            .collect();
        let fams = family_of(r, &row.kernel);
        let kin: Vec<(usize, String)> = holding
            .iter()
            .filter_map(|&i| {
                // 2026-10-22: The planned kernel of the same family that shares the longest
                // name prefix with the captured one is the one it stands in for.
                let prefix = |k: &KernelId| {
                    k.func
                        .chars()
                        .zip(row.kernel.chars())
                        .take_while(|(a, b)| a == b)
                        .count()
                };
                sites[i]
                    .kernels
                    .iter()
                    .filter_map(|k| {
                        let theirs = family_of(r, &k.func);
                        fams.iter().find(|f| theirs.contains(f)).map(|f| (k, *f))
                    })
                    .max_by_key(|(k, _)| prefix(k))
                    .map(|(k, f)| {
                        (i, format!("runs `{}`, the plan names `{k}` (family `{f}`)", row.kernel))
                    })
            })
            .collect();
        let same_op: Vec<(usize, String)> = holding
            .iter()
            .filter_map(|&i| {
                r.families
                    .families
                    .iter()
                    .filter(|f| f.kernels.iter().any(|k| k.func == row.kernel))
                    .find(|f| f.ops.iter().any(|o| sites[i].ops.contains(&o.op)))
                    .map(|f| {
                        (i, format!(
                            "runs `{}` of family `{}`, another implementation of the site's op than the plan's `{}`",
                            row.kernel, f.id, sites[i].what
                        ))
                    })
            })
            .collect();
        let novel: Vec<usize> = holding
            .iter()
            .copied()
            .filter(|&i| sites[i].kernels.is_empty() && sites[i].what == novel_what())
            .collect();
        let with = |note: Option<String>| {
            let parts: Vec<String> = note.into_iter().chain(row.note.clone()).collect();
            (row.clone(), (!parts.is_empty()).then(|| parts.join("; ")))
        };
        let kin = if kin.is_empty() { same_op } else { kin };
        match (exact.as_slice(), kin.as_slice(), novel.as_slice()) {
            ([i], _, _) => a.measured.entry(*i).or_default().push(with(None)),
            ([], [(i, note)], _) => a.measured.entry(*i).or_default().push(with(Some(note.clone()))),
            ([], [], [i]) => a.measured.entry(*i).or_default().push(with(Some(format!(
                "runs `{}` where the plan has no rule (a placeholder)",
                row.kernel
            )))),
            ([], [], []) => refused.push(format!(
                "`{}` is attributed to site `{site}`, whose launch sites run {}: neither that \
                 kernel nor one of its family",
                row.kernel,
                holding
                    .iter()
                    .map(|&i| format!("`{}`", sites[i].what))
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
            _ => refused.push(format!(
                "`{}` at site `{site}` matches several launch sites ({}); the capture must say \
                 which",
                row.kernel,
                holding
                    .iter()
                    .map(|&i| sites[i].label.clone())
                    .collect::<Vec<_>>()
                    .join(", ")
            )),
        }
    }
    if refused.is_empty() {
        Ok(a)
    } else {
        Err(refused)
    }
}
