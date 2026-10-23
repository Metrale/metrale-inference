// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The cost model a parallel layout is compared by: per rank, the weight bytes its
//! one-row decode step reads over the device bandwidth, plus each derived collective at the
//! link's measured per-collective latency and its payload over the link bandwidth. Ranks wait
//! for each other at every collective, so the step is not the slowest rank's whole-step total:
//! each collective starts when the last rank of its group arrives, and the step is the sum over
//! segments (the work between two collectives) of the slowest rank in that segment
//! ([`segmented`]). It ranks layouts; it does not predict a step time (no activation, KV or
//! launch terms), so a layout it prefers is a candidate to measure, never a change to the serve.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: the link and the bandwidth arrive as values (kernels/gb10/HARDWARE.toml `[collective]`,
//!   kernels/DEVICES.toml), each with its basis.
//! - A broadcast counts as one collective like the others: the f14 captures measured only the
//!   one-shot exchange, so a sub-group's extra broadcast is priced at the same latency.
//! - The segments are the rank circuits' own collective nodes (SSOT): a collective is matched
//!   across ranks by its node id and synchronises exactly the ranks its `params.group` names;
//!   a collective some member never reaches, or reaches in another order, is refused.
//! - One rank: no wait, so the step is that rank's [`RankCost::total_us`] (up to rounding).

use crate::prelude::*;

use crate::hardware::estimate::{main_weight_bytes, weight_floor_bytes};
use crate::ir::{Circuit, Node, OpKind};

/// 2026-10-23: The link between ranks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    /// 2026-10-23: Per collective, microseconds (kernel plus the idle before it).
    pub latency_us: f64,
    /// 2026-10-23: Payload bandwidth, GB/s.
    pub gbps: f64,
}

/// 2026-10-23: One rank's step under the model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RankCost {
    /// 2026-10-23: Weight bytes over the device bandwidth, microseconds.
    pub weight_us: f64,
    /// 2026-10-23: The collectives, microseconds.
    pub comm_us: f64,
    /// 2026-10-23: How many collectives.
    pub collectives: u64,
    /// 2026-10-23: Their payload bytes at one row.
    pub comm_bytes: u64,
}

impl RankCost {
    /// 2026-10-23: The rank's modelled step.
    pub fn total_us(&self) -> f64 {
        self.weight_us + self.comm_us
    }
}

/// 2026-10-23: `c`'s one-row step on a device of `dram_gbps` over `link`, as if it never waited.
pub fn rank_cost(c: &Circuit, dram_gbps: f64, link: Link) -> Result<RankCost, String> {
    let weight_us = weight_floor_bytes(c)? / (dram_gbps * 1e3);
    let (mut collectives, mut comm_bytes) = (0u64, 0u64);
    for n in c.nodes.iter().filter(|n| is_collective(n)) {
        collectives += 1;
        comm_bytes += payload(c, n)?;
    }
    let comm_us = collectives as f64 * link.latency_us + comm_bytes as f64 / (link.gbps * 1e3);
    Ok(RankCost {
        weight_us,
        comm_us,
        collectives,
        comm_bytes,
    })
}

fn is_collective(n: &Node) -> bool {
    matches!(
        n.op,
        OpKind::AllReduce | OpKind::AllGather | OpKind::Broadcast
    )
}

fn payload(c: &Circuit, n: &Node) -> Result<u64, String> {
    let e = &c.edges[n.outputs[0]];
    e.format.bytes(1, e.dim_value).ok_or_else(|| {
        format!(
            "collective `{}`: no byte size for {}",
            n.id,
            e.format.name()
        )
    })
}

/// 2026-10-23: One collective and the segment of work it closes.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// 2026-10-23: The collective's node id (the same on every member).
    pub collective: String,
    /// 2026-10-23: The ranks it synchronises (indices into the ranks priced).
    pub group: Vec<usize>,
    /// 2026-10-23: Per member (in `group` order), the weight time since its previous collective.
    pub work_us: Vec<f64>,
    /// 2026-10-23: Per member, how long it waits for the group's last arrival.
    pub wait_us: Vec<f64>,
}

/// 2026-10-23: A layout's step under synchronising collectives.
#[derive(Debug, Clone, PartialEq)]
pub struct Segmented {
    /// 2026-10-23: The step: when the last rank finishes.
    pub step_us: f64,
    /// 2026-10-23: Every collective in the order it completes.
    pub segments: Vec<Segment>,
    /// 2026-10-23: Per rank, the weight time after its last collective.
    pub tail_us: Vec<f64>,
}

impl Segmented {
    /// 2026-10-23: Per rank, the time it spends waiting at collectives.
    pub fn wait_per_rank(&self) -> Vec<f64> {
        let mut w = vec![0.0; self.tail_us.len()];
        for s in &self.segments {
            for (&r, x) in s.group.iter().zip(&s.wait_us) {
                w[r] += x;
            }
        }
        w
    }
}

struct Pending {
    work_bytes: f64,
    id: String,
    group: Vec<usize>,
    cost_us: f64,
}

/// 2026-10-23: Rank `r`'s collectives in program order, each with the weight bytes before it
/// and its cost, and the bytes after the last.
fn timeline(
    r: usize,
    ranks: usize,
    c: &Circuit,
    link: Link,
) -> Result<(Vec<Pending>, f64), String> {
    let bytes = main_weight_bytes(c)?;
    let (mut out, mut acc) = (Vec::new(), 0.0);
    for (i, n) in c.nodes.iter().enumerate() {
        acc += bytes[i];
        if !is_collective(n) {
            continue;
        }
        let text = n
            .params
            .get("group")
            .ok_or_else(|| format!("rank {r} collective `{}`: no `group`", n.id))?;
        let group = text
            .split(',')
            .map(|g| {
                g.trim()
                    .parse::<u64>()
                    .map_err(|e| e.to_string())
                    .and_then(|g| usize::try_from(g).map_err(|e| e.to_string()))
                    .and_then(|g| {
                        (g < ranks)
                            .then_some(g)
                            .ok_or(format!("rank {g} of {ranks}"))
                    })
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| format!("rank {r} collective `{}`: group `{text}`: {e}", n.id))?;
        if !group.contains(&r) {
            return Err(format!(
                "rank {r} holds collective `{}` of group {{{text}}}",
                n.id
            ));
        }
        let cost_us = link.latency_us + payload(c, n)? as f64 / (link.gbps * 1e3);
        out.push(Pending {
            work_bytes: core::mem::take(&mut acc),
            id: n.id.clone(),
            group,
            cost_us,
        });
    }
    Ok((out, acc))
}

/// 2026-10-23: The step of `ranks` (rank `r` is `ranks[r]`): each rank runs its work up to its
/// next collective; a collective starts when the last member of its group arrives and ends
/// after its cost (the largest any member states); the step ends with the last rank's tail.
pub fn segmented(ranks: &[Circuit], dram_gbps: f64, link: Link) -> Result<Segmented, String> {
    let to_us = |b: f64| b / (dram_gbps * 1e3);
    let mut lines = Vec::with_capacity(ranks.len());
    for (r, c) in ranks.iter().enumerate() {
        lines.push(timeline(r, ranks.len(), c, link)?);
    }
    let n = ranks.len();
    let (mut clock, mut at) = (vec![0.0f64; n], vec![0usize; n]);
    let mut segments = Vec::new();
    loop {
        let next = (0..n).find_map(|r| {
            let p = lines[r].0.get(at[r])?;
            let ready = p.group.iter().all(|&g| {
                lines
                    .get(g)
                    .and_then(|l| l.0.get(at[g]))
                    .is_some_and(|q| q.id == p.id && q.group == p.group)
            });
            ready.then(|| p.group.clone())
        });
        let Some(group) = next else { break };
        let arrive: Vec<f64> = group
            .iter()
            .map(|&g| clock[g] + to_us(lines[g].0[at[g]].work_bytes))
            .collect();
        let start = arrive.iter().copied().fold(f64::MIN, f64::max);
        let cost = group
            .iter()
            .map(|&g| lines[g].0[at[g]].cost_us)
            .fold(0.0f64, f64::max);
        let first = &lines[group[0]].0[at[group[0]]];
        segments.push(Segment {
            collective: first.id.clone(),
            group: group.clone(),
            work_us: group
                .iter()
                .map(|&g| to_us(lines[g].0[at[g]].work_bytes))
                .collect(),
            wait_us: arrive.iter().map(|a| start - a).collect(),
        });
        for &g in &group {
            clock[g] = start + cost;
            at[g] += 1;
        }
    }
    if let Some(r) = (0..n).find(|&r| at[r] < lines[r].0.len()) {
        let p = &lines[r].0[at[r]];
        return Err(format!(
            "rank {r} waits forever at collective `{}`: a member of group {:?} never reaches it \
             next",
            p.id, p.group
        ));
    }
    let tail_us: Vec<f64> = lines.iter().map(|l| to_us(l.1)).collect();
    let step_us = (0..n).map(|r| clock[r] + tail_us[r]).fold(0.0f64, f64::max);
    Ok(Segmented {
        step_us,
        segments,
        tail_us,
    })
}

/// 2026-10-23: The modelled step of a layout ([`segmented`]).
pub fn step_us(ranks: &[Circuit], dram_gbps: f64, link: Link) -> Result<f64, String> {
    Ok(segmented(ranks, dram_gbps, link)?.step_us)
}

/// 2026-10-23: Per node of `c`, the segment it runs in: how many collectives precede it (a
/// collective belongs to the segment it closes).
pub fn segment_of_nodes(c: &Circuit) -> Vec<usize> {
    let mut k = 0;
    c.nodes
        .iter()
        .map(|n| {
            let here = k;
            k += usize::from(is_collective(n));
            here
        })
        .collect()
}

/// 2026-10-23: The least slowest-rank work a segment can have when each split part may rotate
/// which rank takes which slice (`Remainder::Rotate { offset }`, priced only): rank `r` reads
/// `base[r]` plus, for each part `p`, slice `(r + offset_p) % d` of `parts[p]`. Returns the best
/// slowest rank and the offsets that reach it (all zero, today's `Front` placement, on a tie).
/// Exhaustive over `d^parts` offsets, refused above `limit` combinations.
pub fn rotate_floor(
    base: &[f64],
    parts: &[Vec<f64>],
    limit: u64,
) -> Result<(f64, Vec<usize>), String> {
    let d = base.len();
    if d == 0 || parts.iter().any(|p| p.len() != d) {
        return Err(format!("rotate: {d} ranks but a part of another width"));
    }
    let limit = usize::try_from(limit).unwrap_or(usize::MAX);
    parts
        .iter()
        .try_fold(1usize, |n, _| n.checked_mul(d).filter(|n| *n <= limit))
        .ok_or_else(|| format!("rotate: {d}^{} offsets exceed {limit}", parts.len()))?;
    let slowest = |off: &[usize]| {
        (0..d)
            .map(|r| {
                base[r]
                    + parts
                        .iter()
                        .zip(off)
                        .map(|(p, o)| p[(r + o) % d])
                        .sum::<f64>()
            })
            .fold(f64::MIN, f64::max)
    };
    let mut off = vec![0usize; parts.len()];
    let mut best = (slowest(&off), off.clone());
    // 2026-10-23: An odometer over every offset vector, today's (all zero) first.
    while let Some(p) = off.iter().position(|o| o + 1 < d) {
        off[p] += 1;
        off[..p].fill(0);
        let m = slowest(&off);
        if m < best.0 {
            best = (m, off.clone());
        }
    }
    Ok(best)
}

/// 2026-10-23: The `[collective]` link of a class's HARDWARE.toml text, with its basis; `None`
/// when the class states none (no layout is priced on it).
pub fn parse_link(hardware_toml: &str) -> Result<Option<(Link, String)>, String> {
    let t: toml::Table = toml::from_str(hardware_toml).map_err(|e| e.to_string())?;
    let Some(c) = t.get("collective") else {
        return Ok(None);
    };
    let num = |k: &str| {
        c.get(k)
            .and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64)))
            .filter(|x| *x > 0.0)
            .ok_or_else(|| format!("[collective] `{k}` must be a positive number"))
    };
    let basis = c
        .get("basis")
        .and_then(|v| v.as_str())
        .filter(|s| !s.trim().is_empty())
        .ok_or("[collective] states no `basis`")?;
    Ok(Some((
        Link {
            latency_us: num("latency_us")?,
            gbps: num("gbps")?,
        },
        basis.to_string(),
    )))
}
