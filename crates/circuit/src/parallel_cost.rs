// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The cost model a parallel layout is compared by: per rank, the weight bytes its
//! one-row decode step reads over the device bandwidth, plus each derived collective at the
//! link's measured per-collective latency and its payload over the link bandwidth; the step is
//! its slowest rank. It ranks layouts; it does not predict a step time (no activation, KV or
//! launch terms), so a layout it prefers is a candidate to measure, never a change to the serve.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: the link and the bandwidth arrive as values (kernels/gb10/HARDWARE.toml `[collective]`,
//!   kernels/DEVICES.toml), each with its basis.
//! - A broadcast counts as one collective like the others: the f14 captures measured only the
//!   one-shot exchange, so a sub-group's extra broadcast is priced at the same latency.

use crate::prelude::*;

use crate::hardware::estimate::weight_floor_bytes;
use crate::ir::{Circuit, OpKind};

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

/// 2026-10-23: `c`'s one-row step on a device of `dram_gbps` over `link`.
pub fn rank_cost(c: &Circuit, dram_gbps: f64, link: Link) -> Result<RankCost, String> {
    let weight_us = weight_floor_bytes(c)? / (dram_gbps * 1e3);
    let (mut collectives, mut comm_bytes) = (0u64, 0u64);
    for n in &c.nodes {
        if !matches!(n.op, OpKind::AllReduce | OpKind::AllGather | OpKind::Broadcast) {
            continue;
        }
        let e = &c.edges[n.outputs[0]];
        let bytes = e
            .format
            .bytes(1, e.dim_value)
            .ok_or_else(|| format!("collective `{}`: no byte size for {}", n.id, e.format.name()))?;
        collectives += 1;
        comm_bytes += bytes;
    }
    let comm_us = collectives as f64 * link.latency_us + comm_bytes as f64 / (link.gbps * 1e3);
    Ok(RankCost {
        weight_us,
        comm_us,
        collectives,
        comm_bytes,
    })
}

/// 2026-10-23: The modelled step of a layout: its slowest rank.
pub fn step_us(ranks: &[Circuit], dram_gbps: f64, link: Link) -> Result<f64, String> {
    ranks.iter().try_fold(0.0f64, |m, c| {
        Ok(m.max(rank_cost(c, dram_gbps, link)?.total_us()))
    })
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
