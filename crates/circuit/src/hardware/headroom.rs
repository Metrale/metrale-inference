// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Headroom (`met circuit headroom`): for one step of a plan on a device, how far each
//! launch site is from its floor, what the step and tok/s would be if that site alone reached its
//! floor (or a stated fraction of it), and where the time not spent in kernels goes.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - A site's floor is `max(bytes / bandwidth, FLOPs / peak)` over its nodes
//!   ([`super::headroom_sites`]), raised to `launches × launch floor` when a launch floor is
//!   stated; its bound names the larger term.
//! - Measured means captured: a site's time comes only from a trace's kernels attributed to it.
//!   Without one it is `Unmeasured` and carries its projection (the floor itself); it never enters
//!   a measured sum, a pct-of-floor, or an upside.
//! - Accounting closes: measured step = Σ sites' exclusive time + Σ outside-the-plan kernels'
//!   exclusive time + residual (GPU idle: host work, syncs, waits, launch gaps), exactly.
//! - Upside is Amdahl's law under the exclusive model: a site that falls from `t` to `floor / p`
//!   saves `min(t - floor / p, its exclusive time)`, never more than its time on the critical path
//!   alone. Sites' exclusive times are disjoint, so savings add along the cumulative curve.
//! - Energy columns are filled only from captured energy; with none they are `None` ("–").

use crate::prelude::*;

use alloc::collections::BTreeMap;

pub use super::headroom_model::*;
pub use super::headroom_sites::LaunchSite;
use super::headroom_sites::{attribute, launch_sites};
use super::headroom_svg::Ceilings;
use super::headroom_trace::{Stated, Trace, TraceKernel};
use super::plan::{Planned, Resolved};
use crate::ir::Circuit;

fn run_name(p: &Planned) -> String {
    format!("{} n={}", p.plan.mode.name(), p.plan.rows)
}

fn tok_s(tokens: f64, step_us: f64) -> f64 {
    if step_us > 0.0 { tokens * 1e6 / step_us } else { 0.0 }
}

fn per_step(rows: &[(TraceKernel, Option<String>)], steps: f64) -> Measured {
    let mut kernels: Vec<String> = rows.iter().map(|(k, _)| k.kernel.clone()).collect();
    kernels.sort();
    kernels.dedup();
    Measured {
        calls_per_step: rows.iter().map(|(k, _)| k.calls as f64).sum::<f64>() / steps,
        total_us: rows.iter().map(|(k, _)| k.total_us).sum::<f64>() / steps,
        exclusive_us: rows.iter().map(|(k, _)| k.exclusive_us).sum::<f64>() / steps,
        kernels,
        drift: rows.iter().filter_map(|(_, d)| d.clone()).collect(),
    }
}

/// 2026-10-22: Time saved by `m` falling to `floor / p` (p in (0, 1]), capped at its exclusive
/// time.
pub fn saved_us(m: &Measured, floor_us: f64, p: f64) -> f64 {
    (m.total_us - floor_us / p).clamp(0.0, m.exclusive_us)
}

fn energy_cols(
    rows: &[(TraceKernel, Option<String>)],
    site: &LaunchSite,
    steps: f64,
    e: Option<&EnergyRoofline>,
) -> EnergyCols {
    let captured: Option<f64> = rows.iter().map(|(k, _)| k.energy_uj).sum::<Option<f64>>();
    let per_step = captured.filter(|_| !rows.is_empty()).map(|j| j / steps);
    let floor = e.map(|e| (site.bytes * e.pj_per_byte + site.flops * e.pj_per_flop) * 1e-6);
    EnergyCols {
        uj_per_launch: per_step
            .filter(|_| site.launches > 0)
            .map(|j| j / site.launches as f64),
        floor_uj: floor,
        pct_of_floor: per_step
            .zip(floor)
            .filter(|(j, _)| *j > 0.0)
            .map(|(j, f)| 100.0 * f / j),
        upside_uj: per_step.zip(floor).map(|(j, f)| (j - f).max(0.0)),
    }
}

/// 2026-10-22: The headroom of the step that runs `runs` (each a plan of `c` on `r`).
pub fn headroom(
    r: &Resolved,
    c: &Circuit,
    settings: &BTreeMap<String, String>,
    runs: &[Planned],
    inputs: &Inputs<'_>,
) -> Result<Headroom, HeadroomError> {
    if !(inputs.target_pct > 0.0 && inputs.target_pct <= 100.0) {
        return Err(HeadroomError::Trace(format!(
            "target {}% of the floor is outside (0, 100]",
            inputs.target_pct
        )));
    }
    let kv = inputs.kv_tokens.clone().unwrap_or_else(|| Stated {
        value: r.roofline.roofline.context_tokens as f64,
        basis: "the class's [roofline] context_tokens assumption".into(),
    });
    let kv_tokens = inputs.kv_tokens.as_ref().map(|_| {
        let v = kv.value.max(1.0);
        crate::num::sat_u64(v)
    });
    let mut sites = Vec::new();
    for p in runs {
        sites.extend(launch_sites(r, c, settings, p, kv_tokens)?);
    }
    let trace = inputs.trace;
    if let Some(t) = trace {
        t.check().map_err(HeadroomError::Trace)?;
    }
    let attr = match trace {
        Some(t) => Some(attribute(r, &sites, &t.kernels).map_err(HeadroomError::Mismatch)?),
        None => None,
    };
    let steps = trace.map_or(1.0, |t| t.steps as f64);
    let step_us = trace.map(Trace::step_us);
    let tokens = inputs.tokens_per_step.value;
    let p_target = inputs.target_pct / 100.0;
    let launch = inputs.launch_floor_us.as_ref().map(|l| l.value);
    let mut rows: Vec<SiteRow> = sites
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let launch_us = launch.map(|l| l * s.launches as f64);
            let roof = s.mem_us.max(s.compute_us);
            let (floor_us, bound) = match launch_us {
                Some(l) if l > roof => (l, Bound::Launch),
                _ if s.compute_us > s.mem_us => (roof, Bound::Compute),
                _ => (roof, Bound::Memory),
            };
            let captured = attr.as_ref().and_then(|a| a.measured.get(&i));
            let measured = captured.map(|rows| per_step(rows, steps));
            let upside = |p: f64| {
                measured.as_ref().zip(step_us).map(|(m, step)| {
                    let saved = saved_us(m, floor_us, p);
                    Upside {
                        saved_us: saved,
                        step_us: step - saved,
                        tok_s: tok_s(tokens, step - saved),
                    }
                })
            };
            SiteRow {
                run: format!("{} n={}", s.run.mode.name(), s.run.rows),
                site: s.label.clone(),
                kernels: s.what.clone(),
                rules: s.rules.iter().cloned().collect(),
                groups: s.groups,
                launches: s.launches,
                bytes: s.bytes,
                flops: s.flops,
                fused: s.fused,
                floor_us,
                bound,
                node_sum_us: s.node_sum_us,
                pct_of_floor: measured
                    .as_ref()
                    .filter(|m| m.total_us > 0.0)
                    .map(|m| 100.0 * floor_us / m.total_us),
                headroom_us: measured.as_ref().map(|m| m.total_us - floor_us),
                upside_full: upside(1.0),
                upside_target: upside(p_target),
                energy: energy_cols(
                    captured.map_or(&[][..], Vec::as_slice),
                    s,
                    steps,
                    inputs.energy.as_ref(),
                ),
                measured,
            }
        })
        .collect();
    let key = |r: &SiteRow| r.upside_full.map_or(0.0, |u| u.saved_us);
    rows.sort_by(|a, b| {
        key(b)
            .total_cmp(&key(a))
            .then_with(|| b.floor_us.total_cmp(&a.floor_us))
            .then_with(|| a.run.cmp(&b.run))
            .then_with(|| a.site.cmp(&b.site))
    });
    let outside = attr.as_ref().map_or_else(Vec::new, |a| {
        let mut o: Vec<OutsideRow> = a
            .outside
            .iter()
            .map(|(k, why)| OutsideRow {
                kernel: k.kernel.clone(),
                label: k.label.clone().or_else(|| k.site.clone()),
                calls_per_step: k.calls as f64 / steps,
                total_us: k.total_us / steps,
                exclusive_us: k.exclusive_us / steps,
                why: why.clone(),
            })
            .collect();
        o.sort_by(|a, b| {
            b.exclusive_us
                .total_cmp(&a.exclusive_us)
                .then_with(|| a.kernel.cmp(&b.kernel))
        });
        o
    });
    let totals = totals(&rows, &outside, step_us, tokens);
    let curve = curve(&rows, step_us, tokens, p_target);
    Ok(Headroom {
        schema: "headroom/v1".into(),
        model: inputs.model.clone(),
        device: r.device.id.clone(),
        roofline: r.roofline.basis.clone(),
        ceilings: Ceilings {
            dram_gbps: r.roofline.roofline.dram_gbps,
            bf16_tflops: r.roofline.roofline.bf16_tflops,
            fp8_tflops: r.roofline.roofline.fp8_tflops,
            nvfp4_tflops: r.roofline.roofline.nvfp4_tflops,
        },
        runs: runs.iter().map(run_name).collect(),
        kv_tokens: kv,
        tokens_per_step: inputs.tokens_per_step.clone(),
        launch_floor_us: inputs.launch_floor_us.clone(),
        dims: inputs.dims.0.clone(),
        dims_basis: inputs.dims.1.clone(),
        target_pct: inputs.target_pct,
        trace: trace.map(|t| t.source.clone()),
        energy: inputs.energy.as_ref().map(|e| e.basis.clone()),
        sites: rows,
        outside,
        curve,
        totals,
    })
}

fn totals(rows: &[SiteRow], outside: &[OutsideRow], step: Option<f64>, tokens: f64) -> Totals {
    let ceiling: f64 = rows.iter().map(|r| r.floor_us).sum();
    let measured = || rows.iter().filter_map(|r| r.measured.as_ref());
    let gate = |x: f64| step.map(|_| x);
    let sites_total: f64 = measured().map(|m| m.total_us).sum();
    let sites_excl: f64 = measured().map(|m| m.exclusive_us).sum();
    let out_total: f64 = outside.iter().map(|o| o.total_us).sum();
    let out_excl: f64 = outside.iter().map(|o| o.exclusive_us).sum();
    Totals {
        ceiling_us: ceiling,
        ceiling_tok_s: tok_s(tokens, ceiling),
        measured_floor_us: gate(
            rows.iter()
                .filter(|r| r.measured.is_some())
                .map(|r| r.floor_us)
                .sum(),
        ),
        unmeasured_projection_us: rows
            .iter()
            .filter(|r| r.measured.is_none())
            .map(|r| r.floor_us)
            .sum(),
        step_us: step,
        step_tok_s: step.map(|s| tok_s(tokens, s)),
        sites_total_us: gate(sites_total),
        sites_exclusive_us: gate(sites_excl),
        outside_exclusive_us: gate(out_excl),
        overlap_us: gate(sites_total + out_total - sites_excl - out_excl),
        residual_us: step.map(|s| s - sites_excl - out_excl),
    }
}

fn curve(rows: &[SiteRow], step: Option<f64>, tokens: f64, p: f64) -> Vec<CurvePoint> {
    let Some(step) = step else {
        return Vec::new();
    };
    let mut gains: Vec<(&SiteRow, f64, f64)> = rows
        .iter()
        .filter_map(|r| {
            let m = r.measured.as_ref()?;
            Some((r, saved_us(m, r.floor_us, 1.0), saved_us(m, r.floor_us, p)))
        })
        .filter(|(_, full, _)| *full > 0.0)
        .collect();
    gains.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.site.cmp(&b.0.site)));
    let (mut full, mut target) = (0.0, 0.0);
    gains
        .iter()
        .enumerate()
        .map(|(i, (r, f, t))| {
            full += f;
            target += t;
            let at = |saved: f64| Upside {
                saved_us: saved,
                step_us: step - saved,
                tok_s: tok_s(tokens, step - saved),
            };
            CurvePoint {
                n: i + 1,
                site: format!("{} · {}", r.run, r.site),
                full: at(full),
                target: at(target),
            }
        })
        .collect()
}
