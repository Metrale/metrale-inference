// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The roadmap matrix's pieces of a hardware report: its summary row and the
//! per-device port lists (`kernels/circuits/plans/hw/MATRIX.md`).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants: deterministic text, formatted as [`super::render`] formats the reports.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use super::HwReport;
use super::render::{FIT_CONTEXTS, FIT_SEQS, pct};
use crate::venn::Class;

/// 2026-09-30: One device's port list: kernel to (why the device cannot run it, how many of the
/// matrix's models need it).
pub type PortList = BTreeMap<String, (String, usize)>;

/// 2026-09-30: The port lists of a matrix: per device (in the order given), every rule kernel
/// some model's policy could select that the device cannot run, with why and how many of the
/// matrix's models are affected.
pub fn port_lists(per_device: &[(String, PortList)]) -> String {
    let mut s = String::from(
        "## Port lists\n\nRule kernels a matrix model's policy could select that the device cannot run (the class does not compile them, its build compiles them out, or the device lacks the instruction), with the number of matrix models whose policy names them.\n",
    );
    for (device, kernels) in per_device {
        let _ = writeln!(s, "\n### {device}\n");
        if kernels.is_empty() {
            let _ = writeln!(s, "None.");
            continue;
        }
        let _ = writeln!(s, "| kernel | why | models |\n|---|---|---:|");
        for (k, (why, n)) in kernels {
            let _ = writeln!(s, "| {k} | {} | {n} |", why.replace('|', "\\|"));
        }
    }
    s
}

/// 2026-09-30: The report's row of the roadmap matrix: coverage, the top five gaps at C=16,
/// and the memory fit.
pub fn summary_row(r: &HwReport) -> String {
    let usable = r.resolved.device.memory_bytes * r.resolved.device.usable_fraction;
    let fits = |c: u64| r.footprint.bytes(c, FIT_CONTEXTS[0]) <= usable;
    let max_c = FIT_SEQS.iter().copied().filter(|&c| fits(c)).max();
    let tok = |i: usize| {
        r.tables.get(i).map_or_else(
            || "-".into(),
            |t| format!("{:.0}", t.planned.plan.rows as f64 / (t.total_us / 1e6)),
        )
    };
    let wide = r.tables.get(1);
    let mut gaps: Vec<(f64, String)> = wide
        .map(|t| {
            t.rows
                .iter()
                .filter(|x| !matches!(x.class, Class::Shared | Class::SharedUnmeasured))
                .map(|x| {
                    (
                        x.share,
                        format!("{} ({}, {})", x.site, x.class.name(), pct(x.share)),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    if let Some(t) = wide {
        for f in &t.flags {
            let kind = f.layer_kind.map_or("prologue/epilogue", |k| k.name());
            gaps.push((f.share, format!("per-row {kind} ({})", pct(f.share))));
        }
    }
    gaps.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let top: Vec<String> = gaps.into_iter().take(5).map(|(_, g)| g).collect();
    format!(
        "| {} | {} | {} / {} | {} | {} / {} / {} | {} | {} | {} |",
        r.model.label,
        r.resolved.device.id,
        pct(r.covered(0, true)),
        pct(r.covered(1, true)),
        pct(r.covered(1, false)),
        tok(0),
        tok(1),
        tok(2),
        if fits(1) {
            "yes".to_string()
        } else {
            let need = r.footprint.bytes(1, FIT_CONTEXTS[0]);
            format!("no: TP={}", (need / usable).ceil() as u64)
        },
        max_c.map_or_else(|| "none".into(), |c| c.to_string()),
        if top.is_empty() {
            "none".into()
        } else {
            top.join("; ")
        }
    )
}
