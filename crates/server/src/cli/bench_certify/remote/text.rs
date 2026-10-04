// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The fleet plan, for a human (stderr) and for a script (JSON).
//!
//! Owner: server CLI (`met benchmark certify`).
//! Invariants: none beyond the types.

use metrale_bench::hardware::energy::JOULES_PER_TOKEN_KEY;

use super::super::plan::Unit;
use super::super::text::human;
use super::Fleet;
use super::schedule::{Plan, SpeedMode};

pub fn fleet_json(f: &Fleet, units: &[Unit], plan: &Plan) -> serde_json::Value {
    serde_json::json!({
        "nodes": f.nodes.iter().map(|n| serde_json::json!({
            "addr": n.addr, "name": n.name, "node_id": n.node_id, "signer": n.signer,
            "local": n.local, "built": n.built,
            "gpu": n.hardware.gpu, "driver_major": n.hardware.driver_major,
            "sm_clock_max_mhz": n.hardware.sm_clock_max_mhz, "mem_total_kb": n.hardware.mem_total_kb,
            "thermal_alert": n.hardware.thermal_alert, "hottest_chassis_c": n.hardware.hottest_chassis_c,
            "free_fraction": n.free_fraction,
        })).collect::<Vec<_>>(),
        "rejected": f.rejected.iter().map(|r| serde_json::json!({ "addr": r.addr, "why": r.why })).collect::<Vec<_>>(),
        "speed_mode": match &f.mode {
            SpeedMode::Spread => serde_json::json!({ "mode": "spread" }),
            SpeedMode::Bundle { node, why } => serde_json::json!({
                "mode": "bundle", "node": f.nodes[*node].addr, "why": why,
            }),
        },
        "energy_pin": f.energy.as_ref().map(|p| serde_json::json!({
            "node": f.nodes[p.node].addr, "gates": p.gates,
        })),
        "queues": plan.queues.iter().enumerate().map(|(k, q)| serde_json::json!({
            "node": f.nodes[k].addr,
            "units": q.iter().map(|&i| units[i].label()).collect::<Vec<_>>(),
            "finish_at_secs": plan.finish_at[k],
        })).collect::<Vec<_>>(),
        "makespan_secs": plan.makespan_secs,
    })
}

pub fn print_fleet(f: &Fleet, units: &[Unit], plan: &Plan) {
    eprintln!("certify: fleet");
    for (k, n) in f.nodes.iter().enumerate() {
        eprintln!(
            "  {:<18} {:<12} signer {}  {}{}  chassis {}  free {}",
            n.addr,
            n.name,
            &n.signer[..n.signer.len().min(12)],
            n.hardware.gpu,
            if n.built { "  (anchor built)" } else { "" },
            n.hardware
                .hottest_chassis_c
                .map_or("n/a".to_owned(), |c| format!("{c:.0} °C")),
            n.free_fraction
                .map_or("n/a".to_owned(), |x| format!("{:.0} %", x * 100.0)),
        );
        let q: Vec<String> = plan.queues[k]
            .iter()
            .map(|&i| format!("{} ({})", units[i].label(), human(units[i].secs())))
            .collect();
        eprintln!(
            "    plan: {}  → done at ~{}",
            if q.is_empty() {
                "idle".to_owned()
            } else {
                q.join(", ")
            },
            human(plan.finish_at[k])
        );
    }
    for r in &f.rejected {
        eprintln!("  REJECTED {}", r);
    }
    match &f.mode {
        SpeedMode::Spread => eprintln!(
            "  speed-class gates SPREAD across {} node(s): every pair is one box by \
             the equivalence policy",
            f.nodes.len()
        ),
        SpeedMode::Bundle { node, why } => {
            eprintln!(
                "  WARNING speed-class gates BUNDLED on {}: the nodes are not one box —",
                f.nodes[*node].addr
            );
            for w in why {
                eprintln!("    {w}");
            }
        }
    }
    for line in energy_pin_lines(f) {
        eprintln!("  {line}");
    }
    eprintln!("  makespan ~{}", human(plan.makespan_secs));
}

/// 2026-10-04: The reference `--energy-reference-node` named, and which
/// energy-bounded gates it pins, and why.
fn energy_pin_lines(f: &Fleet) -> Vec<String> {
    let Some(pin) = &f.energy else {
        return Vec::new();
    };
    let addr = &f.nodes[pin.node].addr;
    let gates = if pin.gates.is_empty() {
        "no energy-bounded gate in this campaign".to_string()
    } else {
        format!(
            "energy-bounded gates PINNED: {}. Their default entries bound a \
             *{JOULES_PER_TOKEN_KEY} metric, and the GPU rail reads differently box to box",
            pin.gates.iter().copied().collect::<Vec<_>>().join(", ")
        )
    };
    vec![format!(
        "reference {addr} (--energy-reference-node) hosts every Speed-class gate; {gates}"
    )]
}

#[cfg(test)]
#[path = "text_tests.rs"]
mod text_tests;
