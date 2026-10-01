// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The batched MTP verify's plans against the legacy forward's launches. For every
//! traced batch (`fixtures/verify_batch_traces.toml`: nsys traces of the legacy
//! `decode_verify_batched` at 8 to 80 rows, carried and not, under both weight tiers of the dense
//! 27B), the reference plan of its row table launches exactly the traced kernels, each as many
//! times.
//!
//! One known difference is stated, not hidden: under the declared tier the legacy attention
//! projects Q|K|V with one stacked W8A8 launch (`qwen3_attention/w8a8_decode_arm.rs:105-131`),
//! where every circuit plan (decode, multi-sequence, verify) launches the three segments
//! apart. Each output column is its own dot product, so the bytes agree (the parity diffs), but
//! the plan launches two more per attention layer; [`STACKED_QKV_EXTRA`] accounts for them.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeMap;

use metrale_circuit::{Instance, Numerics, RowTable, fuse_table};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Traces {
    schema: u32,
    trace: Vec<Trace>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Trace {
    tier: String,
    table: String,
    launches: u64,
    kernels: BTreeMap<String, u64>,
}

/// 2026-09-30: The circuit's extra W8A8 launches per attention layer under the declared tier:
/// Q, K and V apart where legacy stacks them.
const STACKED_QKV_EXTRA: u64 = 2;

fn instance(tier: &str) -> Instance {
    let recipe = match tier {
        "declared" => "qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared",
        "nvfp4" => "qwen3.8/qwen3.8-27b-nvfp4-unsloth",
        other => panic!("no instance for tier `{other}`"),
    };
    common::instances()
        .into_iter()
        .find(|i| i.recipe == recipe)
        .expect("instance")
}

/// 2026-09-30: The launches of `plan` by kernel function.
fn launches_by_kernel(plan: &metrale_circuit::FusionPlan) -> BTreeMap<String, u64> {
    let mut out = BTreeMap::new();
    for g in &plan.groups {
        if g.runs.is_empty() {
            let reps = g.repeat.count(plan.rows).expect("a row-counted group");
            for k in &g.kernels {
                *out.entry(k.func.clone()).or_default() += reps;
            }
        } else {
            for r in &g.runs {
                for (k, t) in &r.launches {
                    *out.entry(k.func.clone()).or_default() += t.count(r.run);
                }
            }
        }
    }
    out
}

// 2026-09-30: Mutation: widening any batched-verify rule's rows, or giving a run another
// selector, moves a launch to another kernel and fails this.
#[test]
fn every_traced_batch_plans_exactly_the_legacy_launches() {
    let traces: Traces = toml::from_str(&common::read(
        "crates/circuit/tests/fixtures/verify_batch_traces.toml",
    ))
    .expect("traces parse");
    assert_eq!(traces.schema, 1);
    assert!(traces.trace.len() >= 20, "{} traces", traces.trace.len());
    let mut failures = Vec::new();
    for t in &traces.trace {
        let inst = instance(&t.tier);
        let loaded = common::load(&inst);
        let mut avail = common::available(&inst, &loaded.rules);
        // 2026-09-30: The reference plan: no `bit_identical` fusion, as the legacy forward runs.
        for r in &loaded.rules {
            if matches!(r.numerics, Numerics::BitIdentical { .. }) {
                for k in &r.kernels {
                    avail.kernels.remove(k);
                }
            }
        }
        let table = RowTable::parse(&t.table).expect("table parses");
        let plan = fuse_table(&loaded.circuit, &loaded.rules, &avail, &inst.policy, &table)
            .unwrap_or_else(|e| panic!("{} `{}`: {e}", t.tier, t.table));
        let mut got = launches_by_kernel(&plan);
        let mut launches = plan.launches();
        if t.tier == "declared" {
            let attn = loaded
                .circuit
                .layer_kinds
                .iter()
                .filter(|k| **k == metrale_circuit::LayerKind::FullAttention)
                .count() as u64;
            let gemv = got
                .keys()
                .find(|k| k.starts_with("w8a8_gemv_rowscale"))
                .cloned()
                .expect("a declared plan launches W8A8 GEMVs");
            *got.get_mut(&gemv).expect("present") -= STACKED_QKV_EXTRA * attn;
            launches -= STACKED_QKV_EXTRA * attn;
        }
        if got != t.kernels || launches != t.launches {
            let diff: Vec<String> = got
                .keys()
                .chain(t.kernels.keys())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .filter(|k| got.get(*k) != t.kernels.get(*k))
                .map(|k| format!("{k}: plan {:?} legacy {:?}", got.get(k), t.kernels.get(k)))
                .collect();
            failures.push(format!(
                "{} `{}`: {} launches, legacy {}; {}",
                t.tier,
                t.table,
                plan.launches(),
                t.launches,
                diff.join(", ")
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
