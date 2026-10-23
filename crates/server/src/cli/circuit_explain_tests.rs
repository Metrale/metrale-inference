// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met circuit plan --format explain` on the checked-out tree: every node of a
//! golden gb10 plan that a family runs is named with that family (multi-kernel groups included),
//! every named family and stored format has its math, and every edge its size.
//!
//! Owner: server CLI tests.
//! Invariants: reads the repository at the workspace root; writes nothing.

use std::path::PathBuf;

use metrale_circuit::hardware::model::model_of;
use metrale_circuit::hardware::{self, PrecisionChoice};
use metrale_circuit::venn::Run;
use metrale_circuit::{Mode, parse_instances};
use serde_json::Value;

use super::explain_json;
use crate::cli::circuit_hw::{FsTree, registry};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn plans() -> Vec<(hardware::ModelUnderPlan, hardware::OnePlan)> {
    let t = FsTree::new(root());
    let reg = registry(&t).unwrap();
    let text = std::fs::read_to_string(root().join("kernels/circuits/INSTANCES.toml")).unwrap();
    parse_instances(&text)
        .unwrap()
        .iter()
        .filter(|i| i.golden)
        .map(|inst| {
            let m = model_of(&t, inst, "recipe".into(), PrecisionChoice::Recipe).unwrap();
            let run = Run {
                mode: Mode::Decode,
                rows: 1,
            };
            let one = hardware::plan_one(&reg, "gb10", &t, &m, run).unwrap();
            (m, one)
        })
        .collect()
}

#[test]
fn every_planned_node_a_family_runs_names_it_and_its_math() {
    let mut multi = 0;
    for (m, one) in plans() {
        let doc: Value = serde_json::from_str(&explain_json(&m.circuit, &one).unwrap()).unwrap();
        assert_eq!(doc["schema"], "kernel-math/v1");
        let fams = doc["families"].as_object().unwrap();
        for g in &one.planned.plan.groups {
            for &i in &g.nodes {
                let n = &m.circuit.nodes[i];
                let want = one
                    .resolved
                    .families
                    .of_group(&g.kernels, &g.emitter, &n.op);
                let got = doc["nodes"].get(&n.id).and_then(Value::as_str);
                assert_eq!(got, want.map(|f| f.id.as_str()), "{}", n.id);
                if let Some(f) = got {
                    assert!(!fams[f]["math"].as_array().unwrap().is_empty(), "{f}");
                    multi += usize::from(g.kernels.len() > 1);
                }
            }
        }
        for (i, e) in m.circuit.edges.iter().enumerate() {
            let t = &doc["tensors"][&e.id];
            let stored = one.planned.plan.edge_formats[i].name();
            assert_eq!(t["format"], stored.as_str(), "{}", e.id);
            assert_eq!(t["dim_value"], e.dim_value, "{}", e.id);
            assert!(
                doc["formats"][stored.as_str()]["math"].is_array(),
                "{stored}"
            );
        }
    }
    assert!(multi > 0, "some golden node runs in a multi-kernel group");
}
