// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `owed_lkb_golden` (queries doc §5): six kernels, one fully promoted, one missing
//! each of the four facets, one not launched by any plan and not compiled into the target, give
//! exactly the four owed rows with the right booleans; removing a `LAUNCHES` edge drops the row.
//! Plus the `--json --partial` inventory contract the host relies on.

use metrale_kg_derive::bundle::{Builder, Bundle};
use metrale_kg_derive::id::{KgId, Label, Rel};
use metrale_kg_derive::lkb::inventory;
use metrale_kg_derive::owed::owed_lkb_rows;
use serde_json::Value;
use std::collections::BTreeMap;

fn id(l: Label, k: &str) -> KgId {
    KgId::new(l, k).unwrap()
}

fn props(kv: &[(&str, Value)]) -> BTreeMap<String, Value> {
    kv.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

/// Kernels: `full` (all four facets), `nopoint`, `noclass`, `nomeas`, `noev`, `stray` (fully
/// promoted facets missing, but neither launched nor compiled into the target).
fn fixture(launch: &[&str]) -> Bundle {
    let mut b = Builder::default();
    let inst = id(Label::Instance, "i");
    let arch = id(Label::Model, "glm");
    let hw = id(Label::Hardware, "gb10");
    let plan = id(Label::Plan, "p");
    b.define(
        inst.clone(),
        "f".into(),
        "i",
        props(&[("target", "gb10/glm".into()), ("arch", "glm".into())]),
    );
    b.define(arch.clone(), "f".into(), "glm", BTreeMap::new());
    b.define(hw.clone(), "f".into(), "gb10", BTreeMap::new());
    b.define(
        id(Label::Model, "other"),
        "f".into(),
        "other",
        BTreeMap::new(),
    );
    b.define(plan.clone(), "f".into(), "p", BTreeMap::new());
    b.edge(&inst, Rel::Plans, &plan, 1.0, "f");
    let point = id(Label::FamilyPoint, "fp");
    b.define(point.clone(), "f".into(), "fp", BTreeMap::new());
    for k in ["full", "nopoint", "noclass", "nomeas", "noev", "stray"] {
        let kid = id(Label::Kernel, k);
        let comp = if k == "noclass" {
            Value::Null
        } else {
            Value::String("gemm".into())
        };
        let targets: Vec<&str> = if k == "stray" { vec!["b200/x"] } else { vec![] };
        b.define(
            kid.clone(),
            "f".into(),
            k,
            props(&[
                ("taxonomy_component", comp),
                ("targets", Value::from(targets)),
            ]),
        );
        if k != "nopoint" {
            b.edge(&kid, Rel::Instantiates, &point, 1.0, "f");
        }
        let m = id(Label::Measurement, &format!("{k} @ decode"));
        b.define(m.clone(), "f".into(), k, BTreeMap::new());
        b.edge(&m, Rel::Measures, &kid, 1.0, "f");
        let model = if k == "nomeas" {
            id(Label::Model, "other")
        } else {
            arch.clone()
        };
        b.edge(&m, Rel::ForModel, &model, 1.0, "f");
        b.edge(&m, Rel::Targets, &hw, 1.0, "f");
        if k != "noev" {
            let ev = id(Label::Evidence, k);
            b.define(ev.clone(), "f".into(), k, BTreeMap::new());
            b.edge(&ev, Rel::KeyedTo, &m, 1.0, "f");
        }
    }
    for k in launch {
        b.edge(&plan, Rel::Launches, &id(Label::Kernel, k), 1.0, "f");
    }
    b.seal(Vec::new())
}

fn rows(b: &Bundle) -> Vec<(String, [bool; 5])> {
    owed_lkb_rows(b, "i")
        .unwrap()
        .into_iter()
        .map(|o| {
            (
                o.kernel.key().to_string(),
                [o.landed, o.has_point, o.classified, o.measured, o.evidenced],
            )
        })
        .collect()
}

#[test]
fn exactly_the_four_kernels_missing_a_facet_owe_with_the_right_booleans() {
    let all = ["full", "nopoint", "noclass", "nomeas", "noev"];
    let got = rows(&fixture(&all));
    let want: Vec<(String, [bool; 5])> = vec![
        ("noclass".into(), [true, true, false, true, true]),
        ("noev".into(), [true, true, true, true, false]),
        ("nomeas".into(), [true, true, true, false, false]),
        ("nopoint".into(), [true, false, true, true, true]),
    ];
    assert_eq!(got, want);
}

#[test]
fn removing_the_launches_edge_drops_the_row() {
    let got = rows(&fixture(&["full", "nopoint", "noclass", "nomeas"]));
    assert!(got.iter().all(|(k, _)| k != "noev"), "{got:?}");
    assert_eq!(got.len(), 3);
}

#[test]
fn a_kernel_compiled_into_the_target_owes_even_when_no_plan_launches_it() {
    let mut b = fixture(&[]);
    let n = b
        .nodes
        .iter_mut()
        .find(|n| n.id == id(Label::Kernel, "noclass"))
        .unwrap();
    n.p.insert("targets".into(), Value::from(vec!["gb10/glm"]));
    assert_eq!(
        rows(&b),
        vec![("noclass".to_string(), [false, true, false, true, true])]
    );
}

const ROW: &str = r#"{"kernel":"scan::K","file":"kernels/gb10/common/scan.cu","line":3,"hardware":["gb10"],"targets":["gb10/glm"],"component":COMP,"kind":null,"engine_call_sites":[]}"#;

fn inv_json(comp: &str, problems: &str) -> String {
    format!(
        r#"{{"kernels":[{}],"problems":[{problems}]}}"#,
        ROW.replace("COMP", comp)
    )
}

#[test]
fn a_partial_inventory_keeps_an_unclassified_kernel_with_a_null_component() {
    let p = r#"{"kind":"unclassified","kernel":"kernels/gb10/common/scan.cu::K"}"#;
    let inv = inventory(&inv_json("null", p), "gb10").unwrap();
    assert_eq!(inv.kernels["scan::K"].component, None);
    assert!(inv.problems.is_empty());
}

#[test]
fn an_unclassified_problem_naming_a_classified_kernel_is_refused() {
    let p = r#"{"kind":"unclassified","kernel":"kernels/gb10/common/scan.cu::K"}"#;
    let e = inventory(&inv_json(r#""gemm""#, p), "gb10").unwrap_err();
    assert!(e.contains("no kernel row has a null component"), "{e}");
}

#[test]
fn other_problems_are_kept_and_the_strict_array_form_is_refused() {
    let p = r#"{"kind":"inconsistency","text":"rule x is stale"}"#;
    let inv = inventory(&inv_json(r#""gemm""#, p), "gb10").unwrap();
    assert_eq!(
        inv.problems,
        vec!["inconsistency: rule x is stale".to_string()]
    );
    let strict = format!("[{}]", ROW.replace("COMP", r#""gemm""#));
    assert!(
        inventory(&strict, "gb10")
            .unwrap_err()
            .contains("--partial")
    );
}
