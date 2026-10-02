// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: FP4 costing follows the class's compiled kernel set, not the device's datasheet:
//! an NVFP4-activation node is costed at the NVFP4 peak only where a rule runs it through a
//! compiled FP4 block-scale kernel, the report's mark names the ops that have none, and every
//! number estimated from datasheet constants is labelled a roofline projection.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use std::collections::{BTreeMap, BTreeSet};

use super::avail::{Absence, guard_of};
use super::device::{Guard, Instr, MmaKind, Polarity};
use super::estimate::prefill_us;
use super::exec::Exec;
use super::gaps::gap_table;
use super::plan::Resolved;
use super::test_fixture::{self as fx, Dev};
use super::{Registry, build_report, plan_one, render_report, summary_row};
use crate::ir::{Circuit, NodeIdx};
use crate::rules::{KernelId, Mode};
use crate::venn::Run;
use crate::venn::roofline::nvfp4_mma;

const W4A4: &str = r#"
schema = 1
checkpoint = "toy"
tier = "nvfp4"
[[linear]]
match = "lm_head"
weight = "bf16"
activation = "bf16"
[[linear]]
match = "*"
weight = "nvfp4/g16"
activation = "nvfp4/g16"
"#;

const FP8_PATH: &str = "costed at the FP8 peak (exact E2M1->E4M3)";

/// 2026-10-01: The fixture's devices plus two `child` devices that run the FP4 block-scale MMA
/// natively, in a family the class's FP4 region is not written for: the instruction exists, the
/// class compiles no kernel for it. `tcslow` is `slowmath` with that instruction, so every node
/// is compute-bound. `fp4dev`'s constants are measured on its own class.
fn registry() -> Registry {
    let mut devs = fx::devices();
    let dev = |id, bw, bf16, fp8, fp4| Dev {
        id,
        class: "child",
        family: "tcgen05",
        bw,
        bf16,
        fp8,
        fp4,
        gib: 64.0,
    };
    devs.push(dev("tcfp4", 1000.0, 100.0, 200.0, 400.0));
    devs.push(dev("tcslow", 1e6, 1e-6, 2e-6, 4e-6));
    let text = fx::registry_text(&devs).replacen(
        "usable_why = \"test\"",
        "usable_why = \"test\"\nmeasured = \"test manifest\"",
        1,
    );
    super::parse_devices(&text).unwrap_or_else(|e| panic!("registry: {e}"))
}

fn resolved(reg: &Registry, dev: &str, w: &str) -> (Resolved, Circuit) {
    let model = fx::model(w);
    let run = Run {
        mode: Mode::Decode,
        rows: 1,
    };
    let one = plan_one(reg, dev, &fx::tree(), &model, run).unwrap();
    (one.resolved, model.circuit)
}

fn nodes(c: &Circuit, local: &str) -> Vec<NodeIdx> {
    (0..c.nodes.len())
        .filter(|&i| c.nodes[i].local == local)
        .collect()
}

fn k(func: &str) -> KernelId {
    KernelId {
        module: "m".into(),
        func: func.into(),
    }
}

// 2026-10-01: The defect: a device whose datasheet has the FP4 MMA was marked and costed native
// although its class compiles no FP4 kernel. Mutation: deriving the mark or the per-node peak
// from the device's peaks (an empty `without_fp4`) marks `tcfp4` native and costs its
// projections at 400 TFLOPS.
#[test]
fn a_device_whose_class_compiles_no_fp4_kernel_is_not_costed_native() {
    let reg = registry();
    let (tc, c) = resolved(&reg, "tcfp4", W4A4);
    assert_eq!(
        tc.availability.absent.get(&k("up_a4")),
        Some(&Absence::CompiledOut {
            macro_name: "METRALE_NO_WARP_BLOCKSCALE_MMA".into(),
            requires: Instr::parse("mma_sync.fp4_block_scale").unwrap(),
        })
    );
    assert_eq!(
        tc.fp4_costing(&c),
        format!(
            "no path for linear:down, linear:gate_up (the class compiles no FP4 block-scale kernel for them): {FP8_PATH}"
        )
    );
    for n in nodes(&c, "up").into_iter().chain(nodes(&c, "down")) {
        assert_eq!(tc.roofline_of(n).nvfp4_tflops, 200.0, "node {n}");
    }
}

// 2026-10-01: A class that compiles the FP4 kernel for an op is native for that op only, and
// the mark names the op it lacks. Mutation: marking per device instead of per op marks
// `linear:down` native too.
#[test]
fn a_class_that_compiles_an_fp4_kernel_is_native_for_that_op_only() {
    let reg = registry();
    let (base, c) = resolved(&reg, "fp4dev", W4A4);
    assert_eq!(
        base.fp4_costing(&c),
        format!(
            "no path for linear:down (the class compiles no FP4 block-scale kernel for them): {FP8_PATH}; native for linear:gate_up"
        )
    );
    let peak = base.roofline.roofline.nvfp4_tflops;
    for n in nodes(&c, "up") {
        assert_eq!(base.roofline_of(n).nvfp4_tflops, peak);
    }
    for n in nodes(&c, "down") {
        assert_eq!(
            base.roofline_of(n).nvfp4_tflops,
            base.roofline.roofline.fp8_tflops
        );
    }
    let (w4a16, c16) = resolved(&reg, "fp4dev", super::hardware_tests::W4A16);
    assert_eq!(
        w4a16.fp4_costing(&c16),
        "not used: the model has no NVFP4-activation node"
    );
}

// 2026-10-01: Either source of the class's compiled set gives an op its FP4 path: a rule that
// runs a compiled FP4 kernel on it, or a family the device runs that implements it with one.
// Mutation: dropping either source leaves `linear:gate_up` without a path in one of the trees.
#[test]
fn a_rule_or_a_family_with_a_compiled_fp4_kernel_gives_the_op_its_path() {
    let (reg, model) = (registry(), fx::model(W4A4));
    let cut = |text: &str, from: &str, to: &str| {
        let at = text
            .find(from)
            .unwrap_or_else(|| panic!("{from} not in the fixture"));
        let end = text[at + from.len()..]
            .find(to)
            .map_or(text.len(), |e| at + from.len() + e);
        format!("{}{}", &text[..at], &text[end..])
    };
    let (mut no_rule, mut no_family) = (fx::tree(), fx::tree());
    let fusions = "kernels/base/common/FUSIONS.toml";
    let rules = cut(
        &no_rule.files[fusions],
        "[[rule]]\nid = \"up_a4\"",
        "[[rule]]",
    );
    no_rule.files.insert(fusions.into(), rules);
    let manifest = "kernels/base/common/KERNEL_FAMILIES.toml";
    let fams = cut(
        &no_family.files[manifest],
        "[[family]]\nid = \"f_up_fp4\"",
        "[[family]]",
    );
    no_family.files.insert(manifest.into(), fams);
    let run = Run {
        mode: Mode::Decode,
        rows: 1,
    };
    for (what, tree) in [("rule", no_rule), ("family", no_family)] {
        let one = plan_one(&reg, "fp4dev", &tree, &model, run).unwrap();
        let without = &one.resolved.without_fp4;
        assert!(
            nodes(&model.circuit, "up")
                .iter()
                .all(|n| !without.contains(n)),
            "without the {what}"
        );
        assert!(
            nodes(&model.circuit, "down")
                .iter()
                .all(|n| without.contains(n))
        );
    }
}

// 2026-10-01: Compute-bound, a device with the FP4 MMA its class has no kernel for estimates
// exactly as its twin without the instruction: decode steps and prefill. Mutation: costing
// gap tables or prefill with the device's roofline instead of `Resolved::roofline_of` makes
// `tcslow` faster than `slowmath`.
#[test]
fn without_a_compiled_fp4_kernel_the_estimate_equals_the_device_without_fp4() {
    let (reg, tree, model) = (registry(), fx::tree(), fx::model(W4A4));
    let settings = BTreeMap::from([
        ("kv_cache_dtype".to_string(), "bf16".to_string()),
        ("ssm_h_dtype".to_string(), "f32".to_string()),
    ]);
    let est = |dev: &str| {
        let run = Run {
            mode: Mode::MultiSeq,
            rows: 16,
        };
        let one = plan_one(&reg, dev, &tree, &model, run).unwrap();
        let r = &one.resolved;
        let step = gap_table(r, &model.circuit, &settings, "toy", one.planned.clone())
            .unwrap()
            .total_us;
        let pre = prefill_us(&model.circuit, &settings, &|n| r.roofline_of(n), 4096).unwrap();
        (step, pre)
    };
    assert_eq!(est("tcslow"), est("slowmath"));
}

// 2026-10-01: A kernel a macro instantiates takes the guard of the invocation, wherever the
// macro is defined and whichever parameter names it; a macro that defines no kernel instantiates
// none. Mutation: reading only `__global__ ... func(` definitions finds no guard for `fp4_k`
// and leaves it available where its region is compiled out.
#[test]
fn a_macro_instantiated_kernel_takes_the_guard_of_its_invocation() {
    let guards = [Guard {
        macro_name: "METRALE_NO_WARP_BLOCKSCALE_MMA".into(),
        polarity: Polarity::Ifndef,
        requires: Instr::parse("mma_sync.fp4_block_scale").unwrap(),
    }];
    let text = r#"
#define ENTRY(ROWS, NAME, K) \
    extern "C" __global__ \
    void NAME(int x) { body<ROWS, K>(x); }
#define HELPER(NAME) int NAME(int x) { return x; }
ENTRY(1, plain_k, 4)
#ifndef METRALE_NO_WARP_BLOCKSCALE_MMA
ENTRY(8, fp4_k, 4)
HELPER(not_a_kernel)
#endif
"#;
    assert_eq!(guard_of(text, "fp4_k", &guards), Some(&guards[0]));
    assert_eq!(guard_of(text, "plain_k", &guards), None);
    assert_eq!(guard_of(text, "not_a_kernel", &guards), None);
}

// 2026-10-01: Every number estimated from datasheet constants is labelled where it is shown,
// and none from constants measured on the class is. Mutation: dropping the label from any of
// the estimate rows, the weight floor, a gap report's step or the matrix row fails here.
#[test]
fn every_datasheet_estimate_is_labelled_a_roofline_projection() {
    const LABEL: &str = "(roofline projection, unmeasured)";
    let (reg, tree) = (registry(), fx::tree());
    let report = |dev: &str| {
        build_report(&reg, dev, &tree, fx::model(W4A4), "cmd".into())
            .unwrap_or_else(|e| panic!("{dev}: {e}"))
    };
    let (datasheet, measured) = (report("tcfp4"), report("fp4dev"));
    assert!(!datasheet.resolved.roofline.measured);
    let text = render_report(&datasheet);
    let shown: Vec<&str> = text
        .lines()
        .filter(|l| {
            l.starts_with("| decode C=")
                || l.starts_with("| prefill ")
                || l.starts_with("Weight floor")
                || l.starts_with("Estimated step")
        })
        .collect();
    assert_eq!(shown.len(), 3 + 2 + 1 + 3, "{text}");
    for l in shown {
        assert!(l.contains(LABEL), "unlabelled: {l}");
    }
    assert!(summary_row(&datasheet).contains(LABEL));
    assert!(measured.resolved.roofline.measured);
    assert!(!render_report(&measured).contains("roofline projection"));
    assert!(!summary_row(&measured).contains("roofline projection"));
}

/// 2026-10-01: The ops the "FP4 costing" row marks native.
fn marked_native(costing: &str, all: &BTreeSet<String>) -> BTreeSet<String> {
    if costing == "native" {
        return all.clone();
    }
    costing
        .split("; ")
        .filter_map(|p| p.strip_prefix("native for "))
        .flat_map(|ops| ops.split(", ").map(String::from))
        .collect()
}

// 2026-10-01: The declared-formats table, every gap row's execution column, the "FP4 costing" row
// and each node's cost tell one story: an NVFP4-activation op is "native fp4_block_scale" in the
// columns exactly when the row marks it native, and is costed at the peak its column names; a
// pair whose nodes run differently is one row per execution. Mutation: computing either column
// from the device's instruction alone (`exec_of`) prints `native fp4_block_scale` on `tcfp4`
// and for `fp4dev`'s down projection, which the row marks as having no path.
#[test]
fn the_execution_columns_the_fp4_costing_row_and_the_cost_agree() {
    let (reg, tree) = (registry(), fx::tree());
    let native = Exec::Native(MmaKind::Fp4BlockScale);
    for dev in ["fp4dev", "tcfp4", "nofp4", "nofp8"] {
        let r = build_report(&reg, dev, &tree, fx::model(W4A4), "cmd".into()).unwrap();
        let (c, res) = (&r.model.circuit, &r.resolved);
        let fp4: Vec<NodeIdx> = (0..c.nodes.len())
            .filter(|&i| nvfp4_mma(c, &c.nodes[i]))
            .collect();
        let ops: BTreeSet<String> = fp4.iter().map(|&i| c.nodes[i].op.name()).collect();
        let marked = marked_native(&res.fp4_costing(c), &ops);
        for &i in &fp4 {
            let e = res.exec[i].unwrap();
            let op = c.nodes[i].op.name();
            assert_eq!(e == native, marked.contains(&op), "{dev} {op}: {e:?}");
            let base = res.roofline.roofline;
            assert_eq!(res.roofline_of(i).nvfp4_tflops, e.peak(&base).0, "{dev}");
        }
        for t in &r.tables {
            for row in t.rows.iter().filter(|x| x.op.starts_with("linear:")) {
                let op_native = marked.contains(&row.op);
                assert_eq!(row.exec == Some(native), op_native, "{dev} {}", row.site);
            }
        }
        let text = render_report(&r);
        let w4a4: Vec<&str> = text.lines().filter(|l| l.starts_with("| W4A4 |")).collect();
        let native_rows = w4a4
            .iter()
            .filter(|l| l.ends_with("| native fp4_block_scale |"))
            .count();
        let expected = match dev {
            "fp4dev" => (1, 2),
            _ => (0, 1),
        };
        assert_eq!((native_rows, w4a4.len()), expected, "{dev}: {w4a4:?}");
    }
}
