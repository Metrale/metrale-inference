// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The hardware axis on an in-memory tree ([`super::test_fixture`]): the FP4 gate,
//! class inheritance and overrides, roofline-only differences between SKUs of one class, and the
//! explicit refusals (unknown device, a class without rules, a format no MMA can run, a build
//! that contradicts its device).
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::avail::Absence;
use super::class::{ClassRules, class_rules};
use super::device::{Instr, MmaKind};
use super::exec::{Exec, exec_of};
use super::gaps::gap_table;
use super::plan::{NOVEL_EMITTER, report_runs};
use super::test_fixture::{self as fx, Dev};
use super::{HwError, plan_one};
use crate::format::Format;
use crate::rules::{KernelId, Mode};
use crate::venn::{Class, Run};

const W4A16: &str = r#"
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
activation = "bf16"
"#;

fn decode() -> Run {
    Run {
        mode: Mode::Decode,
        rows: 1,
    }
}

fn kernel_of(one: &super::OnePlan, c: &crate::ir::Circuit, local: &str) -> Vec<String> {
    one.planned
        .plan
        .groups
        .iter()
        .filter(|g| g.nodes.iter().any(|&n| c.nodes[n].local == local))
        .map(|g| {
            g.kernels
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join("+")
        })
        .collect()
}

fn k(func: &str) -> KernelId {
    KernelId {
        module: "m".into(),
        func: func.into(),
    }
}

// 2026-09-30: Path A. Mutation: dropping the build-define check in `avail::kernel_status` (or
// the guard parse) lets `m::up_fp4` into the `child` plan.
#[test]
fn an_fp4_mma_kernel_runs_only_where_the_device_has_the_instruction() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    let c = &model.circuit;
    let native = plan_one(&reg, "fp4dev", &tree, &model, decode()).unwrap();
    assert!(kernel_of(&native, c, "up").iter().all(|x| x == "m::up_fp4"));
    for dev in ["nofp4", "nofp4big", "nofp8"] {
        let one = plan_one(&reg, dev, &tree, &model, decode()).unwrap();
        assert!(
            kernel_of(&one, c, "up").iter().all(|x| x == "m::up"),
            "{dev} ran an FP4-MMA kernel"
        );
        assert_eq!(
            one.resolved.availability.absent.get(&k("up_fp4")),
            Some(&Absence::CompiledOut {
                macro_name: "METRALE_NO_WARP_BLOCKSCALE_MMA".into(),
                requires: Instr::parse("mma_sync.fp4_block_scale").unwrap(),
            })
        );
    }
}

// 2026-09-30: Path B. Mutation: appending instead of replacing a same-id rule in `rules_of`
// leaves `m::act` on the child (two rules with one id, the base one first).
#[test]
fn a_class_override_wins_over_the_inherited_rule() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    let c = &model.circuit;
    let base = plan_one(&reg, "fp4dev", &tree, &model, decode()).unwrap();
    let child = plan_one(&reg, "nofp4", &tree, &model, decode()).unwrap();
    assert!(kernel_of(&base, c, "act").iter().all(|x| x == "m::act"));
    assert!(
        kernel_of(&child, c, "act")
            .iter()
            .all(|x| x == "m::act_child")
    );
    match class_rules(&tree, "child").unwrap() {
        ClassRules::Rules { rules, files } => {
            assert_eq!(
                files,
                [
                    "kernels/base/common/FUSIONS.toml",
                    "kernels/child/common/FUSIONS.toml"
                ]
            );
            assert_eq!(rules.iter().filter(|r| r.id == "act").count(), 1);
        }
        ClassRules::None => panic!("child inherits rules"),
    }
}

// 2026-09-30: Path B. Evidence is per class: the base family's microbench makes `up` Shared on
// the base class and only Shared-unmeasured on the child that inherits it.
#[test]
fn inherited_families_keep_their_points_and_lose_their_evidence() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    let class_of = |dev: &str| {
        let one = plan_one(&reg, dev, &tree, &model, decode()).unwrap();
        let t = gap_table(
            &one.resolved,
            &model.circuit,
            &one.policy.settings,
            "toy",
            one.planned.clone(),
        )
        .unwrap();
        let row = t
            .rows
            .iter()
            .find(|r| r.site == "ffn.down")
            .unwrap()
            .clone();
        (row.class, one.resolved.families.families.len())
    };
    assert_eq!(class_of("fp4dev").0, Class::Shared);
    let (child, n) = class_of("nofp4");
    assert_eq!(child, Class::SharedUnmeasured);
    // 2026-09-30: f_up_fp4's only kernel cannot run on the child: the family is not offered.
    assert_eq!(n, 7);
}

// 2026-09-30: Path B. Two SKUs of one class: same plan (digest), different estimate, and the
// ranking follows the roofline. Mutation: costing with the manifest's roofline instead of the
// device's makes the two step estimates equal.
#[test]
fn a_device_roofline_changes_the_ranking_and_the_estimate_but_not_the_plan() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    let run = Run {
        mode: Mode::MultiSeq,
        rows: 64,
    };
    let table = |dev: &str| {
        let one = plan_one(&reg, dev, &tree, &model, run).unwrap();
        gap_table(
            &one.resolved,
            &model.circuit,
            &one.policy.settings,
            "toy",
            one.planned.clone(),
        )
        .unwrap()
    };
    let (a, b, slow) = (table("nofp4"), table("nofp4big"), table("slowmath"));
    assert_eq!(a.planned.plan.digest, b.planned.plan.digest);
    assert_eq!(a.planned.plan.digest, slow.planned.plan.digest);
    assert!(
        b.total_us < a.total_us,
        "twice the bandwidth must be faster"
    );
    let order =
        |t: &super::gaps::GapTable| t.rows.iter().map(|r| r.site.clone()).collect::<Vec<_>>();
    assert_ne!(
        order(&a),
        order(&slow),
        "compute-bound ranking equals memory-bound ranking"
    );
}

// 2026-09-30: Path C.
#[test]
fn an_unknown_device_is_a_typed_error_that_lists_the_known_ones() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    match plan_one(&reg, "tpu-v9", &tree, &model, decode()) {
        Err(HwError::UnknownDevice { id, known }) => {
            assert_eq!(id, "tpu-v9");
            assert!(known.contains(&"nofp4".to_string()));
        }
        other => panic!("expected UnknownDevice, got {other:?}"),
    }
}

// 2026-09-30: Path C. Mutation: treating a missing FUSIONS.toml as an error (or as the parent's
// without `inherits`) fails this test.
#[test]
fn a_class_without_rules_plans_every_node_as_an_explicit_gap() {
    let (reg, tree, model) = (fx::registry(), fx::tree(), fx::model(W4A16));
    assert_eq!(class_rules(&tree, "lonely").unwrap(), ClassRules::None);
    for run in report_runs() {
        let one = plan_one(&reg, "alone", &tree, &model, run).unwrap();
        assert!(
            one.planned
                .plan
                .groups
                .iter()
                .all(|g| g.emitter == NOVEL_EMITTER && g.kernels.is_empty())
        );
        assert!(!one.planned.novel.is_empty());
    }
}

// 2026-09-30: Path C. A declared W4A4 layer on a device with neither the FP4 nor the FP8 MMA:
// the plan keeps the declared formats and says there is no path; with the FP8 MMA it names the
// exact E2M1 -> E4M3 path; with the FP4 MMA it is native. Never re-planned wider.
#[test]
fn a_declared_format_no_mma_can_run_is_reported_as_no_path() {
    let reg = fx::registry();
    let (w, a) = (
        Format::parse("nvfp4/g16").unwrap(),
        Format::parse("nvfp4/g16").unwrap(),
    );
    let dev = |id: &str| reg.device(id).unwrap();
    assert_eq!(
        exec_of(dev("nofp8"), w, a),
        Exec::NoPath(MmaKind::Fp4BlockScale)
    );
    assert_eq!(exec_of(dev("nofp4"), w, a), Exec::ExactFp8Emulation);
    assert_eq!(
        exec_of(dev("fp4dev"), w, a),
        Exec::Native(MmaKind::Fp4BlockScale)
    );
    // 2026-09-30: W4A8 with group-16 E4M3 scales has no single instruction unless declared.
    let fp8 = Format::parse("fp8/token").unwrap();
    assert_eq!(exec_of(dev("fp4dev"), w, fp8), Exec::ExactFp8Emulation);
    assert_eq!(exec_of(dev("nofp8"), w, fp8), Exec::NoPath(MmaKind::Fp8));
}

// 2026-09-30: The build and the registry must agree. Mutation: a `child` device that claims the
// warp-level FP4 MMA its class compiles out is refused, not planned.
#[test]
fn a_device_claiming_an_instruction_its_class_compiles_out_is_refused() {
    let mut devs = fx::devices();
    devs.push(Dev {
        id: "liar",
        class: "child",
        family: "mma_sync",
        bw: 1000.0,
        bf16: 100.0,
        fp8: 200.0,
        fp4: 400.0,
        gib: 64.0,
    });
    let reg = super::parse_devices(&fx::registry_text(&devs)).unwrap();
    let e = plan_one(&reg, "liar", &fx::tree(), &fx::model(W4A16), decode()).unwrap_err();
    assert!(
        matches!(e, HwError::BuildContradictsDevice { ref device, defined: true, .. } if device == "liar"),
        "{e}"
    );
}

// 2026-09-30: The registry refuses a native_mma pair its dense peak contradicts.
#[test]
fn a_native_pair_without_a_peak_is_refused() {
    let text = fx::registry_text(&fx::devices()).replacen("fp4_nvfp4 = 400", "fp4_nvfp4 = 0", 1);
    let e = super::parse_devices(&text).unwrap_err();
    assert!(e.to_string().contains("fp4_block_scale"), "{e}");
}

// 2026-09-30: Path B. The manifest's discover rules run over the class's own tree too: a
// class's own copy of a kernel file is a point of its family there, and the inherited tree is
// not re-rooted onto itself. Mutation: skipping the re-rooting finds nothing.
#[test]
fn a_class_s_own_copy_of_a_kernel_is_discovered_as_a_point() {
    use super::class::{ClassInfo, class_discovered};
    use super::sources::ClassSources;
    let manifest = r#"
schema = 1
hardware = "base"
[roofline]
dram_gbps = 1.0
bf16_tflops = 1.0
fp8_tflops = 1.0
nvfp4_tflops = 1.0
context_tokens = 1
[[family]]
id = "attn"
description = "test"
kernels = ["attn_a::attn"]
rows = [1, 1]
op = [{ op = "paged_attention" }]
[[family.point]]
values = {}
how = "copy"
files = ["kernels/base/common/attn_a.cu"]
[[family.discover]]
kind = "file"
glob = "kernels/base/common/attn_*.cu"
values = {}
"#;
    let fams = crate::venn::parse_families(manifest).unwrap();
    let info = |name: &str| ClassInfo {
        name: name.into(),
        arch: "sm".into(),
        inherits: None,
        defines: Default::default(),
        defaults: Default::default(),
    };
    let sources = ClassSources {
        files: [
            "kernels/base/common/attn_a.cu".to_string(),
            "kernels/child/common/attn_b.cu".to_string(),
        ]
        .into(),
        ..ClassSources::default()
    };
    let found = class_discovered(&fams, "base", &[info("child"), info("base")], &sources);
    let files: Vec<&str> = found.iter().map(|f| f.file.as_str()).collect();
    assert_eq!(files, ["kernels/child/common/attn_b.cu"]);
}

// 2026-09-30: A recipe's value for a class-decided setting stands on the class it was stated for,
// even away from that class's default; every other class, and a model planned from its
// checkpoint on any class, takes the class's value; a class that does not state the setting is
// refused on the recipe's own class too. Mutation: dropping the own-class check turns the first
// assertion's `off` into `on`; applying it to every class keeps `off` on `hopper`.
#[test]
fn a_recipe_pin_stands_on_its_own_class_only() {
    use std::collections::{BTreeMap, BTreeSet};
    let class = |name: &str| super::class::ClassInfo {
        name: name.into(),
        arch: "sm_0".into(),
        inherits: None,
        defines: BTreeSet::new(),
        defaults: BTreeMap::from([
            ("ssm_batched_recurrent".to_string(), "true".to_string()),
            ("decode_split_silu".to_string(), "true".to_string()),
        ]),
    };
    let policy = crate::fuser::Policy {
        opt_in_levers: BTreeSet::new(),
        settings: BTreeMap::from([
            ("ssm_batched_recurrent".to_string(), "off".to_string()),
            ("decode_split_silu".to_string(), "on".to_string()),
        ]),
    };
    let (kept, changed) =
        super::model::policy_on_class(&policy, Some("gb10"), &class("gb10")).unwrap();
    assert_eq!(kept.settings, policy.settings);
    assert!(changed.is_empty(), "{changed:?}");
    for (stated_for, on) in [(Some("gb10"), "hopper"), (None, "hopper"), (None, "gb10")] {
        let (p, changed) = super::model::policy_on_class(&policy, stated_for, &class(on)).unwrap();
        assert_eq!(
            p.settings["ssm_batched_recurrent"], "on",
            "{stated_for:?} on {on}"
        );
        assert_eq!(p.settings["decode_split_silu"], "on");
        assert_eq!(
            changed,
            BTreeMap::from([("ssm_batched_recurrent".to_string(), "on".to_string())])
        );
    }
    let mut bare = class("gb10");
    bare.defaults.clear();
    let e = super::model::policy_on_class(&policy, Some("gb10"), &bare).unwrap_err();
    assert!(matches!(e, HwError::Class(_)), "{e}");
}
