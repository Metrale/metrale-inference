// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The LKB of the toy model on the in-memory tree: the residual is exactly the
//! class's own unnamed modules (tier, lines and shadow reason), coverage is the hardware
//! report's own shares, relations are the `bit_identical` groups the plans apply, and the
//! ledger TOML parses back to the same numbers.
//!
//! Owner: metrale-circuit tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use super::{Tier, lkb, render_markdown, render_toml};
use crate::hardware::hardware_tests::W4A16;
use crate::hardware::sources::{ClassSources, KernelTree, Module};
use crate::hardware::test_fixture::{self as fx, Tree};
use crate::hardware::{HwError, HwReport, build_report};
use crate::venn::Class;
use crate::venn::repo::Repo;

/// 2026-10-05: The fixture tree, plus the `child` class's own sources: `extra` in its common
/// tier (declared as a shadow) and `leaf` in its model tier.
struct Own {
    tree: Tree,
}

const EXTRA: &str = "__global__ void extra(int x) {}\n// two\n// three\n";
const LEAF: &str = "__global__ void leaf(int x) {}\n";

impl Repo for Own {
    fn read(&self, rel: &str) -> Result<String, String> {
        self.tree.read(rel)
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        self.tree.list(rel)
    }
}

impl KernelTree for Own {
    fn class_sources(&self, class: &str, model: &str, quant: &str) -> Result<ClassSources, String> {
        let mut s = self.tree.class_sources(class, model, quant)?;
        // 2026-10-05: The entry points of the copy-point families (`COPY_FAMILY`).
        if let Some(m) = s.modules.get_mut("m") {
            m.text
                .push_str("__global__ void copy_k(int x) {}\n__global__ void one_k(int x) {}\n");
        }
        if class == "child" {
            for (name, path, text) in [
                ("extra", "kernels/child/common/extra.cu", EXTRA),
                ("leaf", "kernels/child/toy/q/leaf.cu", LEAF),
            ] {
                s.modules.insert(
                    name.into(),
                    Module {
                        path: path.into(),
                        text: text.into(),
                    },
                );
                s.files.insert(path.into());
            }
        }
        Ok(s)
    }

    fn kernel_target(&self, config: &str, refs: &[&str]) -> Result<Option<String>, String> {
        self.tree.kernel_target(config, refs)
    }

    fn as_repo(&self) -> &dyn Repo {
        self
    }
}

fn own(extra_rules: &str, shadow: &str) -> Own {
    let mut tree = fx::tree();
    tree.files
        .get_mut("kernels/child/common/FUSIONS.toml")
        .expect("child rules")
        .push_str(extra_rules);
    tree.files.insert(
        "kernels/child/common/KERNEL.toml".into(),
        format!("[shadow]\n{shadow}\n"),
    );
    Own { tree }
}

fn report(tree: &Own, device: &str) -> Result<HwReport, HwError> {
    build_report(
        &fx::registry(),
        device,
        tree,
        fx::model(W4A16),
        "cmd".into(),
    )
}

const SHADOW: &str = r#"extra = "replaces base/common/extra.cu: measured faster here""#;

// 2026-10-05: Mutation: dropping the `kernels/<class>/` filter lists the inherited `m`; dropping
// the family-name filter lists `m` on the base class; reading no KERNEL.toml loses the reason;
// a tier test on `common/` alone files `leaf` as class.
#[test]
fn the_residual_is_the_class_own_modules_no_family_names() {
    let tree = own("", SHADOW);
    let child = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap();
    let got: Vec<_> = child
        .residual
        .iter()
        .map(|r| (r.module.as_str(), r.tier, r.lines, r.shadow.is_some()))
        .collect();
    assert_eq!(
        got,
        [
            ("extra", Tier::Class, 3, true),
            ("leaf", Tier::Model, 1, false)
        ]
    );
    assert_eq!(child.residual_lines(), 4);
    assert_eq!(child.chain, ["child", "base"]);
    let base = lkb(&report(&tree, "fp4dev").unwrap(), &tree, "c".into()).unwrap();
    assert!(
        base.residual.is_empty(),
        "the base class's own `m` is named by families: {:?}",
        base.residual
    );
}

// 2026-10-05: Mutation: computing coverage from anything but the report's gap tables (or
// swapping `Shared` for `SharedUnmeasured`) breaks the equalities. A class with no rules plans
// every node with a placeholder: coverage must read zero there, although the report files
// those rows under the family that implements the op (counting only `Novel` rows reads 100%).
#[test]
fn coverage_is_the_report_own_shares() {
    let tree = own("", SHADOW);
    let r = report(&tree, "nofp4").unwrap();
    let l = lkb(&r, &tree, "c".into()).unwrap();
    assert_eq!(l.coverage.len(), r.tables.len());
    for (c, t) in l.coverage.iter().zip(&r.tables) {
        assert_eq!(c.run, t.planned.run);
        assert!(
            t.rows.iter().all(|r| !r.placeholder),
            "every node has a rule here"
        );
        assert_eq!(c.lkb, 1.0 - t.share_of(&[Class::Novel]));
        assert_eq!(c.measured, t.share_of(&[Class::Shared]));
    }
    let alone = lkb(&report(&tree, "alone").unwrap(), &tree, "c".into()).unwrap();
    for c in &alone.coverage {
        assert!(c.lkb.abs() < 1e-12, "{c:?}");
        assert!((c.uncovered - 1.0).abs() < 1e-12, "{c:?}");
        assert_eq!(c.groups.keys().copied().collect::<Vec<_>>(), ["uncovered"]);
    }
}

const EXACT_ADD: &str = r#"
[[rule]]
id = "add_exact"
pattern = [{ op = "residual_add" }]
kernels = [{ module = "m", func = "add" }]
repeat = "once"
emitter = "add_exact"
rows = [1, 128]
modes = ["decode", "multi_seq", "verify"]
numerics = "bit_identical"
microtest = "add_exact_microtest"
priority = 20
cite = "test"
"#;

// 2026-10-05: Mutation: counting every group's rule as a relation lists `embed` and the rest;
// ignoring the numerics tag of groups leaves `bit_identical` at zero.
#[test]
fn relations_are_the_bit_identical_groups_the_plans_apply() {
    let tree = own(EXACT_ADD, SHADOW);
    let l = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap();
    assert_eq!(
        l.relations_used
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["add_exact"]
    );
    assert!(
        l.coverage
            .iter()
            .all(|c| c.groups.get("bit_identical") > Some(&0))
    );
    let base = lkb(&report(&tree, "fp4dev").unwrap(), &tree, "c".into()).unwrap();
    assert!(
        base.relations_used.is_empty(),
        "the rule is the child's only"
    );
}

// 2026-10-05: Mutation: reordering the runs, or printing a share as a fraction, changes the
// parsed values; dropping a ledger key fails the lookup.
#[test]
fn the_ledger_toml_parses_back_to_the_same_numbers() {
    let tree = own("", SHADOW);
    let l = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap();
    let doc: toml::Table = toml::from_str(&render_toml(&l)).unwrap();
    let want: Vec<String> = l
        .coverage
        .iter()
        .map(|c| format!("{:.1}", 100.0 * c.lkb))
        .collect();
    assert_eq!(
        doc["lkb_coverage_pct"].as_str(),
        Some(want.join("/").as_str())
    );
    assert_eq!(doc["residual_count"].as_integer(), Some(2));
    assert_eq!(doc["residual_loc"].as_integer(), Some(4));
    assert_eq!(doc["lkb_class"].as_str(), Some("child"));
    let md = render_markdown(&l);
    assert!(
        md.contains("# LKB on child (realization inherits base)"),
        "{md}"
    );
    assert!(
        md.contains("## LKB residual on child: 2 sources, 4 lines"),
        "{md}"
    );
    assert!(
        md.contains("| extra | `kernels/child/common/extra.cu` | 3 | class |"),
        "{md}"
    );
}

// 2026-10-05: Mutation: listing a family with one copy point, or none, as a candidate; or a
// placeholder row (a point no rule of the class runs) as a missing point.
#[test]
fn promotion_lists_copy_families_and_missing_points_only() {
    let tree = own("", SHADOW);
    let l = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap();
    let copies = l
        .promotion
        .iter()
        .filter(|c| matches!(c, super::Candidate::CopyPoints { .. }))
        .count();
    assert_eq!(
        copies, 0,
        "the toy families have no copy points: {:?}",
        l.promotion
    );
    assert!(render_markdown(&l).contains("## Promotion candidates\n\nNone."));

    let mut tree = own("", SHADOW);
    tree.tree
        .files
        .get_mut("kernels/base/common/KERNEL_FAMILIES.toml")
        .expect("families")
        .push_str(COPY_FAMILY);
    let l = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap();
    assert_eq!(
        l.promotion,
        [super::Candidate::CopyPoints {
            family: "f_copies".into(),
            count: 2
        }]
    );
}

/// 2026-10-05: A family whose two points are per-point copies (`t` = 1 and 2), and one with a
/// single copy point, which is no candidate.
const COPY_FAMILY: &str = r#"
[[family]]
id = "f_copies"
description = "test"
compute = "cuda_core"
kernels = ["m::copy_k"]
rows = [1, 128]
op = [{ op = "argmax" }]
pipeline.argmax = { in = ["bf16"], compare = "bf16", out = ["i32"] }
[[family.param]]
name = "t"
kind = "compile"
from = "dim:hidden"
[[family.point]]
values = { t = "1" }
how = "copy"
files = ["kernels/base/common/m.cu"]
[[family.point]]
values = { t = "2" }
how = "copy"
files = ["kernels/base/common/m.cu"]

[[family]]
id = "f_one_copy"
description = "test"
compute = "cuda_core"
kernels = ["m::one_k"]
rows = [1, 128]
op = [{ op = "argmax" }]
pipeline.argmax = { in = ["bf16"], compare = "bf16", out = ["i32"] }
[[family.param]]
name = "t"
kind = "compile"
from = "dim:hidden"
[[family.point]]
values = { t = "1" }
how = "copy"
files = ["kernels/base/common/m.cu"]
"#;

// 2026-10-05: Mutation: accepting a non-string reason would let a table or a number stand in
// for the evidence a shadow must carry.
#[test]
fn a_shadow_without_a_written_reason_is_refused() {
    let tree = own("", "extra = 1");
    let e = lkb(&report(&tree, "nofp4").unwrap(), &tree, "c".into()).unwrap_err();
    assert!(
        e.to_string()
            .contains("kernels/child/common/KERNEL.toml [shadow] `extra`"),
        "{e}"
    );
}
