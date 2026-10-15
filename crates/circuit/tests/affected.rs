// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `met circuit affected` against the repo: every golden instance's family usage
//! derives from its checked-in plans without falling back to "every family", a kernel file one
//! instance's families list and another's do not reaches only the first, a shared one reaches
//! both, and the manifest itself (listed by no family) is unmapped.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use std::collections::BTreeSet;

use metrale_circuit::venn::affected::{InstanceUse, Usage, affected};
use metrale_circuit::venn::affected_repo::{affected_in, instance_uses};
use metrale_circuit::venn::families::Families;
use venn_common::{DENSE, MANIFEST, MOE, Tree};

fn on_tree() -> (Vec<Families>, Vec<InstanceUse>) {
    instance_uses(&Tree).expect("instances load")
}

fn usage<'a>(uses: &'a [InstanceUse], recipe: &str) -> &'a BTreeSet<String> {
    match &uses
        .iter()
        .find(|u| u.recipe == recipe)
        .expect(recipe)
        .usage
    {
        Usage::Known(s) => s,
        Usage::Unknown(why) => panic!("{recipe}: {why}"),
    }
}

/// 2026-10-15: The first file a point of `family` lists on gb10.
fn file_of(fams: &[Families], family: &str) -> String {
    fams.iter()
        .filter(|f| f.hardware == "gb10")
        .flat_map(|f| &f.families)
        .find(|f| f.id == family)
        .and_then(|f| f.points.iter().flat_map(|p| &p.files).next())
        .unwrap_or_else(|| panic!("family {family} lists a file"))
        .clone()
}

fn checkpoints(paths: &[String], fams: &[Families], uses: &[InstanceUse]) -> BTreeSet<String> {
    affected(paths, fams, uses)
        .models
        .into_iter()
        .map(|m| m.checkpoint)
        .collect()
}

#[test]
fn every_golden_instance_has_a_derived_usage() {
    let (_, uses) = on_tree();
    let golden = metrale_circuit::parse_instances(&common::read("kernels/circuits/INSTANCES.toml"))
        .expect("instances");
    for inst in golden.iter().filter(|i| i.golden) {
        assert!(
            !usage(&uses, &inst.recipe).is_empty(),
            "{} runs no family",
            inst.recipe
        );
    }
}

#[test]
fn a_family_only_the_moe_runs_does_not_reach_the_dense_model() {
    let (fams, uses) = on_tree();
    let (moe, dense) = (usage(&uses, MOE), usage(&uses, DENSE));
    let only_moe = moe
        .difference(dense)
        .next()
        .expect("the MoE runs a family the dense model does not");
    let reached = checkpoints(&[file_of(&fams, only_moe)], &fams, &uses);
    let dense_ckpt = uses.iter().find(|u| u.recipe == DENSE).expect("dense");
    let moe_ckpt = uses.iter().find(|u| u.recipe == MOE).expect("moe");
    assert!(reached.contains(&moe_ckpt.checkpoint), "{reached:?}");
    assert!(!reached.contains(&dense_ckpt.checkpoint), "{reached:?}");
    // 2026-10-15: Control: a family both run reaches both.
    let shared = moe.intersection(dense).next().expect("a shared family");
    let reached = checkpoints(&[file_of(&fams, shared)], &fams, &uses);
    assert!(reached.contains(&moe_ckpt.checkpoint) && reached.contains(&dense_ckpt.checkpoint));
}

#[test]
fn the_manifest_itself_is_unmapped() {
    let a = affected_in(&Tree, &[MANIFEST.to_string()]).expect("affected");
    assert_eq!(a.unmapped, vec![MANIFEST.to_string()]);
    assert!(a.models.is_empty());
}
