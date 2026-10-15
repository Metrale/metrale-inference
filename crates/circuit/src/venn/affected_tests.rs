// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `affected`: a kernel only one model runs reaches that model alone, a shared
//! kernel reaches every user, an unlisted kernel file and an unknown usage fail closed, and the
//! answer moves when a family's `files` entry does (the mutation control).
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use super::super::families::parse_families;
use super::super::venn_tests::manifest;
use super::{Affected, InstanceUse, Usage, affected, op_usage, plan_usage};
use crate::test_toy;

const GEMV: &str = "kernels/gb10/common/gemv.cu";
const HEAD: &str = "kernels/gb10/glm-5.3-flash/nvfp4/head.cu";
const NORM: &str = "kernels/gb10/common/norm.cu";

/// 2026-10-15: The toy manifest on gb10 with one file per family; the head family's file sits
/// in the GLM model directory. `head_files` is the head point's `files` line, so a test can
/// drop the entry.
fn families(head_files: &str) -> super::super::Families {
    let gemv = format!(
        "[[family.point]]\nvalues = {{ hidden = \"64\" }}\nhow = \"instantiation\"\nfiles = [\"{GEMV}\"]\n"
    );
    let mut text = manifest(&gemv).replacen("hardware = \"toy\"", "hardware = \"gb10\"", 1);
    for file in [
        "kernels/gb10/common/embed.cu",
        NORM,
        "kernels/gb10/common/act.cu",
        "kernels/gb10/common/add.cu",
    ] {
        let before = text.clone();
        text = text.replacen("files = [\"toy.cu\"]", &format!("files = [\"{file}\"]"), 1);
        assert_ne!(text, before, "anchor for {file} must match");
    }
    let before = text.clone();
    text = text.replacen("files = [\"toy.cu\"]", head_files, 1);
    assert_ne!(text, before, "the head anchor must match");
    assert!(!text.contains("toy.cu"), "every toy file must be replaced");
    parse_families(&text).unwrap_or_else(|e| panic!("fixture: {e}"))
}

fn uses() -> Vec<InstanceUse> {
    let set = |ids: &[&str]| Usage::Known(ids.iter().map(|s| s.to_string()).collect());
    vec![
        InstanceUse {
            recipe: "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k".into(),
            checkpoint: "nvidia/GLM-5.3-Flash-NVFP4".into(),
            target: "gb10/glm-5.3-flash/nvfp4".into(),
            usage: set(&["gemv", "head", "norm"]),
        },
        InstanceUse {
            recipe: "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head".into(),
            checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
            target: "gb10/qwen3.6-35b-a3b/nvfp4".into(),
            usage: set(&["gemv", "norm"]),
        },
    ]
}

fn checkpoints(a: &Affected) -> BTreeSet<&str> {
    a.models.iter().map(|m| m.checkpoint.as_str()).collect()
}

fn run(paths: &[&str], fams: super::super::Families, uses: &[InstanceUse]) -> Affected {
    let changed: Vec<String> = paths.iter().map(|s| s.to_string()).collect();
    affected(&changed, &[fams], uses)
}

const HEAD_FILES: &str = "files = [\"kernels/gb10/glm-5.3-flash/nvfp4/head.cu\"]";

/// 2026-10-15: The GLM-only claim, as a predicate the mutation control reuses.
fn glm_only(a: &Affected) -> bool {
    checkpoints(a) == BTreeSet::from(["nvidia/GLM-5.3-Flash-NVFP4"]) && a.unmapped.is_empty()
}

#[test]
fn a_kernel_only_glm_runs_reaches_glm_alone_with_its_chain() {
    let a = run(&[HEAD], families(HEAD_FILES), &uses());
    assert!(glm_only(&a), "{a:?}");
    let hop = &a.models[0].reason[0];
    assert_eq!(
        (
            hop.file.as_str(),
            hop.family.as_str(),
            hop.instance.as_str()
        ),
        (HEAD, "head", "glm-5.3/glm-5.3-flash-nvfp4-tp3-512k")
    );
    assert!(!hop.assumed);
    assert_eq!(
        (a.models[0].hw.as_str(), a.models[0].model.as_str()),
        ("gb10", "glm-5.3-flash")
    );
}

#[test]
fn mutation_control_dropping_the_files_entry_changes_the_answer() {
    // 2026-10-15: The same diff against the manifest with head.cu dropped from `files` (the
    // parser requires a point to list a file, so the entry names another one): the path is no
    // longer mapped, so it lands in `unmapped` (every model) and the GLM-only claim fails.
    let dropped = "files = [\"kernels/gb10/glm-5.3-flash/nvfp4/head_old.cu\"]";
    let a = run(&[HEAD], families(dropped), &uses());
    assert!(!glm_only(&a));
    assert_eq!(a.unmapped, vec![HEAD.to_string()]);
    assert!(a.models.is_empty());
}

#[test]
fn a_shared_kernel_reaches_every_user() {
    let a = run(&[GEMV], families(HEAD_FILES), &uses());
    assert_eq!(
        checkpoints(&a),
        BTreeSet::from(["nvidia/GLM-5.3-Flash-NVFP4", "Qwen/Qwen3.6-35B-A3B-FP8"])
    );
    // 2026-10-15: Control: a model that does not run the family is not reached by it.
    let mut only_glm = uses();
    only_glm[1].usage = Usage::Known(BTreeSet::from(["norm".to_string()]));
    let a = run(&[GEMV], families(HEAD_FILES), &only_glm);
    assert!(glm_only(&a));
}

#[test]
fn an_unlisted_kernel_file_is_unmapped_and_a_listed_unused_one_is_unused() {
    let fresh = "kernels/gb10/common/brand_new.cu";
    let a = run(&[fresh], families(HEAD_FILES), &uses());
    assert_eq!(a.unmapped, vec![fresh.to_string()]);
    assert!(a.models.is_empty());
    // 2026-10-15: A hardware without a manifest is unmapped too, as are circuit files.
    let a = run(
        &[
            "kernels/hopper/common/x.cu",
            "kernels/circuits/qwen3_5.toml",
        ],
        families(HEAD_FILES),
        &uses(),
    );
    assert_eq!(a.unmapped.len(), 2);
    // 2026-10-15: The add family is listed and nobody runs it.
    let a = run(
        &["kernels/gb10/common/add.cu"],
        families(HEAD_FILES),
        &uses(),
    );
    assert_eq!(a.unused, vec!["kernels/gb10/common/add.cu".to_string()]);
    assert!(a.unmapped.is_empty() && a.models.is_empty());
}

#[test]
fn a_path_outside_kernels_is_left_to_the_caller() {
    let a = run(
        &["crates/server/src/main.rs", "Cargo.lock"],
        families(HEAD_FILES),
        &uses(),
    );
    assert_eq!(a.non_kernel.len(), 2);
    assert!(a.models.is_empty() && a.unmapped.is_empty());
}

#[test]
fn an_unknown_usage_counts_as_running_every_family_on_its_hardware() {
    let mut u = uses();
    u[1].usage = Usage::Unknown("test".into());
    let a = run(&[HEAD], families(HEAD_FILES), &u);
    assert_eq!(checkpoints(&a).len(), 2);
    let qwen = a
        .models
        .iter()
        .find(|m| m.checkpoint.starts_with("Qwen/"))
        .expect("qwen");
    assert!(qwen.reason.iter().all(|h| h.assumed));
    assert_eq!(
        a.assumed
            .get("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head")
            .map(String::as_str),
        Some("test")
    );
    // 2026-10-15: An instance on another hardware is never reached by a gb10 file.
    u[1].target = "hopper/qwen3.6-35b-a3b/nvfp4".into();
    let a = run(&[HEAD], families(HEAD_FILES), &u);
    assert!(glm_only(&a));
}

#[test]
fn a_discovery_glob_maps_the_files_it_matches() {
    let mut fams = families(HEAD_FILES);
    let head = fams
        .families
        .iter_mut()
        .find(|f| f.id == "head")
        .expect("head");
    head.discover.push(super::super::families::Discover::File {
        glob: "kernels/gb10/glm-5.3-flash/nvfp4/head_*.cu".into(),
        values: Default::default(),
    });
    let a = run(
        &["kernels/gb10/glm-5.3-flash/nvfp4/head_v2.cu"],
        fams,
        &uses(),
    );
    assert!(glm_only(&a), "{a:?}");
}

#[test]
fn a_fused_plan_names_its_families_and_an_unlisted_group_makes_it_unknown() {
    let c = test_toy::circuit(2);
    let plan = test_toy::plan(&c, &test_toy::rules(""), &test_toy::policy(), 1);
    let fams = families(HEAD_FILES);
    let Usage::Known(set) = plan_usage(&c, &plan, &fams) else {
        panic!("every toy group is listed")
    };
    assert_eq!(
        set,
        ["act", "add", "embed", "gemv", "head", "norm"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    );
    // 2026-10-15: Control: without the head family the lm_head group is unlisted.
    let mut fewer = fams.clone();
    fewer.families.retain(|f| f.id != "head");
    assert!(matches!(
        plan_usage(&c, &plan, &fewer),
        Usage::Unknown(why) if why.contains("lm_head")
    ));
    // 2026-10-15: The op view over-approximates the same circuit with the same families, and
    // fails closed the same way.
    assert_eq!(op_usage(&c, &fams), Usage::Known(set));
    assert!(matches!(op_usage(&c, &fewer), Usage::Unknown(_)));
}

#[test]
fn union_is_absorbed_by_unknown() {
    let a = Usage::Known(BTreeSet::from(["x".to_string()]));
    let b = Usage::Known(BTreeSet::from(["y".to_string()]));
    assert_eq!(
        a.clone().union(b),
        Usage::Known(BTreeSet::from(["x".to_string(), "y".to_string()]))
    );
    assert!(matches!(
        a.union(Usage::Unknown("u".into())),
        Usage::Unknown(_)
    ));
}
