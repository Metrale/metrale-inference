// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `reaching`: a path outside `kernels/` reaches every model, a kernel one model runs
//! reaches that model alone, and every way the kernel map cannot answer fails closed.
//!
//! Owner: bench gate tests.
//! Invariants: none beyond the types.

use std::collections::BTreeSet;

use metrale_circuit::venn::affected::{InstanceUse, Usage};
use metrale_circuit::venn::parse_families;

use super::{KernelMap, Subject, reaching};

pub(super) const SHARED: &str = "kernels/gb10/common/shared.cu";
pub(super) const GLM_ONLY: &str = "kernels/gb10/glm-5.3-flash/nvfp4/mla.cu";

fn family(id: &str, file: &str) -> String {
    format!(
        "[[family]]\nid = \"{id}\"\ndescription = \"t\"\ncompute = \"cuda_core\"\n\
         kernels = [\"m::{id}\"]\nrows = [1, 128]\n\
         pipeline.rms_norm = {{ in = [\"bf16\"], compute = \"f32\", out = [\"bf16\"] }}\n\
         op = [{{ op = \"rms_norm\" }}]\n\
         [[family.point]]\nvalues = {{}}\nhow = \"instantiation\"\nfiles = [\"{file}\"]\n"
    )
}

/// 2026-10-15: A gb10 manifest with a family both models run and one only GLM runs.
pub(super) fn map() -> KernelMap {
    let text = format!(
        "schema = 1\nhardware = \"gb10\"\n[roofline]\ndram_gbps = 1.0\nbf16_tflops = 1.0\n\
         fp8_tflops = 1.0\nnvfp4_tflops = 1.0\ncontext_tokens = 4096\n{}{}",
        family("shared", SHARED),
        family("glm_only", GLM_ONLY)
    );
    let manifest = parse_families(&text).unwrap_or_else(|e| panic!("fixture: {e}"));
    let set = |ids: &[&str]| Usage::Known(ids.iter().map(|s| s.to_string()).collect());
    KernelMap::Derived {
        manifests: vec![manifest],
        uses: vec![
            InstanceUse {
                recipe: "glm".into(),
                checkpoint: "nvidia/GLM-5.3-Flash-NVFP4".into(),
                target: "gb10/glm-5.3-flash/nvfp4".into(),
                usage: set(&["shared", "glm_only"]),
            },
            InstanceUse {
                recipe: "qwen".into(),
                checkpoint: "Qwen/Qwen3.6-35B-A3B-FP8".into(),
                target: "gb10/qwen3.6-35b-a3b/nvfp4".into(),
                usage: set(&["shared"]),
            },
        ],
    }
}

pub(super) fn subject(hw: &str, model: &str, checkpoint: &str) -> Subject {
    Subject {
        hw: hw.into(),
        model: model.into(),
        checkpoint: checkpoint.into(),
    }
}

pub(super) fn glm() -> Subject {
    subject("gb10", "glm-5.3-flash", "nvidia/GLM-5.3-Flash-NVFP4")
}

pub(super) fn qwen() -> Subject {
    subject("gb10", "qwen3.6-35b-a3b", "Qwen/Qwen3.6-35B-A3B-FP8")
}

/// 2026-10-15: A registered model with no INSTANCES.toml entry.
fn bare() -> Subject {
    subject("gb10", "nemotron", "nvidia/Nemotron")
}

fn reached(paths: &[&str], map: &KernelMap) -> BTreeSet<Subject> {
    let paths: Vec<String> = paths.iter().map(|s| s.to_string()).collect();
    reaching(&paths, &[glm(), qwen(), bare()], map)
        .into_iter()
        .filter(|(_, p)| !p.is_empty())
        .map(|(s, _)| s)
        .collect()
}

#[test]
fn a_path_outside_kernels_reaches_every_model() {
    assert_eq!(
        reached(&["crates/server/src/main.rs"], &map()),
        BTreeSet::from([glm(), qwen(), bare()])
    );
}

#[test]
fn a_kernel_only_glm_runs_reaches_glm_and_the_models_without_an_instance() {
    let got = reached(&[GLM_ONLY], &map());
    assert_eq!(got, BTreeSet::from([glm(), bare()]));
    // 2026-10-15: Control: a kernel both run reaches both.
    assert!(reached(&[SHARED], &map()).contains(&qwen()));
}

#[test]
fn every_unanswerable_case_fails_closed() {
    let all = BTreeSet::from([glm(), qwen(), bare()]);
    assert_eq!(
        reached(&["kernels/gb10/common/brand_new.cu"], &map()),
        all,
        "an unmapped kernel file reaches every model"
    );
    assert_eq!(
        reached(&[GLM_ONLY], &KernelMap::Unavailable("t".into())),
        all,
        "no map: every model"
    );
    let KernelMap::Derived {
        manifests,
        mut uses,
    } = map()
    else {
        unreachable!("fixture")
    };
    uses[1].usage = Usage::Unknown("t".into());
    assert!(reached(&[GLM_ONLY], &KernelMap::Derived { manifests, uses }).contains(&qwen()));
}

#[test]
fn a_model_on_other_hardware_is_not_reached_by_a_gb10_kernel() {
    let hopper = subject("hopper", "nemotron", "nvidia/Nemotron");
    let r = reaching(
        &[GLM_ONLY.to_string()],
        std::slice::from_ref(&hopper),
        &map(),
    );
    assert!(r[&hopper].is_empty());
    // 2026-10-15: Control: a non-kernel path reaches it.
    let r = reaching(
        &["Cargo.lock".to_string()],
        std::slice::from_ref(&hopper),
        &map(),
    );
    assert!(!r[&hopper].is_empty());
}
