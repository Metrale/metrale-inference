// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: `met circuit venn` against the repo: the kernel-family manifest agrees with the
//! kernel sources, FUSIONS.toml, measurements.toml, KERNEL-PERF.md and the legacy code it
//! cites; the checked-in Lightning report is what the tool produces today (the `--check` the
//! CLI runs, through the same `report_text`); and a checkpoint-directory target is checked
//! against its declarations. Regenerate the report after an intended change with
//! `cargo test -p metrale-circuit --test venn -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use metrale_circuit::venn::discover::{KernelSources, drift, macro_files};
use metrale_circuit::venn::{self, Repo};
use venn_common::*;

#[test]
fn the_checked_in_report_is_current() {
    let text = venn::report_text(&Tree, &args(), None).expect("report");
    let on_disk = common::read(REPORT);
    assert!(
        text == on_disk,
        "{REPORT} is stale; regenerate with `cargo test -p metrale-circuit --test venn -- \
         --ignored regenerate` and review the diff"
    );
}

#[test]
fn the_manifest_matches_the_kernel_sources() {
    let fams = families();
    let mut src = KernelSources {
        paths: Tree
            .list("kernels")
            .unwrap()
            .into_iter()
            .chain(Tree.list("crates").unwrap())
            .collect(),
        ..KernelSources::default()
    };
    for f in macro_files(&fams) {
        src.texts.insert(f.clone(), common::read(&f));
    }
    assert_eq!(drift(&fams, &src), Vec::<String>::new());
    // 2026-09-29: A new per-head_dim copy is drift until the manifest lists it.
    src.paths
        .insert("kernels/gb10/common/paged_decode_attn_turbo5_128.cu".into());
    let d = drift(&fams, &src);
    assert!(d.len() == 1 && d[0].contains("turbo5_128"), "{d:?}");
}

#[test]
fn every_fusions_kernel_and_emitter_belongs_to_one_family() {
    let fams = families();
    let rules = metrale_circuit::parse_rule_set(&common::read("kernels/gb10/common/FUSIONS.toml"))
        .unwrap()
        .rules;
    for r in &rules {
        for k in &r.kernels {
            let n = fams
                .families
                .iter()
                .filter(|f| f.kernels.contains(k))
                .count();
            assert_eq!(n, 1, "rule {}: kernel {k} is in {n} families", r.id);
        }
        if r.kernels.is_empty() {
            assert!(
                fams.families
                    .iter()
                    .any(|f| f.emitters.contains(&r.emitter)),
                "rule {}: emitter {} is in no family",
                r.id,
                r.emitter
            );
        }
    }
}

#[test]
fn every_legacy_citation_holds_its_line() {
    for l in &families().legacy {
        let (path, line) = l.cite.rsplit_once(':').expect("path:line");
        let text = common::read(path);
        let at = text
            .lines()
            .nth(line.parse::<usize>().unwrap() - 1)
            .unwrap_or_default();
        assert!(
            at.contains(&l.holds),
            "{}: `{at}` does not hold `{}`",
            l.cite,
            l.holds
        );
    }
}

#[test]
fn the_roofline_constants_are_kernel_perf_mds_peaks() {
    let r = families().roofline;
    let doc = common::read("KERNEL-PERF.md");
    for v in [
        format!("**{:.1} GB/s**", r.dram_gbps),
        format!("**{:.1} TFLOPS**", r.bf16_tflops),
        format!("**{:.1} TFLOPS**", r.fp8_tflops),
        format!("**{:.1} TFLOPS**", r.nvfp4_tflops),
    ] {
        assert!(doc.contains(&v), "KERNEL-PERF.md has no {v}");
    }
}

#[test]
fn a_checkpoint_directory_target_is_checked_against_its_declarations() {
    let (config, hfq) = lightning_checkpoint();
    let (c, h) = (config.to_string(), hfq.to_string());
    let text = venn::report_text(&Tree, &args(), Some((&c, Some(&h)))).expect("agrees");
    assert_eq!(text, common::read(REPORT));
    // 2026-09-29: One layer of another kind.
    let mut other = config.clone();
    other["layers_block_type"][0] = "attention".into();
    let e = venn::report_text(&Tree, &args(), Some((&other.to_string(), Some(&h)))).unwrap_err();
    assert!(
        matches!(&e, venn::VennError::Checkpoint(m) if m.contains("layer kinds")),
        "{e}"
    );
    // 2026-09-29: A checkpoint whose lm_head is BF16 disagrees with the NVFP4 head the circuit runs.
    let mut bf16_head = hfq.clone();
    bf16_head["quantization"]["quantized_layers"]
        .as_object_mut()
        .unwrap()
        .remove("lm_head");
    let e =
        venn::report_text(&Tree, &args(), Some((&c, Some(&bf16_head.to_string())))).unwrap_err();
    assert!(
        matches!(&e, venn::VennError::Checkpoint(m) if m.contains("lm_head: the circuit runs nvfp4/g16/bf16, the checkpoint declares bf16/bf16")),
        "{e}"
    );
}

#[test]
fn target_specs_resolve_by_recipe_checkpoint_or_unique_arch() {
    let all = common::instances();
    for spec in [
        LIGHTNING,
        "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        "nemotron_h",
    ] {
        assert_eq!(
            venn::resolve(&all, spec).unwrap().recipe,
            LIGHTNING,
            "{spec}"
        );
    }
    assert!(venn::resolve(&all, "llama").is_err());
    assert_eq!(
        venn::checkpoint_id_of(
            "/c/hub/models--nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4/snapshots/ab12/"
        ),
        Some("nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4".into())
    );
    assert_eq!(
        venn::checkpoint_id_of("/models/nvidia/Foo"),
        Some("nvidia/Foo".into())
    );
}

#[test]
#[ignore = "writes the checked-in report; run explicitly to regenerate"]
fn regenerate() {
    let text = venn::report_text(&Tree, &args(), None).expect("report");
    std::fs::write(common::root().join(REPORT), text).expect("write report");
}
