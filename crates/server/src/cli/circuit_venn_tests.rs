// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: `met circuit venn`: the flag surface, a checkpoint directory resolved to the
//! checkpoint it holds, and `--check` against the checked-in report (current, and stale).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::Parser;

use super::*;
use crate::cli::{CircuitAction, Cli, Command};

const REPORT: &str = "kernels/circuits/venn/nemotron-3.5-lightning-vs-qwen3.6-35b-a3b.md";

fn parse(extra: &[&str]) -> Result<CircuitVennArgs, clap::Error> {
    let mut argv = vec!["met", "circuit", "venn"];
    argv.extend_from_slice(extra);
    match Cli::try_parse_from(argv)?.command {
        Command::Circuit(c) => match c.action {
            CircuitAction::Venn(v) => Ok(*v),
            other => panic!("parsed {other:?}"),
        },
        other => panic!("parsed {other:?}"),
    }
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

fn lightning(out: &str) -> CircuitVennArgs {
    parse(&[
        "--target=nemotron-3.5/nemotron-3.5-lightning-30b-a3b-nvfp4",
        "--against=qwen3.6/qwen3.6-35b-a3b-fp8-bf16head,qwen3.8/qwen3.8-27b-nvfp4-unsloth",
        "--out",
        out,
        "--check",
        "--root",
        repo_root().to_str().unwrap(),
    ])
    .unwrap()
}

#[test]
fn the_defaults_are_the_representative_rungs_and_every_mode() {
    let a = parse(&["--target=t", "--against=a,b", "--out=o.md"]).unwrap();
    assert_eq!(a.against, ["a", "b"]);
    assert_eq!(a.rows, [1, 16, 128]);
    assert_eq!(a.verify_rows, [2]);
    assert_eq!(a.mode.len(), 4);
    assert!(!a.check);
    assert!(
        parse(&["--target=t", "--out=o.md"]).is_err(),
        "--against is required"
    );
    assert!(
        parse(&["--target=t", "--against=a"]).is_err(),
        "--out is required"
    );
}

#[test]
fn a_checkpoint_directory_resolves_to_the_checkpoint_it_holds() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp
        .path()
        .join("models--nvidia--Toy-Model/snapshots/0123abcd");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("config.json"), "{\"layers_block_type\": []}").unwrap();
    let mut a = parse(&["--target=x", "--against=a", "--out=o.md"]).unwrap();
    a.target = dir.to_string_lossy().into_owned();
    let (args, texts) = resolve_args(&a).unwrap();
    assert_eq!(args.target, "nvidia/Toy-Model");
    let t = texts.expect("config texts");
    assert!(t.config.contains("layers_block_type"));
    assert!(t.hf_quant.is_none());
    // 2026-09-29: A recipe id is passed through untouched, with no checkpoint texts.
    let (args, texts) =
        resolve_args(&parse(&["--target=r/x", "--against=a", "--out=o.md"]).unwrap()).unwrap();
    assert_eq!(args.target, "r/x");
    assert!(texts.is_none());
}

#[test]
fn check_passes_on_the_checked_in_report_and_fails_on_a_stale_one() {
    run(lightning(REPORT)).unwrap();
    // 2026-09-29: Any other file stands in for a stale report; --check must not write it.
    let before = std::fs::read_to_string(repo_root().join("README.md")).unwrap();
    let e = run(lightning("README.md")).unwrap_err().to_string();
    assert!(e.contains("README.md is stale"), "{e}");
    assert_eq!(
        std::fs::read_to_string(repo_root().join("README.md")).unwrap(),
        before
    );
}

/// 2026-10-05: `--hardware` plans every side on the device's class: on the H100 (class `hopper`,
/// no microbench records of its own) no row is measured-shared and the estimate uses the H100's
/// bandwidth; on the GB10 (whose records these are) measured-shared rows remain, so the device
/// path does not simply drop evidence. The offline report refuses a device.
#[test]
fn venn_on_a_device_plans_with_its_class_and_counts_only_its_evidence() {
    let root = repo_root();
    let tree = super::super::circuit_hw::FsTree::new(root.clone());
    let reg = super::super::circuit_hw::registry(&tree).unwrap();
    let on = |device: &str| {
        let mut a = lightning("unused.md");
        a.device.hardware = Some(device.into());
        let (args, _) = resolve_args(&a).unwrap();
        metrale_circuit::hardware::venn_text(&tree, &reg, &args, None).unwrap()
    };
    let shared = |text: &str| text.matches("| Shared |").count();
    let h100 = on("h100-sxm");
    assert_eq!(shared(&h100), 0, "hopper has no records of its own");
    assert!(h100.matches("| Shared, unmeasured |").count() > 0);
    assert!(h100.contains("--hardware h100-sxm --out unused.md"));
    assert!(h100.contains("| Planned on | `h100-sxm` (class `hopper`"));
    assert!(
        h100.contains("max(bytes / 3350.0 GB/s"),
        "the H100's bandwidth"
    );
    let gb10 = on("gb10");
    assert!(shared(&gb10) > 0, "gb10's own records still count on gb10");
    let offline = std::fs::read_to_string(root.join(REPORT)).unwrap();
    assert!(shared(&offline) > 0);
    assert!(!offline.contains("| Planned on |"));
    let mut a = lightning("unused.md");
    a.device.hardware = Some("h100-sxm".into());
    let (args, _) = resolve_args(&a).unwrap();
    let repo = FsRepo { root };
    assert!(
        metrale_circuit::venn::report_text(&repo, &args, None).is_err(),
        "the offline report never silently ignores --hardware"
    );
}
