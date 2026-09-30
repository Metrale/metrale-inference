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
