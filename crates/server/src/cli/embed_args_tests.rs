// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for `met embed`'s rewrite into the serve its blueprint pins.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;
use clap::Parser as _;

fn recipe() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../recipes/llama-nemotron-embed/llama-nemotron-embed-vl-1b-v2-bf16.yaml")
}

fn args(blueprint: PathBuf, idle: u64) -> EmbedArgs {
    EmbedArgs {
        blueprint,
        input_type: "passage".into(),
        idle_exit_ms: idle,
    }
}

/// 2026-10-24: The embed recipe becomes its own `met serve` argv (model and limits from the
/// recipe, the dashboard off), and the hash is the file's.
#[test]
fn the_recipe_becomes_its_serve() {
    let (serve, stdio) = into_serve(&args(recipe(), 0)).unwrap();
    assert_eq!(
        serve.model.as_deref(),
        Some("nvidia/llama-nemotron-embed-vl-1b-v2")
    );
    assert_eq!(serve.max_seq_len, 4096);
    let want = super::super::blueprint_build::sha_file(&recipe())
        .unwrap()
        .hex();
    assert_eq!(stdio.blueprint_sha256, want);
    assert_eq!(stdio.blueprint_sha256.len(), 64);
    assert_eq!(stdio.input_type, "passage");
    assert_eq!(stdio.idle_exit, None);
    assert_eq!(
        EmbedStdio::new(&args(recipe(), 1500), want).idle_exit,
        Some(Duration::from_millis(1500))
    );
}

/// 2026-10-24: A signed blueprint and a missing file are refused, never served from a guess.
#[test]
fn a_kbp_or_missing_blueprint_is_refused() {
    let e = into_serve(&args(PathBuf::from("/x/y/z.kbp"), 0)).unwrap_err();
    assert!(e.to_string().contains("signed blueprint"), "{e}");
    assert!(into_serve(&args(PathBuf::from("/nonexistent/f/r.yaml"), 0)).is_err());
}

/// 2026-10-24: `met embed` parses with every flag required, and any other command passes
/// through the rewrite unchanged.
#[test]
fn every_flag_is_required() {
    let ok = super::super::Cli::try_parse_from([
        "met",
        "embed",
        "--blueprint",
        "/b.yaml",
        "--input-type",
        "query",
        "--idle-exit-ms",
        "0",
    ]);
    assert!(matches!(
        ok.unwrap().command,
        super::super::Command::Embed(_)
    ));
    for drop in ["--blueprint", "--input-type", "--idle-exit-ms"] {
        let mut argv = vec![
            "met",
            "embed",
            "--blueprint",
            "/b.yaml",
            "--input-type",
            "query",
            "--idle-exit-ms",
            "0",
        ];
        let i = argv.iter().position(|a| *a == drop).unwrap();
        argv.drain(i..i + 2);
        assert!(super::super::Cli::try_parse_from(argv).is_err(), "{drop}");
    }
    let circuit = super::super::Cli::try_parse_from(["met", "sync-recipes"]).unwrap();
    let (back, embed) = rewrite(circuit).unwrap();
    assert!(embed.is_none());
    assert!(matches!(back.command, super::super::Command::SyncRecipes));
}
