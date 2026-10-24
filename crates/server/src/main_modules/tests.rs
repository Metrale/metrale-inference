// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `serve` command-line parsing and of the per-layer KV
//! dtype helpers in `kv_dtypes.rs`.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use clap::Parser;

use crate::cli::{Cli, Command};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_kv_codec::{BlockGeometry, KvCachePlan, KvCodecId, KvPair, KvPlanInputs};

use crate::main_modules::legacy_layer_dtypes;

#[test]
fn test_cli_parse_positional_model() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "--port",
        "9999",
        "--max-seq-len",
        "8192",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(
                args.model.as_deref(),
                Some("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"),
            );
            assert!(args.model_from_path.is_none());
            assert_eq!(args.port, 9999);
            assert_eq!(args.max_seq_len, 8192);
            assert_eq!(args.gpu_memory_utilization, 0.90);
            assert_eq!(args.scheduler, "fifo");
            assert_eq!(args.scheduler_config, "sync");
            assert_eq!(args.tbt_deadline_ms, 100);
        }
    }
}

#[test]
fn test_cli_parse_model_from_path() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "--model-from-path",
        "/tmp/model",
        "--port",
        "8888",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert!(args.model.is_none());
            assert_eq!(
                args.model_from_path,
                Some(std::path::PathBuf::from("/tmp/model")),
            );
        }
    }
}

#[test]
fn test_cli_parse_slai_policy() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/model",
        "--scheduler",
        "slai",
        "--tbt-deadline-ms",
        "50",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.scheduler, "slai");
            assert_eq!(args.tbt_deadline_ms, 50);
        }
    }
}

/// 2026-10-18: The boundary rule and auto count are tested in `metrale-kv-codec`
/// (`tests/resolve.rs`); these pin the legacy list derived from a plan.
fn legacy_list(requested: &str, n: usize, boundary_layers: u32) -> Vec<KvCacheDtype> {
    let requested: KvPair = requested.parse().unwrap();
    let boundary = KvPair::symmetric(KvCodecId::Bf16);
    let geometry = vec![
        BlockGeometry {
            block_size: 16,
            num_kv_heads: 4,
            head_dim: 256,
        };
        n
    ];
    let plan = KvCachePlan::resolve(&KvPlanInputs {
        requested,
        boundary_layers,
        boundary,
        geometry: &geometry,
    });
    legacy_layer_dtypes(&plan, requested, boundary).unwrap()
}

#[test]
fn a_plan_uniform_at_the_requested_pair_has_an_empty_legacy_list() {
    assert!(legacy_list("nvfp4", 12, 0).is_empty());
    assert!(legacy_list("bf16", 12, 2).is_empty());
    assert!(legacy_list("turbo4", 0, 2).is_empty());
}

#[test]
fn a_boundary_plan_lists_every_layer() {
    use KvCacheDtype as D;
    let mut want = vec![D::Nvfp4; 12];
    for i in [0, 1, 10, 11] {
        want[i] = D::Bf16;
    }
    assert_eq!(legacy_list("nvfp4", 12, 2), want);
    assert_eq!(
        legacy_list("turbo4k3v", 3, 1),
        [D::Bf16, D::Turbo4KTurbo3V, D::Bf16]
    );
}

#[test]
fn a_boundary_covering_every_layer_still_lists_them() {
    // 2026-10-18: turbo2's auto count on 16 attention layers is 13, so the first 13 and the
    // last 13 cover all 16. The list must still be sent: an empty one would make every layer
    // the base dtype (turbo2) instead of bf16.
    let n = 16;
    let auto = "turbo2"
        .parse::<KvPair>()
        .unwrap()
        .auto_boundary()
        .layers(n)
        .unwrap();
    assert_eq!(auto, 13);
    let list = legacy_list("turbo2", 16, auto);
    assert_eq!(list, vec![KvCacheDtype::Bf16; 16]);
    assert_eq!(legacy_list("fp8", 4, 3), vec![KvCacheDtype::Bf16; 4]);
}

#[test]
fn test_cli_parse_kv_high_precision_layers() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/model",
        "--kv-high-precision-layers",
        "3",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "3");
        }
    }
}

#[test]
fn test_cli_default_kv_high_precision_layers() {
    let cli = Cli::try_parse_from(["met", "serve", "nvidia/model"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor
        | Command::Circuit(_)
        | Command::Blueprint(_)
        | Command::Embed(_) => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "0");
        }
    }
}
