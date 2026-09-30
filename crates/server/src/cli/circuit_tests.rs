// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit` against the embedded circuit files: every instance loads, the
//! CLI's plan is the checked-in golden plan, and bad requests are refused by name.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::Parser;

use super::*;
use crate::cli::{Cli, Command};
use metrale_model_layers::circuit_exec::sources::{BLOCKS, CIRCUITS, INSTANCES, lookup};

fn parse(args: &[&str]) -> CircuitArgs {
    let mut argv = vec!["met", "circuit"];
    argv.extend_from_slice(args);
    match Cli::try_parse_from(argv).expect("parses").command {
        Command::Circuit(c) => c,
        other => panic!("parsed {other:?}"),
    }
}

fn plan_args(args: &[&str]) -> CircuitPlanArgs {
    match parse(args).action {
        CircuitAction::Show(p) => p,
        CircuitAction::Display(d) => d.plan,
        CircuitAction::Diff(_) => panic!("parsed diff"),
        CircuitAction::Venn(_) => panic!("parsed venn"),
        CircuitAction::Plan(_) => panic!("parsed plan"),
    }
}

#[test]
fn every_instance_source_is_embedded_and_loads() {
    let all = metrale_circuit::parse_instances(INSTANCES).unwrap();
    assert!(all.len() >= 2);
    for inst in &all {
        let src = sources(inst).unwrap_or_else(|e| panic!("{}: {e}", inst.recipe));
        metrale_circuit::load(inst, src).unwrap_or_else(|e| panic!("{}: {e}", inst.recipe));
        let circuit = lookup(&CIRCUITS, &inst.arch, "circuit").unwrap();
        for name in metrale_circuit::includes_of(circuit).unwrap() {
            lookup(&BLOCKS, &name, "block library").unwrap();
        }
    }
}

#[test]
fn the_cli_plan_is_the_checked_in_golden_plan() {
    let inst = instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth").unwrap();
    let loaded = metrale_circuit::load(&inst, sources(&inst).unwrap()).unwrap();
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    for (mode, rows, file) in [
        (Mode::Decode, 1, "qwen3_5-decode-n1.txt"),
        (Mode::Verify, 4, "qwen3_5-verify-n4.txt"),
    ] {
        let shown = metrale_circuit::render_plan(&inst, &loaded, &avail, mode, rows).unwrap();
        let path = format!(
            "{}/../../kernels/circuits/plans/{file}",
            env!("CARGO_MANIFEST_DIR")
        );
        assert_eq!(shown, std::fs::read_to_string(path).unwrap(), "{file}");
    }
}

#[test]
fn an_unknown_recipe_names_the_known_ones() {
    let err = instance("qwen3.6/not-a-recipe").unwrap_err().to_string();
    assert!(err.contains("qwen3.6/not-a-recipe"), "{err}");
    assert!(err.contains("qwen3.8/qwen3.8-27b-nvfp4-unsloth"), "{err}");
}

#[test]
fn rows_default_only_where_one_row_is_the_plan() {
    let inst = instance("qwen3.6/qwen3.6-35b-a3b-fp8-bf16head").unwrap();
    let r = "--recipe=qwen3.6/qwen3.6-35b-a3b-fp8-bf16head";
    assert_eq!(rows_of(&inst, &plan_args(&["show", r])).unwrap(), 1);
    assert_eq!(
        rows_of(&inst, &plan_args(&["show", r, "--mode", "draft"])).unwrap(),
        1
    );
    assert_eq!(
        rows_of(
            &inst,
            &plan_args(&["display", r, "--mode", "multi_seq", "--rows", "96"])
        )
        .unwrap(),
        96
    );
    let err = rows_of(&inst, &plan_args(&["show", r, "--mode", "verify"]))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("needs --rows") && err.contains("[2, 3, 4]"),
        "{err}"
    );
    assert!(rows_of(&inst, &plan_args(&["show", r, "--rows", "0"])).is_err());
}

#[test]
fn display_flags_parse_and_conflict() {
    let r = "--recipe=qwen3.6/qwen3.6-35b-a3b-fp8-bf16head";
    match parse(&["display", r, "--layer", "7", "--ascii", "--color", "never"]).action {
        CircuitAction::Display(d) => {
            assert_eq!(d.layer, Some(7));
            assert!(d.ascii && !d.all_layers);
            assert_eq!(d.color, crate::cli::ColorChoice::Never);
        }
        other => panic!("parsed {other:?}"),
    }
    let both = Cli::try_parse_from([
        "met",
        "circuit",
        "display",
        r,
        "--layer",
        "1",
        "--all-layers",
    ]);
    assert!(
        both.is_err(),
        "--layer and --all-layers together must be refused"
    );
    assert!(Cli::try_parse_from(["met", "circuit", "show", r, "--mode", "prefill"]).is_err());
}

const DENSE_CONFIG: &str = r#"{
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "hidden_size": 5120, "intermediate_size": 17408, "vocab_size": 248320,
    "num_hidden_layers": 8, "num_attention_heads": 24, "num_key_value_heads": 4,
    "head_dim": 256, "full_attention_interval": 4,
    "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention",
                    "linear_attention", "linear_attention", "linear_attention", "full_attention"],
    "linear_num_key_heads": 16, "linear_key_head_dim": 128,
    "linear_num_value_heads": 48, "linear_value_head_dim": 128, "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6, "max_position_embeddings": 262144
  }
}"#;

#[test]
fn the_config_adapter_matches_the_instance_and_names_drift() {
    let cfg = metrale_config::parse_config(DENSE_CONFIG).expect("fixture config parses");
    let from_config = arch_shape(&cfg).unwrap();
    let mut stated = instance("qwen3.8/qwen3.8-27b-nvfp4-unsloth").unwrap().shape;
    stated.layer_kinds.truncate(8);
    assert_eq!(shape_drift(&stated, &from_config), Vec::<String>::new());
    let wider = DENSE_CONFIG.replace("\"hidden_size\": 5120", "\"hidden_size\": 6144");
    let drift = shape_drift(
        &stated,
        &arch_shape(&metrale_config::parse_config(&wider).unwrap()).unwrap(),
    );
    assert_eq!(
        drift,
        ["hidden: INSTANCES.toml 5120, config.json Some(6144)"]
    );
    stated.layer_kinds.pop();
    assert_eq!(shape_drift(&stated, &from_config).len(), 1);
}

#[test]
fn diff_takes_its_counts_explicitly_and_the_serve_flags_after_them() {
    let d = match parse(&[
        "diff",
        "--steps",
        "64",
        "--prompts",
        "4",
        "--out",
        "r.json",
        "m/x",
    ])
    .action
    {
        CircuitAction::Diff(d) => d,
        other => panic!("parsed {other:?}"),
    };
    assert_eq!((d.steps, d.prompts), (64, 4));
    assert_eq!(d.serve.model.as_deref(), Some("m/x"));
    for missing in ["--steps", "--prompts", "--out"] {
        let mut argv = vec![
            "met",
            "circuit",
            "diff",
            "--steps",
            "64",
            "--prompts",
            "4",
            "--out",
            "r",
        ];
        let at = argv.iter().position(|a| *a == missing).unwrap();
        argv.drain(at..at + 2);
        assert!(Cli::try_parse_from(argv).is_err(), "{missing} is required");
    }
}
