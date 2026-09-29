// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Tests of the Mamba-2 speculative-decoding refusal.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::*;
use clap::Parser as _;

const PROPOSERS: [&str; 4] = [
    "--speculative",
    "--self-speculative",
    "--ngram-speculative",
    "--dflash",
];

fn args(extra: &[&str]) -> cli::ServeArgs {
    let mut argv = vec!["met", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"];
    argv.extend_from_slice(extra);
    cli::ServeArgs::parse_from(argv)
}

/// 2026-09-29: Nemotron-H with one Mamba-2, one MoE and one attention layer.
fn nemotron_h() -> ModelConfig {
    metrale_config::parse_config(
        r#"{
            "model_type": "nemotron_h",
            "hidden_size": 2688,
            "num_hidden_layers": 3,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "n_routed_experts": 128,
            "num_experts_per_tok": 6,
            "moe_intermediate_size": 1856,
            "vocab_size": 131072,
            "hybrid_override_pattern": "ME*",
            "mamba_num_heads": 64,
            "mamba_head_dim": 64,
            "ssm_state_size": 128,
            "n_groups": 8,
            "conv_kernel": 4
        }"#,
    )
    .unwrap()
}

#[test]
fn every_speculative_proposer_is_refused_over_mamba2() {
    let config = nemotron_h();
    for flag in PROPOSERS {
        let err = refuse_speculation_over_mamba2(&args(&[flag]), &config)
            .expect_err(flag)
            .to_string();
        assert!(err.contains("Mamba-2"), "{flag}: {err}");
        assert!(err.contains("nemotron_h"), "{flag}: {err}");
    }
}

/// 2026-09-29: Path B: the same model without a proposer boots.
#[test]
fn mamba2_without_a_proposer_passes() {
    refuse_speculation_over_mamba2(&args(&[]), &nemotron_h()).unwrap();
}

/// 2026-09-29: Path B: GatedDeltaNet SSM models keep every proposer (their rollback exists).
#[test]
fn gated_delta_net_keeps_its_proposers() {
    let gdn = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(gdn.num_ssm_layers() > 0, "the control must have SSM layers");
    for flag in PROPOSERS {
        refuse_speculation_over_mamba2(&args(&[flag]), &gdn).unwrap();
    }
}
