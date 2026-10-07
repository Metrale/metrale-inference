// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-06: Experimental GPT admission controls, separate from production field scans.
use super::*;
use clap::Parser;
fn fixture() -> (ServeArgs, ModelConfig) {
    let crate::cli::Command::Serve(args) = crate::cli::Cli::parse_from([
        "met",
        "serve",
        "checkpoint",
        "--experimental-gpt-oss",
        "--max-batch-size",
        "1",
        "--swap-space-gb",
        "0",
        "--gpu-memory-utilization",
        "0.85",
        "--lm-head-dtype",
        "bf16",
    ])
    .command
    else {
        panic!("serve fixture")
    };
    let config = metrale_config::parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json"
    )))
    .unwrap();
    (args, config)
}
#[test]
fn opt_in_is_required_and_does_not_expand_to_other_models() {
    let (mut args, mut config) = fixture();
    assert!(validate(&args, &config).is_ok());
    args.experimental_gpt_oss = false;
    assert!(validate(&args, &config).is_err());
    args.experimental_gpt_oss = true;
    config.model_type = "qwen3".into();
    assert!(validate(&args, &config).is_err());
}
#[test]
fn incompatible_runtime_options_are_refused_individually() {
    let (args, config) = fixture();
    let mutations: &[fn(&mut ServeArgs)] = &[
        |a| a.max_batch_size = SlotRequest::Count(2),
        |a| a.max_batch_size = SlotRequest::Auto,
        |a| a.kv_cache_dtype = Some("fp8".into()),
        |a| a.enable_prefix_caching = true,
        |a| a.swap_space_gb = 1,
        |a| a.high_speed_swap = true,
        |a| a.tp_size = 2,
        |a| a.ep_size = 2,
        |a| a.world_size = 2,
        |a| a.speculative = true,
        |a| a.prefill_varlen_batch = true,
        |a| a.prefill_codispatch = true,
        |a| a.gpu_memory_utilization = 0.86,
        |a| a.gpu_memory_utilization = f64::NAN,
        |a| a.gpu_memory_utilization = 0.0,
        |a| a.draft_model = Some("draft".into()),
        |a| a.lora_adapter.push(("adapter".into(), "path".into())),
        |a| a.forward = crate::cli::flag_values::ForwardArg::Circuit,
        |a| a.lm_head_dtype = "default".into(),
    ];
    for mutate in mutations {
        let mut candidate = args.clone();
        mutate(&mut candidate);
        assert!(
            validate(&candidate, &config).is_err(),
            "accepted {candidate:?}"
        );
    }
}

#[test]
fn hermetic_resolves_prefix_caching_before_runtime_admission() {
    let (mut args, config) = fixture();
    args.enable_prefix_caching = true;
    assert!(validate(&args, &config).is_err());
    args.hermetic = true;
    assert!(validate(&args, &config).is_ok());
}

#[test]
fn chunk_prefill_requires_explicit_parent_opt_in_and_preserves_refusals() {
    let (mut args, mut config) = fixture();
    assert!(!args.experimental_gpt_oss_chunk_prefill);
    args.experimental_gpt_oss_chunk_prefill = true;
    assert!(validate(&args, &config).is_ok());
    args.experimental_gpt_oss = false;
    assert!(validate(&args, &config).is_err());
    args.experimental_gpt_oss = true;
    args.max_batch_size = SlotRequest::Count(2);
    assert!(validate(&args, &config).is_err());
    args.max_batch_size = SlotRequest::Count(1);
    config.model_type = "qwen3".into();
    assert!(validate(&args, &config).is_err());
    assert!(
        crate::cli::Cli::try_parse_from([
            "met",
            "serve",
            "checkpoint",
            "--experimental-gpt-oss-chunk-prefill"
        ])
        .is_err()
    );
}

#[test]
fn explicit_chunk_capacity_requires_parent_and_retains_c1() {
    let (mut args, mut config) = fixture();
    assert_eq!(args.experimental_gpt_oss_chunk_tokens, None);
    args.experimental_gpt_oss_chunk_tokens = Some(64);
    assert!(validate(&args, &config).is_err());
    args.experimental_gpt_oss_chunk_prefill = true;
    args.experimental_gpt_oss_chunk_tokens = None;
    assert_eq!(
        crate::main_modules::serve_phases::experimental_policy(&args)
            .chunk_tokens()
            .unwrap(),
        Some(16)
    );
    for tokens in [16, 64, 128] {
        args.experimental_gpt_oss_chunk_tokens = Some(tokens);
        assert!(validate(&args, &config).is_ok());
        assert_eq!(
            crate::main_modules::serve_phases::experimental_policy(&args)
                .chunk_tokens()
                .unwrap(),
            Some(tokens)
        );
        args.max_batch_size = SlotRequest::Count(2);
        assert!(validate(&args, &config).is_err());
        args.max_batch_size = SlotRequest::Count(1);
    }
    for tokens in [0, 1, 17, 63, 65, 129, usize::MAX] {
        args.experimental_gpt_oss_chunk_tokens = Some(tokens);
        assert!(validate(&args, &config).is_err());
    }
    args.experimental_gpt_oss_chunk_tokens = Some(128);
    config.model_type = "qwen3".into();
    assert!(validate(&args, &config).is_err());
    assert!(
        crate::cli::Cli::try_parse_from([
            "met",
            "serve",
            "checkpoint",
            "--experimental-gpt-oss-chunk-tokens",
            "64"
        ])
        .is_err()
    );
}
