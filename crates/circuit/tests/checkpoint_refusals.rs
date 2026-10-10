// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `instantiate_from_checkpoint` over the real checkpoint fixtures (see
//! checkpoints.rs, Path A), split out of it when it passed 500 lines:
//! - Path B: for each golden instance, the config-derived shape IS the one INSTANCES.toml states,
//!   and the golden plans rendered under it are the checked-in ones.
//! - Path C: every other checkpoint is refused, and each refusal names its reason: an unserved
//!   model type, an unmapped key, a refused value, or a precision the circuit cannot express.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use checkpoint_fixtures::*;
use metrale_circuit::{
    CheckpointError, QuantMetadata, ServePrecision, map_checkpoint, resolve_checkpoint,
};

#[test]
fn path_b_golden_instances_restate_the_config_derived_shape() {
    for inst in common::instances().iter().filter(|i| i.golden) {
        let (config, _) = fixture(&inst.checkpoint.replace('/', "--"));
        let mapped = map_checkpoint(&config).unwrap_or_else(|e| panic!("{}: {e}", inst.recipe));
        assert_eq!(mapped.arch, inst.arch, "{}", inst.recipe);
        assert_eq!(
            mapped.shape, inst.shape,
            "{}: INSTANCES.toml != config.json",
            inst.recipe
        );
        let mut derived = inst.clone();
        derived.shape = mapped.shape;
        let loaded = common::load(&derived);
        let avail = common::available(&derived, &loaded.rules);
        let fams = common::families(&derived);
        for (&mode, rows) in &derived.plans {
            for &n in rows {
                let text = metrale_circuit::render_plan(&derived, &loaded, &avail, mode, n, &fams)
                    .unwrap_or_else(|e| panic!("{} {} n={n}: {e}", inst.recipe, mode.name()));
                let file = common::plans_dir().join(derived.plan_file(mode, n));
                assert_eq!(
                    text,
                    std::fs::read_to_string(&file).unwrap(),
                    "{}",
                    file.display()
                );
            }
        }
    }
}

/// 2026-09-30: Every other fixture is refused, for the reason named.
#[test]
fn path_c_every_other_checkpoint_is_refused_with_its_reason() {
    let cases: [(&str, &str); 10] = [
        (
            "Inferact--Qwen3.8-Flash-Next-NVFP4",
            "model_type `qwen4_exp`",
        ),
        ("Qwen--Qwen3-Coder-Next-FP8", "model_type `qwen3_next`"),
        (
            "nvidia--Qwen3-Next-80B-A3B-Instruct-NVFP4",
            "model_type `qwen3_next`",
        ),
        (
            "ig1--Qwen3-VL-30B-A3B-Instruct-NVFP4",
            "model_type `qwen3_vl_moe`",
        ),
        // 2026-10-10: Block diffusion is refused by what the circuit model lacks (a canvas mode,
        // non-causal read-only attention, the self-conditioning state, the denoising loop), not
        // by a missing map; the FP8 checkpoint's quantization is never reached.
        (
            "google--diffusiongemma-26B-A4B-it",
            "model_type `diffusion_gemma` has no circuit: block diffusion",
        ),
        (
            "RedHatAI--diffusiongemma-26B-A4B-it-FP8-dynamic",
            "model_type `diffusion_gemma` has no circuit: block diffusion",
        ),
        ("lukealonso--MiniMax-M2.7-NVFP4", "model_type `minimax_m2`"),
        ("stepfun-ai--Step-3.7-Flash-NVFP4", "model_type `step3p7`"),
        // 2026-09-30: DFlash drafts reuse `qwen3`; their extra keys refuse them.
        (
            "incoai--Qwen3.8-27B-DFlash2",
            "`dflash_config` is not mapped",
        ),
        ("z-lab--Qwen3.6-27B-DFlash", "`auto_map` is not mapped"),
    ];
    for (name, want) in cases {
        let e = resolve(name).expect_err(name).to_string();
        assert!(e.contains(want), "{name}: {e}");
    }
}

#[test]
fn path_c_an_unmapped_math_key_or_a_missing_quant_group_is_refused() {
    let (config, _) = fixture("NousResearch--Meta-Llama-3.1-8B-Instruct");
    let with = |k: &str, v: serde_json::Value| {
        let mut j: serde_json::Value = serde_json::from_str(&config).unwrap();
        j[k] = v;
        resolve_checkpoint(
            &j.to_string(),
            QuantMetadata::default(),
            &ServePrecision::Declared,
        )
    };
    // 2026-09-30: An unknown key, an unmodelled RoPE type, a sliding window, a GELU FFN.
    let e = with("logit_softcapping", 30.0.into()).unwrap_err();
    assert!(
        matches!(&e, CheckpointError::Map(m) if m.to_string().contains("logit_softcapping")),
        "{e}"
    );
    let e = with(
        "rope_scaling",
        serde_json::json!({"rope_type": "longrope", "factor": 4.0}),
    )
    .unwrap_err();
    assert!(e.to_string().contains("longrope"), "{e}");
    assert!(
        with("sliding_window", 4096.into())
            .unwrap_err()
            .to_string()
            .contains("sliding-window")
    );
    assert!(
        with("hidden_act", "gelu".into())
            .unwrap_err()
            .to_string()
            .contains("SiLU")
    );
    // 2026-09-30: A quantization group whose scheme the circuit has no format for (integer
    // weights), and malformed sidecar metadata.
    let e = with(
        "quantization_config",
        serde_json::json!({
            "quant_method": "compressed-tensors",
            "config_groups": { "group_0": {
                "weights": { "num_bits": 8, "type": "int", "strategy": "channel" },
                "targets": ["Linear"] } },
        }),
    )
    .unwrap_err();
    assert!(
        matches!(&e, CheckpointError::Quant(m) if m.contains("no format")),
        "{e}"
    );
    // 2026-09-30: A quantization block that names a method but no scheme for any layer (a
    // missing quant group) would read every projection as 16-bit; it is refused.
    let e = with(
        "quantization_config",
        serde_json::json!({ "quant_method": "compressed-tensors", "config_groups": {} }),
    )
    .unwrap_err();
    assert!(
        matches!(&e, CheckpointError::Quant(m) if m.contains("missing quant group")),
        "{e}"
    );
    let e = resolve_checkpoint(
        &config,
        QuantMetadata {
            hf_quant_config: Some(
                r#"{"producer": {"name": "modelopt"}, "quantization": {"quant_algo": "MIXED_PRECISION", "quantized_layers": {}}}"#,
            ),
        },
        &ServePrecision::Declared,
    )
    .unwrap_err();
    assert!(
        matches!(&e, CheckpointError::Quant(m) if m.contains("without quantized_layers")),
        "{e}"
    );
    let e = resolve_checkpoint(
        &config,
        QuantMetadata {
            hf_quant_config: Some("{ not json"),
        },
        &ServePrecision::Declared,
    )
    .unwrap_err();
    assert!(
        matches!(
            e,
            CheckpointError::Json {
                file: "hf_quant_config.json",
                ..
            }
        ),
        "{e}"
    );
    // 2026-09-30: A Nemotron-H layer schedule with a dense MLP layer ('-') is refused.
    let (nano, hq) = fixture("nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4");
    let mut j: serde_json::Value = serde_json::from_str(&nano).unwrap();
    let p = j["hybrid_override_pattern"]
        .as_str()
        .unwrap()
        .replacen('E', "-", 1);
    j["hybrid_override_pattern"] = p.into();
    let e = resolve_checkpoint(
        &j.to_string(),
        QuantMetadata {
            hf_quant_config: hq.as_deref(),
        },
        &ServePrecision::Declared,
    )
    .unwrap_err();
    assert!(e.to_string().contains("'-'"), "{e}");
}
