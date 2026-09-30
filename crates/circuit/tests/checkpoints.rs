// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: `instantiate_from_checkpoint` over real checkpoints: the config.json (and the
//! trimmed `hf_quant_config.json`, where the checkpoint ships one) of every checkpoint cached
//! on the three GB10 boxes, and of the G1 and Nemotron-H checkpoints fetched from the Hub, under
//! tests/fixtures/checkpoints/<org>--<name>/.
//!
//! - Path A: each served checkpoint instantiates with the right arch, layer kinds, dims, switches,
//!   params and declared edge formats.
//! - Path B and Path C: checkpoint_refusals.rs.
//!
//! Fixture trimming: per-expert quantization entries for experts 1.. are dropped (the circuit
//! asks expert 0 for a stacked expert projection, `instantiate.rs` `resolve`), which keeps the
//! Nemotron-3 Super and Lightning fixtures small without changing any node's declared format.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod checkpoint_fixtures;
mod common;

use checkpoint_fixtures::*;
use std::collections::BTreeMap;

use metrale_circuit::{Format, OpKind, Scale};

#[test]
fn the_unsloth_27b_declares_w8a8_attention_w4a4_ffn_and_fp8_kv() {
    for name in ["unsloth--Qwen3.8-27B-NVFP4", "unsloth--Qwen3.6-27B-NVFP4"] {
        let r = ok(name);
        assert_eq!(
            (r.arch.as_str(), r.model_type.as_str()),
            ("qwen3_5", "qwen3_5")
        );
        assert_eq!(
            kinds(&r),
            BTreeMap::from([("full_attention", 16), ("linear_attention", 48)])
        );
        assert_eq!(
            (dim(&r, "hidden"), dim(&r, "inter"), dim(&r, "head_dim")),
            (5120, 17408, 256)
        );
        assert_eq!(r.kv_cache, Some(FP8_TENSOR), "{name}");
        let c = &r.circuit;
        // 2026-09-30: q/k/v share one dynamic per-token quantizer of the normed input.
        let q = node(c, "l3.attn.q");
        assert_eq!(
            q.weight,
            Some(Format::Fp8E4m3 {
                scale: Scale::PerChannel
            })
        );
        assert_eq!(input_format(c, "l3.attn.q"), FP8_TOKEN);
        assert_eq!(node(c, "l3.attn.q").inputs, node(c, "l3.attn.k").inputs);
        assert!(matches!(node(c, "l3.attn.xn_quant").op, OpKind::ActQuant(f) if f == FP8_TOKEN));
        // 2026-09-30: The FFN of layers 0-55 is W4A4; layers 56-63 are W8A8.
        assert_eq!(node(c, "l0.dense_ffn.down").weight, Some(NVFP4));
        assert_eq!(input_format(c, "l0.dense_ffn.down"), NVFP4);
        assert_eq!(input_format(c, "l60.dense_ffn.down"), FP8_TOKEN);
        // 2026-09-30: The MTP head is 16-bit.
        assert_eq!(node(c, "draft.mtp_in.fc").weight, Some(Format::Bf16));
        assert_eq!(
            r.params.get("rope.rope_theta").map(String::as_str),
            Some("10000000")
        );
    }
}

#[test]
fn the_qwen36_35b_fp8_is_the_moe_circuit_with_block_scaled_experts() {
    let r = ok("Qwen--Qwen3.6-35B-A3B-FP8");
    assert_eq!(r.arch, "qwen3_6_moe");
    assert_eq!(
        kinds(&r),
        BTreeMap::from([("full_attention", 10), ("linear_attention", 30)])
    );
    assert_eq!(
        (dim(&r, "experts"), dim(&r, "top_k"), dim(&r, "moe_inter")),
        (256, 8, 512)
    );
    let c = &r.circuit;
    assert_eq!(
        node(c, "l0.moe_ffn.experts_down").weight,
        Some(Format::Fp8E4m3 {
            scale: Scale::Block(128, 128)
        })
    );
    // 2026-09-30: HF `fp8` declares dynamic per-(token, 128) activations.
    assert_eq!(
        input_format(c, "l0.moe_ffn.experts_down"),
        Format::Fp8E4m3 {
            scale: Scale::Group(128)
        }
    );
    assert_eq!(node(c, "l0.moe_ffn.router").weight, Some(Format::Bf16));
    assert_eq!(r.kv_cache, None);
}

#[test]
fn nemotron_h_takes_either_layer_schedule_and_its_own_precision() {
    let nano = ok("nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4");
    let lightning = ok("nvidia--NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4");
    for r in [&nano, &lightning] {
        assert_eq!(r.arch, "nemotron_h");
        assert_eq!(
            kinds(r),
            BTreeMap::from([("full_attention", 6), ("mamba", 23), ("moe", 23)])
        );
        assert_eq!(
            (dim(r, "mamba_heads"), dim(r, "ssm_state"), dim(r, "top_k")),
            (64, 128, 6)
        );
        assert_eq!(r.kv_cache, Some(FP8_TENSOR));
        // 2026-09-30: No RoPE anywhere, and the routed top-k is the config's.
        assert!(!r.circuit.nodes.iter().any(|n| n.op == OpKind::Rope));
        let topk = r
            .circuit
            .nodes
            .iter()
            .find(|n| n.op == OpKind::TopK)
            .unwrap();
        assert_eq!(topk.params.get("top_k").map(String::as_str), Some("6"));
    }
    // 2026-09-30: Nano (hybrid_override_pattern) has no MTP layer and NVFP4 mixers.
    assert_eq!(dim(&nano, "mtp"), 0);
    assert!(
        !nano
            .circuit
            .nodes
            .iter()
            .any(|n| n.id.starts_with("draft."))
    );
    assert_eq!(node(&nano.circuit, "l0.mamba.in_proj").weight, Some(NVFP4));
    // 2026-09-30: Lightning (layers_block_type) has the MTP head and FP8 W8A8 mixers whose static
    // activation scales are bound to each projection.
    assert_eq!(dim(&lightning, "mtp"), 1);
    assert!(
        lightning
            .circuit
            .nodes
            .iter()
            .any(|n| n.id.starts_with("draft."))
    );
    let c = &lightning.circuit;
    assert_eq!(node(c, "l0.mamba.in_proj").weight, Some(FP8_TENSOR));
    assert_eq!(input_format(c, "l0.mamba.in_proj"), FP8_TENSOR);
    assert_eq!(
        node(c, "l0.mamba.in_proj_quant").binding,
        ["backbone.layers.0.mixer.in_proj.input_scale"]
    );
    assert_eq!(node(c, "l1.moe.experts_up").weight, Some(NVFP4));
    assert_eq!(
        lightning
            .params
            .get("routed_scaling_factor")
            .map(String::as_str),
        Some("2.5")
    );
}

#[test]
fn nemotron_3_super_runs_its_routed_experts_in_the_latent_space() {
    let r = ok("nvidia--NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4");
    assert_eq!(
        kinds(&r),
        BTreeMap::from([("full_attention", 8), ("mamba", 40), ("moe", 40)])
    );
    assert_eq!(
        (dim(&r, "moe_latent"), dim(&r, "moe_io"), dim(&r, "top_k")),
        (1, 1024, 22)
    );
    let c = &r.circuit;
    // 2026-09-30: hidden -> fc1 -> experts at 1024 -> routed sum -> fc2 -> + shared (hidden).
    let fc1 = node(c, "l1.moe_latent.latent_in");
    assert_eq!(c.edges[fc1.outputs[0]].dim_value, 1024);
    assert_eq!(fc1.weight, Some(FP8_TENSOR));
    assert_eq!(input_format(c, "l1.moe_latent.latent_in"), FP8_TENSOR);
    // 2026-09-30: The experts are declared W4A4: they read the latent input quantized.
    assert_eq!(input_format(c, "l1.moe_latent.experts_up"), NVFP4);
    let q = node(c, "l1.moe_latent.xl_quant");
    assert_eq!(c.edges[q.inputs[0]].id, "l1.moe_latent.xl");
    let shared = node(c, "l1.moe_latent.shared_up");
    assert_eq!(c.edges[shared.inputs[0]].dim_value, 4096);
    let combine = node(c, "l1.moe_latent.combine");
    assert_eq!(
        combine.inputs,
        [
            node(c, "l1.moe_latent.latent_out").outputs[0],
            node(c, "l1.moe_latent.shared_down").outputs[0]
        ]
    );
    // 2026-09-30: The draft MoE layer is the latent block at mtp.layers.1.
    assert_eq!(
        node(c, "draft.moe_latent.latent_in").binding,
        ["mtp.layers.1.mixer.fc1_latent_proj"]
    );
    assert!(c.node("l1.moe.experts_up").is_none());
    assert_eq!(
        node(c, "l1.moe_latent.top_k")
            .params
            .get("top_k")
            .map(String::as_str),
        Some("22")
    );
}

#[test]
fn the_g1_dense_families_map_their_switches_and_rope() {
    let llama = ok("NousResearch--Meta-Llama-3.1-8B-Instruct");
    let qwen3 = ok("Qwen--Qwen3-8B");
    let qwen2 = ok("Qwen--Qwen2.5-7B-Instruct");
    let mistral = ok("mistralai--Mistral-Small-24B-Instruct-2501");
    for (r, layers) in [(&llama, 32), (&qwen3, 36), (&qwen2, 28), (&mistral, 40)] {
        assert_eq!(r.arch, "dense_gqa");
        assert_eq!(kinds(r), BTreeMap::from([("full_attention", layers)]));
        assert!(
            r.circuit
                .nodes
                .iter()
                .all(|n| n.weight.is_none_or(|w| w == Format::Bf16))
        );
    }
    // 2026-09-30: Llama 3.1: head_dim derived (4096 / 32), llama3 RoPE scaling.
    assert_eq!((dim(&llama, "head_dim"), dim(&llama, "kv_heads")), (128, 8));
    assert_eq!(
        llama
            .params
            .get("rope_scaling.rope_type")
            .map(String::as_str),
        Some("\"llama3\"")
    );
    assert_eq!(
        llama.params.get("rope_scaling.factor").map(String::as_str),
        Some("8.0")
    );
    // 2026-09-30: Mistral Small states head_dim 128, not 5120 / 32.
    assert_eq!(dim(&mistral, "head_dim"), 128);
    // 2026-09-30: Qwen3's Q/K norms are present; elsewhere RoPE reads the projection directly.
    assert_eq!(dim(&qwen3, "qk_norm"), 1);
    assert!(qwen3.circuit.node("l0.attn.q_norm").is_some());
    assert!(llama.circuit.node("l0.attn.q_norm").is_none());
    let rope = node(&llama.circuit, "l0.attn.rope");
    assert_eq!(rope.inputs[0], node(&llama.circuit, "l0.attn.q").outputs[0]);
    // 2026-09-30: Qwen2 carries Q/K/V biases.
    assert_eq!(dim(&qwen2, "attn_bias"), 1);
    assert_eq!(
        node(&qwen2.circuit, "l0.attn.q")
            .params
            .get("bias")
            .map(String::as_str),
        Some("true")
    );
    assert!(
        !node(&llama.circuit, "l0.attn.q")
            .params
            .contains_key("bias")
    );
}

#[test]
fn a_qwen_checkpoint_without_an_mtp_layer_has_no_draft_head() {
    let holo = ok("Hcompany--Holo-3.1-35B-A3B-NVFP4");
    assert_eq!((holo.arch.as_str(), dim(&holo, "mtp")), ("qwen3_6_moe", 0));
    assert!(
        !holo
            .circuit
            .nodes
            .iter()
            .any(|n| n.id.starts_with("draft."))
    );
    let with = ok("Qwen--Qwen3.6-35B-A3B-FP8");
    assert!(
        with.circuit
            .nodes
            .iter()
            .any(|n| n.id.starts_with("draft."))
    );
}

/// 2026-09-30: Qwen/Qwen3.6-27B-FP8 lists MoE router names (`...mlp.gate`) in
/// modules_to_not_convert; matched on whole segments (crates/config, HF fp8 targets) they do
/// not exclude the dense gate_proj, so every FFN projection is block-scaled FP8.
#[test]
fn the_hf_fp8_dense_27b_declares_block_scaled_w8a8_everywhere() {
    let r = ok("Qwen--Qwen3.6-27B-FP8");
    let block = Format::Fp8E4m3 {
        scale: Scale::Block(128, 128),
    };
    for id in ["l0.dense_ffn.gate_up", "l0.dense_ffn.down", "l3.attn.q"] {
        assert_eq!(node(&r.circuit, id).weight, Some(block), "{id}");
    }
    assert_eq!(node(&r.circuit, "l0.gdn.ba").weight, Some(Format::Bf16));
}

#[test]
fn a_tied_head_and_the_other_qwen35_checkpoints_instantiate() {
    let small = ok("Qwen--Qwen3.5-0.8B");
    assert_eq!(
        small.params.get("tie_word_embeddings").map(String::as_str),
        Some("true")
    );
    for name in [
        "Qwen--Qwen3.6-27B",
        "Qwen--Qwen3.8-27B",
        "Kbenkhaled--Qwen3.5-27B-NVFP4",
        "nvidia--Qwen3.6-27B-NVFP4",
        "centml--Qwen3.6-27B-NVFP4-W4A4-mlpinf",
        "Sehyo--Qwen3.5-35B-A3B-NVFP4",
        "Sehyo--Qwen3.5-122B-A10B-NVFP4",
        "nvidia--Qwen3.6-35B-A3B-NVFP4",
        "Hcompany--Holo-3.1-35B-A3B-NVFP4",
        "Qwen--Qwen3-32B",
    ] {
        ok(name);
    }
}
