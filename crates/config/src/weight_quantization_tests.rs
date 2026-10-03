// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The weight-quantization policy against the checkpoints' own
//! `quantization_config` blocks (`precision_plan/fixtures`), one per declaration dialect,
//! plus a checkpoint that declares nothing.

use super::*;
use crate::precision_plan::DeclaredPrecisionPlan;

fn plan(name: &str) -> DeclaredPrecisionPlan {
    let text = match name {
        "unsloth" => include_str!("precision_plan/fixtures/unsloth_qwen3_8_27b_nvfp4.json"),
        "nvidia27" => include_str!("precision_plan/fixtures/nvidia_qwen3_6_27b_nvfp4.json"),
        "fp8_moe" => include_str!("precision_plan/fixtures/qwen3_6_35b_a3b_fp8.json"),
        "fp8_dense" => include_str!("precision_plan/fixtures/qwen3_6_27b_fp8.json"),
        "nvidia35" => include_str!("precision_plan/fixtures/nvidia_qwen3_6_35b_a3b_nvfp4.json"),
        "modelopt_nvfp4" => {
            include_str!("precision_plan/fixtures/nvidia_qwen3_next_80b_nvfp4.json")
        }
        other => panic!("no fixture {other}"),
    };
    let raw: serde_json::Value = serde_json::from_str(text).expect("fixture parses");
    DeclaredPrecisionPlan::from_quantization_config(&raw).expect("plan")
}

fn declared() -> WeightQuantTier {
    WeightQuantTier::new(WeightQuantization::Declared, W4a4Downcast::Off).expect("tier")
}

fn nvfp4(downcast: W4a4Downcast) -> WeightQuantTier {
    WeightQuantTier::new(WeightQuantization::Nvfp4, downcast).expect("tier")
}

const L: &str = "model.language_model.layers";
const CAPS: KernelCaps = KernelCaps {
    w8a8_decode: false,
    w8a8_moe_decode: false,
    w8a8_block_scaled_decode: false,
    fp8_lm_head_batched: false,
};

/// 2026-09-28: The tiers' names are the flag values, `declared` is the default, and the
/// W4A4 lever is refused under `declared` whatever its width.
#[test]
fn tiers_name_default_and_refuse_the_lever_under_declared() {
    let names: Vec<_> = WeightQuantization::ALL.iter().map(|t| t.name()).collect();
    assert_eq!(names, ["declared", "nvfp4"]);
    assert_eq!(WeightQuantTier::default(), declared());
    for d in [W4a4Downcast::Narrow, W4a4Downcast::Wide] {
        assert!(WeightQuantTier::new(WeightQuantization::Declared, d).is_err());
        assert_eq!(nvfp4(d).downcast(), d);
    }
    assert_eq!(W4a4Downcast::from_flags(false, true), W4a4Downcast::Off);
    assert_eq!(W4a4Downcast::from_flags(true, false), W4a4Downcast::Narrow);
    assert_eq!(W4a4Downcast::from_flags(true, true), W4a4Downcast::Wide);
}

/// 2026-09-28: unsloth/Qwen3.8-27B-NVFP4 (compressed-tensors, mixed). Under `declared`, MLP
/// 0-55 is stamped A4, and every FP8-declared projection's NVFP4 copy (MLP 56-63, attention,
/// GDN) is stamped `Wide`, so no FP4 activations run below the declared FP8. The FP8 weights
/// are asked for, and the head resolves to the checkpoint's FP8 once the batched FP8 head
/// kernel is present.
#[test]
fn unsloth_mixed_checkpoint_under_declared() {
    let p = plan("unsloth");
    let pol = WeightQuantPolicy::new(declared(), &p, CAPS);
    assert!(pol.follows_plan());
    for layer in [0, 55] {
        let m = format!("{L}.{layer}.mlp.down_proj");
        assert_eq!(pol.nvfp4_act(&m), Nvfp4Act::A4, "{m}");
        assert!(!pol.wants_fp8_weights(&m), "{m}");
    }
    for m in [
        format!("{L}.56.mlp.gate_proj"),
        format!("{L}.63.mlp.down_proj"),
        format!("{L}.3.self_attn.o_proj"),
        format!("{L}.0.linear_attn.in_proj_qkv"),
        format!("{L}.0.linear_attn.out_proj"),
    ] {
        assert_eq!(pol.nvfp4_act(&m), Nvfp4Act::Wide, "{m}");
        assert!(pol.wants_fp8_weights(&m), "{m}");
    }
    // 2026-09-28: Path A: without the batched FP8 head the engine default head stays.
    assert_eq!(pol.lm_head(), LmHeadChoice::PendingFp8Kernel);
    // 2026-09-28: Path B: with it, the declared FP8 head.
    let head_caps = KernelCaps {
        fp8_lm_head_batched: true,
        ..CAPS
    };
    assert_eq!(
        WeightQuantPolicy::new(declared(), &p, head_caps).lm_head(),
        LmHeadChoice::Declared(LmHeadFormat::Fp8)
    );
    let attn = format!("{L}.3.self_attn.q_proj");
    assert_eq!(pol.fp8_decode_act(&attn), Some(ActFormat::Bf16));
    assert_eq!(pol.fp8_decode_act(&format!("{L}.0.mlp.up_proj")), None);
    let dense_caps = KernelCaps {
        w8a8_decode: true,
        ..CAPS
    };
    let with_w8a8 = WeightQuantPolicy::new(declared(), &p, dense_caps);
    assert_eq!(with_w8a8.fp8_decode_act(&attn), Some(ActFormat::Fp8));
    assert_eq!(with_w8a8.fp8_decode_act("lm_head"), Some(ActFormat::Fp8));
}

/// 2026-09-28: The same checkpoint under `nvfp4`: nothing is stamped, no FP8 weight is asked
/// for and the head keeps the engine default, whatever the lever, so the load and dispatch
/// are those from before the plan existed.
#[test]
fn nvfp4_tier_ignores_the_plan() {
    let p = plan("unsloth");
    for d in [W4a4Downcast::Off, W4a4Downcast::Narrow, W4a4Downcast::Wide] {
        let pol = WeightQuantPolicy::new(nvfp4(d), &p, CAPS);
        assert!(!pol.follows_plan());
        for m in [
            format!("{L}.0.mlp.gate_proj"),
            format!("{L}.60.mlp.gate_proj"),
            format!("{L}.3.self_attn.q_proj"),
        ] {
            assert_eq!(pol.nvfp4_act(&m), Nvfp4Act::Unstamped, "{m}");
            assert!(!pol.wants_fp8_weights(&m), "{m}");
            assert_eq!(pol.fp8_decode_act(&m), None, "{m}");
        }
        // 2026-09-28: Path C: the `nvfp4` tier keeps the engine default head, with or
        // without the batched FP8 head.
        assert_eq!(pol.lm_head(), LmHeadChoice::EngineDefault);
        let head_caps = KernelCaps {
            fp8_lm_head_batched: true,
            ..CAPS
        };
        assert_eq!(
            WeightQuantPolicy::new(nvfp4(d), &p, head_caps).lm_head(),
            LmHeadChoice::EngineDefault
        );
    }
}

/// 2026-09-28: nvidia/Qwen3.6-27B-NVFP4 (ModelOpt MIXED_PRECISION): the MLP declares W4A16,
/// so it is stamped `Wide` (no W4A4 decode and no FP4 MMQ prefill); GDN and attention are
/// per-tensor FP8; the head is declared NVFP4.
#[test]
fn nvidia_weight_only_mlp_under_declared() {
    let p = plan("nvidia27");
    let pol = WeightQuantPolicy::new(declared(), &p, CAPS);
    assert_eq!(
        pol.nvfp4_act(&format!("{L}.0.mlp.gate_proj")),
        Nvfp4Act::Wide
    );
    assert!(!Nvfp4Act::Wide.allows_fp4_prefill());
    assert!(pol.wants_fp8_weights(&format!("{L}.0.linear_attn.in_proj_qkv")));
    assert_eq!(pol.lm_head(), LmHeadChoice::Declared(LmHeadFormat::Nvfp4));
}

/// 2026-09-28: A ModelOpt NVFP4 checkpoint (nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4, every
/// `Linear` W4A4): a quantized projection is stamped A4; an ignored one declares 16-bit
/// activations, so an NVFP4 copy the engine makes of it is stamped `Wide`; the ignored head
/// resolves to BF16.
#[test]
fn modelopt_nvfp4_is_a4_where_quantized() {
    let p = plan("modelopt_nvfp4");
    let pol = WeightQuantPolicy::new(declared(), &p, CAPS);
    assert_eq!(
        pol.nvfp4_act("model.layers.3.mlp.experts.7.gate_proj"),
        Nvfp4Act::A4
    );
    assert_eq!(
        pol.nvfp4_act("model.layers.0.linear_attn.in_proj_qkvz"),
        Nvfp4Act::Wide
    );
    assert_eq!(pol.lm_head(), LmHeadChoice::Declared(LmHeadFormat::Bf16));
}

/// 2026-09-28: Qwen/Qwen3.6-35B-A3B-FP8 (HF fp8, 128x128 blocks): every projection asks for
/// its FP8 weights (the MoE loader already serves them so), and the head, which
/// `modules_to_not_convert` leaves BF16, resolves to BF16.
#[test]
fn fp8_block_moe_under_declared() {
    let p = plan("fp8_moe");
    let pol = WeightQuantPolicy::new(declared(), &p, CAPS);
    for m in [
        format!("{L}.0.mlp.experts.3.down_proj"),
        format!("{L}.3.self_attn.q_proj"),
        format!("{L}.0.linear_attn.out_proj"),
    ] {
        assert!(pol.wants_fp8_weights(&m), "{m}");
        assert_eq!(pol.nvfp4_act(&m), Nvfp4Act::Wide, "{m}");
    }
    assert_eq!(pol.lm_head(), LmHeadChoice::Declared(LmHeadFormat::Bf16));
    // 2026-09-28: Each W8A8 family's bit serves its own modules only.
    let experts = format!("{L}.0.mlp.experts.3.down_proj");
    let shared = format!("{L}.0.mlp.shared_expert.up_proj");
    let attn = format!("{L}.3.self_attn.q_proj");
    for (caps, want_expert, want_attn) in [
        (
            KernelCaps {
                w8a8_moe_decode: true,
                ..CAPS
            },
            ActFormat::Fp8,
            ActFormat::Bf16,
        ),
        (
            KernelCaps {
                w8a8_decode: true,
                ..CAPS
            },
            ActFormat::Bf16,
            ActFormat::Fp8,
        ),
    ] {
        let pol = WeightQuantPolicy::new(declared(), &p, caps);
        assert_eq!(pol.fp8_decode_act(&experts), Some(want_expert), "{caps:?}");
        assert_eq!(pol.fp8_decode_act(&shared), Some(want_expert), "{caps:?}");
        assert_eq!(pol.fp8_decode_act(&attn), Some(want_attn), "{caps:?}");
    }
}

/// 2026-09-28: The block-scaled and MoE W8A8 caps on Qwen/Qwen3.6-35B-A3B-FP8 under `declared`.
/// Path A: with the dense bit on and both others off (the caps shipped until 2026-09-29), the
/// experts and the block-scaled attention/GDN decode BF16 (W8A16) while still declaring FP8
/// activations. Path B: with all three on (shipped since), FP8. Path C: the dense per-channel
/// checkpoint answers FP8 with the block-scaled bit off, so it is unaffected; the `nvfp4` tier
/// asks for nothing.
#[test]
fn block_scaled_and_moe_w8a8_wait_for_their_own_caps() {
    let p = plan("fp8_moe");
    let experts = format!("{L}.0.mlp.experts.0.gate_proj");
    let attn = format!("{L}.3.self_attn.q_proj");
    let gdn = format!("{L}.0.linear_attn.in_proj_qkv");
    let dense_only = KernelCaps {
        w8a8_decode: true,
        ..CAPS
    };
    let held = WeightQuantPolicy::new(declared(), &p, dense_only);
    for m in [&experts, &attn, &gdn] {
        assert_eq!(
            held.fp8_block_scaled_decode_act(m),
            Some(ActFormat::Bf16),
            "{m}"
        );
        assert!(held.declares_fp8_activations(m), "{m}");
    }
    let all = KernelCaps {
        w8a8_decode: true,
        w8a8_moe_decode: true,
        w8a8_block_scaled_decode: true,
        ..CAPS
    };
    let open = WeightQuantPolicy::new(declared(), &p, all);
    for m in [&experts, &attn, &gdn] {
        assert_eq!(
            open.fp8_block_scaled_decode_act(m),
            Some(ActFormat::Fp8),
            "{m}"
        );
    }
    let dense = plan("unsloth");
    let d = WeightQuantPolicy::new(declared(), &dense, dense_only);
    assert_eq!(
        d.fp8_decode_act(&format!("{L}.3.self_attn.q_proj")),
        Some(ActFormat::Fp8)
    );
    assert!(!d.declares_fp8_activations(&format!("{L}.0.mlp.down_proj")));
    let nv = WeightQuantPolicy::new(nvfp4(W4a4Downcast::Off), &p, all);
    assert_eq!(nv.fp8_block_scaled_decode_act(&attn), None);
    assert!(!nv.declares_fp8_activations(&attn));
}

/// 2026-10-03: The held block-scaled W8A8 lifts only for a MoE checkpoint without FP8 experts
/// (the NVFP4 35B). The FP8 35B with its experts at BF16 activations (`moe:bf16`) and the dense
/// FP8 27B keep W8A16 attention/GDN, as main serves them: lifting there changed greedy text
/// (FP8 35B, main vs branch, 16 of 19 essays).
#[test]
fn block_scaled_w8a8_lifts_only_without_fp8_experts() {
    let experts = |n: usize| -> Vec<String> {
        [0, 3]
            .iter()
            .take(n)
            .map(|i| format!("{L}.{i}.mlp.experts.0.gate_proj"))
            .collect()
    };
    let shipped = KernelCaps {
        w8a8_decode: true,
        w8a8_moe_decode: true,
        ..CAPS
    };
    let nv = plan("nvidia35");
    let nvp = WeightQuantPolicy::new(declared(), &nv, shipped);
    assert!(nvp.lifts_block_scaled_w8a8(&experts(2), false));
    assert!(!nvp.lifts_block_scaled_w8a8(&experts(2), true));
    assert!(!nvp.lifts_block_scaled_w8a8(&[], false));
    let tier4 = WeightQuantPolicy::new(nvfp4(W4a4Downcast::Off), &nv, shipped);
    assert!(!tier4.lifts_block_scaled_w8a8(&experts(2), false));
    let fp8 = plan("fp8_moe");
    for caps in [shipped, CAPS] {
        let p = WeightQuantPolicy::new(declared(), &fp8, caps);
        assert!(!p.lifts_block_scaled_w8a8(&experts(2), false), "{caps:?}");
        assert!(!p.lifts_block_scaled_w8a8(&experts(2), true), "{caps:?}");
    }
    let dense = plan("fp8_dense");
    let d = WeightQuantPolicy::new(declared(), &dense, shipped);
    assert!(d.declares_fp8_activations(&format!("{L}.3.self_attn.q_proj")));
    assert!(!d.lifts_block_scaled_w8a8(&[], false));
}

/// 2026-09-28: A checkpoint without `quantization_config` declares nothing, so both tiers
/// answer as before the plan existed.
#[test]
fn unquantized_checkpoint_keeps_the_engine_defaults() {
    let p = DeclaredPrecisionPlan::default();
    for tier in [declared(), nvfp4(W4a4Downcast::Off)] {
        let pol = WeightQuantPolicy::new(tier, &p, CAPS);
        assert!(!pol.follows_plan());
        assert_eq!(
            pol.nvfp4_act(&format!("{L}.0.mlp.gate_proj")),
            Nvfp4Act::Unstamped
        );
        assert!(!pol.wants_fp8_weights(&format!("{L}.0.mlp.gate_proj")));
        assert_eq!(pol.lm_head(), LmHeadChoice::EngineDefault);
    }
}

/// 2026-09-28: The decode W4A4 row edge. `nvfp4` without the lever: no weight; with it:
/// every weight, 32 or 64 rows, whatever its stamp and the kernels. `declared`: only an A4
/// weight, up to what the kernels serve.
#[test]
fn w4a4_rows_per_tier() {
    use Nvfp4Act::*;
    let rows = |t: WeightQuantTier, a| t.w4a4_rows(a, 32, 64, 48);
    for a in [Unstamped, A4, Wide] {
        assert_eq!(rows(nvfp4(W4a4Downcast::Off), a), 0);
        assert_eq!(rows(nvfp4(W4a4Downcast::Narrow), a), 32);
        assert_eq!(rows(nvfp4(W4a4Downcast::Wide), a), 64);
    }
    assert_eq!(rows(declared(), A4), 48);
    assert_eq!(rows(declared(), Wide), 0);
    assert_eq!(rows(declared(), Unstamped), 0);
    assert_eq!(declared().w4a4_rows(A4, 32, 64, 0), 0, "kernels absent");
    assert!(!nvfp4(W4a4Downcast::Off).uses_w4a4_decode());
    assert!(nvfp4(W4a4Downcast::Narrow).uses_w4a4_decode());
    assert!(declared().uses_w4a4_decode());
}

/// 2026-09-28: The dense FFN's 1- to 3-row steps. The lever reached 2 and 3 rows only; the
/// declared single-row W4A4 needs an A4 FFN.
#[test]
fn ffn_narrow_steps_per_tier() {
    for any_a4 in [false, true] {
        assert!(!nvfp4(W4a4Downcast::Off).ffn_small_batch_w4a4(any_a4));
        assert!(nvfp4(W4a4Downcast::Narrow).ffn_small_batch_w4a4(any_a4));
        for d in [W4a4Downcast::Off, W4a4Downcast::Narrow, W4a4Downcast::Wide] {
            assert!(!nvfp4(d).ffn_single_row_w4a4(any_a4));
        }
        assert_eq!(declared().ffn_small_batch_w4a4(any_a4), any_a4);
        assert_eq!(declared().ffn_single_row_w4a4(any_a4), any_a4);
    }
}

/// 2026-09-28: A fused weight is A4 only when every part is, and `Wide` when any part is.
#[test]
fn combined_stamps() {
    use Nvfp4Act::*;
    assert_eq!(Nvfp4Act::combine([A4, A4]), A4);
    assert_eq!(Nvfp4Act::combine([A4, Wide]), Wide);
    assert_eq!(Nvfp4Act::combine([Unstamped, Wide]), Wide);
    assert_eq!(Nvfp4Act::combine([A4, Unstamped]), Unstamped);
    assert_eq!(Nvfp4Act::combine([]), Unstamped);
    assert!(A4.allows_fp4_prefill() && Unstamped.allows_fp4_prefill());
}
