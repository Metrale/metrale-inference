// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The W8A8 block-scaled decode serves only the block its kernels read: the
//! DeepSeek-V4.1 attention, declared FP8 at 32x32, keeps BF16 activations with every cap open.
//! The V4.1 routed experts are MXFP4, so no FP8 answer applies to them. 2026-10-22: Kept here,
//! not in metrale-precision, because it reads the plan through the engine's config parse.
//!
//! Owner: config (quantization).
//! Invariants: none beyond the types.

use crate::precision_plan::Granularity;
use crate::weight_quantization::{ActFormat, KernelCaps};
use crate::{W4a4Downcast, WeightQuantPolicy, WeightQuantTier, WeightQuantization};

#[test]
fn deepseek_v41_32x32_fp8_attention_keeps_bf16_activations() {
    let all = KernelCaps {
        w8a8_decode: true,
        w8a8_moe_decode: true,
        w8a8_block_scaled_decode: true,
        fp8_lm_head_batched: false,
    };
    let declared =
        WeightQuantTier::new(WeightQuantization::Declared, W4a4Downcast::Off).expect("tier");
    let v41 = crate::parse_config(crate::tests::deepseek_v41::CONFIG)
        .unwrap()
        .quantization_config
        .expect("V4.1 declares a quantization_config");
    let p = WeightQuantPolicy::new(declared, &v41.precision, all);
    let attn = "layers.0.attn.wq_a";
    assert_eq!(
        p.declared(attn).weight.map(|w| w.granularity),
        Some(Granularity::Block(32, 32))
    );
    assert_eq!(p.fp8_decode_act(attn), Some(ActFormat::Fp8));
    assert_eq!(p.fp8_block_scaled_decode_act(attn), Some(ActFormat::Bf16));
    assert_eq!(
        p.fp8_block_scaled_decode_act("layers.0.ffn.experts.7.w1"),
        None
    );
}
