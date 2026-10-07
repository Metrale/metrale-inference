// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Production YaRN parameter derivation and native launcher controls.
#[path = "../src/layers/ops/gpt_oss_rope.rs"]
mod rope;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
const CONFIG: &str =
    include_str!("../../circuit/tests/fixtures/checkpoints/openai--gpt-oss-20b/config.json");

#[test]
fn pinned_parameters_and_missing_policy_control() {
    let mut c = metrale_config::parse_config(CONFIG).unwrap();
    let p = rope::GptOssYarn::from_config(&c).unwrap();
    assert_eq!(p.head_dim, 64);
    assert_eq!(p.factor, 32.0);
    assert!(p.low.fract() != 0.0);
    assert!(p.span.fract() != 0.0);
    println!(
        "YARN_PARAMETERS={}",
        serde_json::json!({"head_dim":p.head_dim,"base":p.base,"factor":p.factor,"low":p.low,"span":p.span,"attention_factor":p.attention_factor})
    );
    c.gpt_oss = None;
    assert!(rope::GptOssYarn::from_config(&c).is_err());
}
#[test]
fn invalid_parameters_never_take_a_default() {
    for case in 0..6 {
        let mut c = metrale_config::parse_config(CONFIG).unwrap();
        match case {
            0 => c.rope_theta = f64::NAN,
            1 => c.yarn_beta_fast = 0.0,
            2 => c.yarn_original_max_position_embeddings = 0,
            3 => c.head_dim = 128,
            4 => c.yarn_factor = 0.0,
            _ => c.yarn_attention_factor = f32::INFINITY,
        };
        assert!(rope::GptOssYarn::from_config(&c).is_err());
    }
}
#[test]
fn geometry_and_buffer_refusals_precede_launch() {
    let gpu = MockGpuBackend::new();
    let c = metrale_config::parse_config(CONFIG).unwrap();
    let p = rope::GptOssYarn::from_config(&c).unwrap();
    let ptrs = [
        DevicePtr(0x1000),
        DevicePtr(0x2000),
        DevicePtr(0x3000),
        DevicePtr(0x4000),
    ];
    assert!(rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), ptrs, 0, 64, 8, &p, 0).is_err());
    let mut bad = ptrs;
    bad[3] = DevicePtr::NULL;
    assert!(rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), bad, 2, 64, 8, &p, 0).is_err());
    let mut overflow = ptrs;
    overflow[0] = DevicePtr(u64::MAX - 1);
    assert!(rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), overflow, 2, 64, 8, &p, 0).is_err());
    assert!(rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), ptrs, u32::MAX, 64, 8, &p, 0).is_err());
    assert!(gpu.launches_snapshot().is_empty());
    rope::gpt_oss_yarn_frequencies(&gpu, KernelHandle(8), ptrs[3], &p, 4).unwrap();
    rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), ptrs, 2, 64, 8, &p, 4).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 2);
}
