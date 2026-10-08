// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Production YaRN parameter derivation, host frequency table and
//! native launcher controls.
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
    rope::gpt_oss_rope_bf16(&gpu, KernelHandle(7), ptrs, 2, 64, 8, &p, 4).unwrap();
    assert_eq!(gpu.launches_snapshot().len(), 1);
}

/// 2026-10-07: Torch 2.13.0+cu130 `_compute_yarn_parameters` on NVIDIA GB10 (Transformers
/// v4.55 bodies), FP32 bits. The GB10 device kernel produced the same bits.
const TORCH_CUDA_BITS: [u32; 32] = [
    0x3f800000, 0x3f306535, 0x3ef316a2, 0x3ea77faa, 0x3e66d3fa, 0x3e1f0cfd, 0x3ddb2f9f, 0x3d970764,
    0x3d502194, 0x3d01ddd5, 0x3c9e646f, 0x3c3dec94, 0x3bdea840, 0x3b7cfe14, 0x3b093803, 0x3a89f778,
    0x39ef543c, 0x390799b9, 0x3820adc6, 0x37dd6dfc, 0x37989328, 0x375242fa, 0x3710e12c, 0x36c7a82b,
    0x3689928c, 0x363d9647, 0x3602a245, 0x35b40668, 0x35781728, 0x352af200, 0x34eb93eb, 0x34a252d3,
];
/// 2026-10-07: Entries where the correctly rounded table differs from Torch CUDA by one ulp
/// (CUDA `powf` rounding). Derived from a 200-bit evaluation of the same FP32 op order.
const CORRECTLY_ROUNDED_OVERRIDES: [(usize, u32); 2] = [(6, 0x3ddb2f9e), (10, 0x3c9e6470)];

fn expected_bits() -> [u32; 32] {
    let mut bits = TORCH_CUDA_BITS;
    for (i, b) in CORRECTLY_ROUNDED_OVERRIDES {
        bits[i] = b;
    }
    bits
}

fn table_bits(p: &rope::GptOssYarn) -> [u32; 32] {
    rope::gpt_oss_yarn_frequency_table(p)
        .unwrap()
        .map(f32::to_bits)
}

/// 2026-10-07: Written apart from production: the power is exp(ln(base) * i/32) in f64, and
/// every FP32 step is an f64 operation on FP32 operands followed by one rounding to FP32,
/// which equals the FP32 IEEE result for add/sub/mul/div (53 >= 2*24 + 2 bits).
fn independent_bits(base: f32, factor: f32, low: f32, span: f32) -> [u32; 32] {
    let r = |x: f64| f64::from(x as f32);
    let (base, factor, low, span) = (
        f64::from(base),
        f64::from(factor),
        f64::from(low),
        f64::from(span),
    );
    std::array::from_fn(|i| {
        let i = i as f64;
        let pos = r((base.ln() * (i / 32.0)).exp());
        let extrapolation = r(1.0 / pos);
        let interpolation = r(1.0 / r(factor * pos));
        let linear = r(r(i - low) / span);
        let ramp = linear.clamp(0.0, 1.0);
        let extra = r(1.0 - ramp);
        let mixed = r(r(interpolation * r(1.0 - extra)) + r(extrapolation * extra));
        (mixed as f32).to_bits()
    })
}

#[test]
fn host_table_equals_independent_reimplementation() {
    let c = metrale_config::parse_config(CONFIG).unwrap();
    let p = rope::GptOssYarn::from_config(&c).unwrap();
    // 2026-10-07: Parameters the Torch reference run recorded (FP32 bits of low/span).
    assert_eq!((p.base, p.factor), (150000.0, 32.0));
    assert_eq!(
        (p.low.to_bits(), p.span.to_bits()),
        (0x41017c06, 0x4114e249)
    );
    assert_eq!(
        table_bits(&p),
        independent_bits(p.base, p.factor, p.low, p.span)
    );
}

#[test]
fn host_table_known_answers() {
    let c = metrale_config::parse_config(CONFIG).unwrap();
    let p = rope::GptOssYarn::from_config(&c).unwrap();
    let table = rope::gpt_oss_yarn_frequency_table(&p).unwrap();
    let bits = table.map(f32::to_bits);
    // 2026-10-07: Index 0: base^0 = 1 and ramp 0, so the mix is exactly 1.
    assert_eq!(table[0], 1.0);
    // 2026-10-07: Index 16: ramp (16 - low)/span = 0.8492..., so the result lies strictly
    // between the interpolated 1/(32*sqrt(150000)) and extrapolated 1/sqrt(150000).
    let root = 150000f64.sqrt();
    assert!(f64::from(table[16]) > 1.0 / (32.0 * root) && f64::from(table[16]) < 1.0 / root);
    // 2026-10-07: Indices 18..32 have ramp 1, so the entry is 1/(32*pos) = (1/pos)/32 exactly.
    for (i, f) in table.iter().enumerate().skip(18) {
        let pos = 150000f64.powf(i as f64 / 32.0) as f32;
        assert_eq!(*f, (1.0 / pos) / 32.0, "index {i}");
    }
    // 2026-10-07: Pinned bits from the 200-bit evaluation, e.g. index 8 (ramp 0) is the
    // correctly rounded 1/150000^(1/4), index 31 is 1/(32*150000^(31/32)).
    for i in [6, 8, 10, 12, 17, 31] {
        assert_eq!(bits[i], expected_bits()[i], "index {i}");
    }
    assert_eq!(bits, expected_bits());
    // 2026-10-07: Device independence is the point: only the two CUDA powf entries move.
    let moved: Vec<usize> = (0..32).filter(|&i| bits[i] != TORCH_CUDA_BITS[i]).collect();
    assert_eq!(moved, [6, 10]);
}

// 2026-10-07: Known-bad controls: a wrong beta, a wrong factor, or one f64 evaluation of
// the whole expression (skipping the FP32 op order) must each change the table.
#[test]
fn host_table_detects_wrong_beta_factor_and_op_order() {
    let mut c = metrale_config::parse_config(CONFIG).unwrap();
    let good = rope::GptOssYarn::from_config(&c).unwrap();
    let want = expected_bits();
    c.yarn_beta_fast = 16.0;
    let wrong_beta = rope::GptOssYarn::from_config(&c).unwrap();
    assert_ne!(wrong_beta.low, good.low);
    assert_ne!(table_bits(&wrong_beta), want);
    assert_ne!(
        independent_bits(
            wrong_beta.base,
            wrong_beta.factor,
            wrong_beta.low,
            wrong_beta.span
        ),
        want
    );
    let wrong_factor = rope::GptOssYarn {
        factor: 16.0,
        ..good
    };
    assert_ne!(table_bits(&wrong_factor), want);
    let whole_f64: [u32; 32] = std::array::from_fn(|i| {
        let pos = f64::from(good.base).powf(i as f64 / 32.0);
        let ramp = ((i as f64 - f64::from(good.low)) / f64::from(good.span)).clamp(0.0, 1.0);
        let extra = 1.0 - ramp;
        ((1.0 / (f64::from(good.factor) * pos)) * (1.0 - extra) + (1.0 / pos) * extra) as f32
    })
    .map(f32::to_bits);
    assert_ne!(whole_f64, want);
    let bad = rope::GptOssYarn { span: 0.0, ..good };
    assert!(rope::gpt_oss_yarn_frequency_table(&bad).is_err());
}
