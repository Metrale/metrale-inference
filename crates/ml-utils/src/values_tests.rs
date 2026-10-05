// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Tensor classes by name, residual writers, fan-in, the documented constants, and
//! thread-count independence of generation.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;
use crate::index::Dtype;

fn t(name: &str, shape: Vec<u64>) -> TensorEntry {
    TensorEntry {
        name: name.into(),
        dtype: Dtype::Bf16,
        shape,
        shard: "s".into(),
        offset: 0,
    }
}

#[test]
fn classes_follow_the_table() {
    let c = |n: &str, s: Vec<u64>| classify_plain(&t(n, s), 1.0).unwrap();
    assert_eq!(c("a.input_layernorm.weight", vec![8]), ValueClass::Ones);
    assert_eq!(c("a.linear_attn.norm.weight", vec![8]), ValueClass::Ones);
    assert_eq!(c("v.blocks.0.norm1.bias", vec![8]), ValueClass::Zeros);
    assert_eq!(c("v.attn.proj.bias", vec![8]), ValueClass::Zeros);
    assert_eq!(c("a.linear_attn.A_log", vec![8]), ValueClass::ALog);
    assert_eq!(c("a.linear_attn.dt_bias", vec![8]), ValueClass::DtBias);
    assert_eq!(c("a.self_attn.k_scale", vec![1]), ValueClass::KvScale);
    assert_eq!(
        c("a.linear_attn.conv1d.weight", vec![64, 1, 4]),
        ValueClass::Normal {
            std: 0.5,
            row_len: 4,
            zero_row: None
        }
    );
    assert!(classify_plain(&t("a.mystery", vec![3]), 1.0).is_err());
}

#[test]
fn fused_experts_read_the_last_dim_and_writers_are_named() {
    assert_eq!(
        rows_and_fan_in(&t("m.experts.down_proj", vec![4, 8, 16])).unwrap(),
        (32, 16)
    );
    assert_eq!(
        rows_and_fan_in(&t("p.patch_embed.proj.weight", vec![8, 3, 2, 4, 4])).unwrap(),
        (8, 96)
    );
    assert!(is_residual_writer("l.mlp.experts.3.down_proj"));
    assert!(is_residual_writer("l.linear_attn.out_proj"));
    assert!(!is_residual_writer("l.mlp.up_proj"));
    assert!((residual_gain(100) - 0.01).abs() < 1e-9);
}

#[test]
fn the_written_out_constants_are_the_functions_they_name() {
    for (k, v) in A_LOG.iter().enumerate() {
        assert!(
            (f64::from(*v) - ((k + 1) as f64).ln()).abs() < 1e-6,
            "ln({})",
            k + 1
        );
    }
    for (dt, v) in [0.001f64, 0.002, 0.005, 0.01, 0.02, 0.05, 0.07, 0.1]
        .iter()
        .zip(DT_BIAS)
    {
        assert!((f64::from(v) - dt.exp_m1().ln()).abs() < 1e-5, "dt {dt}");
    }
}

#[test]
fn generation_does_not_depend_on_chunking_and_zero_rows_are_zero() {
    let class = ValueClass::Normal {
        std: 0.1,
        row_len: 1000,
        zero_row: Some(1500),
    };
    let s = Stream::for_tensor(1, "w", "BF16", &[3000, 1000]);
    let n = 3 * CHUNK + 17;
    let all = fill(&class, s, n);
    let mut parts = vec![0.0f32; n];
    fill_range(&class, s, 0, &mut parts[..5]);
    fill_range(&class, s, 5, &mut parts[5..]);
    assert_eq!(all, parts);
    assert!(all[1500 * 1000..1501 * 1000].iter().all(|&v| v == 0.0));
    assert!(all[..1000].iter().any(|&v| v != 0.0));
}
