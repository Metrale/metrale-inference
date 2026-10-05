// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: `--kv-cache-dtype declared` resolves to the checkpoint's declared KV-cache format and
//! has no default: a checkpoint that declares none is refused.
//!
//! Owner: server startup.
//! Invariants: none beyond the types.

use super::{declared_kv_dtype, fp8_kv_calibration_tokens};

fn config_with(kv_cache_format: Option<&str>) -> metrale_config::ModelConfig {
    let mut c = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    c.quantization_config = kv_cache_format.map(|f| {
        let mut q = metrale_config::parse_quantization_config(&serde_json::json!({
            "quantization_config": { "quant_method": "modelopt", "quant_algo": "NVFP4" }
        }))
        .expect("parses")
        .expect("declares");
        q.kv_cache_format = Some(f.to_string());
        q
    });
    c
}

#[test]
fn declared_takes_the_checkpoint_format_and_refuses_none() {
    assert_eq!(
        declared_kv_dtype(&config_with(Some("FP8"))).expect("fp8"),
        "fp8"
    );
    let none = declared_kv_dtype(&config_with(None)).expect_err("no quantization config");
    assert!(
        none.to_string().contains("declares no KV-cache format"),
        "{none}"
    );
    let mut no_kv = config_with(Some("FP8"));
    no_kv.quantization_config.as_mut().unwrap().kv_cache_format = None;
    assert!(declared_kv_dtype(&no_kv).is_err());
    assert!(declared_kv_dtype(&config_with(Some("INT4"))).is_err());
}

#[test]
fn declared_kv_does_not_calibrate_unless_asked() {
    assert_eq!(fp8_kv_calibration_tokens(None, true, 256), 0);
    assert_eq!(fp8_kv_calibration_tokens(None, false, 256), 256);
    assert_eq!(fp8_kv_calibration_tokens(Some(128), true, 256), 128);
    assert_eq!(fp8_kv_calibration_tokens(Some(0), false, 256), 0);
}
