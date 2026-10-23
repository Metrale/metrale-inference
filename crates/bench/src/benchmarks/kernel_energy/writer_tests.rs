// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The writer derives the floors from the roofline, appends a row that parses, and
//! refuses a duplicate key or a roofline without the class it was asked for.

use std::collections::BTreeMap;

use metrale_abstract_io::power::{Rail, SourceKind};
use metrale_power::row::{Ci, EnergyRow, Label};

use super::*;

const FAMILIES: &str = "schema = 1\nhardware = \"gb10\"\n[roofline]\ndram_gbps = 249.0\nbf16_tflops = 123.7\n\
fp8_tflops = 243.6\nnvfp4_tflops = 490.8\ncontext_tokens = 4096\ndram_pj_per_byte = 23.2\n\
cuda_core_fp32_pj_per_flop = 1.7\nstatic_w = 12.5\n";

pub(super) fn energy() -> EnergyRow {
    let by = BTreeMap::from([("nvml_gpu_rail_counter".to_string(), 0.0016)]);
    EnergyRow {
        energy_j: 0.0016,
        energy_dyn_j: 0.0005,
        avg_w: 17.0,
        static_w: 12.5,
        energy_ci_j: Ci {
            lo: 0.0015,
            hi: 0.0017,
        },
        energy_dyn_ci_j: Ci {
            lo: 0.0004,
            hi: 0.0006,
        },
        avg_w_ci: Ci { lo: 16.5, hi: 17.5 },
        energy_ci_level: 0.95,
        energy_ci_method: "envelope".to_string(),
        energy_method: "steady_state",
        energy_samples: 3,
        energy_boxes: 1,
        energy_launches: 3000,
        energy_window_ms: 3000.0 * 0.0016 / 17.0 * 1000.0,
        energy_ess: 40.0,
        energy_outliers: 0,
        energy_seed: 1,
        energy_order: "abba:I L0 I".to_string(),
        energy_label: Label::Measured,
        sm_mhz: Some(2400.0),
        temp_c: Some(50.0),
        throttle_pct: Some(0.0),
        power_source: SourceKind::GpuDriverCounter,
        power_rail: Rail::GpuCore,
        energy_by_source: by.clone(),
        energy_dyn_by_source: BTreeMap::from([("nvml_gpu_rail_counter".to_string(), 0.0005)]),
        sensor_delta_pct: BTreeMap::new(),
        energy_flags: Vec::new(),
        energy_agreement: BTreeMap::new(),
        energy_fingerprint: "fp".to_string(),
        energy_source: "src".to_string(),
        trace_sha256: "0".repeat(64),
        energy_floor_class: None,
        energy_floor_j: None,
        pct_of_energy_floor: None,
    }
}

fn base() -> BaseRow {
    BaseRow {
        kernel: "glm-5.3-flash::w4a16_gemv_pipe_m1".to_string(),
        file: "kernels/gb10/common/w4a16_gemv_pipe.cu".to_string(),
        hardware: "gb10".to_string(),
        model: "nvidia/GLM-5.3-Flash-NVFP4".to_string(),
        regime: "microbench steady-state R=1 · KDA in-proj".to_string(),
        time_us: 85.0,
        bytes: 20_138_536,
        flops: 71_483_392,
        source: "s".to_string(),
        notes: "n".to_string(),
    }
}

#[test]
fn the_floors_come_from_the_roofline_and_the_row_parses() {
    let roof = roofline(FAMILIES, "bf16", "cuda_core_fp32").unwrap();
    let block = render(&base(), &roof, &energy(), Some("cuda_core_fp32")).unwrap();
    let t: toml::Table = toml::from_str(&block).unwrap();
    let m = &t["m"].as_array().unwrap()[0];
    let floor_us = 20_138_536.0 / 249.0e3;
    assert!((m["floor_us"].as_float().unwrap() - floor_us).abs() < 1e-3);
    assert_eq!(m["bound"].as_str(), Some("memory"));
    let ej = (20_138_536.0 * 23.2 + 71_483_392.0 * 1.7) * 1e-12;
    assert_eq!(m["energy_floor_j"].as_float(), Some(ej));
    assert!((m["pct_of_energy_floor"].as_float().unwrap() - ej / 0.0005 * 100.0).abs() < 0.01);
    assert_eq!(m["energy_j"].as_float(), Some(0.0016));
}

#[test]
fn a_duplicate_key_is_refused_and_a_new_one_appends() {
    let roof = roofline(FAMILIES, "bf16", "dram_only").unwrap();
    let block = render(&base(), &roof, &energy(), Some("dram_only")).unwrap();
    let once = append("# header\n", &block).unwrap();
    let e = append(&once, &block).unwrap_err();
    assert!(e.to_string().contains("already has a row"), "{e}");
    let mut other = base();
    other.regime = "microbench steady-state R=2 · KDA in-proj".to_string();
    let twice = append(&once, &render(&other, &roof, &energy(), None).unwrap()).unwrap();
    let t: toml::Table = toml::from_str(&twice).unwrap();
    assert_eq!(t["m"].as_array().unwrap().len(), 2);
}

#[test]
fn a_roofline_without_the_asked_class_is_refused() {
    assert!(roofline(FAMILIES, "bf16", "nvfp4_mma").is_err());
    assert!(roofline(FAMILIES, "fp6", "dram_only").is_err());
}
