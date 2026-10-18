// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of the store selectors with every tier lever at its default.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: none beyond the types.

use metrale_model_layers::lever_equivalence::defaults;

use super::*;

fn fp() -> ModelFingerprint {
    let cfg = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    ModelFingerprint::derive_with_id(&cfg, 4, "").unwrap()
}

fn levers() -> SsmTierLevers {
    SsmTierLevers::from_settings(&defaults())
}

#[test]
fn decode_tier_defaults_to_host_ram_non_dropping() {
    let s = build_decode_tier_store(fp(), 4, 8, &levers()).unwrap();
    for k in 0..2000u64 {
        assert!(s.put(k, &[0; 4]).unwrap(), "non-dropping: nothing refused");
    }
    assert_eq!(s.len(), 2000);
}

#[test]
fn disk_cap_defaults_to_unbounded() {
    let levers = levers();
    assert_eq!(
        disk_slots_from(levers.disk_gb.as_deref(), 4).unwrap(),
        0,
        "unset budget must resolve to the unbounded sentinel"
    );
    let s = build_tier_store(fp(), 4, &levers).unwrap();
    assert!(s.put(u64::MAX, &[1, 2, 3, 4]).unwrap());
    for k in 0..1000u64 {
        assert!(s.put(k, &[0; 4]).unwrap(), "unbounded: nothing dropped");
    }
    assert_eq!(s.len(), 1001);
    let mut out = [0u8; 4];
    assert!(s.get(u64::MAX, &mut out).unwrap());
    assert_eq!(out, [1, 2, 3, 4], "default store preserves payloads");
}

/// 2026-10-17: A decode tier value other than `nvme`, `peer` or unset is refused, `""`
/// included, and `nvme` / `peer` without their companion lever are refused too.
#[test]
fn decode_tier_refuses_unknown_and_incomplete_selections() {
    for env in [
        &[("METRALE_SSM_DECODE_TIER", "")][..],
        &[("METRALE_SSM_DECODE_TIER", "NVME")],
        &[("METRALE_SSM_DECODE_TIER", "nvme")],
        &[
            ("METRALE_SSM_DECODE_TIER", "nvme"),
            ("METRALE_SSM_DECODE_NVME_DIR", ""),
        ],
        &[("METRALE_SSM_DECODE_TIER", "peer")],
        &[
            ("METRALE_SSM_DECODE_TIER", "peer"),
            ("METRALE_SSM_DECODE_RDMA_TIER", ""),
        ],
    ] {
        let s = metrale_model_layers::lever_equivalence::settings_from(env).unwrap();
        let levers = SsmTierLevers::from_settings(&s);
        assert!(
            build_decode_tier_store(fp(), 4, 8, &levers).is_err(),
            "{env:?}"
        );
    }
}
