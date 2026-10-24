// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `WeightsSettings` properties: the declared settings, the RDMA fallback chains and
//! the projection errors.
//!
//! Owner: settings.
//! Invariants: never reads or writes the process environment.

use metrale_error::{Error, ErrorCode};

use super::{Rail, WeightsSettings};
use crate::groups::equivalence::resolved;

fn weights(pairs: &[(&str, &str)]) -> Result<WeightsSettings, Error> {
    WeightsSettings::try_from(&resolved(pairs).expect("every value here parses"))
}

fn rail(dev: &str, gid: u32) -> Rail {
    Rail {
        dev: dev.into(),
        gid,
    }
}

#[test]
fn declared_settings_load_bf16_through_direct_reads_on_the_built_in_rails() {
    assert_eq!(
        WeightsSettings::declared().expect("declared defaults resolve"),
        WeightsSettings {
            gguf_native_q2: false,
            q2_group: 128,
            gguf_force_cpu: false,
            keep_page_cache: false,
            direct_reads: true,
            kquant_resident: true,
            q6k_head: true,
            evict_random_pct: 8,
            weight_rail0: rail("roceP2p1s0f1", 3),
            weight_rail1: None,
            lora_rail: rail("roceP2p1s0f1", 3),
        }
    );
}

#[test]
fn rail_devices_fall_back_weight_then_expert_then_declared_skipping_empty_names() {
    let on = ("METRALE_WEIGHT_DUAL_RAIL", "1");
    let w = weights(&[
        on,
        ("METRALE_WEIGHT_RDMA_DEV", ""),
        ("METRALE_EXPERT_RDMA_DEV", ""),
        ("METRALE_WEIGHT_RAIL2_DEV", ""),
    ])
    .expect("valid");
    assert_eq!(w.weight_rail0.dev, "roceP2p1s0f1");
    assert_eq!(w.weight_rail1.map(|r| r.dev).as_deref(), Some("rocep1s0f1"));
    let w = weights(&[
        on,
        ("METRALE_WEIGHT_RDMA_DEV", ""),
        ("METRALE_EXPERT_RDMA_DEV", "e0"),
        ("METRALE_EXPERT_RAIL2_DEV", "e1"),
    ])
    .expect("valid");
    assert_eq!(w.weight_rail0.dev, "e0");
    assert_eq!(w.weight_rail1.map(|r| r.dev).as_deref(), Some("e1"));
    let w = weights(&[
        on,
        ("METRALE_WEIGHT_RDMA_DEV", "w0"),
        ("METRALE_EXPERT_RDMA_DEV", "e0"),
        ("METRALE_WEIGHT_RAIL2_DEV", "w1"),
        ("METRALE_EXPERT_RAIL2_DEV", "e1"),
    ])
    .expect("valid");
    assert_eq!(w.weight_rail0.dev, "w0");
    assert_eq!(w.weight_rail1.map(|r| r.dev).as_deref(), Some("w1"));
}

#[test]
fn the_lora_device_takes_the_first_set_name_even_when_empty() {
    let lora = |pairs: &[(&str, &str)]| weights(pairs).expect("valid").lora_rail.dev;
    assert_eq!(lora(&[("METRALE_EXPERT_RDMA_DEV", "e0")]), "e0");
    assert_eq!(
        lora(&[
            ("METRALE_WEIGHT_RDMA_DEV", ""),
            ("METRALE_EXPERT_RDMA_DEV", "e0")
        ]),
        ""
    );
    assert_eq!(
        lora(&[
            ("METRALE_LORA_RDMA_DEV", "l0"),
            ("METRALE_WEIGHT_RDMA_DEV", "w0")
        ]),
        "l0"
    );
}

#[test]
fn gids_take_the_weight_lever_over_the_expert_one_and_lora_reads_only_its_own() {
    let w = weights(&[
        ("METRALE_WEIGHT_DUAL_RAIL", "1"),
        ("METRALE_EXPERT_RDMA_GID", "5"),
        ("METRALE_EXPERT_RAIL2_GID", "6"),
        ("METRALE_WEIGHT_RAIL2_GID", "7"),
    ])
    .expect("valid");
    assert_eq!(w.weight_rail0.gid, 5);
    assert_eq!(w.weight_rail1.map(|r| r.gid), Some(7));
    assert_eq!(w.lora_rail.gid, 3);
    let w = weights(&[
        ("METRALE_WEIGHT_RDMA_GID", "4"),
        ("METRALE_EXPERT_RDMA_GID", "5"),
        ("METRALE_LORA_RDMA_GID", "9"),
    ])
    .expect("valid");
    assert_eq!((w.weight_rail0.gid, w.lora_rail.gid), (4, 9));
}

#[test]
fn a_value_the_loader_cannot_use_is_projection_out_of_range_naming_the_lever() {
    for (env, raw) in [
        ("METRALE_GGUF_Q2_GROUP", "32"),
        ("METRALE_GGUF_Q2_GROUP", "0"),
        ("METRALE_DS41_EVICT_RANDOM_PCT", "101"),
        ("METRALE_DS41_EVICT_RANDOM_PCT", "-1"),
        ("METRALE_WEIGHT_RDMA_GID", "-1"),
        ("METRALE_EXPERT_RDMA_GID", "4294967296"),
        ("METRALE_LORA_RDMA_GID", "-1"),
    ] {
        let e = weights(&[(env, raw)]).expect_err("out of range");
        assert_eq!(e.code(), ErrorCode::ProjectionOutOfRange, "{env}={raw}");
        assert!(
            alloc::format!("{e}").starts_with(env),
            "{env}={raw}: names the lever: {e}"
        );
    }
    let e = weights(&[
        ("METRALE_WEIGHT_DUAL_RAIL", "1"),
        ("METRALE_WEIGHT_RAIL2_GID", "-1"),
    ])
    .expect_err("out of range");
    assert!(alloc::format!("{e}").starts_with("METRALE_WEIGHT_RAIL2_GID"));
    assert_eq!(
        weights(&[
            ("METRALE_GGUF_Q2_GROUP", "64"),
            ("METRALE_DS41_EVICT_RANDOM_PCT", "100")
        ])
        .map(|w| (w.q2_group, w.evict_random_pct)),
        Ok((64, 100))
    );
}
