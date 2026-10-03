// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The cache plan of real checkpoints equals what the legacy allocators allocate:
//! the snapshot pool's prefix slots (h and conv at FP32 per layer, plus the last-hidden row), its
//! ring slots, and the carried-state verify's buffers (`GdnCarrySizes`). A model without a
//! circuit keeps the transitional arithmetic.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants: none beyond the types.

use super::*;
use crate::layers::ops::{GdnCarrySizes, gdn_carry_conv_seq_elems, gdn_carry_seq_floats};

/// 2026-10-03: The engine-parsed config of a checked-in checkpoint fixture.
fn config(name: &str) -> ModelConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints")
        .join(name)
        .join("config.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{name}: {e}"));
    metrale_config::parse_config(&json).unwrap_or_else(|e| panic!("{name}: {e}"))
}

const CIRCUIT_MODELS: [&str; 2] = ["unsloth--Qwen3.8-27B-NVFP4", "Qwen--Qwen3.6-35B-A3B-FP8"];

#[test]
fn the_prefix_cache_and_the_ring_equal_the_snapshot_pools_allocation() {
    for name in CIRCUIT_MODELS {
        let c = config(name);
        let plan = CachePlan::new(&c, false).unwrap();
        assert_eq!(plan.source, UnitSource::Circuit, "{name}");
        let layer = c.ssm_h_state_bytes() + c.ssm_conv_state_bytes();
        let hidden = c.hidden_size * 2;
        for slots in [0, 1, 256] {
            assert_eq!(
                plan.marconi_bytes(slots).unwrap(),
                slots * (c.num_ssm_layers() * layer + hidden),
                "{name} slots={slots}"
            );
        }
        assert_eq!(plan.ring_seq_bytes().unwrap(), c.num_ssm_layers() * layer, "{name}");
        assert_eq!(
            plan.prefix_units(&c).unwrap(),
            PrefixUnits {
                h: c.ssm_h_state_bytes(),
                conv: c.ssm_conv_state_bytes(),
                hidden,
            },
            "{name}"
        );
    }
}

/// 2026-10-03: The carry's reserve equals `GdnCarrySizes::total`, the bytes `gdn_carry_bind`
/// allocates, at every slot count the serves use.
#[test]
fn the_carry_equals_the_allocators_sizes() {
    for name in CIRCUIT_MODELS {
        let c = config(name);
        let plan = CachePlan::new(&c, false).unwrap();
        let (nv, kd, vd) = (
            c.linear_num_value_heads,
            c.linear_key_head_dim,
            c.linear_value_head_dim,
        );
        let conv_dim = c.linear_num_key_heads * kd * 2 + nv * vd;
        let rows = crate::layer::VERIFY_WY_TABLE_SEQS;
        for slots in [2, 33, 129] {
            let legacy = GdnCarrySizes::new(
                c.num_ssm_layers(),
                slots,
                gdn_carry_seq_floats(nv, kd, vd),
                gdn_carry_conv_seq_elems(conv_dim),
                rows,
            )
            .total();
            assert_eq!(plan.carry_bytes(slots, rows).unwrap(), legacy, "{name} slots={slots}");
        }
    }
}

/// 2026-10-03: A model without a circuit is sized as before 2026-10-03: no last-hidden row.
#[test]
fn a_model_without_a_circuit_keeps_the_transitional_arithmetic() {
    let c = ModelConfig::qwen3_next_80b_nvfp4();
    let plan = CachePlan::new(&c, false).unwrap();
    assert_eq!(plan.source, UnitSource::Transitional);
    let layer = c.ssm_h_state_bytes() + c.ssm_conv_state_bytes();
    assert_eq!(plan.marconi_bytes(8).unwrap(), 8 * c.num_ssm_layers() * layer);
    assert_eq!(plan.ring_seq_bytes().unwrap(), c.num_ssm_layers() * layer);
}
