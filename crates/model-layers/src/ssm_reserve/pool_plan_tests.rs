// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: GLM-5.3-Flash (`glm5_next`) has a circuit, so its SSM pool is sized from the
//! circuit's KDA declarations instead of the transitional `ModelConfig` sizes it used before.
//! The two must agree byte for byte on every tensor-parallel rank (the uneven split gives 22,
//! 21 and 21 local heads at TP=3), and the sparse-attention layers' indexer pool tail must not
//! enter the pool.
//!
//! Owner: model-layers (SSM reserve).
//! Invariants: none beyond the types.

use metrale_circuit::state::Holding;
use metrale_config::{ModelConfig, TpSupport};

use super::*;

/// 2026-10-09: The checkpoint's own config.json (crates/circuit/tests/fixtures/checkpoints/).
fn glm_config() -> ModelConfig {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../circuit/tests/fixtures/checkpoints/nvidia--GLM-5.3-Flash-NVFP4/config.json");
    let json = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    metrale_config::parse_config(&json).expect("GLM-5.3-Flash config parses")
}

/// 2026-10-09: The process-local config of `rank` of `tp`, sharded as the serve's topology
/// phase does (`shard_heads_for_tp` with the GLM loader's uneven support).
fn local(tp: usize, rank: usize) -> ModelConfig {
    let mut c = glm_config();
    c.tp_world_size = tp;
    c.tp_rank = rank;
    c.shard_heads_for_tp(TpSupport::Uneven {
        linear_channel_unit: 1,
    })
    .expect("GLM heads shard");
    c
}

fn shapes() -> Vec<PoolCounts> {
    let shape = |max_slots, spec, rollback| PoolShape {
        max_slots,
        spec,
        num_intermediates: 4,
        num_drafts: 3,
        uniform_h: false,
        rollback,
    };
    [
        shape(1, false, SsmRollbackMode::Snapshot),
        shape(16, true, SsmRollbackMode::Snapshot),
        shape(8, true, SsmRollbackMode::Replay),
    ]
    .iter()
    .map(|s| pool_counts_tiered(s, None))
    .collect()
}

#[test]
fn glm_pool_from_its_circuit_is_the_transitional_pool_on_every_tp_rank() {
    for tp in [1, 2, 3] {
        for rank in 0..tp {
            let c = local(tp, rank);
            let at = format!("tp={tp} rank={rank}");
            let (decls, source) = recurrent_units(&c).expect(&at);
            assert_eq!(source, UnitSource::Circuit, "{at}");
            let ids: Vec<&str> = decls.iter().map(|d| d.id.as_str()).collect();
            assert_eq!(ids, ["kda.h", "kda.conv"], "{at}");
            let old = transitional_units(&c);
            for counts in shapes() {
                for f16 in [false, true] {
                    let new = PoolPlan::new(&c, &counts, f16).expect(&at);
                    let was =
                        PoolPlan::from_units(&c, &old, UnitSource::Transitional, &counts, f16)
                            .expect(&at);
                    let at = format!("{at} {counts:?} f16={f16}");
                    assert_eq!(new.h_f32_unit, c.ssm_h_state_bytes(), "{at}");
                    assert_eq!(new.conv_unit, c.ssm_conv_state_bytes(), "{at}");
                    assert_eq!(
                        (new.layers, new.h_f32_unit, new.h_stored_unit, new.conv_unit),
                        (was.layers, was.h_f32_unit, was.h_stored_unit, was.conv_unit),
                        "{at}"
                    );
                    assert_eq!(new.total(), was.total(), "{at}");
                    for state in [PoolState::H, PoolState::Conv] {
                        for holding in [
                            Holding::Live,
                            Holding::Steps,
                            Holding::Checkpoint,
                            Holding::Blocks,
                        ] {
                            assert_eq!(
                                new.layer_bytes(state, holding),
                                was.layer_bytes(state, holding),
                                "{at} {state:?} {holding:?}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// 2026-10-09: The comparison above reads the sharded config: the uneven TP=3 split gives
/// rank 0 one head more, so its units are larger than rank 1's.
#[test]
fn the_tp3_ranks_size_their_own_heads() {
    let heads: Vec<usize> = (0..3).map(|r| local(3, r).linear_num_value_heads).collect();
    assert_eq!(heads, [22, 21, 21]);
    let unit = |r| recurrent_units(&local(3, r)).unwrap().0[0].elements;
    assert_eq!(unit(0), 22 * 128 * 128);
    assert_eq!(unit(1), 21 * 128 * 128);
}
