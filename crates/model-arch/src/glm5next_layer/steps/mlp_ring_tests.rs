// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The ring order of `mlp_ring`. The bytes of the ring against the serial section
//! are the SP accuracy check's job; the shard launches' rows are `topk_combine`'s tests.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use super::order;
use crate::glm5next_sp_mhc::{ring_step_plan, shards};

/// 2026-10-14: Launch `i < world - 1` is ring step `i + 1`'s shard `(rank + i + 1) % world`, the
/// one that step sends (`glm5next_sp_mhc::ring_step_plan`); the own shard is last; every shard
/// is combined once.
#[test]
fn ring_order_is_step_shards_then_own() {
    assert_eq!(order(0, 3), [1, 2, 0]);
    assert_eq!(order(1, 3), [2, 0, 1]);
    assert_eq!(order(2, 3), [0, 1, 2]);
    assert_eq!(order(1, 2), [0, 1]);
    for world in 2..=8 {
        for rank in 0..world {
            let o = order(rank, world);
            for (i, &shard) in o.iter().enumerate().take(world - 1) {
                assert_eq!(shard, (rank + i + 1) % world);
            }
            assert_eq!(o[world - 1], rank);
            let mut seen = o.clone();
            seen.sort_unstable();
            assert_eq!(seen, (0..world).collect::<Vec<_>>());
        }
    }
}

/// 2026-10-14: Launch `s - 1` is the shard ring step `s` sends: `ring_step_plan(..).to` at every
/// rank and step, at shard sizes even and ragged.
#[test]
fn ring_order_matches_the_step_plan() {
    for world in 2..=8 {
        for k in [world * 64, world * 64 + world - 1] {
            let sh = shards(k, world);
            for rank in 0..world {
                let o = order(rank, world);
                for step in 1..world {
                    let st = ring_step_plan(rank, &sh, 16, 1 << 20, step).unwrap();
                    assert_eq!(o[step - 1], st.to, "rank {rank} of {world}, step {step}");
                }
            }
        }
    }
}
