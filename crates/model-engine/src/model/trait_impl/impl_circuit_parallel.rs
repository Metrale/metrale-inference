// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Which parallel rank a circuit build plans (LIFECYCLE-DESIGN.md 15.10, TP/EP):
//! the communicator's rank and world, as legacy's collectives and `lmhead_vocab_shard` read them.
//!
//! Owner: model-engine (FEATURES workstream).
//! Invariants:
//! - The rank and world are the communicator's; the config's TP or EP world must equal it (the
//!   two together are refused by `circuit_unmodelled`).

use metrale_circuit::parallel::{Parallel, TpRank};

use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-03: This process's parallel rank; `None` without a communicator of two or more.
    pub(super) fn circuit_parallel(&self) -> Option<Parallel> {
        let comm = self.comm.as_ref().filter(|c| c.world_size() > 1)?;
        let rank = TpRank {
            rank: comm.rank() as u64,
            world: comm.world_size() as u64,
        };
        Some(if self.config.tp_world_size > 1 {
            Parallel::Tensor(rank)
        } else {
            Parallel::Expert(rank)
        })
    }
}
