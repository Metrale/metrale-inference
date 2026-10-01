// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The state programs of a circuit (M5, LIFECYCLE-DESIGN.md section 3.5): the small
//! programs that move recurrent state between its places around a speculative verify, and reset
//! a slot, as nodes over the circuit's declared states. Legacy runs them as host code that sizes
//! each copy itself (`verify_a_ssm.rs`, `async_chkpt.rs`, `ssm_pool_slots.rs`), with the
//! GatedDeltaNet conv formula, which is 0 bytes for a Mamba2 layer; here every node is sized from
//! its state's declaration, so a Mamba2 layer's copies are its own.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The programs are generic over recurrent state kinds: every recurrent state of the target
//!   (`Section::Main`) gets one node in each program, in state order.
//! - A node's bytes are one unit of its state ([`StatePlan`] at one slot), so a program's bytes
//!   are what one sequence's copies move.

use std::collections::BTreeMap;

use crate::ir::{Circuit, Section};
use crate::state::{Holding, StateDtype, StateError, StateInputs, StateKind, StatePlan};

/// 2026-09-30: Where one sequence's unit of a recurrent state lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatePlace {
    /// 2026-09-30: The slot's live state.
    Live,
    /// 2026-09-30: The pre-verify checkpoint.
    Checkpoint,
    /// 2026-09-30: The verify intermediate after the last accepted row.
    AcceptedStep,
}

/// 2026-09-30: What a state-program node does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateOp {
    /// 2026-09-30: Copy one unit from `from` to `to`.
    Copy { from: StatePlace, to: StatePlace },
    /// 2026-09-30: Zero the live unit.
    Zero,
}

/// 2026-09-30: The state programs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateProgramId {
    /// 2026-09-30: Before a verify: live to checkpoint (`checkpoint_ssm_states_dispatch`).
    VerifyCheckpoint,
    /// 2026-09-30: A verify that accepted nothing: checkpoint to live (`rollback_ssm_states`).
    VerifyRollback,
    /// 2026-09-30: A verify that accepted a prefix: the accepted row's intermediate to live
    /// (`commit_accepted_prefix_dispatch`).
    CommitAccepted,
    /// 2026-09-30: A slot handed to a new sequence: zero its live state (`zero_slot`).
    SlotZero,
}

impl StateProgramId {
    /// 2026-09-30: Every program, in this order.
    pub const ALL: [StateProgramId; 4] = [
        Self::VerifyCheckpoint,
        Self::VerifyRollback,
        Self::CommitAccepted,
        Self::SlotZero,
    ];

    /// 2026-09-30: The op each of its nodes runs.
    pub fn op(self) -> StateOp {
        match self {
            Self::VerifyCheckpoint => StateOp::Copy {
                from: StatePlace::Live,
                to: StatePlace::Checkpoint,
            },
            Self::VerifyRollback => StateOp::Copy {
                from: StatePlace::Checkpoint,
                to: StatePlace::Live,
            },
            Self::CommitAccepted => StateOp::Copy {
                from: StatePlace::AcceptedStep,
                to: StatePlace::Live,
            },
            Self::SlotZero => StateOp::Zero,
        }
    }

    /// 2026-09-30: The spelling used in node ids.
    pub fn name(self) -> &'static str {
        match self {
            Self::VerifyCheckpoint => "verify_checkpoint",
            Self::VerifyRollback => "verify_rollback",
            Self::CommitAccepted => "commit_accepted",
            Self::SlotZero => "slot_zero",
        }
    }
}

/// 2026-09-30: One node of a state program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateOpNode {
    /// 2026-09-30: `<program>.<state id>` (`verify_rollback.l4.mamba.conv`).
    pub id: String,
    /// 2026-09-30: Index into [`Circuit::states`].
    pub state: usize,
    /// 2026-09-30: What it does.
    pub op: StateOp,
}

/// 2026-09-30: One state program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateProgram {
    pub id: StateProgramId,
    pub nodes: Vec<StateOpNode>,
}

impl StateProgram {
    /// 2026-09-30: Bytes one sequence's run of the program moves (or zeroes): one unit of each
    /// node's state, its keyed format from `formats`.
    pub fn bytes(
        &self,
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
    ) -> Result<u64, StateError> {
        let decls: Vec<_> = self
            .nodes
            .iter()
            .map(|n| circuit.states[n.state].clone())
            .collect();
        let plan = StatePlan::new(
            &decls,
            &StateInputs {
                formats: formats.clone(),
                slots: 1,
                verify: None,
                kv: None,
                draft_kv: None,
            },
        )?;
        Ok(plan.bytes_where(|t| t.holding == Holding::Live))
    }
}

/// 2026-09-30: Every state program of `circuit`, one node per recurrent target state.
pub fn state_programs(circuit: &Circuit) -> Vec<StateProgram> {
    StateProgramId::ALL
        .into_iter()
        .map(|id| StateProgram {
            id,
            nodes: circuit
                .states
                .iter()
                .enumerate()
                .filter(|(_, s)| s.kind == StateKind::Recurrent && s.section == Section::Main)
                .map(|(i, s)| StateOpNode {
                    id: format!("{}.{}", id.name(), s.id),
                    state: i,
                    op: id.op(),
                })
                .collect(),
        })
        .collect()
}

#[cfg(test)]
#[path = "state_ops_tests.rs"]
mod state_ops_tests;
