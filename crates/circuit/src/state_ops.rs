// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The state programs of a circuit (M5, LIFECYCLE-DESIGN.md section 3.5): the small
//! programs that move recurrent state between its places around a speculative verify, reset a
//! slot, and copy it to and from the decode-rollback ring and the prefix cache, as nodes over the
//! circuit's declared states. Legacy runs them as host code that sizes each copy itself
//! (`verify_a_ssm.rs`, `async_chkpt.rs`, `ssm_pool_slots.rs`, `ssm_snapshot*.rs`), with the
//! GatedDeltaNet conv formula, which is 0 bytes for a Mamba2 layer; here every node is sized from
//! its state's declaration, so a Mamba2 layer's copies are its own.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - The programs are generic over recurrent state kinds: every recurrent state of the target
//!   (`Section::Main`) gets one node in each verify and slot program, in state order.
//! - A ring or prefix program has one node per recurrent target state whose block declares a
//!   snapshot of that kind copying it (`of`); a state without one is left out, and the executor
//!   refuses a program it needs that came out empty.
//! - A node's bytes are what it writes for one sequence: one unit of its destination.

use std::collections::BTreeMap;

use crate::ir::{Circuit, Section};
use crate::state::{StateDtype, StateError, StateKind};

/// 2026-09-30: Where one sequence's unit of a recurrent state lives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StatePlace {
    /// 2026-09-30: The slot's live state.
    Live,
    /// 2026-09-30: The pre-verify checkpoint.
    Checkpoint,
    /// 2026-09-30: The verify intermediate after the last accepted row.
    AcceptedStep,
    /// 2026-10-03: A decode-rollback ring slot of the sequence (the `ring_snapshot` cache).
    Ring,
    /// 2026-10-03: A prefix-cache slot (the `prefix_snapshot` cache).
    Prefix,
}

/// 2026-09-30: What a state-program node does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum StateOp {
    /// 2026-09-30: Copy one unit's stored bytes from `from` to `to`, bit for bit.
    Copy { from: StatePlace, to: StatePlace },
    /// 2026-10-03: Copy one unit from `from` to `to`, converting each element from the source's
    /// dtype to the destination's (a plain copy when they are equal): the prefix cache keeps
    /// FP32 whatever the pool stores.
    Convert { from: StatePlace, to: StatePlace },
    /// 2026-09-30: Zero the live unit.
    Zero,
}

impl StateOp {
    /// 2026-10-03: The place this op writes.
    pub fn writes(self) -> StatePlace {
        match self {
            Self::Copy { to, .. } | Self::Convert { to, .. } => to,
            Self::Zero => StatePlace::Live,
        }
    }

    /// 2026-10-03: The snapshot cache kind this op reads or writes, when it touches one.
    pub fn cache_kind(self) -> Option<StateKind> {
        let place = |p| match p {
            StatePlace::Ring => Some(StateKind::RingSnapshot),
            StatePlace::Prefix => Some(StateKind::PrefixSnapshot),
            _ => None,
        };
        match self {
            Self::Copy { from, to } | Self::Convert { from, to } => place(from).or(place(to)),
            Self::Zero => None,
        }
    }
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
    /// 2026-10-03: Live to a ring slot, the stored bytes as they are (`save_decode`).
    RingSave,
    /// 2026-10-03: A ring slot back to live (`restore_decode`).
    RingRestore,
    /// 2026-10-03: Live to a prefix-cache slot, widened to the cache's FP32 (Marconi `save`).
    PrefixSave,
    /// 2026-10-03: A prefix-cache slot to live, narrowed to the pool's storage (Marconi
    /// `restore`).
    PrefixRestore,
}

impl StateProgramId {
    /// 2026-09-30: Every program, in this order.
    pub const ALL: [StateProgramId; 8] = [
        Self::VerifyCheckpoint,
        Self::VerifyRollback,
        Self::CommitAccepted,
        Self::SlotZero,
        Self::RingSave,
        Self::RingRestore,
        Self::PrefixSave,
        Self::PrefixRestore,
    ];

    /// 2026-09-30: The op each of its nodes runs.
    pub fn op(self) -> StateOp {
        use StatePlace::*;
        match self {
            Self::VerifyCheckpoint => StateOp::Copy {
                from: Live,
                to: Checkpoint,
            },
            Self::VerifyRollback => StateOp::Copy {
                from: Checkpoint,
                to: Live,
            },
            Self::CommitAccepted => StateOp::Copy {
                from: AcceptedStep,
                to: Live,
            },
            Self::SlotZero => StateOp::Zero,
            Self::RingSave => StateOp::Copy {
                from: Live,
                to: Ring,
            },
            Self::RingRestore => StateOp::Copy {
                from: Ring,
                to: Live,
            },
            Self::PrefixSave => StateOp::Convert {
                from: Live,
                to: Prefix,
            },
            Self::PrefixRestore => StateOp::Convert {
                from: Prefix,
                to: Live,
            },
        }
    }

    /// 2026-09-30: The spelling used in node ids.
    pub fn name(self) -> &'static str {
        match self {
            Self::VerifyCheckpoint => "verify_checkpoint",
            Self::VerifyRollback => "verify_rollback",
            Self::CommitAccepted => "commit_accepted",
            Self::SlotZero => "slot_zero",
            Self::RingSave => "ring_save",
            Self::RingRestore => "ring_restore",
            Self::PrefixSave => "prefix_save",
            Self::PrefixRestore => "prefix_restore",
        }
    }
}

/// 2026-09-30: One node of a state program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateOpNode {
    /// 2026-09-30: `<program>.<state id>` (`verify_rollback.l4.mamba.conv`).
    pub id: String,
    /// 2026-09-30: Index into [`Circuit::states`]: the recurrent state.
    pub state: usize,
    /// 2026-10-03: Index into [`Circuit::states`] of the snapshot cache the op reads or writes
    /// (`l4.gdn.ring_h`); `None` for the verify and slot programs.
    pub cache: Option<usize>,
    /// 2026-09-30: What it does.
    pub op: StateOp,
}

impl StateOpNode {
    /// 2026-10-03: The (source, destination) element types under `formats`: the recurrent
    /// state's on the live, checkpoint and step side, the cache's on the ring and prefix side.
    pub fn dtypes(
        &self,
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
    ) -> Result<(StateDtype, StateDtype), StateError> {
        let state = circuit.states[self.state].dtype(formats)?;
        let side = |p: StatePlace| -> Result<StateDtype, StateError> {
            match (p, self.cache) {
                (StatePlace::Ring | StatePlace::Prefix, Some(c)) => {
                    circuit.states[c].dtype(formats)
                }
                (StatePlace::Ring | StatePlace::Prefix, None) => Err(StateError::Inputs(format!(
                    "node `{}` touches a snapshot and names no cache",
                    self.id
                ))),
                _ => Ok(state),
            }
        };
        match self.op {
            StateOp::Copy { from, to } | StateOp::Convert { from, to } => {
                Ok((side(from)?, side(to)?))
            }
            StateOp::Zero => Ok((state, state)),
        }
    }

    /// 2026-10-03: Bytes the node writes for one sequence. A `Copy` moves the source's stored
    /// bytes (a ring slot keeps an f16 pool's bytes as they are, in a slot sized for FP32); a
    /// `Convert` writes one unit of its destination's dtype; a `Zero` clears the live unit.
    pub fn bytes(
        &self,
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
    ) -> Result<u64, StateError> {
        let elements = circuit.states[self.state].elements;
        let (src, dst) = self.dtypes(circuit, formats)?;
        let width = match self.op {
            StateOp::Copy { .. } => src.size().min(dst.size()),
            StateOp::Convert { .. } | StateOp::Zero => dst.size(),
        };
        elements
            .checked_mul(width)
            .ok_or_else(|| StateError::Overflow(self.id.clone()))
    }
}

/// 2026-09-30: One state program.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateProgram {
    pub id: StateProgramId,
    pub nodes: Vec<StateOpNode>,
}

impl StateProgram {
    /// 2026-09-30: Bytes one sequence's run of the program writes, keyed formats from `formats`.
    pub fn bytes(
        &self,
        circuit: &Circuit,
        formats: &BTreeMap<String, StateDtype>,
    ) -> Result<u64, StateError> {
        self.nodes.iter().try_fold(0u64, |acc, n| {
            acc.checked_add(n.bytes(circuit, formats)?)
                .ok_or_else(|| StateError::Overflow(n.id.clone()))
        })
    }
}

/// 2026-10-03: The prefix-cache snapshot of the last hidden row: the target's one
/// `prefix_snapshot` that copies no state (`head.prefix_hidden`), when the circuit declares it.
/// Its source is the last row of the head's input, not a state, so it is a cache of its own
/// beside the state programs.
pub fn prefix_hidden(circuit: &Circuit) -> Option<usize> {
    circuit.states.iter().position(|s| {
        s.kind == StateKind::PrefixSnapshot && s.copies.is_none() && s.section == Section::Main
    })
}

/// 2026-10-03: The snapshot of `kind` that copies state `of` (by id), when its block declares
/// one.
fn snapshot_of(circuit: &Circuit, kind: StateKind, of: &str) -> Option<usize> {
    circuit
        .states
        .iter()
        .position(|s| s.kind == kind && s.copies.as_deref() == Some(of))
}

/// 2026-09-30: Every state program of `circuit`, one node per recurrent target state (per
/// recurrent target state with a snapshot of the program's kind, for the ring and prefix
/// programs).
pub fn state_programs(circuit: &Circuit) -> Vec<StateProgram> {
    StateProgramId::ALL
        .into_iter()
        .map(|id| {
            let op = id.op();
            StateProgram {
                id,
                nodes: circuit
                    .states
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.kind == StateKind::Recurrent && s.section == Section::Main)
                    .filter_map(|(i, s)| {
                        let cache = match op.cache_kind() {
                            Some(kind) => Some(snapshot_of(circuit, kind, &s.id)?),
                            None => None,
                        };
                        Some(StateOpNode {
                            id: format!("{}.{}", id.name(), s.id),
                            state: i,
                            cache,
                            op,
                        })
                    })
                    .collect(),
            }
        })
        .collect()
}

#[cfg(test)]
#[path = "state_ops_tests.rs"]
mod state_ops_tests;
