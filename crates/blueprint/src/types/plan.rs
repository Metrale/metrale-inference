// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Plans: one fused arm per `(mode, rows)`.
//!
//! Owner: blueprint.
//! Invariants: wire types only; [`Mode`], [`Repeat`] and [`Numerics`] mirror
//! `metrale_circuit`'s and are converted exhaustively at the adapter.

use alloc::string::String;
use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

use super::kernel::KernelId;
use crate::ids::Sha256;

/// 2026-10-22: A plan mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Mode {
    /// 2026-10-22: One sequence, one row.
    Decode,
    /// 2026-10-22: `n` sequences, one row each.
    MultiSeq,
    /// 2026-10-22: `K` draft rows of one sequence.
    Verify,
    /// 2026-10-22: The draft head.
    Draft,
    /// 2026-10-22: Batched verifies over a row table.
    VerifyBatch,
}

impl Mode {
    /// 2026-10-22: The plan-file spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Mode::Decode => "decode",
            Mode::MultiSeq => "multi_seq",
            Mode::Verify => "verify",
            Mode::Draft => "draft",
            Mode::VerifyBatch => "verify_batch",
        }
    }

    /// 2026-10-22: The mode a spelling names.
    pub fn parse(s: &str) -> Option<Self> {
        [
            Mode::Decode,
            Mode::MultiSeq,
            Mode::Verify,
            Mode::Draft,
            Mode::VerifyBatch,
        ]
        .into_iter()
        .find(|m| m.name() == s)
    }
}

/// 2026-10-22: How often a group's kernels launch per step.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Repeat {
    /// 2026-10-22: Once.
    Once,
    /// 2026-10-22: Once per row.
    PerRow,
    /// 2026-10-22: Once per `n`-row chunk.
    Chunk(u64),
    /// 2026-10-22: Once per row but the last.
    PerRowButLast,
    /// 2026-10-22: Per run of a batched verify.
    PerRun,
}

impl Repeat {
    /// 2026-10-22: `once`, `per_row`, `chunk<n>`, `per_row_but_last`, `per_run`.
    pub fn name(self) -> String {
        match self {
            Repeat::Once => String::from("once"),
            Repeat::PerRow => String::from("per_row"),
            Repeat::Chunk(n) => alloc::format!("chunk{n}"),
            Repeat::PerRowButLast => String::from("per_row_but_last"),
            Repeat::PerRun => String::from("per_run"),
        }
    }
}

/// 2026-10-22: A group's numerics against the reference.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Numerics {
    /// 2026-10-22: Byte-identical, proven by the microtest.
    BitIdentical {
        /// 2026-10-22: The microtest.
        microtest: String,
    },
    /// 2026-10-22: The reference definition.
    Reference,
    /// 2026-10-22: Changes numerics; selected only under `lever`.
    Differs {
        /// 2026-10-22: The opt-in lever.
        lever: String,
    },
}

impl Numerics {
    /// 2026-10-22: `bit_identical`, `reference` or `differs`.
    pub const fn class(&self) -> &'static str {
        match self {
            Numerics::BitIdentical { .. } => "bit_identical",
            Numerics::Reference => "reference",
            Numerics::Differs { .. } => "differs",
        }
    }
}

/// 2026-10-22: An edge's state under a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeStateSpec {
    /// 2026-10-22: Outside this mode's section.
    Absent,
    /// 2026-10-22: Written to memory.
    Materialized,
    /// 2026-10-22: Kept inside the group with this index.
    Fused(u32),
}

/// 2026-10-22: One fused arm.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanSpec {
    /// 2026-10-22: Mode.
    pub mode: Mode,
    /// 2026-10-22: Padded rows.
    pub rows: u64,
    /// 2026-10-22: Groups in execution order.
    pub groups: Vec<GroupSpec>,
    /// 2026-10-22: Per edge.
    pub edge_states: Vec<EdgeStateSpec>,
    /// 2026-10-22: Per edge: the stored format spelling.
    pub edge_formats: Vec<String>,
    /// 2026-10-22: The plan digest (`metrale_circuit::digest`).
    pub digest: Sha256,
}

/// 2026-10-22: One launch group.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GroupSpec {
    /// 2026-10-22: The rule that formed it.
    pub rule: String,
    /// 2026-10-22: Its kernels.
    pub kernels: Vec<KernelId>,
    /// 2026-10-22: Launch repetition.
    pub repeat: Repeat,
    /// 2026-10-22: Copy-engine repetition, if any.
    pub copies: Option<Repeat>,
    /// 2026-10-22: The executor emitter.
    pub emitter: String,
    /// 2026-10-22: Numerics.
    pub numerics: Numerics,
    /// 2026-10-22: The nodes it covers.
    pub nodes: Vec<u32>,
}
