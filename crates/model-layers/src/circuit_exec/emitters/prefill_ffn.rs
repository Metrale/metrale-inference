// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The dense FFN, embedding and head emitters of the prefill modes (LIFECYCLE-DESIGN.md 15.4). Each mirrors
//! the legacy prefill call site it names: the same `ops::*` function, the same arguments, the
//! row count from the step (`StepEnv::prefill`).
//!
//! Owner: model-layers circuit executor.
//! Invariants: see `mod.rs`.

use super::super::compile::OpEmitter;

/// 2026-10-03: This module's emitters, for the registry in `mod.rs`.
pub(super) static ALL: &[&dyn OpEmitter] = &[];
