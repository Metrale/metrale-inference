// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The compute unit a kernel runs on: tensor cores (with the MMA atom and the
//! operand format it issues), CUDA cores, or memory (data movement, element-wise work and
//! reductions). Declared per family in `KERNEL_FAMILIES.toml` (`compute`, `mma`), per point where
//! a family's points differ, and per kernel where one family's kernels differ; read by the
//! tensor-core policy (`hardware::tc_policy`) and shown on every planned group.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Nothing defaults: every family states its unit; `tensor_core` names its atom, the other
//!   units name none.
//! - A kernel's unit is its `kernel_compute` entry, else its family's. A family whose points
//!   state different units names every kernel's unit, so no kernel's unit is inferred.

use std::collections::BTreeMap;

use crate::rules::KernelId;

/// 2026-10-02: Where a kernel's arithmetic runs.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeUnit {
    /// 2026-10-02: Tensor cores, issuing `atom` (e.g. `mma.sync.m16n8k32.e4m3`).
    TensorCore {
        /// 2026-10-02: The MMA instruction shape and operand format.
        atom: String,
    },
    /// 2026-10-02: CUDA-core FMA arithmetic (GEMV dot products, recurrences, convolutions).
    CudaCore,
    /// 2026-10-02: Data movement, element-wise work and reductions.
    Memory,
}

impl ComputeUnit {
    /// 2026-10-02: `tensor_core`, `cuda_core` or `memory`.
    pub fn kind(&self) -> &'static str {
        match self {
            ComputeUnit::TensorCore { .. } => "tensor_core",
            ComputeUnit::CudaCore => "cuda_core",
            ComputeUnit::Memory => "memory",
        }
    }

    /// 2026-10-02: The unit as a report shows it: `tensor_core <atom>`, `cuda_core`, `memory`.
    pub fn name(&self) -> String {
        match self {
            ComputeUnit::TensorCore { atom } => format!("tensor_core {atom}"),
            other => other.kind().to_string(),
        }
    }

    /// 2026-10-02: The unit as one word, for a plan line: `tensor_core:<atom>`, `cuda_core`,
    /// `memory`.
    pub fn tag(&self) -> String {
        match self {
            ComputeUnit::TensorCore { atom } => format!("tensor_core:{atom}"),
            other => other.kind().to_string(),
        }
    }

    /// 2026-10-02: Runs on tensor cores.
    pub fn is_tensor_core(&self) -> bool {
        matches!(self, ComputeUnit::TensorCore { .. })
    }

    /// 2026-10-02: Parse a `compute` value and its `mma` atom.
    pub fn parse(compute: &str, mma: Option<&str>) -> Result<Self, String> {
        match (compute, mma.map(str::trim)) {
            ("tensor_core", Some(atom)) if !atom.is_empty() => Ok(ComputeUnit::TensorCore {
                atom: atom.to_string(),
            }),
            ("tensor_core", _) => Err("compute `tensor_core` needs its `mma` atom".into()),
            ("cuda_core", None) => Ok(ComputeUnit::CudaCore),
            ("memory", None) => Ok(ComputeUnit::Memory),
            ("cuda_core" | "memory", Some(_)) => Err(format!(
                "compute `{compute}` issues no MMA, so it names no `mma`"
            )),
            (other, _) => Err(format!(
                "compute `{other}` (tensor_core | cuda_core | memory)"
            )),
        }
    }
}

/// 2026-10-02: A family's units: its own, and the kernels that differ from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FamilyCompute {
    /// 2026-10-02: The family's unit.
    pub unit: ComputeUnit,
    /// 2026-10-02: Kernels whose unit differs from the family's.
    pub kernels: BTreeMap<KernelId, ComputeUnit>,
}

impl FamilyCompute {
    /// 2026-10-02: The unit `kernel` runs on.
    pub fn of(&self, kernel: &KernelId) -> &ComputeUnit {
        self.kernels.get(kernel).unwrap_or(&self.unit)
    }
}

/// 2026-10-02: The unit a plan group runs its node on, from its kernels' units: tensor cores
/// when any kernel issues MMAs (the others quantize, stage or reduce around it), else CUDA
/// cores when any kernel computes, else memory. `None` for a group without kernels.
pub fn group_unit<'a>(units: impl IntoIterator<Item = &'a ComputeUnit>) -> Option<ComputeUnit> {
    let mut best: Option<&ComputeUnit> = None;
    for u in units {
        let rank = |x: &ComputeUnit| match x {
            ComputeUnit::TensorCore { .. } => 2,
            ComputeUnit::CudaCore => 1,
            ComputeUnit::Memory => 0,
        };
        if best.is_none_or(|b| rank(u) > rank(b)) {
            best = Some(u);
        }
    }
    best.cloned()
}

#[cfg(test)]
#[path = "compute_tests.rs"]
mod compute_tests;
