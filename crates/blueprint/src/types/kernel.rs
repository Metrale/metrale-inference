// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The kernel set: every module the serve may load, pinned by hash, and the
//! `module::func` entries the plans name.
//!
//! Owner: blueprint.
//! Invariants:
//! - A module's source (internal, inline, disk, remote) is not here: it lives in the unsigned
//!   locator table, so a lite and a full file carry one signature.
//! - `KernelSpec.module` indexes `KernelSet.modules`.
//! - 2026-10-22: Kernel ids and compute units are the circuit's own types, and the compiled
//!   target is metrale-kernel-target's `KernelTarget<String>` (SSOT).

use alloc::string::String;
use alloc::vec::Vec;

use metrale_circuit::KernelId;
use metrale_circuit::venn::compute::ComputeUnit;
pub use metrale_kernel_target::KernelTarget;
use serde::{Deserialize, Serialize};

use crate::ids::ArtifactId;

/// 2026-10-22: What kind of bytes an artifact is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// 2026-10-22: PTX text (NVIDIA).
    PtxText,
    /// 2026-10-22: An AMD GPU code object (`.co`).
    AmdCodeObject,
    /// 2026-10-22: A SCALE object.
    ScaleObject,
    /// 2026-10-22: An Apple GPU shader library.
    AppleGpuLibrary,
    /// 2026-10-24: An NVIDIA fatbin: SASS for the module's arch plus its PTX, uncompressed.
    NvidiaFatbin,
}

impl ArtifactKind {
    /// 2026-10-22: The export spelling.
    pub const fn name(self) -> &'static str {
        match self {
            ArtifactKind::PtxText => "ptx_text",
            ArtifactKind::AmdCodeObject => "amd_code_object",
            ArtifactKind::ScaleObject => "scale_object",
            ArtifactKind::AppleGpuLibrary => "apple_gpu_library",
            ArtifactKind::NvidiaFatbin => "nvidia_fatbin",
        }
    }
}

/// 2026-10-22: The modules and the entries the plans use.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSet {
    /// 2026-10-22: The compiled target.
    pub target: KernelTarget<String>,
    /// 2026-10-22: Every module, by name order.
    pub modules: Vec<ModuleSpec>,
    /// 2026-10-22: Every `module::func` a plan names, sorted.
    pub specs: Vec<KernelSpec>,
}

/// 2026-10-22: One compiled module.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleSpec {
    /// 2026-10-22: The module name the registry loads it under.
    pub name: String,
    /// 2026-10-22: Its kind.
    pub kind: ArtifactKind,
    /// 2026-10-22: The arch it was compiled for, e.g. `sm_121f`.
    pub arch: String,
    /// 2026-10-22: SHA-256 of its raw bytes.
    pub artifact: ArtifactId,
    /// 2026-10-22: Its raw length.
    pub len: u64,
    /// 2026-10-22: Its entry points (`.entry` names for PTX), sorted.
    pub entries: Vec<String>,
}

/// 2026-10-22: One entry a plan names.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KernelSpec {
    /// 2026-10-22: `module::func`.
    pub id: KernelId,
    /// 2026-10-22: Index into `KernelSet.modules`.
    pub module: u32,
    /// 2026-10-22: The family that declares it (`KERNEL_FAMILIES.toml`), if any.
    pub family: Option<String>,
    /// 2026-10-22: The compute unit the family declares (its tag, e.g.
    /// `tensor_core:mma.m16n8k32`, is what the exports show).
    pub compute: Option<ComputeUnit>,
}

impl KernelSet {
    /// 2026-10-22: The module named `name`.
    pub fn module(&self, name: &str) -> Option<&ModuleSpec> {
        self.modules.iter().find(|m| m.name == name)
    }

    /// 2026-10-22: The spec of `id`.
    pub fn spec(&self, id: &KernelId) -> Option<&KernelSpec> {
        self.specs.iter().find(|s| &s.id == id)
    }
}
