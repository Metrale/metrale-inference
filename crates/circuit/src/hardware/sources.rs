// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: What the caller resolves from the kernel tree for one class: the modules a target
//! of the class compiles (after inheritance, shadows and `[modules]` renames), every file the
//! resolution stages, and the kernels its MODEL.tomls declare absent. The pure planner reads
//! nothing else about the sources.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - The caller resolves with the same layout the kernels build uses (metrale-closure), so
//!   "compiled" here is what the build compiles; the crate never walks a directory itself.
//! - A class with no target for the model resolves its common layer only (`target = None`),
//!   and the report says so: its model-specific kernels are absent, not assumed.

use std::collections::{BTreeMap, BTreeSet};

use crate::venn::repo::Repo;

/// 2026-09-30: One compiled module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    /// 2026-09-30: Repo-relative source path.
    pub path: String,
    /// 2026-09-30: Source text.
    pub text: String,
}

/// 2026-09-30: The kernel sources of one class for one model target.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClassSources {
    /// 2026-09-30: Class.
    pub class: String,
    /// 2026-09-30: `class/model/quant` resolved, or `None` when the class has no target for the
    /// model and only its common layer was resolved.
    pub target: Option<String>,
    /// 2026-09-30: Module name (source stem after renames) to its source.
    pub modules: BTreeMap<String, Module>,
    /// 2026-09-30: Every repo-relative file the resolution compiles or stages (headers too).
    pub files: BTreeSet<String>,
    /// 2026-09-30: `[expected_absent.<module>]` entries: module to (function to reason).
    pub expected_absent: BTreeMap<String, BTreeMap<String, String>>,
}

/// 2026-09-30: A repository that can also resolve a class's kernel sources.
pub trait KernelTree: Repo {
    /// 2026-09-30: The sources `class` compiles for `model`/`quant`, or its common layer when it
    /// has no such target.
    fn class_sources(&self, class: &str, model: &str, quant: &str)
    -> Result<ClassSources, String>;

    /// 2026-09-30: This tree as a plain [`Repo`].
    fn as_repo(&self) -> &dyn Repo;
}
