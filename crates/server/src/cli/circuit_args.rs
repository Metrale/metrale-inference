// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The clap options of `met circuit affected`, kept apart from its logic in
//! `circuit_affected.rs` because their `///` comments are the `--help` text, which carries no
//! date (.comment-ttl.toml exempts this file for that reason).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use std::path::PathBuf;

/// The models a change reaches through the kernel families it edits. A path under kernels/
/// that no family lists is unmapped and reaches every model; a path outside kernels/ is not
/// scoped by this map (the gate treats it as reaching every model).
#[derive(clap::Args, Debug)]
pub struct CircuitAffectedArgs {
    /// The revision the change is measured from (a branch's merge target).
    #[arg(long)]
    pub base: String,
    /// The revision whose kernels and plans are read, and whose change is reported.
    #[arg(long)]
    pub head: String,
    /// Print JSON: {models: [{model, hw, checkpoint, reason: [{file, family, instance,
    /// assumed}]}], unmapped, unused, non_kernel, assumed}.
    #[arg(long)]
    pub json: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<PathBuf>,
}
