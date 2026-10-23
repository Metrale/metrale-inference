// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `met circuit` and its subcommands (split from `cli.rs` for the 500-line cap).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

/// `met circuit`: inspect an architecture circuit.
#[derive(clap::Args, Debug)]
pub struct CircuitArgs {
    #[command(subcommand)]
    pub action: CircuitAction,
}

/// The `met circuit` views.
#[derive(clap::Subcommand, Debug)]
pub enum CircuitAction {
    /// Print the plan as stable text, one kernel group per line (the format of the checked-in
    /// plans under kernels/circuits/plans/).
    Show(CircuitPlanArgs),
    /// Draw the architecture for a terminal: the layer strip, one diagram per distinct layer
    /// plan with fused groups framed, and the per-step totals.
    Display(CircuitDisplayArgs),
    /// Load a model as `met serve` would and compare its decode logits, byte for byte, under
    /// the legacy forward and the circuit forward (reference rules only, then every rule).
    /// Set METRALE_DEBUG_NO_GRAPH=1 for the eager comparison; without it decode is graphed.
    Diff(Box<CircuitDiffArgs>),
    /// Classify a new model's ops against the kernel families of supported models (shared,
    /// parameterization opportunity, policy variant, novel), ranked by estimated step share,
    /// and write or check the Markdown report (kernels/circuits/venn/).
    Venn(Box<CircuitVennArgs>),
    /// Plan a checkpoint on a target device (kernels/DEVICES.toml): the fused plan its kernel
    /// class can run, the gap report against that class ranked by the device's roofline,
    /// decode and prefill estimates, and the memory fit. `--matrix` writes or checks every
    /// report of the roadmap matrix instead.
    Plan(Box<CircuitHwArgs>),
    /// The memory a checkpoint needs on a device at a serve's settings: per node (weights and
    /// their load-time copies, activations, workspace, state), per state and cache (KV, SSM,
    /// snapshots, speculative caches) as functions of ISL, OSL, concurrency and slots, the total
    /// against the util budget, and the largest concurrency and prompt that fit.
    Memory(Box<CircuitMemoryArgs>),
    /// List the precision each node needs: its input formats, the steps inside it (activation
    /// and weight preparation, multiply operands, accumulation, scales, compute, cache and state
    /// precision) and its output formats, as the plan on a device requires them and its kernels
    /// declare them (KERNEL_FAMILIES.toml `pipeline`).
    Precision(Box<CircuitPrecisionArgs>),
    /// How far each launch site of a step is from its roofline floor, what the step and tok/s
    /// would be at that floor (Amdahl), and the time no kernel runs; from a capture when given.
    Headroom(Box<circuit_headroom::CircuitHeadroomArgs>),
    /// Critical path, slack and exact per-site upside of a captured step; what-if rewrites.
    Whatif(Box<circuit_whatif::CircuitWhatIfArgs>),
    /// Check the fusion rules themselves (`--critical-pairs`: overlaps settled by priority alone).
    Rules(Box<circuit_rules::CircuitRulesArgs>),
}
