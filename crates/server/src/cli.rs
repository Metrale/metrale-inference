// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `met` command line: `Cli` and its subcommands. The argument structs are
//! in the `cli/` modules.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::Parser;

#[cfg(feature = "bench")]
pub mod bench_aggregate;
#[cfg(feature = "bench")]
mod bench_args;
#[cfg(feature = "bench")]
pub mod bench_card;
#[cfg(feature = "bench")]
pub(crate) mod bench_cause;
#[cfg(feature = "bench")]
pub mod bench_certify;
#[cfg(feature = "bench")]
pub mod bench_forced_ab;
#[cfg(feature = "bench")]
mod bench_gate_check;
#[cfg(feature = "bench")]
pub mod bench_lease;
#[cfg(feature = "bench")]
pub mod bench_paired_ab;
#[cfg(feature = "bench")]
mod bench_print;
#[cfg(feature = "bench")]
pub mod bench_record;
#[cfg(feature = "bench")]
mod bench_resolve;
#[cfg(feature = "bench")]
pub mod bench_run;
#[cfg(feature = "bench")]
mod bench_selfstart;
#[cfg(feature = "bench")]
mod bench_serve_plan;
#[cfg(feature = "bench")]
pub mod bench_spec_cost;
pub(crate) mod blueprint;
pub mod blueprint_args;
mod blueprint_build;
mod blueprint_seal;
pub(crate) mod circuit;
mod circuit_action;
mod circuit_args;
mod circuit_diff;
pub(crate) mod circuit_headroom;
pub(crate) mod circuit_hw;
pub(crate) mod circuit_hw_tree;
mod circuit_memory;
pub(crate) mod circuit_memory_point;
mod circuit_memory_serve;
pub(crate) mod circuit_memory_tables;
mod circuit_memory_weights;
mod circuit_paint;
mod circuit_precision;
mod circuit_rules;
mod circuit_venn;
mod circuit_whatif;
#[cfg(feature = "bench")]
pub(crate) mod doctor;
pub(crate) mod flag_values;
pub(crate) mod hermetic;
pub(crate) mod manifest;
mod serve_args;
mod serve_args_chat_surface;
pub(crate) mod serve_args_dflash;
mod serve_args_mtp;
mod serve_args_mtp_draft;
mod serve_args_prompt_lookup;
pub(crate) mod serve_args_spec_cost;
pub(crate) mod sync_recipes;
mod validate;
mod validate_spec_cost;
#[cfg(feature = "bench")]
pub use bench_args::BenchmarkArgs;
#[cfg(not(feature = "bench"))]
pub(crate) mod no_bench;
pub use blueprint_args::BlueprintArgs;
pub use circuit_args::{
    CircuitDiffArgs, CircuitDisplayArgs, CircuitHwArgs, CircuitHwFormat, CircuitMemoryArgs,
    CircuitMode, CircuitPlanArgs, CircuitPrecision, CircuitPrecisionArgs, CircuitVennArgs,
    ColorChoice,
};
#[cfg(not(feature = "bench"))]
pub use no_bench::BenchmarkArgs;
pub use serve_args::{DEFAULT_KV_CACHE_DTYPE, DEFAULT_NUM_DRAFTS, ServeArgs};
pub use validate::validate_serve_args;

/// 2026-09-26: The release string, e.g. `1.0.0-beta-preview`: the package version from the
/// workspace `Cargo.toml`, which `met --version` prints. Code that records which engine
/// build produced an artifact should read this constant rather than derive its own.
pub const METRALE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser, Debug)]
#[command(
    name = "met",
    version = METRALE_VERSION,
    about = "Metrale Engine — pure Rust LLM inference server"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(clap::Subcommand, Debug)]
pub enum Command {
    /// Start the inference server.
    Serve(ServeArgs),
    /// Run and inspect the benchmark suite, without the dashboard.
    // 2026-10-17: Hidden in a build without the `bench` feature, where it is refused with a
    // message naming the feature (`cli/no_bench.rs`). A plain comment, so `--help` is unchanged.
    #[command(visible_alias = "bench")]
    #[cfg_attr(not(feature = "bench"), command(hide = true))]
    Benchmark(BenchmarkArgs),
    /// Print the serve flag surface as JSON.
    ///
    /// Hidden because it is a build tool, not part of the supported CLI: it
    /// exists so downstream tooling can be generated from clap rather than
    /// transcribed from it. `ServeArgs` has no `Serialize` derive, and this
    /// does not promise that any flag keeps its name: a rename shows up as a
    /// diff in whatever consumes the output.
    #[command(hide = true)]
    DumpServeOptions,
    /// Populate the local recipe index from the recipe repository.
    ///
    /// `benchmark run` resolves a recipe id against this index. The TUI Library
    /// also fills it, but a CI runner, a container or a machine reached over
    /// ssh cannot open the TUI; this command fills it without one.
    ///
    /// It is a separate command rather than an automatic fetch inside
    /// `benchmark run`, so a benchmark never reaches the network mid-run and
    /// its result depends only on what was declared.
    SyncRecipes,
    /// Report whether this box can run a benchmark, and say what to fix.
    ///
    /// Each check covers one cause of the same symptom, `recipe "..." is not in
    /// the local index (0 cached)`: an `~/.metrale` owned by another uid, a
    /// `sync-recipes` that was never run, or a signing identity created in a
    /// scratch METRALE_HOME whose key was never committed.
    ///
    /// Exits non-zero when anything is wrong, so a provisioning script can gate
    /// on it.
    #[cfg_attr(not(feature = "bench"), command(hide = true))]
    Doctor,
    /// Show a recipe's architecture circuit and the fused kernel plan the engine runs.
    ///
    /// The circuits, precision tables and fusion rules are the ones this binary was built
    /// with (`kernels/circuits/`, `kernels/<hw>/common/FUSIONS.toml`).
    Circuit(CircuitArgs),
    /// Build, sign, verify and convert signed blueprints (`.kbp`): one serve, fully stated.
    Blueprint(BlueprintArgs),
}

pub use circuit_action::{CircuitAction, CircuitArgs};

#[cfg(test)]
mod bool_surface_tests;

#[cfg(test)]
mod version_tests;
