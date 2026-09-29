// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `met` command line: `Cli` and its subcommands. The argument structs are
//! in the `cli/` modules.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use clap::Parser;

pub mod bench_aggregate;
mod bench_args;
pub mod bench_card;
pub(crate) mod bench_cause;
pub mod bench_certify;
mod bench_gate_check;
pub mod bench_lease;
mod bench_print;
pub mod bench_record;
mod bench_resolve;
pub mod bench_run;
mod bench_selfstart;
mod bench_serve_plan;
pub(crate) mod circuit;
mod circuit_diff;
mod circuit_paint;
pub(crate) mod doctor;
pub(crate) mod flag_values;
pub(crate) mod hermetic;
pub(crate) mod manifest;
mod serve_args;
pub(crate) mod sync_recipes;
mod validate;
pub use bench_args::BenchmarkArgs;
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
    #[command(visible_alias = "bench")]
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
    Doctor,
    /// Show a recipe's architecture circuit and the fused kernel plan the engine runs.
    ///
    /// The circuits, precision tables and fusion rules are the ones this binary was built
    /// with (kernels/circuits/, kernels/<hw>/common/FUSIONS.toml).
    Circuit(CircuitArgs),
}

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
}

/// `met circuit diff` options.
#[derive(clap::Args, Debug)]
pub struct CircuitDiffArgs {
    /// Decode steps compared per prompt.
    #[arg(long)]
    pub steps: usize,
    /// Prompts: synthetic token sequences from a fixed generator, each a different length.
    #[arg(long)]
    pub prompts: usize,
    /// Where to write the JSON report.
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Batch widths (comma-separated): diff multi-sequence decode at each width instead of
    /// single-sequence decode. `--prompts` is unused then; a width of `n` decodes `n` prompts.
    #[arg(long, value_delimiter = ',')]
    pub batch: Vec<usize>,
    /// The serve the model is built with. `--forward` is ignored: the diff runs every forward.
    #[command(flatten)]
    pub serve: ServeArgs,
}

/// Which plan to show.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitPlanArgs {
    /// Recipe id, e.g. qwen3.8/qwen3.8-27b-nvfp4-unsloth.
    #[arg(long)]
    pub recipe: String,
    /// The forward to plan.
    #[arg(long, value_enum, default_value_t = CircuitMode::Decode)]
    pub mode: CircuitMode,
    /// Padded rows: the batch rung for multi_seq, K for verify. Required for those two
    /// modes; decode and draft plan one row.
    #[arg(long)]
    pub rows: Option<u64>,
}

/// `met circuit display` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitDisplayArgs {
    #[command(flatten)]
    pub plan: CircuitPlanArgs,
    /// Expand one layer, with its module bindings.
    #[arg(long, conflicts_with = "all_layers")]
    pub layer: Option<usize>,
    /// Draw every layer instead of one diagram per distinct layer plan.
    #[arg(long)]
    pub all_layers: bool,
    /// Draw with 7-bit ASCII only.
    #[arg(long)]
    pub ascii: bool,
    /// When to colour: auto colours a terminal only; NO_COLOR always wins.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,
}

/// A forward `met circuit` can plan.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitMode {
    /// One sequence, one row.
    #[value(name = "decode")]
    Decode,
    /// Many sequences, one row each, padded to the batch ladder.
    #[value(name = "multi_seq")]
    MultiSeq,
    /// K draft rows of one sequence.
    #[value(name = "verify")]
    Verify,
    /// The MTP draft head.
    #[value(name = "draft")]
    Draft,
}

/// `--color`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorChoice {
    /// Colour when stdout is a terminal.
    Auto,
    /// Colour even when piped.
    Always,
    /// Never colour.
    Never,
}

#[cfg(test)]
mod bool_surface_tests;

#[cfg(test)]
mod version_tests {
    use super::*;

    #[test]
    fn version_flag_reports_the_packaged_version() {
        // 2026-09-26: `--version` ends parsing early, so clap returns it as an error whose
        // kind is DisplayVersion and whose rendering is the output.
        let err = Cli::try_parse_from(["met", "--version"]).expect_err("exits early");
        assert_eq!(err.kind(), clap::error::ErrorKind::DisplayVersion);
        assert!(
            err.to_string().contains(METRALE_VERSION),
            "`--version` printed {:?}, which does not carry {METRALE_VERSION}",
            err.to_string()
        );
    }

    #[test]
    fn the_reported_version_is_the_cargo_version() {
        // 2026-09-26: The constant is the package version itself, not a literal copy.
        assert_eq!(METRALE_VERSION, env!("CARGO_PKG_VERSION"));
        assert!(!METRALE_VERSION.is_empty());
    }
}
