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
pub mod bench_spec_cost;
pub(crate) mod bring_up;
mod bring_up_conc;
mod bring_up_core;
mod bring_up_discover;
mod bring_up_energy;
mod bring_up_render;
pub(crate) mod circuit;
mod circuit_args;
mod circuit_diff;
pub(crate) mod circuit_hw;
pub(crate) mod circuit_hw_tree;
mod circuit_memory;
pub(crate) mod circuit_memory_point;
mod circuit_memory_serve;
pub(crate) mod circuit_memory_tables;
mod circuit_memory_weights;
mod circuit_paint;
mod circuit_precision;
mod circuit_venn;
pub(crate) mod doctor;
pub(crate) mod flag_values;
pub(crate) mod hermetic;
pub(crate) mod manifest;
pub(crate) mod ml_utils;
pub(crate) mod ml_utils_calib;
pub(crate) mod ml_utils_io;
mod serve_args;
mod serve_args_mtp;
mod serve_args_mtp_draft;
mod serve_args_prompt_lookup;
pub(crate) mod serve_args_spec_cost;
pub(crate) mod sync_recipes;
mod validate;
mod validate_spec_cost;
pub use bench_args::BenchmarkArgs;
pub use circuit_args::*;
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
    /// with (`kernels/circuits/`, `kernels/<hw>/common/FUSIONS.toml`).
    Circuit(CircuitArgs),
    /// Model utilities: inspect a checkpoint from its metadata, write a mock (rehearsal)
    /// checkpoint that keeps the architecture with fewer layers and synthetic weights,
    /// extrapolate full-model numbers from mock measurements, and bench a model's bring-up
    /// (single-stream TTFT and the concurrency ladder) on any OpenAI-compatible endpoint.
    #[command(visible_alias = "dev-tools")]
    MlUtils(MlUtilsArgs),
}

/// `met ml-utils`: model utilities.
#[derive(clap::Args, Debug)]
pub struct MlUtilsArgs {
    #[command(subcommand)]
    pub action: MlUtilsAction,
}

/// The `met ml-utils` commands.
#[derive(clap::Subcommand, Debug)]
pub enum MlUtilsAction {
    /// Report a checkpoint from its config, quantization metadata and safetensors headers (no
    /// tensor data): arch, layer signatures and their bytes, storage schemes, and for a MoE the
    /// distinct experts a decode step touches per concurrency. With --spec, also the mock plan.
    Inspect(InspectArgs),
    /// Write a mock (rehearsal) checkpoint: the architecture exactly, fewer layers, synthetic
    /// weights (deterministic by seed), in the source's own Hugging Face format.
    Mockify(MockifyArgs),
    /// Estimate a full-model metric from mock measurements (one per resolved spec), and its error
    /// against a full-model measurement when given.
    Extrapolate(ExtrapolateArgs),
    /// Read a sample of every tensor class's stored bytes (a few tensors per class, large ones
    /// in spaced chunks) and write the class bit-pattern statistics a `values.mode = "stats"`
    /// mock samples from.
    ValueStats(ValueStatsArgs),
    /// Fit per-layer router gains from the expert loads a histogram-routed mock produced
    /// (`met serve --mock <spec> --record-routing <file>`), so the next mock built with the
    /// calibration reproduces the profile under its own activations.
    CalibrateRouting(CalibrateRoutingArgs),
    /// Bring-up bench of a served model, vLLM or Metrale alike: single-stream (C=1) cold, warm,
    /// high-ISL cold and high-ISL warm TTFT, run sequentially with prompts cut to exact token
    /// counts of the model's own tokenizer, then the concurrency ladder; prints one table and
    /// writes a JSON record. `--compare A B` renders two records side by side with the winner
    /// per metric and every asymmetry between the runs.
    ModelBringUpBench(BringUpArgs),
}

/// `met ml-utils model-bring-up-bench` options.
#[derive(clap::Args, Debug)]
pub struct BringUpArgs {
    /// The OpenAI-compatible base URL to measure, e.g. http://127.0.0.1:8888. Omitted, the
    /// local `met serve` that confirms its own identity (`/serve-config`) on the port its
    /// argv names; none or several is an error, never a guessed port.
    #[arg(long)]
    pub url: Option<String>,
    /// The model to request. Omitted, the one model the endpoint's /v1/models lists.
    #[arg(long)]
    pub model: Option<String>,
    /// The served model's tokenizer: a tokenizer.json, its checkpoint directory, or a Hub id in
    /// the local cache. Omitted, the discovered local serve's --model-from-path.
    #[arg(long)]
    pub tokenizer: Option<String>,
    /// Samples per TTFT bench. Omitted, each TTFT gate's own default (printed in the plan).
    #[arg(long)]
    pub reps: Option<usize>,
    /// Cold and warm prompt sizes in tokens, comma-separated. Omitted, the TTFT gates' own
    /// `prompt_lengths` default.
    #[arg(long, value_delimiter = ',')]
    pub isl: Option<Vec<usize>>,
    /// High-ISL prompt size in tokens. Omitted, the high-ISL gates' own `min_prompt_tokens`
    /// default (32768).
    #[arg(long)]
    pub high_isl: Option<usize>,
    /// Concurrency rungs. One standard rung N runs every standard rung up to and including N
    /// (standard: 1,2,4,8,12,16,32,64,128; e.g. `--concs 128`); a comma-separated, strictly
    /// increasing list runs exactly those (e.g. `--concs 1,4,16,64,128`).
    #[arg(long, default_value = bring_up_conc::DEFAULT_CONCS_ARG)]
    pub concs: String,
    /// Concurrency ladder prompt size in tokens (the GLM campaign's ladder instrument).
    #[arg(long, default_value_t = 128)]
    pub conc_isl: usize,
    /// Concurrency ladder output tokens per request (the GLM campaign's ladder instrument).
    #[arg(long, default_value_t = 1024)]
    pub conc_osl: usize,
    /// Read GPU-rail energy during the ladder from the NVML counters of these hosts
    /// (`localhost` or ssh destinations, comma-separated; every box the serve runs on). Omitted,
    /// J/tok is reported as not measured.
    #[arg(long, value_delimiter = ',')]
    pub energy: Vec<String>,
    /// Skip the TTFT benches.
    #[arg(long)]
    pub skip_ttft: bool,
    /// Skip the concurrency ladder.
    #[arg(long)]
    pub skip_concurrency: bool,
    /// Where to write the JSON record (created if absent). Required when measuring.
    #[arg(long, required_unless_present = "compare")]
    pub out: Option<std::path::PathBuf>,
    /// The engine's name in the table and the record. Omitted, the kind the endpoint reports
    /// (metrale, vllm or unknown).
    #[arg(long)]
    pub label: Option<String>,
    /// Render two records (A B) side by side instead of measuring.
    #[arg(long, num_args = 2, value_names = ["A", "B"])]
    pub compare: Option<Vec<std::path::PathBuf>>,
}

/// `met ml-utils value-stats` options.
#[derive(clap::Args, Debug)]
pub struct ValueStatsArgs {
    /// The checkpoint: a directory, or a Hub id (org/name) in the local cache.
    #[arg(long)]
    pub checkpoint: String,
    /// Where to write the statistics (JSON); it must not exist.
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Read from huggingface.co when the checkpoint is not cached (byte ranges only).
    #[arg(long)]
    pub allow_network: bool,
}

/// `met ml-utils calibrate-routing` options.
#[derive(clap::Args, Debug)]
pub struct CalibrateRoutingArgs {
    /// The source checkpoint the mock was made from.
    #[arg(long)]
    pub checkpoint: String,
    /// The histogram-routed mock spec the loads were recorded on (its own calibration, if any,
    /// is the starting point).
    #[arg(long)]
    pub spec: std::path::PathBuf,
    /// The recorded expert loads of that mock (one row per mock MoE layer).
    #[arg(long)]
    pub measured: std::path::PathBuf,
    /// Where to write the calibration (JSON); it must not exist.
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Read the source's metadata and headers from huggingface.co when it is not cached.
    #[arg(long)]
    pub allow_network: bool,
}

/// `met ml-utils inspect` options.
#[derive(clap::Args, Debug)]
pub struct InspectArgs {
    /// The checkpoint: a directory, or a Hub id (org/name) in the local cache.
    #[arg(long)]
    pub checkpoint: String,
    /// A mock spec (TOML) to plan and report.
    #[arg(long)]
    pub spec: Option<std::path::PathBuf>,
    /// Read the checkpoint's metadata and headers from huggingface.co when it is not cached.
    #[arg(long)]
    pub allow_network: bool,
    /// Print JSON instead of text.
    #[arg(long)]
    pub json: bool,
}

/// `met ml-utils mockify` options.
#[derive(clap::Args, Debug)]
pub struct MockifyArgs {
    /// The source checkpoint: a directory, or a Hub id (org/name).
    #[arg(long)]
    pub checkpoint: String,
    /// The mock spec (TOML). Every key is required; see `metrale_ml_utils::spec`.
    #[arg(long)]
    pub spec: std::path::PathBuf,
    /// The output directory; it must not exist.
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Read the source's metadata and headers from huggingface.co when it is not cached.
    #[arg(long)]
    pub allow_network: bool,
}

/// `met ml-utils extrapolate` options.
#[derive(clap::Args, Debug)]
pub struct ExtrapolateArgs {
    /// One mock measurement: `<mock dir or mock.resolved.toml>=<value>`. Give at least one more
    /// than the source has layer signatures, varying each signature's count on its own.
    #[arg(long = "point", required = true)]
    pub points: Vec<String>,
    /// How the metric scales with layers: `rate` (tok/s: its reciprocal is affine) or `linear`
    /// (J/tok at a fixed concurrency, TTFT).
    #[arg(long)]
    pub scaling: String,
    /// The full model's measured value, to report the estimate's error.
    #[arg(long)]
    pub full: Option<f64>,
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
