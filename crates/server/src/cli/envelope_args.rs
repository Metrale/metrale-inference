// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The `met accuracy envelope` argument types: the kernel envelope sweep (time every
//! contracted candidate entry point at every projection cell the described models need, each
//! judged by its own accuracy contract) and the SCHEDULES.toml generator.
//!
//! Owner: server CLI.
//! Invariants: every timing and thermal knob is stated on the command line (no defaults).

/// `met accuracy envelope`: the kernel envelope sweep.
#[derive(clap::Args, Debug)]
pub struct EnvelopeArgs {
    #[command(subcommand)]
    pub action: EnvelopeAction,
}

/// The `met accuracy envelope` actions.
#[derive(clap::Subcommand, Debug)]
pub enum EnvelopeAction {
    /// Print the grid (cells, candidates, defaults, shards) without a GPU.
    Grid(EnvelopeGridArgs),
    /// Time and judge every candidate of this shard's cells on this box's GPU; appends one JSON
    /// record per (cell, candidate) to --out and skips records already there (resumable).
    Sweep(EnvelopeSweepArgs),
    /// Select the winners from the sweep's records and write SCHEDULES.toml and its report
    /// (byte-identical winners enabled by default, numerics-changing ones opt-in only). CPU only.
    Schedules(EnvelopeSchedulesArgs),
    /// Check FUSIONS.toml's row ranges against SCHEDULES.toml and print the report. CPU only.
    Fusions(EnvelopeFusionsArgs),
}

/// `met accuracy envelope schedules` options.
#[derive(clap::Args, Debug, Clone)]
pub struct EnvelopeSchedulesArgs {
    /// Hardware class (`kernels/<class>/`).
    #[arg(long)]
    pub hardware: String,
    /// The sweep's record files (comma-separated; every box's).
    #[arg(long, value_delimiter = ',', required = true)]
    pub records: Vec<std::path::PathBuf>,
    /// Where to write SCHEDULES.toml (normally kernels/<hw>/common/SCHEDULES.toml).
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Where to write the selection report (Markdown).
    #[arg(long)]
    pub report: std::path::PathBuf,
    /// Repository root; default: the first ancestor of the working directory holding
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met accuracy envelope fusions` options.
#[derive(clap::Args, Debug, Clone)]
pub struct EnvelopeFusionsArgs {
    /// Hardware class (`kernels/<class>/`).
    #[arg(long)]
    pub hardware: String,
    /// The SCHEDULES.toml to check against.
    #[arg(long)]
    pub schedules: std::path::PathBuf,
    /// Repository root; default as for `schedules`.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// What the grid covers.
#[derive(clap::Args, Debug, Clone)]
pub struct EnvelopeGridArgs {
    /// Hardware class (`kernels/<class>/`).
    #[arg(long)]
    pub hardware: String,
    /// Add the x0.5 / x2 margin corners of every served shape (after every served cell).
    #[arg(long)]
    pub margin: bool,
    /// This box's shard index (0-based).
    #[arg(long)]
    pub shard: usize,
    /// Shards in all (boxes).
    #[arg(long)]
    pub shards: usize,
    /// Only candidates of these families (comma-separated KERNEL_FAMILIES.toml ids).
    #[arg(long, value_delimiter = ',')]
    pub families: Vec<String>,
    /// Repository root; default: the first ancestor of the working directory holding
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met accuracy envelope sweep` options.
#[derive(clap::Args, Debug, Clone)]
pub struct EnvelopeSweepArgs {
    #[command(flatten)]
    pub grid: EnvelopeGridArgs,
    /// The JSON-lines record file (appended; existing non-throttled records are skipped).
    #[arg(long)]
    pub out: std::path::PathBuf,
    /// Untimed launches before the timed repetitions.
    #[arg(long)]
    pub warmup: usize,
    /// Launches per timed repetition.
    #[arg(long)]
    pub iters: usize,
    /// Timed repetitions (their median is the record's time).
    #[arg(long)]
    pub reps: usize,
    /// Before each cell, wait while the GPU is hotter than this (Celsius)...
    #[arg(long)]
    pub max_temp_c: u32,
    /// ...until it has cooled to this.
    #[arg(long)]
    pub resume_temp_c: u32,
    /// Stop cleanly (after the current candidate) at this UNIX time.
    #[arg(long)]
    pub deadline_unix: u64,
}
