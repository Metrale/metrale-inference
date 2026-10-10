// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The `met accuracy` argument types (split from `cli.rs` under the 500-line rule).
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

/// `met accuracy`: kernel accuracy contracts.
#[derive(clap::Args, Debug)]
pub struct AccuracyArgs {
    #[command(subcommand)]
    pub action: AccuracyAction,
}

/// The `met accuracy` actions.
#[derive(clap::Subcommand, Debug)]
pub enum AccuracyAction {
    /// List every (family, point, shape) the models in kernels/circuits/INSTANCES.toml run on a
    /// hardware class, with the contract that covers it or the reason none does. CPU only.
    Points(AccuracySelectArgs),
    /// Run the contracts on this binary's GPU kernels and write the record; exits non-zero
    /// unless every check passes (good arm inside its contract, every mutation caught).
    Check(AccuracyRunArgs),
    /// As `check`, and also print the `[[contract.calibration]]` rows the passing checks
    /// measured, for review into ACCURACY.toml.
    Calibrate(AccuracyRunArgs),
    /// Judge a model-level accuracy check over two engine-neutral logprob dumps (teacher-forced
    /// prompt logprobs `tf`, greedy decode at one and four rows `dec1`/`dec4`, top-k per
    /// position, taken over the OpenAI-compatible API): `exact` requires byte equality,
    /// `numerics` judges top-1, top-k KL, |dlogprob| p99 and the divergence margin against the
    /// limits given. Exits non-zero on a failure.
    Model(AccuracyModelArgs),
    /// The kernel envelope sweep: time every contracted candidate at every projection cell the
    /// described models need (plus a margin), each judged by its own contract.
    Envelope(EnvelopeArgs),
}

#[path = "envelope_args.rs"]
mod envelope_args;
pub use envelope_args::{EnvelopeAction, EnvelopeArgs, EnvelopeGridArgs, EnvelopeSweepArgs};

/// `met accuracy model` options.
#[derive(clap::Args, Debug, Clone)]
pub struct AccuracyModelArgs {
    /// The pinned reference dump (JSON).
    #[arg(long = "ref")]
    pub reference: std::path::PathBuf,
    /// SHA-256 the reference file must have.
    #[arg(long = "ref-sha256")]
    pub reference_sha256: String,
    /// The dump under test (JSON, same corpus).
    #[arg(long)]
    pub test: std::path::PathBuf,
    /// `exact` (bit-identical levers) or `numerics` (precision-changing levers).
    #[arg(long)]
    pub mode: String,
    /// numerics: least tf top-1 agreement.
    #[arg(long)]
    pub tf_min_top1: Option<f64>,
    /// numerics: largest tf mean KL.
    #[arg(long)]
    pub tf_max_kl: Option<f64>,
    /// numerics: largest tf p99 |dlogprob|.
    #[arg(long)]
    pub tf_max_dlp_p99: Option<f64>,
    /// numerics: largest decode mean KL.
    #[arg(long)]
    pub dec_max_kl: Option<f64>,
    /// numerics: largest decode p99 |dlogprob|.
    #[arg(long)]
    pub dec_max_dlp_p99: Option<f64>,
    /// numerics: largest reference top-1/top-2 margin at a decode divergence.
    #[arg(long)]
    pub max_divergence_margin: Option<f64>,
    /// numerics: most decode divergences without a measurable margin.
    #[arg(long)]
    pub max_unmeasured_divergences: Option<f64>,
}

/// `met accuracy points` options.
#[derive(clap::Args, Debug, Clone)]
pub struct AccuracySelectArgs {
    /// Hardware class (`kernels/<class>/`).
    #[arg(long)]
    pub hardware: String,
    /// Only this family (a KERNEL_FAMILIES.toml id).
    #[arg(long)]
    pub family: Option<String>,
    /// Only points a recipe or checkpoint containing this text runs.
    #[arg(long)]
    pub model: Option<String>,
    /// Repository root; default: the first ancestor of the working directory holding
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met accuracy check|calibrate` options.
#[derive(clap::Args, Debug, Clone)]
pub struct AccuracyRunArgs {
    #[command(flatten)]
    pub select: AccuracySelectArgs,
    /// `quick` (every distinct shape on the gaussian class with its mutations, the adversarial
    /// classes at the largest shape) or `full` (every point on every class).
    #[arg(long)]
    pub scope: String,
    /// Directory the record is written to (`<out>/<hardware>/<closure>.toml`).
    #[arg(long)]
    pub out: std::path::PathBuf,
}
