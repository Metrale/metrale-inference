// SPDX-License-Identifier: MIT OR Apache-2.0

//! The `met circuit lkb` arguments. Their doc comments are the `--help` text, so they carry no
//! date (`.comment-ttl.toml` exempts this file).

use crate::cli::CircuitPrecision;

/// `met circuit lkb` options.
#[derive(clap::Args, Debug, Clone)]
pub struct CircuitLkbArgs {
    /// The model: a checkpoint id (org/name), a checkpoint directory, or a recipe id.
    #[arg(long)]
    pub checkpoint: String,
    /// Target device id from kernels/DEVICES.toml; the LKB is read on its kernel class.
    #[arg(long)]
    pub hardware: String,
    /// Formats served: the recipe's pinned formats or the checkpoint's declared ones.
    #[arg(long, value_enum)]
    pub precision: CircuitPrecision,
    /// What to print: the Markdown report, or the campaign-ledger fields as TOML.
    #[arg(long, value_enum, default_value_t = CircuitLkbFormat::Report)]
    pub format: CircuitLkbFormat,
    /// Fetch config.json / hf_quant_config.json from huggingface.co when the checkpoint is not
    /// in the local cache.
    #[arg(long)]
    pub allow_network: bool,
    /// Repository root; by default the nearest directory above the working directory that has
    /// kernels/circuits/INSTANCES.toml.
    #[arg(long)]
    pub root: Option<std::path::PathBuf>,
}

/// `met circuit lkb --format`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum CircuitLkbFormat {
    /// The Markdown report.
    #[value(name = "report")]
    Report,
    /// The campaign-ledger fields (`lkb_coverage_pct`, `residual_count`, ...).
    #[value(name = "toml")]
    Toml,
}
