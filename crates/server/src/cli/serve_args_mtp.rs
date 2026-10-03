// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The serve flags that shape MTP verify rounds per batch width
//! (`--mtp-k-ladder`, `--mtp-dcut-ratio`), flattened into `ServeSchedulingArgs` after
//! `--num-drafts`.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and
//! carries no date.

use clap::Args;

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeMtpShapeArgs {
    /// MTP drafts per verify at each batch width, as `n_max:drafts` steps (default:
    /// `4:3,8:3,16:1,32:1`). A step applies up to `n_max` active sequences; wider
    /// batches take the last step. Counts are capped at `--num-drafts`. Example:
    /// `2:2,4:1,32:1` drafts twice at one or two sequences and once from three up.
    /// Setting it also pins the adaptive rung to these counts. The verify pools are
    /// sized from it.
    ///
    /// Environment fallback: `METRALE_MTP_K_LADDER` when this flag is absent.
    #[arg(long, value_name = "STEPS")]
    pub mtp_k_ladder: Option<String>,

    /// D-Cut retention ratio for batched MTP verify (default: 0.75): the fraction
    /// of the prunable draft positions (depth 2 and deeper) a batched verify keeps,
    /// ranked by the drafter's confidence, snapped to 0.25, 0.5, 0.75 or 1.0; 1.0
    /// keeps every draft. It acts only on steps with 2 or more drafts per sequence,
    /// so a `--num-drafts 1` serve is unaffected. Measured on GB10,
    /// Qwen3.6-35B-A3B-FP8, `--num-drafts 3`: 1.0 against 0.75 is +1.9% tok/s at
    /// C2 and +4.3% at C4, with the same verify time.
    ///
    /// Environment fallback: `METRALE_MTP_DCUT_RATIO` when this flag is absent.
    #[arg(long, value_name = "RATIO")]
    pub mtp_dcut_ratio: Option<f32>,
}
