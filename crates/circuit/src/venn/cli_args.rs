// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The report's runs and regenerating command, from the `met circuit venn`
//! arguments. Here, not in the CLI, so the CLI and the tests that check a report is current build
//! the same header from the same arguments.
//!
//! Owner: metrale-circuit (venn).
//! Invariants: decode and draft plan one row; multi_seq plans every requested count above one;
//! verify plans the requested K. A mode left with no row count is an error, never skipped.

use super::{Run, VennError};
use crate::rules::Mode;

/// 2026-09-29: What the CLI was asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VennArgs {
    /// 2026-09-29: `--target`, as given.
    pub target: String,
    /// 2026-09-29: `--against`, in order.
    pub against: Vec<String>,
    /// 2026-09-29: `--mode`, in order.
    pub modes: Vec<Mode>,
    /// 2026-09-29: `--rows`: the concurrency rungs.
    pub rows: Vec<u64>,
    /// 2026-09-29: `--verify-rows`: K for the MTP verify.
    pub verify_rows: Vec<u64>,
    /// 2026-09-29: `--out`, repo-relative.
    pub out: String,
}

impl VennArgs {
    /// 2026-09-29: The runs, in mode then row order.
    pub fn runs(&self) -> Result<Vec<Run>, VennError> {
        let mut out = Vec::new();
        for &mode in &self.modes {
            let rows: Vec<u64> = match mode {
                Mode::Decode | Mode::Draft => vec![1],
                Mode::MultiSeq => self.rows.iter().copied().filter(|&r| r > 1).collect(),
                Mode::Verify => self.verify_rows.clone(),
                // 2026-09-30: A batched verify is planned for a row table, which the Venn does
                // not take.
                Mode::VerifyBatch => {
                    return Err(VennError::Run(
                        "--mode verify_batch is not classified (its plans take a row table)".into(),
                    ));
                }
            };
            if rows.is_empty() || rows.contains(&0) {
                return Err(VennError::Run(format!(
                    "--mode {} has no row count to plan (multi_seq takes --rows above 1, verify --verify-rows)",
                    mode.name()
                )));
            }
            out.extend(rows.into_iter().map(|rows| Run { mode, rows }));
        }
        Ok(out)
    }

    /// 2026-09-29: The command line that regenerates the report.
    pub fn command(&self) -> String {
        let join = |v: &[u64]| v.iter().map(u64::to_string).collect::<Vec<_>>().join(",");
        let modes: Vec<&str> = self.modes.iter().map(|m| m.name()).collect();
        format!(
            "met circuit venn --target {} --against {} --mode {} --rows {} --verify-rows {} --out {}",
            self.target,
            self.against.join(","),
            modes.join(","),
            join(&self.rows),
            join(&self.verify_rows),
            self.out
        )
    }
}
