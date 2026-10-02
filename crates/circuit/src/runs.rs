// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The row table of a batched MTP verify (`Mode::VerifyBatch`) and the per-run
//! launches a `repeat = "per_run"` rule selects. A batched verify checks `n` sequences
//! together, sequence `i` with `k_i` rows, seq-major. The GatedDeltaNet layers launch once per
//! run: a maximal group of adjacent sequences with equal `k`. Model-engine
//! `batched_conv_gdn_route` forms the runs, and so does [`RowTable::from_seqs`]. Each run is
//! contiguous when its sequences' state slots are consecutive, which the batched kernels need.
//! A table is carried when the verify runs the carried-state GDN kernels (`gdn_carry_begin`),
//! which it does whenever the WY tables are staged.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A row table has at least one run; every run has `k >= 1` and `n >= 1`; adjacent runs differ
//!   in `k` (runs are maximal).
//! - A per-run rule applies to a table only when every run matches one of its selectors; the
//!   first matching selector, in file order, gives the run's launches.

use std::fmt;

use crate::rules::KernelId;

/// 2026-09-30: One run of a batched verify.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VerifyRun {
    /// 2026-09-30: Rows per sequence.
    pub k: u64,
    /// 2026-09-30: Sequences.
    pub n: u64,
    /// 2026-09-30: The sequences' state slots are consecutive in batch order.
    pub contiguous: bool,
}

/// 2026-09-30: The runs of a batched verify, in batch order.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RowTable {
    /// 2026-09-30: The runs.
    pub runs: Vec<VerifyRun>,
    /// 2026-09-30: The verify runs the carried-state GDN kernels.
    pub carried: bool,
}

/// 2026-09-30: Why a row table was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RowTableError {
    /// 2026-09-30: No sequences.
    #[error("a row table needs at least one sequence")]
    Empty,
    /// 2026-09-30: A sequence with no rows, or a run with no sequences.
    #[error("row table: {0}")]
    Shape(String),
    /// 2026-09-30: Text that is not `<k>x<n>[!]` runs.
    #[error("row table `{0}`: runs are `<k>x<n>`, `!` marking a fragmented run")]
    Parse(String),
}

impl RowTable {
    /// 2026-09-30: The table of sequences with `ks` rows each, in batch order; `contiguous` says
    /// for a run (first sequence, sequences) whether its slots are consecutive.
    pub fn from_seqs(
        ks: &[u64],
        carried: bool,
        contiguous: impl Fn(usize, usize) -> bool,
    ) -> Result<Self, RowTableError> {
        if ks.is_empty() {
            return Err(RowTableError::Empty);
        }
        if ks.contains(&0) {
            return Err(RowTableError::Shape("a sequence with no rows".into()));
        }
        let mut runs = Vec::new();
        let mut g0 = 0;
        while g0 < ks.len() {
            let g1 = (g0..ks.len())
                .find(|&i| ks[i] != ks[g0])
                .unwrap_or(ks.len());
            runs.push(VerifyRun {
                k: ks[g0],
                n: (g1 - g0) as u64,
                contiguous: g1 - g0 > 1 && contiguous(g0, g1 - g0),
            });
            g0 = g1;
        }
        Ok(RowTable { runs, carried })
    }

    /// 2026-09-30: Parse [`RowTable::text`].
    pub fn parse(s: &str) -> Result<Self, RowTableError> {
        let bad = || RowTableError::Parse(s.to_string());
        let mut runs = Vec::new();
        let body = s.strip_prefix(UNCARRIED);
        let carried = body.is_none();
        for tok in body.unwrap_or(s).split_whitespace() {
            let (body, contiguous) = match tok.strip_suffix('!') {
                Some(b) => (b, false),
                None => (tok, true),
            };
            let (k, n) = body.split_once('x').ok_or_else(bad)?;
            let (k, n) = (k.parse().map_err(|_| bad())?, n.parse().map_err(|_| bad())?);
            runs.push(VerifyRun { k, n, contiguous });
        }
        let t = RowTable { runs, carried };
        t.check()?;
        Ok(t)
    }

    fn check(&self) -> Result<(), RowTableError> {
        if self.runs.is_empty() {
            return Err(RowTableError::Empty);
        }
        if self.runs.iter().any(|r| r.k == 0 || r.n == 0) {
            return Err(RowTableError::Shape(
                "a run with no rows or no sequences".into(),
            ));
        }
        // 2026-09-30: The batched arm needs two sequences (`multi_run_arm` declines one).
        if self.runs.iter().any(|r| r.n == 1 && r.contiguous) {
            return Err(RowTableError::Shape(format!(
                "`{self}`: a run of one sequence is never batched; mark it `!`"
            )));
        }
        if self.runs.windows(2).any(|w| w[0].k == w[1].k) {
            return Err(RowTableError::Shape(format!(
                "`{self}`: adjacent runs of equal k are one run"
            )));
        }
        Ok(())
    }

    /// 2026-09-30: Rows, `Σ k·n`.
    pub fn rows(&self) -> u64 {
        self.runs.iter().map(|r| r.k * r.n).sum()
    }

    /// 2026-09-30: Sequences, `Σ n`.
    pub fn seqs(&self) -> u64 {
        self.runs.iter().map(|r| r.n).sum()
    }

    /// 2026-09-30: `<k>x<n>` per run, space-separated, `!` after a fragmented run, the whole
    /// prefixed `uncarried: ` when the verify does not carry.
    pub fn text(&self) -> String {
        self.to_string()
    }
}

impl fmt::Display for RowTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let parts: Vec<String> = self
            .runs
            .iter()
            .map(|r| format!("{}x{}{}", r.k, r.n, if r.contiguous { "" } else { "!" }))
            .collect();
        if !self.carried {
            f.write_str(UNCARRIED)?;
        }
        f.write_str(&parts.join(" "))
    }
}

/// 2026-09-30: The text prefix of a table that does not carry.
const UNCARRIED: &str = "uncarried: ";

/// 2026-09-30: How often a per-run launch (or copy) happens within its run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Times {
    /// 2026-09-30: Once for the run.
    Once,
    /// 2026-09-30: Once per sequence.
    PerSeq,
    /// 2026-09-30: Once per row (`k·n`).
    PerRow,
    /// 2026-09-30: Once per row of each sequence but its last (`(k - 1)·n`).
    PerSeqRowButLast,
}

impl Times {
    /// 2026-09-30: The count in `run`.
    pub fn count(self, run: VerifyRun) -> u64 {
        match self {
            Times::Once => 1,
            Times::PerSeq => run.n,
            Times::PerRow => run.k * run.n,
            Times::PerSeqRowButLast => run.k.saturating_sub(1) * run.n,
        }
    }

    /// 2026-09-30: The spelling in FUSIONS.toml.
    pub fn name(self) -> &'static str {
        match self {
            Times::Once => "once",
            Times::PerSeq => "per_seq",
            Times::PerRow => "per_row",
            Times::PerSeqRowButLast => "per_seq_row_but_last",
        }
    }

    /// 2026-09-30: Parse [`Times::name`].
    pub fn parse(s: &str) -> Option<Self> {
        [
            Times::Once,
            Times::PerSeq,
            Times::PerRow,
            Times::PerSeqRowButLast,
        ]
        .into_iter()
        .find(|t| t.name() == s)
    }
}

/// 2026-09-30: One `[[rule.run]]` selector: the runs it serves and what it launches for one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunSelect {
    /// 2026-09-30: Inclusive `k` range.
    pub k: (u64, u64),
    /// 2026-09-30: Inclusive sequence-count range.
    pub n: (u64, u64),
    /// 2026-09-30: The contiguity it serves; `None` for either.
    pub contiguous: Option<bool>,
    /// 2026-09-30: Whether the verify carries; `None` for either.
    pub carried: Option<bool>,
    /// 2026-09-30: Kernel launches, in the order the emitter makes them.
    pub launches: Vec<(KernelId, Times)>,
    /// 2026-09-30: Copy-engine transfers besides them, and how often.
    pub copies: Option<Times>,
}

impl RunSelect {
    /// 2026-09-30: Whether this selector serves `run` of a verify that carries or not.
    pub fn serves(&self, run: VerifyRun, carried: bool) -> bool {
        (self.k.0..=self.k.1).contains(&run.k)
            && (self.n.0..=self.n.1).contains(&run.n)
            && self.contiguous.is_none_or(|c| c == run.contiguous)
            && self.carried.is_none_or(|c| c == carried)
    }
}

/// 2026-09-30: The launches one run of a per-run group makes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunLaunches {
    /// 2026-09-30: The run.
    pub run: VerifyRun,
    /// 2026-09-30: Its launches, in order.
    pub launches: Vec<(KernelId, Times)>,
    /// 2026-09-30: Its copies.
    pub copies: Option<Times>,
}

impl RunLaunches {
    /// 2026-09-30: Kernel launches this run makes.
    pub fn launch_count(&self) -> u64 {
        self.launches.iter().map(|(_, t)| t.count(self.run)).sum()
    }

    /// 2026-09-30: Copies this run makes.
    pub fn copy_count(&self) -> u64 {
        self.copies.map_or(0, |t| t.count(self.run))
    }
}

/// 2026-09-30: Each run of `table` resolved against `selects`: the first selector that serves
/// it; `None` when some run has none.
pub fn resolve_runs(table: &RowTable, selects: &[RunSelect]) -> Option<Vec<RunLaunches>> {
    table
        .runs
        .iter()
        .map(|&run| {
            selects
                .iter()
                .find(|s| s.serves(run, table.carried))
                .map(|s| RunLaunches {
                    run,
                    launches: s.launches.clone(),
                    copies: s.copies,
                })
        })
        .collect()
}

#[cfg(test)]
#[path = "runs_tests.rs"]
mod runs_tests;
