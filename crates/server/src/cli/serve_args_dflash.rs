// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: `--dflash-adaptive-k` and `--draft-tp-size`, flattened into
//! `ServeSchedulingArgs` after `--dflash-draft-ladder`. Both are parsed here and resolved
//! against the drafter and the topology once, by `serve_phases::apply_dflash_gamma`, which is
//! their only reader.
//!
//! Owner: server CLI.
//! Invariants: the `///` text on the struct's fields is the `--help` output and carries no
//! date.

use clap::Args;

/// 2026-10-10: `--draft-tp-size`'s value: the serve's world size, or a rank count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DraftTpSize {
    Auto,
    Ranks(usize),
}

impl DraftTpSize {
    /// 2026-10-10: The rank count on a serve of `world` ranks.
    pub fn ranks(self, world: usize) -> usize {
        match self {
            Self::Auto => world,
            Self::Ranks(n) => n,
        }
    }
}

/// 2026-10-10: `auto`, or a rank count of at least 1.
pub fn parse_draft_tp_size(s: &str) -> Result<DraftTpSize, String> {
    if s == "auto" {
        return Ok(DraftTpSize::Auto);
    }
    match s.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(DraftTpSize::Ranks(n)),
        _ => Err(format!(
            "expected `auto` or a rank count of at least 1, got {s:?}"
        )),
    }
}

#[derive(Args, Debug, Clone, PartialEq)]
pub struct ServeDflashArgs {
    /// Adaptive DFlash draft count for a single stream: the measured step cost in
    /// ms by draft count, `drafts:ms` pairs over consecutive counts from 0 (the plain decode
    /// step) or 1, e.g. `0:32,1:44,2:50,3:56`. Each C = 1 step then verifies the count with the
    /// most expected tokens per ms at the sequence's measured acceptance, capped by the ladder
    /// rung (or γ - 1), and with a `0:` entry suspends speculation (until the re-probe) where
    /// plain decode pays more. Unset (the default): the ladder or the γ resolver decides.
    #[arg(long, requires = "dflash")]
    pub dflash_adaptive_k: Option<String>,

    /// 2026-10-10: `dflash_adaptive_k` parsed and checked against the resolved γ by
    /// `serve_phases::apply_dflash_gamma`.
    #[arg(skip)]
    pub dflash_adaptive_k_resolved: Option<metrale_speculative::dflash_adaptive_k::CostModel>,

    /// Tensor-parallel degree of the DFlash drafter: `auto` (the serve's world size) or a
    /// rank count. Above 1 every rank holds a slice of the drafter (its KV heads with their
    /// query groups, a slice of the MLP intermediate and of the LM-head rows) and each
    /// propose runs on all ranks at once: two all-reduces per drafter layer and one
    /// all-gather of the top-16 candidates, the drafts equal to one rank's up to the
    /// all-reduce summation order. 1 runs the whole drafter on rank 0. The drafter's
    /// collectives span the whole communicator, so the value must be 1 or the world size,
    /// and the drafter's KV heads must cover every rank; startup refuses anything else. Set
    /// the same value on every rank. Unset (the default): 1.
    #[arg(long, requires = "dflash", value_name = "N|auto", value_parser = parse_draft_tp_size)]
    pub draft_tp_size: Option<DraftTpSize>,

    /// 2026-10-10: `draft_tp_size` resolved against the world size and the drafter's heads by
    /// `serve_phases::apply_dflash_gamma` (1 when unset).
    #[arg(skip)]
    pub draft_tp_size_resolved: Option<usize>,
}
