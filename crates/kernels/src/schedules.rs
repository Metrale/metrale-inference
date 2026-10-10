// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The kernel envelope sweep's schedules, baked by `build.rs` from
//! `kernels/<hw>/common/SCHEDULES.toml` into [`TARGET_SCHEDULES`], and the lookup over them.
//!
//! NOTHING IN THE ENGINE CONSUMES THIS TABLE YET. No router calls [`lookup`], and no flag, env
//! var or config passes `opt_in = true`, so baking a SCHEDULES.toml changes no kernel choice and
//! no output bit. Wiring a router to it (and adding the opt-in lever that sets `opt_in`) is a
//! later, separately certified change.
//!
//! Owner: kernels crate (envelope schedules).
//! Invariants:
//! - An `Enabled::Default` entry is `Numerics::Same` or `Numerics::BitIdentical` (the build
//!   refuses anything else), so a default lookup never changes output bytes.
//! - No two entries of one (op, weight, activation, k, n) overlap in rows (the build refuses it),
//!   so [`lookup`] has at most one answer.
//! - A family whose sources changed since the sweep has no entries; its name is in
//!   [`TARGET_SCHEDULES_STALE`].

pub use crate::{TARGET_SCHEDULES, TARGET_SCHEDULES_STALE};

/// 2026-10-10: How a schedule's winner relates to today's routed entry point at its cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Numerics {
    /// The winner is the default.
    Same,
    /// Another entry point whose output bytes equal the default's on every input class.
    BitIdentical,
    /// Faster but numerics-changing (tile, split-K or reduction order).
    Differs,
    /// No default exists at this cell (no served plan has one).
    New,
}

/// 2026-10-10: Whether a schedule applies by default or only to a caller that opted in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Enabled {
    Default,
    OptIn,
}

/// 2026-10-10: One baked `[[schedule]]` entry: the winning entry point for `op` on a
/// `weight`/`activation` format pair at shape `k x n`, for every row count in
/// `rows_lo..=rows_hi`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Schedule {
    pub op: &'static str,
    pub weight: &'static str,
    pub activation: &'static str,
    pub k: u32,
    pub n: u32,
    pub rows_lo: u32,
    pub rows_hi: u32,
    /// `module::entry_point` of the winner.
    pub kernel: &'static str,
    /// The `[sources]` family `kernel` compiles from.
    pub family: &'static str,
    /// Today's routed entry point at this cell, or `None` when no served plan has one.
    pub default: Option<&'static str>,
    pub numerics: Numerics,
    pub enabled: Enabled,
}

/// 2026-10-10: The shape half of a lookup key: an op on a weight/activation format pair at
/// `k x n`. The row count and the opt-in are passed beside it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell<'a> {
    pub op: &'a str,
    pub weight: &'a str,
    pub activation: &'a str,
    pub k: u32,
    pub n: u32,
}

/// 2026-10-10: The entry of `table` covering `cell` at `rows`: an `Enabled::Default` entry
/// always, an `Enabled::OptIn` entry only when `opt_in`. Row ranges are inclusive.
pub fn lookup_in(
    table: &'static [Schedule],
    cell: &Cell<'_>,
    rows: u32,
    opt_in: bool,
) -> Option<&'static Schedule> {
    table.iter().find(|s| {
        s.op == cell.op
            && s.weight == cell.weight
            && s.activation == cell.activation
            && s.k == cell.k
            && s.n == cell.n
            && (s.rows_lo..=s.rows_hi).contains(&rows)
            && (s.enabled == Enabled::Default || opt_in)
    })
}

/// 2026-10-10: [`lookup_in`] over this binary's [`TARGET_SCHEDULES`]. Not called by any router
/// yet (see the module doc).
pub fn lookup(
    op: &str,
    weight: &str,
    activation: &str,
    k: u32,
    n: u32,
    rows: u32,
    opt_in: bool,
) -> Option<&'static Schedule> {
    let cell = Cell {
        op,
        weight,
        activation,
        k,
        n,
    };
    lookup_in(TARGET_SCHEDULES, &cell, rows, opt_in)
}

#[cfg(test)]
#[path = "schedules_tests.rs"]
mod tests;
