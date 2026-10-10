// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: The measured speculative-cost table (`--spec-cost-model measured`): what a
//! verify step and a draft step cost, in wall time and GPU-rail energy, at each batch width
//! and draft depth, on one box class for one recipe; and the drafter's confidence→acceptance
//! calibration. `met bench spec-cost` measures both and writes them; the serve loads them and
//! plans draft depth from them. Pure: parsing and arithmetic only, no I/O.
//!
//! Owner: speculative.
//! Invariants:
//! - A table is used only under the key it was measured under ([`TableKey::check`]): the box
//!   class, the recipe, and the plan digest of every mode it measured. A calibration is used
//!   only for the drafter it was fitted on ([`DrafterKey`]). A mismatch is refused, never
//!   approximated.
//! - A parsed table has, for every width it lists, a cell at every draft depth 0..=its maximum,
//!   all with finite positive costs ([`CostTable::parse`]).
//! - Costs between measured widths are interpolated linearly in width; a width outside the
//!   measured range takes the nearest measured width ([`CostTable::cell`]).

use std::collections::BTreeMap;

use serde::Deserialize;

/// 2026-10-04: Schema version of the table files; bumped when a field changes meaning.
pub const SCHEMA: u32 = 1;

/// 2026-10-04: What a table was measured under. Equal keys are the only licence to use it.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TableKey {
    pub schema: u32,
    /// 2026-10-04: `HARDWARE.toml` box class, e.g. `gb10`.
    pub box_class: String,
    /// 2026-10-04: Recipe id, e.g. `qwen3.6/qwen3.6-35b-a3b-fp8-nvfp4head`.
    pub recipe: String,
    /// 2026-10-04: Circuit plan digest per measured mode (`verify`, `verify_batch`, `draft`),
    /// the plan the serve compiles for that mode; a kernel change outside these plans does
    /// not stale the table.
    pub plan_digests: BTreeMap<String, String>,
}

/// 2026-10-04: One reason a table key does not match the serve's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeyMismatch {
    Schema {
        table: u32,
        serve: u32,
    },
    BoxClass {
        table: String,
        serve: String,
    },
    Recipe {
        table: String,
        serve: String,
    },
    /// 2026-10-04: A mode whose digest differs, or is present on one side only.
    PlanDigest {
        mode: String,
        table: Option<String>,
        serve: Option<String>,
    },
}

impl TableKey {
    /// 2026-10-04: Every way `self` (the table's key) differs from `serve`'s; empty when the
    /// table may be used.
    pub fn check(&self, serve: &TableKey) -> Vec<KeyMismatch> {
        let mut out = Vec::new();
        if self.schema != serve.schema {
            out.push(KeyMismatch::Schema {
                table: self.schema,
                serve: serve.schema,
            });
        }
        if self.box_class != serve.box_class {
            out.push(KeyMismatch::BoxClass {
                table: self.box_class.clone(),
                serve: serve.box_class.clone(),
            });
        }
        if self.recipe != serve.recipe {
            out.push(KeyMismatch::Recipe {
                table: self.recipe.clone(),
                serve: serve.recipe.clone(),
            });
        }
        let modes: std::collections::BTreeSet<&String> = self
            .plan_digests
            .keys()
            .chain(serve.plan_digests.keys())
            .collect();
        for mode in modes {
            let (t, s) = (self.plan_digests.get(mode), serve.plan_digests.get(mode));
            if t != s {
                out.push(KeyMismatch::PlanDigest {
                    mode: mode.clone(),
                    table: t.cloned(),
                    serve: s.cloned(),
                });
            }
        }
        out
    }
}

/// 2026-10-04: The drafter a confidence calibration was fitted on.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct DrafterKey {
    /// 2026-10-04: sha256 over the MTP head's tensors as loaded.
    pub weights_sha256: String,
    /// 2026-10-04: `--mtp-vocab`.
    pub vocab: usize,
    /// 2026-10-04: `--mtp-quantization`.
    pub quantization: String,
    /// 2026-10-04: Whether the drafter's prompt context was active.
    pub context: bool,
}

/// 2026-10-04: One measured cell: `n` sequences, `k` drafts each (`k = 0` is a plain decode
/// step, no draft and no verify rows beyond the one token).
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct Cell {
    pub n: usize,
    pub k: usize,
    /// 2026-10-04: One verify (or, at `k = 0`, decode) step: median wall ms and GPU-rail J.
    pub verify_ms: f64,
    pub verify_j: f64,
    /// 2026-10-04: The `k` chained draft steps that feed the next verify; 0 at `k = 0`.
    pub draft_ms: f64,
    pub draft_j: f64,
}

impl Cell {
    /// 2026-10-04: Wall ms of a whole speculative step at this depth.
    pub fn step_ms(&self) -> f64 {
        self.verify_ms + self.draft_ms
    }

    /// 2026-10-04: GPU-rail joules of a whole speculative step at this depth.
    pub fn step_j(&self) -> f64 {
        self.verify_j + self.draft_j
    }
}

#[derive(Debug, Deserialize)]
struct TableFile {
    key: TableKey,
    cells: Vec<Cell>,
}

/// 2026-10-04: A parsed, validated cost table.
#[derive(Debug, Clone, PartialEq)]
pub struct CostTable {
    pub key: TableKey,
    /// 2026-10-04: Measured widths, ascending.
    widths: Vec<usize>,
    /// 2026-10-04: Deepest measured draft count (the same at every width).
    max_k: usize,
    /// 2026-10-04: `cells[(n, k)]`.
    cells: BTreeMap<(usize, usize), Cell>,
}

impl CostTable {
    /// 2026-10-04: Parses a table file's text and validates it: schema [`SCHEMA`], at least
    /// one width, every listed width measured at every depth `0..=max_k`, no duplicate cell,
    /// every cost finite and positive (a draft cost of exactly 0 is required at `k = 0`).
    pub fn parse(text: &str) -> Result<Self, String> {
        let file: TableFile = toml::from_str(text).map_err(|e| format!("spec-cost table: {e}"))?;
        if file.key.schema != SCHEMA {
            return Err(format!(
                "spec-cost table schema {} (this serve reads {SCHEMA})",
                file.key.schema
            ));
        }
        let mut cells = BTreeMap::new();
        for c in &file.cells {
            let finite = [c.verify_ms, c.verify_j, c.draft_ms, c.draft_j]
                .iter()
                .all(|v| v.is_finite());
            let positive = c.verify_ms > 0.0 && c.verify_j > 0.0;
            let draft_ok = if c.k == 0 {
                c.draft_ms == 0.0 && c.draft_j == 0.0
            } else {
                c.draft_ms > 0.0 && c.draft_j > 0.0
            };
            if c.n == 0 || !finite || !positive || !draft_ok {
                return Err(format!("spec-cost table: bad cell {c:?}"));
            }
            if cells.insert((c.n, c.k), *c).is_some() {
                return Err(format!(
                    "spec-cost table: duplicate cell n={} k={}",
                    c.n, c.k
                ));
            }
        }
        let widths: Vec<usize> = {
            let mut w: Vec<usize> = cells.keys().map(|&(n, _)| n).collect();
            w.dedup();
            w
        };
        let max_k = cells
            .keys()
            .map(|&(_, k)| k)
            .max()
            .ok_or("spec-cost table: no cells")?;
        for &n in &widths {
            for k in 0..=max_k {
                if !cells.contains_key(&(n, k)) {
                    return Err(format!("spec-cost table: width {n} lacks depth {k}"));
                }
            }
        }
        Ok(Self {
            key: file.key,
            widths,
            max_k,
            cells,
        })
    }

    /// 2026-10-04: The table file text for `key` and `cells`, in the format [`Self::parse`]
    /// reads; validated by parsing it back, so a grid that would be refused at serve time is
    /// refused when written.
    pub fn render(key: &TableKey, cells: &[Cell]) -> Result<String, String> {
        use std::fmt::Write as _;
        let mut s = String::new();
        let _ = writeln!(s, "[key]\nschema = {}", key.schema);
        let _ = writeln!(
            s,
            "box_class = {:?}\nrecipe = {:?}",
            key.box_class, key.recipe
        );
        let _ = writeln!(s, "\n[key.plan_digests]");
        for (mode, digest) in &key.plan_digests {
            let _ = writeln!(s, "{mode} = {digest:?}");
        }
        let mut sorted = cells.to_vec();
        sorted.sort_by_key(|c| (c.n, c.k));
        for c in &sorted {
            let _ = write!(
                s,
                "\n[[cells]]\nn = {}\nk = {}\nverify_ms = {:?}\nverify_j = {:?}\ndraft_ms = {:?}\ndraft_j = {:?}\n",
                c.n, c.k, c.verify_ms, c.verify_j, c.draft_ms, c.draft_j
            );
        }
        let parsed = Self::parse(&s)?;
        if parsed.key != *key {
            return Err("spec-cost table: the rendered key does not read back".into());
        }
        Ok(s)
    }

    /// 2026-10-04: Deepest draft count the table measured.
    pub fn max_k(&self) -> usize {
        self.max_k
    }

    /// 2026-10-04: The cost of a step at width `n` and depth `k` (`k <= max_k`), linearly
    /// interpolated between the two measured widths around `n`; the nearest measured width
    /// outside the measured range. `None` above `max_k`.
    pub fn cell(&self, n: usize, k: usize) -> Option<Cell> {
        if k > self.max_k {
            return None;
        }
        let lo = self.widths.iter().copied().filter(|&w| w <= n).max();
        let hi = self.widths.iter().copied().filter(|&w| w >= n).min();
        let at = |w| self.cells[&(w, k)];
        Some(match (lo, hi) {
            (Some(a), Some(b)) if a == b => at(a),
            (Some(a), Some(b)) => {
                let t = (n - a) as f64 / (b - a) as f64;
                let (x, y) = (at(a), at(b));
                let mix = |p: f64, q: f64| p + (q - p) * t;
                Cell {
                    n,
                    k,
                    verify_ms: mix(x.verify_ms, y.verify_ms),
                    verify_j: mix(x.verify_j, y.verify_j),
                    draft_ms: mix(x.draft_ms, y.draft_ms),
                    draft_j: mix(x.draft_j, y.draft_j),
                }
            }
            (Some(a), None) => Cell { n, ..at(a) },
            (None, Some(b)) => Cell { n, ..at(b) },
            (None, None) => unreachable!("parse guarantees at least one width"),
        })
    }
}

#[derive(Debug, Deserialize)]
struct CalibrationFile {
    drafter: DrafterKey,
    acceptance: AcceptanceFields,
}

#[derive(Debug, Deserialize)]
struct AcceptanceFields {
    edges: Vec<f32>,
    p_accept: Vec<f64>,
    prior_by_position: Vec<f64>,
}

/// 2026-10-04: How likely a draft is to be accepted, from the drafter's top-1 log-probability
/// for it, fitted on one drafter ([`DrafterKey`]); and per-position priors for drafts whose
/// confidence was not measured. A draft is accepted only if every draft before it was, so a
/// chain's expected accepted length is the sum of the running products.
#[derive(Debug, Clone, PartialEq)]
pub struct AcceptanceCalibration {
    pub drafter: DrafterKey,
    /// 2026-10-04: Ascending log-probability upper bounds; bucket `i` holds `lp <= edges[i]`
    /// (and above `edges[i - 1]`). The last edge is at least 0, so every log-probability lands.
    edges: Vec<f32>,
    p_accept: Vec<f64>,
    /// 2026-10-04: P(draft j accepted | drafts before it accepted), for j = 1, 2, ..
    prior_by_position: Vec<f64>,
}

impl AcceptanceCalibration {
    /// 2026-10-04: Parses and validates a calibration file: as many probabilities as edges,
    /// edges strictly ascending with the last at least 0, every probability in 0..=1, at least
    /// one position prior.
    pub fn parse(text: &str) -> Result<Self, String> {
        let f: CalibrationFile =
            toml::from_str(text).map_err(|e| format!("acceptance calibration: {e}"))?;
        let a = f.acceptance;
        let in01 = |p: &f64| (0.0..=1.0).contains(p);
        if a.edges.is_empty()
            || a.edges.len() != a.p_accept.len()
            || !a.edges.windows(2).all(|w| w[0] < w[1])
            || a.edges.last().is_some_and(|&e| e < 0.0)
            || !a.p_accept.iter().all(in01)
            || a.prior_by_position.is_empty()
            || !a.prior_by_position.iter().all(in01)
        {
            return Err("acceptance calibration: malformed buckets or priors".into());
        }
        Ok(Self {
            drafter: f.drafter,
            edges: a.edges,
            p_accept: a.p_accept,
            prior_by_position: a.prior_by_position,
        })
    }

    /// 2026-10-04: P(accept) of a draft whose top-1 log-probability is `lp`.
    pub fn p_given_lp(&self, lp: f32) -> f64 {
        let i = self
            .edges
            .iter()
            .position(|&e| lp <= e)
            .unwrap_or(self.edges.len() - 1);
        self.p_accept[i]
    }

    /// 2026-10-04: The prior for draft position `j` (1-based); positions past the last
    /// fitted one take the last prior.
    pub fn prior(&self, j: usize) -> f64 {
        let i = j.saturating_sub(1).min(self.prior_by_position.len() - 1);
        self.prior_by_position[i]
    }

    /// 2026-10-04: Expected accepted drafts of a chain whose drafts have confidences `lps`
    /// (missing entries take the position prior), counting only the first `k`.
    pub fn expected_accepted(&self, lps: &[f32], k: usize) -> f64 {
        crate::spec_ctl::chain::chain_sum(
            |j| {
                lps.get(j - 1)
                    .map_or_else(|| self.prior(j), |&lp| self.p_given_lp(lp))
            },
            k,
            0.0,
        )
    }
}

/// 2026-10-04: `--spec-cost-model measured`'s resolved state, as the serve loaded and checked it
/// at boot (`TableKey::check` and a `DrafterKey` comparison both empty/equal): the table, the
/// drafter's acceptance calibration, and `--spec-cost-slack`.
pub struct SpecCostState {
    pub table: CostTable,
    pub calibration: AcceptanceCalibration,
    pub slack: f64,
}

#[path = "spec_cost_fit.rs"]
mod fit;
pub use fit::MIN_OUTCOMES;

#[cfg(test)]
#[path = "spec_cost_tests.rs"]
mod tests;
