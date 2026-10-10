// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The envelope grid: every projection cell (op, weight and activation formats,
//! N x K) the described models run, plus a margin, at every row count of the ladder, with the
//! contracted entry points that can run it and today's routed default where a served plan has
//! one. Built from the accuracy sweep (`points::sweep`), so a newly described model adds its
//! cells with no new code.
//!
//! Owner: metrale-accuracy (envelope).
//! Invariants:
//! - Candidates are exactly the contracted entry points whose family implements the op at the
//!   cell's weight and activation formats (KERNEL_FAMILIES.toml `op` constraints): a kernel with
//!   no accuracy contract is never a candidate.
//! - A cell's default is the single kernel a golden plan launches at that op, shape and row
//!   count; a cell only plan-only models run has none.
//! - Deterministic: cells sort by (margin, op, weight, activation, k, n, rows).

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::Format;
use metrale_circuit::venn::families::{Families, Family, Roofline};

use super::record::Cell;
use crate::contract::Contracts;
use crate::points::Sweep;

/// 2026-10-10: The row ladder: decode, the verify widths and the multi-sequence ladder.
pub const LADDER: [u64; 13] = [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

/// 2026-10-10: K and N of a margin cell are multiples of these (the tensor-core kernels'
/// K step and the widest N tile), so every margin cell is runnable by the kernels' own rules.
const MARGIN_K_STEP: u64 = 128;
const MARGIN_N_STEP: u64 = 64;

/// 2026-10-10: Ops the grid times: weight-reading projections.
const OPS: [&str; 3] = ["linear", "lm_head", "router"];

/// 2026-10-10: One candidate entry point of a cell.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Candidate {
    /// 2026-10-10: `module::function`.
    pub kernel: String,
    /// 2026-10-10: Its family.
    pub family: String,
    /// 2026-10-10: Index of its contract in the contract file.
    pub contract: usize,
}

/// 2026-10-10: One cell of the grid.
#[derive(Debug, Clone, PartialEq)]
pub struct GridCell {
    /// 2026-10-10: The cell.
    pub cell: Cell,
    /// 2026-10-10: Today's routed entry point, when a served plan has one here.
    pub default: Option<String>,
    /// 2026-10-10: Candidates, sorted.
    pub candidates: Vec<Candidate>,
    /// 2026-10-10: A margin cell (no model runs this N x K; it bounds the envelope).
    pub margin: bool,
    /// 2026-10-10: Recipes that run the shape (empty for margin cells).
    pub users: BTreeSet<String>,
    /// 2026-10-10: Kernel targets of those recipes.
    pub targets: BTreeSet<String>,
    /// 2026-10-10: Roofline floor of one launch, microseconds.
    pub floor_us: f64,
}

/// 2026-10-10: The shape key of a cell without its rows.
type ShapeKey = (String, String, String, u64, u64);

fn base(op: &str) -> &str {
    op.split(':').next().unwrap_or(op)
}

/// 2026-10-10: The contracted entry points whose family implements `op` at `weight` and
/// `activation`.
pub fn candidates(
    contracts: &Contracts,
    fams: &Families,
    op: &str,
    weight: &str,
    activation: &str,
) -> Vec<Candidate> {
    let (Ok(w), Ok(a)) = (Format::parse(weight), Format::parse(activation)) else {
        return Vec::new();
    };
    let mut out = BTreeSet::new();
    for (i, c) in contracts.contracts.iter().enumerate() {
        if c.op != op {
            continue;
        }
        let Some(f) = fams.families.iter().find(|f| f.id == c.family) else {
            continue;
        };
        if !admits(f, op, w, a) {
            continue;
        }
        for k in &c.kernels {
            out.insert(Candidate {
                kernel: k.clone(),
                family: c.family.clone(),
                contract: i,
            });
        }
    }
    out.into_iter().collect()
}

fn admits(f: &Family, op: &str, w: Format, a: Format) -> bool {
    f.ops.iter().any(|s| {
        s.op == op
            && (s.weight.is_empty() || s.weight.contains(&w))
            && (s.activation.is_empty() || s.activation.contains(&a))
    })
}

/// 2026-10-10: The roofline floor of one launch: the weight once, the rows' activations in and
/// outputs out, against the MMA peak of the activation's class.
pub fn floor_us(cell: &Cell, r: &Roofline) -> f64 {
    let (w_bytes, peak) = match cell.weight.as_str() {
        w if w.starts_with("nvfp4") => (0.5625, r.nvfp4_tflops),
        w if w.starts_with("fp8") => (1.0, r.fp8_tflops),
        _ => (2.0, r.bf16_tflops),
    };
    let peak = if cell.activation.starts_with("bf16") {
        r.bf16_tflops
    } else {
        peak
    };
    let (n, k, m) = (cell.n as f64, cell.k as f64, cell.rows as f64);
    let bytes = n * k * w_bytes + m * (n + k) * 2.0;
    let flops = 2.0 * m * n * k;
    (bytes / (r.dram_gbps * 1e3)).max(flops / (peak * 1e6))
}

fn round_up(x: u64, step: u64) -> u64 {
    x.div_ceil(step) * step
}

/// 2026-10-10: The grid of `sweep` at `ladder` rows, with the x0.5 / x2 margin corners of every
/// shape when `margin`.
pub fn grid(
    sweep: &Sweep,
    contracts: &Contracts,
    fams: &Families,
    ladder: &[u64],
    margin: bool,
) -> Vec<GridCell> {
    let mut shapes: BTreeMap<ShapeKey, (BTreeSet<String>, BTreeSet<String>)> = BTreeMap::new();
    let mut defaults: BTreeMap<(ShapeKey, u64), BTreeSet<String>> = BTreeMap::new();
    for p in &sweep.points {
        let op = base(&p.shape.op);
        let (Some(w), Some(a)) = (&p.shape.weight, &p.shape.activation) else {
            continue;
        };
        if !OPS.contains(&op) {
            continue;
        }
        let key = (
            op.to_string(),
            w.clone(),
            a.clone(),
            p.shape.in_dim,
            p.shape.out_dim,
        );
        let e = shapes.entry(key.clone()).or_default();
        e.0.extend(p.users.iter().cloned());
        e.1.extend(p.targets.iter().cloned());
        if !p.kernels.starts_with('(') && !p.kernels.contains(" + ") {
            defaults
                .entry((key, p.shape.rows))
                .or_default()
                .insert(p.kernels.clone());
        }
    }
    let mut keys: BTreeMap<ShapeKey, bool> = shapes.keys().map(|k| (k.clone(), false)).collect();
    if margin {
        for (op, w, a, k, n) in shapes.keys() {
            for kk in [k / 2, k * 2] {
                for nn in [n / 2, n * 2] {
                    let (kk, nn) = (round_up(kk, MARGIN_K_STEP), round_up(nn, MARGIN_N_STEP));
                    let key = (op.clone(), w.clone(), a.clone(), kk, nn);
                    keys.entry(key).or_insert(true);
                }
            }
        }
    }
    let mut out = Vec::new();
    for (key, is_margin) in keys {
        let (op, w, a, k, n) = &key;
        let probe = Cell {
            op: op.clone(),
            weight: w.clone(),
            activation: a.clone(),
            k: *k,
            n: *n,
            rows: 1,
        };
        let mut cands = candidates(contracts, fams, op, w, a);
        cands.retain(|c| runs_formats(sweep, fams, c, &probe));
        if cands.is_empty() {
            continue;
        }
        let (users, targets) = shapes.get(&key).cloned().unwrap_or_default();
        for &rows in ladder {
            let cell = Cell {
                op: op.clone(),
                weight: w.clone(),
                activation: a.clone(),
                k: *k,
                n: *n,
                rows,
            };
            // 2026-10-10: Two served plans that route one cell differently have no single
            // default: the cell is swept without one.
            let default = defaults
                .get(&(key.clone(), rows))
                .filter(|d| d.len() == 1)
                .and_then(|d| d.iter().next().cloned());
            let floor = floor_us(&cell, &fams.roofline);
            out.push(GridCell {
                cell,
                default,
                candidates: cands.clone(),
                margin: is_margin,
                users: users.clone(),
                targets: targets.clone(),
                floor_us: floor,
            });
        }
    }
    out.sort_by(|a, b| (a.margin, &a.cell).cmp(&(b.margin, &b.cell)));
    out
}

/// 2026-10-10: The point values `family` runs a cell's formats at: a swept point of that family
/// at the same op and formats, else an instantiated point whose stated `weight` / `activation`
/// values (where it states them) equal the cell's. `None` when the family has neither.
pub fn point_for(
    sweep: &Sweep,
    fams: &Families,
    family: &str,
    cell: &Cell,
) -> Option<metrale_circuit::venn::families::Values> {
    let swept = sweep.points.iter().find(|p| {
        p.family == family
            && base(&p.shape.op) == cell.op
            && p.shape.weight.as_deref() == Some(cell.weight.as_str())
            && p.shape.activation.as_deref() == Some(cell.activation.as_str())
    });
    if let Some(p) = swept {
        return Some(p.point.clone());
    }
    let f = fams.families.iter().find(|f| f.id == family)?;
    f.points
        .iter()
        .find(|pt| {
            pt.values.get("weight").is_none_or(|w| *w == cell.weight)
                && pt
                    .values
                    .get("activation")
                    .is_none_or(|a| *a == cell.activation)
        })
        .map(|pt| pt.values.clone())
}

/// 2026-10-10: Whether `cand`'s declared pipeline at the cell's formats reads exactly the cell's
/// activation and stored weight: a family whose op spec admits any format (the WxAy engine)
/// offers only the entry points whose pipeline runs these formats.
pub fn runs_formats(sweep: &Sweep, fams: &Families, cand: &Candidate, cell: &Cell) -> bool {
    let (Ok(w), Ok(a)) = (Format::parse(&cell.weight), Format::parse(&cell.activation)) else {
        return false;
    };
    let Some(f) = fams.families.iter().find(|f| f.id == cand.family) else {
        return false;
    };
    let Some(point) = point_for(sweep, fams, &cand.family, cell) else {
        return false;
    };
    let Ok(p) = crate::plan::declared(f, &cand.kernel, &cell.op, &point) else {
        return false;
    };
    let stored = p.steps.iter().find_map(|s| match &s.value {
        metrale_circuit::pipeline::Value::Weight { stored, .. } => Some(*stored),
        _ => None,
    });
    p.inputs.first() == Some(&a) && stored == Some(w)
}

/// 2026-10-10: The proposed cells with no candidate: shapes the models run whose formats no
/// contracted kernel covers (reported, never silently dropped).
pub fn uncovered(sweep: &Sweep, contracts: &Contracts, fams: &Families) -> BTreeSet<ShapeKey> {
    let mut out = BTreeSet::new();
    for p in &sweep.points {
        let op = base(&p.shape.op);
        let (Some(w), Some(a)) = (&p.shape.weight, &p.shape.activation) else {
            continue;
        };
        let probe = Cell {
            op: op.to_string(),
            weight: w.clone(),
            activation: a.clone(),
            k: p.shape.in_dim,
            n: p.shape.out_dim,
            rows: 1,
        };
        if OPS.contains(&op)
            && !candidates(contracts, fams, op, w, a)
                .iter()
                .any(|c| runs_formats(sweep, fams, c, &probe))
        {
            out.insert((
                op.to_string(),
                w.clone(),
                a.clone(),
                p.shape.in_dim,
                p.shape.out_dim,
            ));
        }
    }
    out
}

/// 2026-10-10: Shard `index` of `count`: cells assigned greedily by estimated cost (floor time
/// times candidates), heaviest first, to the least-loaded shard, so boxes finish together. Every
/// cell lands in exactly one shard; the order inside a shard is the grid's.
pub fn shard(cells: &[GridCell], index: usize, count: usize) -> Vec<&GridCell> {
    assert!(count > 0 && index < count, "shard {index} of {count}");
    let mut order: Vec<usize> = (0..cells.len()).collect();
    let cost = |c: &GridCell| c.floor_us.max(5.0) * c.candidates.len() as f64;
    order.sort_by(|&a, &b| cost(&cells[b]).total_cmp(&cost(&cells[a])).then(a.cmp(&b)));
    let mut load = vec![0.0f64; count];
    let mut owner = vec![0usize; cells.len()];
    for i in order {
        let s = (0..count)
            .min_by(|&a, &b| load[a].total_cmp(&load[b]).then(a.cmp(&b)))
            .unwrap_or(0);
        load[s] += cost(&cells[i]);
        owner[i] = s;
    }
    cells
        .iter()
        .enumerate()
        .filter(|(i, _)| owner[*i] == index)
        .map(|(_, c)| c)
        .collect()
}
