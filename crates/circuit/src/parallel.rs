// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Parallelism as a part of the circuit. A circuit TOML declares, once, how each of
//! its dims may split over ranks (`[parallel.<dim>]`: the unit, the remainder policy, padding,
//! where it applies, the degrees it may run at, and the engine site that splits it). A
//! deployment states only N (the device count) and each part's degree; a rank's circuit is the
//! ONE circuit instantiated at that layout and rank, every split dim at the rank's slice, each
//! edge's sharding following from its shape, and the collectives derived where a tensor's
//! sharding changes between producer and consumer (sharding propagation, as GSPMD does it). No
//! collective is written in a circuit file.
//!
//! Parts, degrees and groups. Each declared dim is a part. N is the upper bound; each part runs
//! at its own degree `d` within its bounds and `1..=N`, chosen by the instance or the cost model
//! and recorded in the [`Layout`] (a typed plan field per part): `d = 1` replicates the part on
//! every rank; `d >= 2` splits it near-evenly (remainders under its policy) over a device group of
//! `d` ranks ([`GroupRule::Lowest`]: ranks `0..d`). A rank outside a part's group idles for it:
//! the part's nodes are absent from that rank's circuit and the derived boundary collectives
//! deliver the result (an all-reduce or all-gather over the group's communicator, then a
//! broadcast from the group's first rank to the ranks outside it).
//!
//! Sharding of an edge ([`Shard`]):
//! - `Split { dim, group }`: its shape names a split part declared `scope = "every"`, or the edge
//!   is marked `split = "<dim>"` for a `scope = "marked"` part (a projection whose output rows
//!   are sliced while the dim is whole elsewhere).
//! - `Partial { group }`: a weight-reading node (a contraction) read a `Split` input into an
//!   unsplit output (a row-parallel projection): each rank of the group holds a summand. A
//!   partial passes through an op linear in it ([`passes_partial`]).
//! - `Replicated`: everything else.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - N = 1 is the whole model: no part splits, no edge is sharded, no collective is inserted, so
//!   the rank circuit equals [`crate::instantiate::instantiate`]'s (tested on every instance).
//! - A rank's share is `split(dim, d, index in group, policy)`: the dim (padded up to its unit
//!   when the split pads) in units; [`Remainder::Front`] gives the first `units % d` ranks one
//!   unit more (the engine's `metrale_config::tp_split`), [`Remainder::Back`] gives every rank
//!   `ceil(units / d)` and the last what remains, [`Remainder::Exact`] refuses a degree that does
//!   not divide the units (`TpSupport::Even`). Nothing is defaulted (PCND): a split states
//!   its unit, remainder, pad, scope, degree bounds and `why`, and a deployment every part's d.
//! - A degree outside the part's bounds or `1..=N`, one that leaves a rank of the group no unit,
//!   a zero unit, or an unpadded dim that is not a whole number of units is refused with why.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use serde::Deserialize;

use crate::ir::OpKind;

/// 2026-10-23: Who takes the units left after an even split.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Remainder {
    /// 2026-10-23: The first `units % d` ranks take one unit more (`tp_split`).
    Front,
    /// 2026-10-23: Every rank takes `ceil(units / d)`; the last takes what remains.
    Back,
    /// 2026-10-23: The units must divide evenly (the engine's `TpSupport::Even`); any other
    /// degree is refused.
    Exact,
}

/// 2026-10-23: Where a split dim splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// 2026-10-23: Every edge whose shape names the dim, and the dim's value in the rank's shape.
    Every,
    /// 2026-10-23: Only edges marked `split = "<dim>"`; the dim keeps its value elsewhere.
    Marked,
}

/// 2026-10-23: The degrees a part may run at (its kernels' constraints).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DegreeBounds {
    /// 2026-10-23: At least this many ranks (1: may replicate).
    pub min: u64,
    /// 2026-10-23: At most this many.
    pub max: u64,
}

/// 2026-10-23: How one dim splits.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DimSplit {
    /// 2026-10-23: The split's unit in elements (heads for a head count): its alignment.
    pub unit: u64,
    /// 2026-10-23: The engine pads the dim up to a whole number of units before it splits.
    pub pad: bool,
    /// 2026-10-23: The remainder policy.
    pub remainder: Remainder,
    /// 2026-10-23: Where it splits.
    pub scope: Scope,
    /// 2026-10-23: The degrees it may run at.
    pub degree: DegreeBounds,
    /// 2026-10-23: The part this dim is co-partitioned with (the KDA value heads with their key
    /// heads): it takes that part's placement, so an op mapping one's slice to the other's is
    /// rank-local. Absent: the dim is a part of its own.
    #[serde(default)]
    pub tie: Option<String>,
    /// 2026-10-23: The engine's split site (file:line).
    pub why: String,
}

/// 2026-10-23: The part `dim` belongs to: its tie, or itself.
pub fn part_of<'a>(spec: &'a ParallelSpec, dim: &'a str) -> &'a str {
    spec.get(dim).and_then(|s| s.tie.as_deref()).unwrap_or(dim)
}

/// 2026-10-23: The parts a deployment states a degree for: every dim not tied to another.
pub fn parts(spec: &ParallelSpec) -> impl Iterator<Item = &String> {
    spec.iter().filter(|(_, s)| s.tie.is_none()).map(|(d, _)| d)
}

/// 2026-10-23: A circuit's parallelism: part (dim) to its split. A dim not listed is replicated.
pub type ParallelSpec = BTreeMap<String, DimSplit>;

/// 2026-10-23: How a part's device group was picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupRule {
    /// 2026-10-23: The `d` lowest ranks.
    Lowest,
}

/// 2026-10-23: Who chose a part's degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chosen {
    /// 2026-10-23: The deployment states it (what the serve runs).
    Instance,
    /// 2026-10-23: The cost model.
    Cost,
}

/// 2026-10-23: One part's degree and device group.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// 2026-10-23: `d`: 1 replicates, `>= 2` splits over `group`.
    pub degree: u64,
    /// 2026-10-23: The ranks that hold the part, ascending; every rank when `d = 1`.
    pub group: Vec<u64>,
    /// 2026-10-23: How the group was picked.
    pub rule: GroupRule,
    /// 2026-10-23: Who chose `d`.
    pub chosen: Chosen,
}

impl Placement {
    /// 2026-10-23: The part is split over its group; `false` when replicated.
    pub fn splits(&self) -> bool {
        self.degree > 1
    }
}

/// 2026-10-23: The parallel layout of one deployment: N and every part's placement.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// 2026-10-23: N, the device count.
    pub ranks: u64,
    /// 2026-10-23: Part (dim) to its placement; every declared part is listed.
    pub parts: BTreeMap<String, Placement>,
}

impl Layout {
    /// 2026-10-23: Every rank: the world communicator.
    pub fn world(&self) -> Vec<u64> {
        (0..self.ranks).collect()
    }
}

/// 2026-10-23: Which rank of a layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RankOf<'a> {
    /// 2026-10-23: The layout.
    pub layout: &'a Layout,
    /// 2026-10-23: The rank, below N.
    pub rank: u64,
}

/// 2026-10-23: How an edge is held on a rank.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Shard {
    /// 2026-10-23: The whole tensor on every rank.
    Replicated,
    /// 2026-10-23: The group's ranks each hold a slice along the dim.
    Split {
        /// 2026-10-23: The dim.
        dim: String,
        /// 2026-10-23: The ranks holding slices.
        group: Vec<u64>,
    },
    /// 2026-10-23: The group's ranks each hold a summand.
    Partial {
        /// 2026-10-23: The ranks holding summands.
        group: Vec<u64>,
    },
}

impl Shard {
    /// 2026-10-23: The ranks that hold the tensor (every rank for a replicated one).
    pub fn group<'a>(&'a self, world: &'a [u64]) -> &'a [u64] {
        match self {
            Shard::Replicated => world,
            Shard::Split { group, .. } | Shard::Partial { group } => group,
        }
    }
}

/// 2026-10-23: Why a parallel layout was refused.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParallelError {
    /// 2026-10-23: N = 0 or a rank at or past N.
    #[error("rank {rank} of {ranks}")]
    Rank {
        /// 2026-10-23: The rank.
        rank: u64,
        /// 2026-10-23: N.
        ranks: u64,
    },
    /// 2026-10-23: A split or a degree names a dim the shape or the spec does not define.
    #[error("parallel `{0}`: no such part")]
    UnknownDim(String),
    /// 2026-10-23: The part cannot run at the degree, or its split is malformed.
    #[error("parallel `{dim}` at {ranks} ranks: {detail}")]
    Split {
        /// 2026-10-23: The dim.
        dim: String,
        /// 2026-10-23: N.
        ranks: u64,
        /// 2026-10-23: Why.
        detail: String,
    },
}

/// 2026-10-23: Index `i`'s `(start, len)` of `total` over `d` ranks in `unit`-element units
/// under `remainder`, `total` padded up to the unit first; `None` when some rank would get no
/// unit.
pub fn rank_slice(
    total: u64,
    d: u64,
    i: u64,
    unit: u64,
    remainder: Remainder,
) -> Option<(u64, u64)> {
    if unit == 0 || d == 0 || i >= d {
        return None;
    }
    let units = total.div_ceil(unit);
    let (start, len) = match remainder {
        Remainder::Front => {
            if units < d {
                return None;
            }
            let (base, rem) = (units / d, units % d);
            (i * base + i.min(rem), base + u64::from(i < rem))
        }
        Remainder::Exact => {
            if !units.is_multiple_of(d) {
                return None;
            }
            let per = units / d;
            (i * per, per)
        }
        Remainder::Back => {
            let per = units.div_ceil(d);
            if units <= (d - 1) * per {
                return None;
            }
            let start = i * per;
            (start, per.min(units - start))
        }
    };
    Some((start * unit, len * unit))
}

/// 2026-10-23: Place every part of `spec` at the degree `degrees` states for it over `ranks`
/// devices (every declared part must be stated: nothing is defaulted), checking each degree
/// against the part's bounds, N and its split.
pub fn place(
    spec: &ParallelSpec,
    dims: &BTreeMap<String, u64>,
    ranks: u64,
    degrees: &BTreeMap<String, u64>,
    chosen: Chosen,
) -> Result<Layout, ParallelError> {
    if ranks == 0 {
        return Err(ParallelError::Rank { rank: 0, ranks });
    }
    if let Some(extra) = degrees
        .keys()
        .find(|d| spec.get(*d).is_none_or(|s| s.tie.is_some()))
    {
        return Err(ParallelError::UnknownDim(extra.clone()));
    }
    let mut parts = BTreeMap::new();
    let (tied, free): (Vec<_>, Vec<_>) = spec.iter().partition(|(_, s)| s.tie.is_some());
    for (dim, s) in free.into_iter().chain(tied) {
        let total = *dims
            .get(dim)
            .ok_or_else(|| ParallelError::UnknownDim(dim.clone()))?;
        let bad = |detail: String| ParallelError::Split {
            dim: dim.clone(),
            ranks,
            detail,
        };
        let d = match &s.tie {
            None => *degrees
                .get(dim)
                .ok_or_else(|| bad("no degree stated for this part".into()))?,
            Some(root) => {
                let r = spec
                    .get(root)
                    .filter(|r| r.tie.is_none())
                    .ok_or_else(|| bad(format!("tied to `{root}`, which is no untied part")))?;
                let rt = *dims
                    .get(root)
                    .ok_or_else(|| ParallelError::UnknownDim(root.clone()))?;
                if rt.div_ceil(r.unit) != total.div_ceil(s.unit) {
                    return Err(bad(format!(
                        "tied to `{root}` but splits into another number of units"
                    )));
                }
                let p: &Placement = parts.get(root).expect("untied parts are placed first");
                p.degree
            }
        };
        if s.why.trim().is_empty() {
            return Err(bad("states no `why`".into()));
        }
        if s.unit == 0 {
            return Err(bad("unit 0".into()));
        }
        if s.degree.min == 0 || s.degree.min > s.degree.max {
            return Err(bad(format!(
                "degree bounds {}..={}",
                s.degree.min, s.degree.max
            )));
        }
        if d == 0 || d > ranks || d < s.degree.min || d > s.degree.max {
            return Err(bad(format!(
                "degree {d} is outside 1..={ranks} or the part's bounds {}..={}",
                s.degree.min, s.degree.max
            )));
        }
        if d > 1 {
            if !s.pad && !total.is_multiple_of(s.unit) {
                return Err(bad(format!(
                    "{total} is not a multiple of the {}-element unit and the split does not pad",
                    s.unit
                )));
            }
            if rank_slice(total, d, d - 1, s.unit, s.remainder).is_none() {
                return Err(bad(format!(
                    "{total} in {}-element units leaves a rank of {d} without a unit \
                     ({:?} remainder)",
                    s.unit, s.remainder
                )));
            }
        }
        let group: Vec<u64> = match s.tie.as_ref().and_then(|r| parts.get(r)) {
            Some(p) => p.group.clone(),
            None if d == 1 => (0..ranks).collect(),
            None => (0..d).collect(),
        };
        parts.insert(
            dim.clone(),
            Placement {
                degree: d,
                group,
                rule: GroupRule::Lowest,
                chosen,
            },
        );
    }
    Ok(Layout { ranks, parts })
}

/// 2026-10-23: Rank `at.rank`'s slice of each part: `Some((start, len))` (the whole dim for a
/// replicated part), `None` for a rank outside the part's group (it idles for the part).
pub fn slices(
    spec: &ParallelSpec,
    dims: &BTreeMap<String, u64>,
    at: RankOf<'_>,
) -> Result<BTreeMap<String, Option<(u64, u64)>>, ParallelError> {
    let l = at.layout;
    if at.rank >= l.ranks {
        return Err(ParallelError::Rank {
            rank: at.rank,
            ranks: l.ranks,
        });
    }
    let mut out = BTreeMap::new();
    for (dim, p) in &l.parts {
        let s = spec
            .get(dim)
            .ok_or_else(|| ParallelError::UnknownDim(dim.clone()))?;
        let total = *dims
            .get(dim)
            .ok_or_else(|| ParallelError::UnknownDim(dim.clone()))?;
        let slice = match p.group.iter().position(|&r| r == at.rank) {
            _ if !p.splits() => Some((0, total)),
            None => None,
            Some(i) => Some(
                rank_slice(total, p.degree, i as u64, s.unit, s.remainder).ok_or_else(|| {
                    ParallelError::Split {
                        dim: dim.clone(),
                        ranks: l.ranks,
                        detail: "no slice".into(),
                    }
                })?,
            ),
        };
        out.insert(dim.clone(), slice);
    }
    Ok(out)
}

/// 2026-10-23: The dims a rank's shape has: each split `scope = "every"` part replaced by this
/// rank's slice length (padding included, as the engine allocates it), 0 where the rank idles
/// for the part; a `marked` part keeps its value (only its marked edges are sliced).
pub fn rank_dims(
    spec: &ParallelSpec,
    dims: &BTreeMap<String, u64>,
    at: RankOf<'_>,
) -> Result<BTreeMap<String, u64>, ParallelError> {
    let mut out = dims.clone();
    for (dim, slice) in slices(spec, dims, at)? {
        if spec[&dim].scope == Scope::Every {
            out.insert(dim, slice.map_or(0, |(_, len)| len));
        }
    }
    Ok(out)
}

/// 2026-10-23: `op` is linear in its partial inputs, so its output stays a partial: the expert
/// blend over partial expert rows (and a partial shared expert) with replicated weights, and a
/// sum of partials only.
pub fn passes_partial(op: &OpKind, partial: &[bool]) -> bool {
    match op {
        OpKind::Blend => match partial {
            [experts, weights, shared @ ..] => *experts && !*weights && shared.iter().all(|p| *p),
            _ => false,
        },
        OpKind::ResidualAdd => !partial.is_empty() && partial.iter().all(|p| *p),
        _ => false,
    }
}

#[cfg(test)]
#[path = "parallel_tests.rs"]
mod parallel_tests;
