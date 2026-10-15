// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the kernel tree (pure): every kernel of a segment either sits
//! under a node of the tree or is listed as unattributed, keyed by its full
//! name. Nothing is dropped.
//!
//! Times are kernel-duration sums in integer ns while the tree is built; the
//! JSON carries them as microseconds (ns / 1000) with the unit stated once at
//! the root. A node's total is its own launches plus its children's, exactly
//! (in ns). Only a leaf has launches of its own, so an interior node's
//! `self_time` is always 0. Children are kept in first-seen (execution) order.

pub mod circuit;
pub mod config;
pub mod diff;
pub mod families;
pub mod identity;
pub mod json;

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::error::Result;
use crate::core::model::Kernel;
use crate::core::repr::quote;
use crate::core::tree::identity::{Quantiles, Variants, quantiles, quantiles_json, variants_json};
use crate::fail;

pub const SCHEMA: &str = "metprof.kernel_tree/1";
pub const TIME_UNIT: &str = "us";

/// Launch statistics in ns. `min`/`max` are `None` exactly when `n` is 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Stats {
    pub n: u64,
    pub sum: i64,
    pub min: Option<i64>,
    pub max: Option<i64>,
}

impl Stats {
    pub fn add(&mut self, dur: i64) {
        self.merge(&Stats {
            n: 1,
            sum: dur,
            min: Some(dur),
            max: Some(dur),
        });
    }

    pub fn merge(&mut self, o: &Stats) {
        self.n += o.n;
        self.sum += o.sum;
        self.min = match (self.min, o.min) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        self.max = match (self.max, o.max) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
    }
}

/// The circuit join key of a group node: the group's index in the fusion
/// plan, the rule that formed it and the plan's digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Join {
    pub group: u64,
    pub rule: String,
    pub plan_digest: String,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Node {
    /// What this node absorbed, as declared (never timed separately).
    pub fuses: Vec<String>,
    /// A circuit group's member nodes, in pattern order.
    pub members: Vec<String>,
    pub circuit: Option<Join>,
    /// Launches of this node itself; only a leaf has any.
    pub own: Stats,
    /// The identities of `own` (a leaf's only), with their durations while
    /// the tree is built.
    pub variants: Variants,
    /// Per-launch quantiles over the subtree: computed by `Tree::seal`, or read
    /// back as stated. `None` exactly when the subtree has no launches.
    pub quantiles: Option<Quantiles>,
    /// The baseline total (ns) of the node at the same path; `None` when the
    /// baseline has no such node. Meaningful only when the tree has a baseline.
    pub baseline: Option<i64>,
    pub kernels: IndexMap<String, Node>,
}

impl Node {
    /// Counts one launch as this node's own.
    pub fn record(&mut self, k: &Kernel) {
        self.own.add(k.dur());
        identity::record(&mut self.variants, k);
    }

    /// Takes over the own launches of `o` (a leaf being built).
    pub fn absorb(&mut self, o: Node) {
        self.own.merge(&o.own);
        identity::absorb(&mut self.variants, o.variants);
    }

    /// Fills the quantiles of this subtree from its durations, which it
    /// consumes; returns them.
    fn seal(&mut self) -> Vec<i64> {
        let mut all = identity::seal(&mut self.variants);
        for c in self.kernels.values_mut() {
            all.extend(c.seal());
        }
        self.quantiles = quantiles(&all);
        all
    }

    pub fn total(&self) -> Stats {
        let mut s = self.own;
        for c in self.kernels.values() {
            s.merge(&c.total());
        }
        s
    }
}

/// A child of `map` by key, created on first use. A key is non-empty and has
/// no `/`, so a node path (keys joined by `/`) is unambiguous.
pub fn child<'m>(map: &'m mut IndexMap<String, Node>, key: &str) -> Result<&'m mut Node> {
    if key.is_empty() || key.contains('/') {
        fail!(
            "tree node key {} must be non-empty and contain no '/'",
            quote(key)
        );
    }
    Ok(map.entry(key.to_string()).or_default())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Circuit,
    Families,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Circuit => "circuit",
            Source::Families => "families",
        }
    }

    pub fn parse(s: &str) -> Option<Source> {
        [Source::Circuit, Source::Families]
            .into_iter()
            .find(|x| x.as_str() == s)
    }
}

/// Launches that no node claimed, under one demangled name.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Loose {
    pub n: u64,
    pub sum: i64,
    pub variants: Variants,
    pub quantiles: Option<Quantiles>,
}

impl Loose {
    pub fn record(&mut self, k: &Kernel) {
        self.n += 1;
        self.sum += k.dur();
        identity::record(&mut self.variants, k);
    }
}

/// What a baseline left behind: its root total and the maximal subtrees it
/// had that this tree does not (fused away or renamed), with their totals.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaselineView {
    pub total: i64,
    pub fused_away: Vec<(String, i64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    pub provenance: Map<String, Value>,
    pub source: Source,
    pub kernels: IndexMap<String, Node>,
    pub unattributed: IndexMap<String, Loose>,
    pub baseline: Option<BaselineView>,
}

pub fn us(ns: i64) -> f64 {
    ns as f64 / 1000.0
}

/// The first start and the last end (capture clock, ns) of the launches at
/// `idx`; `None` for none.
pub fn span(kernels: &[Kernel], idx: &[usize]) -> Option<(i64, i64)> {
    let start = idx.iter().map(|&i| kernels[i].start).min()?;
    let end = idx.iter().map(|&i| kernels[i].end).max()?;
    Some((start, end))
}

impl Tree {
    /// Fills every quantile from the launch durations and drops them, so a
    /// sealed tree equals its JSON read back. A built tree is sealed once,
    /// before it is written: a second seal finds no durations, leaves the
    /// quantiles `None`, and `to_json` refuses the tree.
    pub fn seal(&mut self) {
        for n in self.kernels.values_mut() {
            n.seal();
        }
        for l in self.unattributed.values_mut() {
            let durs = identity::seal(&mut l.variants);
            l.quantiles = quantiles(&durs);
        }
    }

    fn check_sealed(&self) -> Result<()> {
        let nodes = self
            .paths()
            .into_iter()
            .map(|(p, n)| (p, n.total().n, n.quantiles));
        let loose = self
            .unattributed
            .iter()
            .map(|(k, l)| (k.clone(), l.n, l.quantiles));
        for (what, n, q) in nodes.chain(loose) {
            if (n > 0) != q.is_some() {
                fail!(
                    "kernel tree: {} has {n} launch(es) but quantiles {q:?}; the tree was not sealed",
                    quote(&what)
                );
            }
        }
        Ok(())
    }

    pub fn attributed(&self) -> Stats {
        let mut s = Stats::default();
        for n in self.kernels.values() {
            s.merge(&n.total());
        }
        s
    }

    pub fn unattributed_ns(&self) -> i64 {
        self.unattributed.values().map(|l| l.sum).sum()
    }

    pub fn total_ns(&self) -> i64 {
        self.attributed().sum + self.unattributed_ns()
    }

    /// Attributed share of the summed kernel time.
    pub fn coverage(&self) -> Result<f64> {
        let total = self.total_ns();
        if total <= 0 {
            fail!("kernel tree: the segment has no kernel time");
        }
        Ok(self.attributed().sum as f64 / total as f64)
    }

    /// The coverage, or an error when it is below `min`.
    pub fn gate(&self, min: f64) -> Result<f64> {
        let c = self.coverage()?;
        if c < min {
            fail!(
                "kernel tree coverage {c:.4} is below [tree] coverage_min {min}: {:.3} us in {} unattributed \
                 kernel name(s); the plan or the family table does not describe this capture",
                us(self.unattributed_ns()),
                self.unattributed.len()
            );
        }
        Ok(c)
    }

    /// Every node with its path, depth first, parents before children.
    pub fn paths(&self) -> Vec<(String, &Node)> {
        fn walk<'n>(
            prefix: &str,
            map: &'n IndexMap<String, Node>,
            out: &mut Vec<(String, &'n Node)>,
        ) {
            for (k, n) in map {
                let p = if prefix.is_empty() {
                    k.clone()
                } else {
                    format!("{prefix}/{k}")
                };
                out.push((p.clone(), n));
                walk(&p, &n.kernels, out);
            }
        }
        let mut out = Vec::new();
        walk("", &self.kernels, &mut out);
        out
    }

    pub fn to_json(&self) -> Result<Value> {
        self.check_sealed()?;
        let total = self.total_ns();
        let mut root = Map::new();
        root.insert("schema".into(), json!(SCHEMA));
        root.insert("time_unit".into(), json!(TIME_UNIT));
        root.insert("provenance".into(), Value::Object(self.provenance.clone()));
        root.insert("source".into(), json!(self.source.as_str()));
        root.insert("total_time".into(), json!(us(total)));
        root.insert("coverage".into(), json!(self.coverage()?));
        if let Some(b) = &self.baseline {
            root.insert("baseline_time".into(), json!(us(b.total)));
            root.insert("delta".into(), json!(us(total - b.total)));
            let away: Vec<Value> = b
                .fused_away
                .iter()
                .map(|(p, t)| json!({"path": p, "baseline_time": us(*t)}))
                .collect();
            root.insert("fused_away".into(), Value::Array(away));
            root.insert("new".into(), json!(self.new_paths()));
        }
        root.insert("kernels".into(), self.nodes_json(&self.kernels));
        let loose: Map<String, Value> = self
            .unattributed
            .iter()
            .map(|(k, l)| {
                let mut m = Map::new();
                m.insert("total_time".into(), json!(us(l.sum)));
                m.insert("launches".into(), json!(l.n));
                quantiles_json(l.quantiles, &mut m);
                m.insert("variants".into(), variants_json(&l.variants));
                (k.clone(), Value::Object(m))
            })
            .collect();
        root.insert("unattributed".into(), Value::Object(loose));
        Ok(Value::Object(root))
    }

    /// Maximal paths with no node in the baseline (empty without a baseline).
    pub fn new_paths(&self) -> Vec<String> {
        if self.baseline.is_none() {
            return Vec::new();
        }
        let mut out: Vec<String> = Vec::new();
        for (p, n) in self.paths() {
            let inside = out.iter().any(|q| p.starts_with(&format!("{q}/")));
            if n.baseline.is_none() && !inside {
                out.push(p);
            }
        }
        out
    }

    fn nodes_json(&self, map: &IndexMap<String, Node>) -> Value {
        Value::Object(
            map.iter()
                .map(|(k, n)| (k.clone(), self.node_json(n)))
                .collect(),
        )
    }

    fn node_json(&self, n: &Node) -> Value {
        let t = n.total();
        let opt = |v: Option<i64>| v.map_or(Value::Null, |x| json!(us(x)));
        let mean = if t.n == 0 {
            Value::Null
        } else {
            json!(t.sum as f64 / t.n as f64 / 1000.0)
        };
        let mut m = Map::new();
        m.insert("fuses".into(), json!(n.fuses));
        m.insert("total_time".into(), json!(us(t.sum)));
        m.insert("self_time".into(), json!(us(n.own.sum)));
        m.insert("launches".into(), json!(t.n));
        m.insert("mean_us".into(), mean);
        m.insert("min_us".into(), opt(t.min));
        m.insert("max_us".into(), opt(t.max));
        quantiles_json(n.quantiles, &mut m);
        if n.kernels.is_empty() {
            m.insert("variants".into(), variants_json(&n.variants));
        }
        if let Some(j) = &n.circuit {
            m.insert(
                "circuit".into(),
                json!({"group": j.group, "rule": j.rule, "plan_digest": j.plan_digest}),
            );
        }
        if !n.members.is_empty() {
            m.insert("members".into(), json!(n.members));
        }
        if self.baseline.is_some() {
            m.insert("baseline_time".into(), opt(n.baseline));
            m.insert(
                "delta".into(),
                n.baseline.map_or(Value::Null, |b| json!(us(t.sum - b))),
            );
        }
        m.insert("kernels".into(), self.nodes_json(&n.kernels));
        Value::Object(m)
    }
}
