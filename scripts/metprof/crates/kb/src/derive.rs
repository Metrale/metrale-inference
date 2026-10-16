// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a metprof run as KB records and edges (pure; DESIGN §2.4).
//!
//! The run becomes one `ProfileRun` (its kernel tree kept whole) linked to its
//! hardware class, campaign and box. Each tree node at depth ≤ `max_depth`
//! becomes a `KernelSample` (`HAS_SAMPLE` weighted by its share of the run's
//! total time, `PARENT_OF` between samples, `FUSES` to each declared member,
//! `SAMPLE_OF` to the kernel it resolves to). Resolution is structural, from
//! the tree's own shape: in a family tree a leaf is a profiler symbol under
//! its family (`KernelSymbol` `SYMBOL_OF` `Kernel(family:<f>)`) and a node
//! whose children are all leaves is that family; in a circuit tree a leaf is
//! the kernel `module::func` itself. Every unattributed name becomes a
//! `KernelSymbol` of `Kernel(family:unclassified)`, so coverage gaps stay
//! visible. A circuit tree's plan digest gives `PLANNED_BY`. Every entity an
//! edge reaches but the run does not describe is a reference stub. All of it
//! is private: it carries campaign numbers.

use std::collections::BTreeMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::ids::{hex_id, natural_key, node_id};
use crate::port::{RunRecord, UNKNOWN};
use crate::schema::{
    Body, CircuitJoin, KernelSample, KernelSymbol, Kind, MappedBy, NodeRef, ProfileRun, Record,
    Relation, Variant,
};
use crate::{Error, Result, bail, canon};

pub const TREE_SCHEMA: &str = "metprof.kernel_tree/1";
pub const UNCLASSIFIED: &str = "family:unclassified";

#[derive(Debug, Clone, PartialEq)]
pub struct GraphEdge {
    pub from: NodeRef,
    pub rel: Relation,
    pub to: NodeRef,
    pub weight: f32,
}

/// Records (complete before stub for one entity; stubs only for entities
/// with no complete record here) and edges, both in a fixed order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    pub records: Vec<Record>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TreeDoc {
    schema: String,
    time_unit: String,
    provenance: Map<String, Value>,
    source: String,
    total_time: f64,
    coverage: f64,
    kernels: Map<String, Value>,
    unattributed: Map<String, Value>,
    #[serde(rename = "baseline_time")]
    _baseline_time: Option<f64>,
    #[serde(rename = "delta")]
    _delta: Option<f64>,
    #[serde(rename = "fused_away")]
    _fused_away: Option<Value>,
    #[serde(rename = "new")]
    _new: Option<Value>,
    /// A slope tree's widths and band; kept whole in the run's tree.
    #[serde(rename = "slope")]
    _slope: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeDoc {
    fuses: Vec<String>,
    total_time: f64,
    self_time: f64,
    launches: u64,
    mean_us: Option<f64>,
    min_us: Option<f64>,
    max_us: Option<f64>,
    #[serde(deserialize_with = "stated")]
    p10_us: Option<f64>,
    #[serde(deserialize_with = "stated")]
    median_us: Option<f64>,
    #[serde(deserialize_with = "stated")]
    p90_us: Option<f64>,
    kernels: Map<String, Value>,
    circuit: Option<CircuitJoin>,
    variants: Option<Vec<Variant>>,
    #[serde(rename = "members")]
    _members: Option<Vec<String>>,
    #[serde(rename = "baseline_time")]
    _baseline_time: Option<f64>,
    #[serde(rename = "delta")]
    _delta: Option<f64>,
    slope: Option<Value>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LooseDoc {
    #[serde(rename = "total_time")]
    _total_time: f64,
    #[serde(rename = "launches")]
    _launches: u64,
    #[serde(rename = "p10_us", deserialize_with = "stated")]
    _p10_us: Option<f64>,
    #[serde(rename = "median_us", deserialize_with = "stated")]
    _median_us: Option<f64>,
    #[serde(rename = "p90_us", deserialize_with = "stated")]
    _p90_us: Option<f64>,
    #[serde(rename = "variants")]
    _variants: Vec<Variant>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Source {
    Circuit,
    Families,
}

struct Builder {
    recorded_at: String,
    records: BTreeMap<NodeRef, Record>,
    edges: BTreeMap<(NodeRef, Relation, NodeRef), f32>,
}

impl Builder {
    fn put(&mut self, r: Record) {
        let n = r.node();
        let keep_old = self.records.get(&n).is_some_and(|old| !old.is_stub());
        if !keep_old {
            self.records.insert(n, r);
        }
    }

    fn edge(&mut self, from: &NodeRef, rel: Relation, to: NodeRef, weight: f32) {
        if !self.records.contains_key(&to) {
            self.records
                .insert(to.clone(), Record::stub(&to, &self.recorded_at, true));
        }
        self.edges.entry((from.clone(), rel, to)).or_insert(weight);
    }

    fn finish(self) -> Graph {
        Graph {
            records: self.records.into_values().collect(),
            edges: self
                .edges
                .into_iter()
                .map(|((from, rel, to), weight)| GraphEdge {
                    from,
                    rel,
                    to,
                    weight,
                })
                .collect(),
        }
    }
}

/// A key the tree always states, whose value may be null: unlike a plain
/// `Option`, a missing key is refused.
fn stated<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Option<f64>, D::Error> {
    Option::<f64>::deserialize(d)
}

fn parse<T: for<'de> Deserialize<'de>>(place: &str, v: &Value) -> Result<T> {
    serde_json::from_value(v.clone()).map_err(|e| Error(format!("{place}: {e}")))
}

fn tree_of(run: &RunRecord) -> Result<Option<(TreeDoc, Value)>> {
    let trees: Vec<&Value> = run
        .files
        .iter()
        .filter_map(|f| f.json.as_ref())
        .filter(|j| j.get("schema").and_then(Value::as_str) == Some(TREE_SCHEMA))
        .collect();
    match trees.as_slice() {
        [] => Ok(None),
        [t] => {
            let doc: TreeDoc = parse("kernel tree", t)?;
            if doc.schema != TREE_SCHEMA || doc.time_unit != "us" {
                bail!("kernel tree: schema/time_unit is not {TREE_SCHEMA:?}/\"us\"");
            }
            if !doc.total_time.is_finite() || doc.total_time <= 0.0 {
                bail!("kernel tree: total_time must be > 0");
            }
            Ok(Some((doc, (*t).clone())))
        }
        _ => bail!(
            "run record: {} kernel trees; a run has at most one",
            trees.len()
        ),
    }
}

/// The run's `ProfileRun` natural key: capture, command, args, config.
pub fn run_key(run: &RunRecord) -> Result<String> {
    let capture = run.stamps.capture_sha256.as_deref().unwrap_or(UNKNOWN);
    let args = canon::digest(&run.args)?;
    Ok(natural_key(&[
        capture,
        &run.command,
        &args,
        &run.stamps.config_sha256,
    ]))
}

/// The records and edges of one run, kernel samples to depth `max_depth`.
pub fn derive(run: &RunRecord, max_depth: usize) -> Result<Graph> {
    run.check()?;
    let tree = tree_of(run)?;
    let s = &run.stamps;
    let mut b = Builder {
        recorded_at: s.recorded_at.clone(),
        records: BTreeMap::new(),
        edges: BTreeMap::new(),
    };
    let key = run_key(run)?;
    let run_node = NodeRef::new(Kind::ProfileRun, key.clone());
    let body = ProfileRun {
        command: run.command.clone(),
        args: run.args.clone(),
        result: run.result.clone(),
        kernel_tree: tree.as_ref().map(|(_, whole)| whole.clone()),
        provenance: tree.as_ref().map(|(t, _)| t.provenance.clone()),
        files: run
            .files
            .iter()
            .map(|f| crate::port::FileRef {
                json: None,
                ..f.clone()
            })
            .collect(),
        metprof_version: s.metprof_version.clone(),
        metprof_git_sha: s.metprof_git_sha.clone(),
        families_digest: s.config_sha256.clone(),
        capture_sha256: s.capture_sha256.clone().unwrap_or_else(|| UNKNOWN.into()),
        identity: s.identity.clone(),
        coverage: tree.as_ref().map(|(t, _)| t.coverage),
        total_time: tree.as_ref().map(|(t, _)| t.total_time),
        time_unit: tree.as_ref().map(|(t, _)| t.time_unit.clone()),
    };
    let summary = match &tree {
        Some((t, _)) => format!(
            "metprof {} run, {} kernel tree, {} top-level nodes, {} unattributed names",
            run.command,
            t.source,
            t.kernels.len(),
            t.unattributed.len()
        ),
        None => format!("metprof {} run", run.command),
    };
    b.put(Record::complete(
        key.clone(),
        summary,
        &s.recorded_at,
        true,
        Body::ProfileRun(Box::new(body)),
    ));
    let id = &s.identity;
    let links = [
        (
            Relation::ON_HARDWARE,
            Kind::HardwareClass,
            &id.hardware_class,
        ),
        (Relation::IN_CAMPAIGN, Kind::Campaign, &id.campaign),
        (Relation::ON_BOX, Kind::Box, &id.box_alias),
    ];
    for (rel, kind, k) in links {
        b.edge(&run_node, rel, NodeRef::new(kind, k.clone()), 1.0);
    }
    if let Some((t, _)) = tree {
        samples(&mut b, &run_node, &t, max_depth)?;
    }
    Ok(b.finish())
}

struct Walk<'a> {
    run: &'a NodeRef,
    run_hex: String,
    source: Source,
    total: f64,
    max_depth: usize,
    ord: u64,
}

fn samples(b: &mut Builder, run: &NodeRef, t: &TreeDoc, max_depth: usize) -> Result<()> {
    let source = match t.source.as_str() {
        "circuit" => Source::Circuit,
        "families" => Source::Families,
        other => bail!("kernel tree: unknown source {other:?}"),
    };
    let mut w = Walk {
        run,
        run_hex: hex_id(node_id(run)),
        source,
        total: t.total_time,
        max_depth,
        ord: 0,
    };
    walk(b, &mut w, &t.kernels, "", None, None, 0)?;
    for (name, loose) in &t.unattributed {
        parse::<LooseDoc>(&format!("unattributed {name:?}"), loose)?;
        symbol(b, name, None, NodeRef::new(Kind::Kernel, UNCLASSIFIED));
    }
    Ok(())
}

fn symbol(b: &mut Builder, name: &str, by: Option<MappedBy>, kernel: NodeRef) {
    let sym = NodeRef::new(Kind::KernelSymbol, name);
    let body = KernelSymbol {
        name: name.to_string(),
        grid_hint: None,
        mapped_by: by,
    };
    let rec = Record::complete(
        name,
        format!("profiler symbol {name}"),
        &b.recorded_at.clone(),
        true,
        Body::KernelSymbol(Box::new(body)),
    );
    b.put(rec);
    b.edge(&sym, Relation::SYMBOL_OF, kernel, 1.0);
}

/// The kernel a node resolves to, and the profiler symbol it is (if a leaf of
/// a family tree).
fn resolve(w: &Walk, key: &str, parent: Option<&str>, n: &NodeDoc) -> Result<Option<NodeRef>> {
    let leaf = n.kernels.is_empty();
    Ok(match w.source {
        Source::Circuit => leaf.then(|| NodeRef::new(Kind::Kernel, key)),
        Source::Families if leaf => match parent {
            Some(f) => Some(NodeRef::new(Kind::Kernel, format!("family:{f}"))),
            None => bail!("family tree: leaf {key:?} has no family above it"),
        },
        Source::Families => {
            let mut all_leaves = true;
            for (k, c) in &n.kernels {
                let c: NodeDoc = parse(&format!("tree node {k:?}"), c)?;
                all_leaves &= c.kernels.is_empty();
            }
            all_leaves.then(|| NodeRef::new(Kind::Kernel, format!("family:{key}")))
        }
    })
}

fn fused_member(source: Source, m: &str) -> NodeRef {
    match source {
        Source::Families => NodeRef::new(Kind::Kernel, format!("family:{m}")),
        Source::Circuit => NodeRef::new(Kind::Kernel, format!("op:{m}")),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk(
    b: &mut Builder,
    w: &mut Walk,
    map: &Map<String, Value>,
    prefix: &str,
    parent_key: Option<&str>,
    parent_sample: Option<&NodeRef>,
    depth: usize,
) -> Result<()> {
    for (k, v) in map {
        let path = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}/{k}")
        };
        let n: NodeDoc = parse(&format!("tree node {path:?}"), v)?;
        if n.variants.is_some() != n.kernels.is_empty() {
            bail!("tree node {path:?}: variants are required on a leaf and refused elsewhere");
        }
        if let Some(c) = &n.circuit {
            b.edge(
                w.run,
                Relation::PLANNED_BY,
                NodeRef::new(Kind::CircuitPlan, c.plan_digest.clone()),
                1.0,
            );
        }
        let kernel = resolve(w, k, parent_key, &n)?;
        let mut here = None;
        if depth <= w.max_depth {
            let key = natural_key(&[&w.run_hex, &path]);
            let sample = NodeRef::new(Kind::KernelSample, key.clone());
            let body = KernelSample {
                node_path: path.clone(),
                depth: depth as u64,
                total_time: n.total_time,
                self_time: n.self_time,
                launches: n.launches,
                mean_us: n.mean_us,
                min_us: n.min_us,
                max_us: n.max_us,
                p10_us: n.p10_us,
                median_us: n.median_us,
                p90_us: n.p90_us,
                union_ms: None,
                fuses: n.fuses.clone(),
                circuit: n.circuit.clone(),
                variants: n.variants.clone().unwrap_or_default(),
                ord: w.ord,
                slope: n.slope.clone(),
            };
            w.ord += 1;
            b.put(Record::complete(
                key,
                format!("kernel sample {path}"),
                &b.recorded_at.clone(),
                true,
                Body::KernelSample(Box::new(body)),
            ));
            b.edge(
                w.run,
                Relation::HAS_SAMPLE,
                sample.clone(),
                (n.total_time / w.total) as f32,
            );
            if let Some(p) = parent_sample {
                b.edge(p, Relation::PARENT_OF, sample.clone(), 1.0);
            }
            for m in &n.fuses {
                b.edge(&sample, Relation::FUSES, fused_member(w.source, m), 1.0);
            }
            if let Some(kernel) = &kernel {
                b.edge(&sample, Relation::SAMPLE_OF, kernel.clone(), 1.0);
            }
            here = Some(sample);
        }
        // A symbol's family is knowledge at any depth, sampled or not.
        if let Some(kernel) = kernel
            && w.source == Source::Families
            && n.kernels.is_empty()
        {
            symbol(b, k, Some(MappedBy::Families), kernel);
        }
        walk(b, w, &n.kernels, &path, Some(k), here.as_ref(), depth + 1)?;
    }
    Ok(())
}
