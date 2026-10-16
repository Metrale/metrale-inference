// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: the kernel tree with per-row cost (pure): one family tree per
//! width, each over its pinned decode steps, joined by node path.
//!
//! The written tree is the widest width's, with every node any narrower width
//! has and the widest lacks grafted in at 0 launches, so each node is listed
//! once whatever width it ran at. Every node carries `slope`: its cost per
//! step at each width (0 where it did not run) and the analysis of
//! `core::slope`; a family node (the parent of its kernels) also carries its
//! declared `cost_driver`. The root's `slope` states the band, the widths
//! (rows, steps, graph share, capture digest) and the whole step's own slope,
//! unattributed kernels included. Nothing in `slope` is stated that the
//! per-width points do not imply: the reader re-derives it.

use indexmap::IndexMap;
use serde_json::{Map, Value, json};

use crate::core::error::{Error, Result};
use crate::core::families::FamilyTable;
use crate::core::repr::quote;
use crate::core::slope::{
    Analysis, DRIVERS, Point, SlopeConfig, UNDECLARED, analyze, check_band, order_widths,
};
use crate::core::tree::{Node, Source, Tree};
use crate::fail;

/// One width: its rows, its pinned steps, and its tree over those steps.
#[derive(Debug, Clone)]
pub struct Width {
    pub rows: u64,
    pub steps: usize,
    pub graph_share: f64,
    pub capture_sha256: String,
    pub tree: Tree,
}

/// A width as the root states it.
#[derive(Debug, Clone, PartialEq)]
pub struct WidthInfo {
    pub rows: u64,
    pub steps: usize,
    pub graph_share: f64,
    pub capture_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeSlope {
    /// One per width, by increasing rows.
    pub points: Vec<Point>,
    pub analysis: Analysis,
    /// Family nodes only: the declared driver or `undeclared`.
    pub cost_driver: Option<String>,
}

impl NodeSlope {
    pub fn new(points: Vec<Point>, band: f64, cost_driver: Option<String>) -> Result<NodeSlope> {
        let analysis = analyze(&points, band)?;
        Ok(NodeSlope {
            points,
            analysis,
            cost_driver,
        })
    }

    pub fn to_json(&self) -> Value {
        let a = &self.analysis;
        let mut m = Map::new();
        m.insert("class".into(), json!(a.class.as_str()));
        if let Some(d) = &self.cost_driver {
            m.insert("cost_driver".into(), json!(d));
        }
        m.insert("absent_at".into(), json!(a.absent_at));
        let widths: Vec<Value> = self
            .points
            .iter()
            .map(|p| {
                json!({
                    "rows": p.rows,
                    "ms_per_step": p.ms_per_step,
                    "launches_per_step": p.launches_per_step,
                    "launches_per_row": p.launches_per_row(),
                })
            })
            .collect();
        m.insert("widths".into(), Value::Array(widths));
        let marginals: Vec<Value> = a
            .marginals
            .iter()
            .map(|x| {
                json!({
                    "from_rows": x.from_rows,
                    "to_rows": x.to_rows,
                    "ms_per_row": x.ms_per_row,
                    "launches_per_row": x.launches_per_row,
                })
            })
            .collect();
        m.insert("marginals".into(), Value::Array(marginals));
        let changes: Vec<Value> = a
            .changes
            .iter()
            .map(|c| {
                json!({
                    "at_rows": c.at_rows,
                    "delta_ms_per_row": c.delta_ms_per_row,
                    "class": c.class.as_str(),
                })
            })
            .collect();
        m.insert("changes".into(), Value::Array(changes));
        Value::Object(m)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RootSlope {
    pub band_ms_per_row: f64,
    pub widths: Vec<WidthInfo>,
    /// The whole step: every launch in the pinned steps, unattributed included.
    pub step: NodeSlope,
}

impl RootSlope {
    pub fn to_json(&self) -> Value {
        let widths: Vec<Value> = self
            .widths
            .iter()
            .map(|w| {
                json!({
                    "rows": w.rows,
                    "steps": w.steps,
                    "graph_share": w.graph_share,
                    "capture_sha256": w.capture_sha256,
                })
            })
            .collect();
        json!({
            "flat_band_ms_per_row": self.band_ms_per_row,
            "widths": widths,
            "step": self.step.to_json(),
        })
    }

    fn widest(&self) -> Result<&WidthInfo> {
        match self.widths.last() {
            Some(w) => Ok(w),
            None => fail!("kernel tree slope: no widths"),
        }
    }

    /// The widest point of `s` must be the tree's own figure at that width:
    /// `total_ns` over `launches` in the widest width's steps.
    pub fn check_widest(&self, place: &str, s: &NodeSlope, total_ns: i64, n: u64) -> Result<()> {
        let w = self.widest()?;
        if s.points.last() != Some(&Point::of(w.rows, w.steps, total_ns, n)) {
            fail!(
                "{place}: the widest slope point is not its total_time and launches over the widest width's steps"
            );
        }
        Ok(())
    }
}

/// Launches in the tree, attributed and not.
pub fn launches(t: &Tree) -> u64 {
    t.attributed().n + t.unattributed.values().map(|l| l.n).sum::<u64>()
}

fn hollow(n: &Node) -> Node {
    Node {
        fuses: n.fuses.clone(),
        members: n.members.clone(),
        circuit: n.circuit.clone(),
        kernels: n
            .kernels
            .iter()
            .map(|(k, c)| (k.clone(), hollow(c)))
            .collect(),
        ..Node::default()
    }
}

/// Add every node of `from` that `into` lacks, at 0 launches.
fn graft(into: &mut IndexMap<String, Node>, from: &IndexMap<String, Node>) {
    for (k, n) in from {
        match into.get_mut(k) {
            Some(d) => graft(&mut d.kernels, &n.kernels),
            None => {
                into.insert(k.clone(), hollow(n));
            }
        }
    }
}

fn find<'t>(map: &'t IndexMap<String, Node>, path: &[&str]) -> Option<&'t Node> {
    let (first, rest) = path.split_first()?;
    let n = map.get(*first)?;
    if rest.is_empty() {
        Some(n)
    } else {
        find(&n.kernels, rest)
    }
}

fn is_family(key: &str, n: &Node, families: &[&str]) -> bool {
    families.contains(&key)
        && !n.kernels.is_empty()
        && n.kernels.values().all(|c| c.kernels.is_empty())
}

struct Ctx<'a> {
    widths: &'a [Width],
    cfg: &'a SlopeConfig,
    families: Vec<&'a str>,
}

fn set(ctx: &Ctx, map: &mut IndexMap<String, Node>, prefix: &[&str]) -> Result<()> {
    for (k, n) in map.iter_mut() {
        let mut path = prefix.to_vec();
        path.push(k.as_str());
        let points = ctx
            .widths
            .iter()
            .map(|w| {
                let t = find(&w.tree.kernels, &path)
                    .map(Node::total)
                    .unwrap_or_default();
                Point::of(w.rows, w.steps, t.sum, t.n)
            })
            .collect();
        let driver = is_family(k, n, &ctx.families).then(|| ctx.cfg.driver(k).to_string());
        n.slope = Some(NodeSlope::new(points, ctx.cfg.band_ms_per_row, driver)?);
        set(ctx, &mut n.kernels, &path)?;
    }
    Ok(())
}

/// The slope tree of `widths` (any order; at least three distinct widths, each
/// a sealed family tree of its pinned steps).
pub fn slope_tree(mut widths: Vec<Width>, cfg: &SlopeConfig, table: &FamilyTable) -> Result<Tree> {
    let rows: Vec<u64> = widths.iter().map(|w| w.rows).collect();
    order_widths(&rows)?;
    widths.sort_by_key(|w| w.rows);
    for w in &widths {
        if w.steps == 0 {
            fail!("slope: width {} rows has no pinned steps", w.rows);
        }
        if w.tree.source != Source::Families || w.tree.slope.is_some() {
            fail!("slope: width {} rows is not a plain family tree", w.rows);
        }
    }
    let Some(widest) = widths.last() else {
        fail!("slope: no widths");
    };
    let mut out = widest.tree.clone();
    for w in widths.iter().rev().skip(1) {
        graft(&mut out.kernels, &w.tree.kernels);
    }
    let ctx = Ctx {
        widths: &widths,
        cfg,
        families: table.families(),
    };
    set(&ctx, &mut out.kernels, &[])?;
    let step = widths
        .iter()
        .map(|w| Point::of(w.rows, w.steps, w.tree.total_ns(), launches(&w.tree)))
        .collect();
    out.slope = Some(RootSlope {
        band_ms_per_row: cfg.band_ms_per_row,
        widths: widths
            .iter()
            .map(|w| WidthInfo {
                rows: w.rows,
                steps: w.steps,
                graph_share: w.graph_share,
                capture_sha256: w.capture_sha256.clone(),
            })
            .collect(),
        step: NodeSlope::new(step, cfg.band_ms_per_row, None)?,
    });
    Ok(out)
}

// ---- reading back ----------------------------------------------------------

fn obj<'v>(place: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error(format!("{place}: must be an object")))
}

fn list<'v>(place: &str, v: Option<&'v Value>) -> Result<&'v Vec<Value>> {
    v.and_then(Value::as_array)
        .ok_or_else(|| Error(format!("{place}: must be a list")))
}

fn num(place: &str, v: Option<&Value>) -> Result<f64> {
    match v.and_then(Value::as_f64).filter(|x| x.is_finite()) {
        Some(x) => Ok(x),
        None => fail!("{place}: must be a finite number"),
    }
}

fn whole(place: &str, v: Option<&Value>) -> Result<u64> {
    v.and_then(Value::as_u64)
        .ok_or_else(|| Error(format!("{place}: must be a non-negative integer")))
}

/// A stated `slope`, rebuilt from its per-width `ms_per_step` and
/// `launches_per_step` at `widths`; every other number and class must be
/// exactly what those imply.
fn slope_from_json(place: &str, v: &Value, widths: &[WidthInfo], band: f64) -> Result<NodeSlope> {
    let place = format!("{place}: slope");
    let m = obj(&place, v)?;
    let stated = list(&format!("{place}: widths"), m.get("widths"))?;
    if stated.len() != widths.len() {
        fail!(
            "{place}: {} widths, the root states {}",
            stated.len(),
            widths.len()
        );
    }
    let mut points = Vec::with_capacity(stated.len());
    for (w, s) in widths.iter().zip(stated) {
        let at = format!("{place}: width {}", w.rows);
        let s = obj(&at, s)?;
        points.push(Point {
            rows: w.rows,
            ms_per_step: num(&format!("{at}: ms_per_step"), s.get("ms_per_step"))?,
            launches_per_step: num(
                &format!("{at}: launches_per_step"),
                s.get("launches_per_step"),
            )?,
        });
    }
    let driver = match m.get("cost_driver") {
        None => None,
        Some(d) => match d.as_str() {
            Some(x) if x == UNDECLARED || DRIVERS.contains(&x) => Some(x.to_string()),
            _ => fail!(
                "{place}: cost_driver {d} is not one of {} or {}",
                DRIVERS.join(", "),
                quote(UNDECLARED)
            ),
        },
    };
    let s = NodeSlope::new(points, band, driver)?;
    if s.to_json() != *v {
        fail!("{place}: is not what its per-width points imply");
    }
    Ok(s)
}

/// A node's stated `slope`, at the root's widths and band.
pub fn node_slope_from_json(place: &str, v: &Value, root: &RootSlope) -> Result<NodeSlope> {
    slope_from_json(place, v, &root.widths, root.band_ms_per_row)
}

/// The root's stated `slope`: its band, its widths (strictly increasing rows,
/// each with steps) and the whole step's slope, checked like a node's.
pub fn root_slope_from_json(v: &Value) -> Result<RootSlope> {
    const PLACE: &str = "kernel tree: slope";
    let m = obj(PLACE, v)?;
    let band = check_band(
        PLACE,
        num(
            &format!("{PLACE}: flat_band_ms_per_row"),
            m.get("flat_band_ms_per_row"),
        )?,
    )?;
    let mut widths = Vec::new();
    for (i, w) in list(&format!("{PLACE}: widths"), m.get("widths"))?
        .iter()
        .enumerate()
    {
        let at = format!("{PLACE}: widths[{i}]");
        let w = obj(&at, w)?;
        let steps = whole(&format!("{at}: steps"), w.get("steps"))?;
        if steps == 0 {
            fail!("{at}: steps must be at least 1");
        }
        widths.push(WidthInfo {
            rows: whole(&format!("{at}: rows"), w.get("rows"))?,
            steps: usize::try_from(steps).map_err(|e| Error(format!("{at}: steps: {e}")))?,
            graph_share: num(&format!("{at}: graph_share"), w.get("graph_share"))?,
            capture_sha256: w
                .get("capture_sha256")
                .and_then(Value::as_str)
                .ok_or_else(|| Error(format!("{at}: capture_sha256 must be a string")))?
                .to_string(),
        });
    }
    let step_v = m
        .get("step")
        .ok_or_else(|| Error(format!("{PLACE}: missing key \"step\"")))?;
    let step = slope_from_json("kernel tree: step", step_v, &widths, band)?;
    if step.cost_driver.is_some() {
        fail!("{PLACE}: the step has no cost_driver");
    }
    let root = RootSlope {
        band_ms_per_row: band,
        widths,
        step,
    };
    if root.to_json() != *v {
        fail!("{PLACE}: is not what its widths and points imply");
    }
    Ok(root)
}
