// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: a kernel tree read back from its JSON (pure), strictly: the
//! schema and unit must match, every key must be known, and every stated
//! number must equal what the tree implies (totals, launches, min/max, mean,
//! coverage, baseline deltas, variant sums, identity hashes). Quantiles are
//! read as stated (the file has no per-launch durations) and checked for
//! order, bounds and presence. A tree that fails any of these is refused, so
//! a baseline or a diff never runs on an edited or truncated file.

use indexmap::IndexMap;
use serde_json::{Map, Value};

use crate::core::error::{Error, Result};
use crate::core::model::{Grid, Launch};
use crate::core::repr::quote;
use crate::core::tree::identity::{Identity, QUANTILE_KEYS, Quantiles, Variant, Variants};
use crate::core::tree::slope::{RootSlope, node_slope_from_json, root_slope_from_json};
use crate::core::tree::{
    BaselineView, Join, Loose, Node, SCHEMA, Source, Stats, TIME_UNIT, Tree, us,
};
use crate::fail;

const ROOT_KEYS: [&str; 8] = [
    "schema",
    "time_unit",
    "provenance",
    "source",
    "total_time",
    "coverage",
    "kernels",
    "unattributed",
];
const ROOT_BASELINE: [&str; 4] = ["baseline_time", "delta", "fused_away", "new"];
const NODE_KEYS: [&str; 11] = [
    "fuses",
    "total_time",
    "self_time",
    "launches",
    "mean_us",
    "min_us",
    "max_us",
    "p10_us",
    "median_us",
    "p90_us",
    "kernels",
];
/// `variants` is required on a leaf and refused on an interior node.
/// `slope` is required on every node of a tree with a root `slope`, refused
/// on every node of one without.
const NODE_OPTIONAL: [&str; 6] = [
    "circuit",
    "members",
    "baseline_time",
    "delta",
    "variants",
    "slope",
];
const LOOSE_KEYS: [&str; 6] = [
    "total_time",
    "launches",
    "p10_us",
    "median_us",
    "p90_us",
    "variants",
];
const VARIANT_KEYS: [&str; 13] = [
    "demangled",
    "grid",
    "block",
    "registers_per_thread",
    "static_shared_bytes",
    "dynamic_shared_bytes",
    "config_sha256",
    "kernel_id",
    "launches",
    "total_time",
    "p10_us",
    "median_us",
    "p90_us",
];
/// A mean is a quotient, so it is compared with a tolerance; every other time
/// is ns / 1000 and is compared exactly in ns.
const MEAN_TOL_US: f64 = 1e-6;

fn keys(place: &str, m: &Map<String, Value>, required: &[&str], optional: &[&str]) -> Result<()> {
    for k in required {
        if !m.contains_key(*k) {
            fail!("{place}: missing key {}", quote(k));
        }
    }
    for k in m.keys() {
        if !required.contains(&k.as_str()) && !optional.contains(&k.as_str()) {
            fail!("{place}: unknown key {}", quote(k));
        }
    }
    Ok(())
}

fn obj<'v>(place: &str, v: &'v Value) -> Result<&'v Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| Error(format!("{place}: must be an object")))
}

fn ns(place: &str, v: &Value) -> Result<i64> {
    match v.as_f64().filter(|x| x.is_finite()) {
        Some(x) => Ok((x * 1000.0).round() as i64),
        None => fail!("{place}: must be a finite number"),
    }
}

fn opt_ns(place: &str, v: &Value) -> Result<Option<i64>> {
    if v.is_null() {
        Ok(None)
    } else {
        ns(place, v).map(Some)
    }
}

fn count(place: &str, v: &Value) -> Result<u64> {
    v.as_u64()
        .ok_or_else(|| Error(format!("{place}: must be a non-negative integer")))
}

fn strings(place: &str, v: &Value) -> Result<Vec<String>> {
    let items = v
        .as_array()
        .ok_or_else(|| Error(format!("{place}: must be a list of strings")))?;
    items
        .iter()
        .map(|x| {
            x.as_str()
                .map(str::to_string)
                .ok_or_else(|| Error(format!("{place}: must be a list of strings")))
        })
        .collect()
}

fn text(place: &str, v: &Value) -> Result<String> {
    v.as_str()
        .map(str::to_string)
        .ok_or_else(|| Error(format!("{place}: must be a string")))
}

fn int(place: &str, v: &Value) -> Result<i64> {
    v.as_i64()
        .ok_or_else(|| Error(format!("{place}: must be an integer")))
}

fn triple(place: &str, v: &Value) -> Result<Grid> {
    match v.as_array().map(Vec::as_slice) {
        Some([x, y, z]) => Ok((int(place, x)?, int(place, y)?, int(place, z)?)),
        _ => fail!("{place}: must be a list of three integers"),
    }
}

/// The stated quantiles: all three null, or three finite, ordered numbers.
fn quantiles_of(place: &str, m: &Map<String, Value>) -> Result<Option<Quantiles>> {
    if QUANTILE_KEYS.iter().all(|(k, _)| m[*k].is_null()) {
        return Ok(None);
    }
    let mut q = [0.0; 3];
    for (o, (k, _)) in q.iter_mut().zip(QUANTILE_KEYS) {
        *o = match m[k].as_f64().filter(|x| x.is_finite()) {
            Some(x) => x,
            None => fail!("{place}: {k} must be a finite number, or every quantile null"),
        };
    }
    if !(q[0] <= q[1] && q[1] <= q[2]) {
        fail!("{place}: p10_us <= median_us <= p90_us does not hold");
    }
    Ok(Some(Quantiles(q)))
}

/// Quantiles are present exactly when there are launches, and lie within
/// [min, max] when those are stated.
fn check_quantiles(place: &str, q: Option<Quantiles>, n: u64, bounds: Option<Stats>) -> Result<()> {
    let Some(q) = q else {
        if n == 0 {
            return Ok(());
        }
        fail!("{place}: quantiles are null but there are {n} launch(es)");
    };
    if n == 0 {
        fail!("{place}: quantiles are stated but there are no launches");
    }
    if let Some(Stats {
        min: Some(lo),
        max: Some(hi),
        ..
    }) = bounds
        && (q.0[0] < us(lo) - MEAN_TOL_US || q.0[2] > us(hi) + MEAN_TOL_US)
    {
        fail!("{place}: quantiles lie outside [min_us, max_us]");
    }
    Ok(())
}

/// A leaf's or an unattributed name's variants; their launches and times must
/// add up to `own`, and with one variant its quantiles are the owner's.
fn variants(place: &str, v: &Value, own: Stats, q: Option<Quantiles>) -> Result<Variants> {
    let items = v
        .as_array()
        .ok_or_else(|| Error(format!("{place}: variants must be a list")))?;
    let mut out = Variants::new();
    for (i, item) in items.iter().enumerate() {
        let place = format!("{place}: variants[{i}]");
        let m = obj(&place, item)?;
        keys(&place, m, &VARIANT_KEYS, &[])?;
        let f = |k: &str| format!("{place}: {k}");
        let id = Identity {
            demangled: text(&f("demangled"), &m["demangled"])?,
            grid: triple(&f("grid"), &m["grid"])?,
            launch: Launch {
                block: triple(&f("block"), &m["block"])?,
                registers_per_thread: int(&f("registers_per_thread"), &m["registers_per_thread"])?,
                static_shared_bytes: int(&f("static_shared_bytes"), &m["static_shared_bytes"])?,
                dynamic_shared_bytes: int(&f("dynamic_shared_bytes"), &m["dynamic_shared_bytes"])?,
            },
        };
        if m["config_sha256"] != id.config_sha256().as_str() {
            fail!("{place}: config_sha256 is not the SHA-256 of its launch configuration");
        }
        if m["kernel_id"] != id.kernel_id().as_str() {
            fail!("{place}: kernel_id is not the SHA-256 of its name and launch configuration");
        }
        let n = count(&f("launches"), &m["launches"])?;
        let quantiles = quantiles_of(&place, m)?;
        check_quantiles(&place, quantiles, n, None)?;
        if n == 0 {
            fail!("{place}: a variant has at least one launch");
        }
        let var = Variant {
            n,
            sum: ns(&f("total_time"), &m["total_time"])?,
            durs: Vec::new(),
            quantiles,
        };
        if out.insert(id, var).is_some() {
            fail!("{place}: the same identity is listed twice");
        }
    }
    let (n, sum) = out.values().fold((0, 0), |(n, s), v| (n + v.n, s + v.sum));
    if (n, sum) != (own.n, own.sum) {
        fail!("{place}: the variants' launches and times do not add up to its own");
    }
    if out.len() == 1 && out[0].quantiles != q {
        fail!("{place}: a single variant's quantiles differ from its owner's");
    }
    Ok(out)
}

fn loose(key: &str, v: &Value) -> Result<Loose> {
    let place = format!("unattributed {}", quote(key));
    let m = obj(&place, v)?;
    keys(&place, m, &LOOSE_KEYS, &[])?;
    let own = Stats {
        n: count(&place, &m["launches"])?,
        sum: ns(&place, &m["total_time"])?,
        min: None,
        max: None,
    };
    let quantiles = quantiles_of(&place, m)?;
    check_quantiles(&place, quantiles, own.n, None)?;
    let variants = variants(&place, &m["variants"], own, quantiles)?;
    if variants.keys().any(|id| id.demangled != key) {
        fail!("{place}: a variant's demangled name is not the entry's");
    }
    Ok(Loose {
        n: own.n,
        sum: own.sum,
        variants,
        quantiles,
    })
}

pub fn tree_from_json(v: &Value) -> Result<Tree> {
    let root = obj("kernel tree", v)?;
    let with_base = root.contains_key("baseline_time");
    let mut optional = vec!["slope"];
    if with_base {
        optional.extend(ROOT_BASELINE);
    }
    keys("kernel tree", root, &ROOT_KEYS, &optional)?;
    if root["schema"] != SCHEMA {
        fail!(
            "kernel tree: schema {} is not {}",
            root["schema"],
            quote(SCHEMA)
        );
    }
    if root["time_unit"] != TIME_UNIT {
        fail!(
            "kernel tree: time_unit {} is not {}",
            root["time_unit"],
            quote(TIME_UNIT)
        );
    }
    let src = text("kernel tree: source", &root["source"])?;
    let Some(source) = Source::parse(&src) else {
        fail!("kernel tree: unknown source {}", quote(&src));
    };
    let mut unattributed = IndexMap::new();
    for (k, l) in obj("kernel tree: unattributed", &root["unattributed"])? {
        unattributed.insert(k.clone(), loose(k, l)?);
    }
    let slope = root.get("slope").map(root_slope_from_json).transpose()?;
    let kernels = nodes("", &root["kernels"], with_base, slope.as_ref())?;
    let baseline = if with_base {
        let mut away = Vec::new();
        let items = root["fused_away"]
            .as_array()
            .ok_or_else(|| Error("kernel tree: fused_away must be a list".into()))?;
        for item in items {
            let o = obj("kernel tree: fused_away", item)?;
            keys(
                "kernel tree: fused_away",
                o,
                &["path", "baseline_time"],
                &[],
            )?;
            away.push((
                text("fused_away path", &o["path"])?,
                ns("fused_away", &o["baseline_time"])?,
            ));
        }
        Some(BaselineView {
            total: ns("kernel tree: baseline_time", &root["baseline_time"])?,
            fused_away: away,
        })
    } else {
        None
    };
    let tree = Tree {
        provenance: obj("kernel tree: provenance", &root["provenance"])?.clone(),
        source,
        kernels,
        unattributed,
        baseline,
        slope,
    };
    if ns("kernel tree: total_time", &root["total_time"])? != tree.total_ns() {
        fail!(
            "kernel tree: total_time {} is not the sum of its nodes and unattributed kernels",
            root["total_time"]
        );
    }
    let cov = root["coverage"].as_f64().unwrap_or(f64::NAN);
    if (cov - tree.coverage()?).abs() > 1e-12 {
        fail!(
            "kernel tree: coverage {} is not attributed / total",
            root["coverage"]
        );
    }
    tree.check_slope()?;
    if let Some(b) = &tree.baseline {
        let delta = ns("kernel tree: delta", &root["delta"])?;
        if delta != tree.total_ns() - b.total {
            fail!("kernel tree: delta is not total_time - baseline_time");
        }
        if strings("kernel tree: new", &root["new"])? != tree.new_paths() {
            fail!("kernel tree: new does not list the nodes without a baseline");
        }
    }
    Ok(tree)
}

fn nodes(
    prefix: &str,
    v: &Value,
    with_base: bool,
    slope: Option<&RootSlope>,
) -> Result<IndexMap<String, Node>> {
    let mut out = IndexMap::new();
    for (k, n) in obj(&format!("{prefix}: kernels"), v)? {
        let path = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}/{k}")
        };
        if k.is_empty() || k.contains('/') {
            fail!(
                "node {}: a key must be non-empty and contain no '/'",
                quote(&path)
            );
        }
        out.insert(k.clone(), node(&path, n, with_base, slope)?);
    }
    Ok(out)
}

fn node(path: &str, v: &Value, with_base: bool, root: Option<&RootSlope>) -> Result<Node> {
    let place = format!("node {}", quote(path));
    let m = obj(&place, v)?;
    keys(&place, m, &NODE_KEYS, &NODE_OPTIONAL)?;
    if with_base != (m.contains_key("baseline_time") && m.contains_key("delta")) {
        fail!(
            "{place}: baseline_time and delta must be present exactly when the tree has a baseline"
        );
    }
    let field = |k: &str| format!("{place}: {k}");
    let kernels = nodes(path, &m["kernels"], with_base, root)?;
    let slope = match (root, m.get("slope")) {
        (Some(r), Some(s)) => Some(node_slope_from_json(&place, s, r)?),
        (None, None) => None,
        _ => fail!("{place}: slope must be present exactly when the tree has one"),
    };
    let launches = count(&field("launches"), &m["launches"])?;
    let total = ns(&field("total_time"), &m["total_time"])?;
    let own = if kernels.is_empty() {
        Stats {
            n: launches,
            sum: ns(&field("self_time"), &m["self_time"])?,
            min: opt_ns(&field("min_us"), &m["min_us"])?,
            max: opt_ns(&field("max_us"), &m["max_us"])?,
        }
    } else {
        if ns(&field("self_time"), &m["self_time"])? != 0 {
            fail!("{place}: an interior node's self_time must be 0");
        }
        Stats::default()
    };
    let quantiles = quantiles_of(&place, m)?;
    let variants = match (kernels.is_empty(), m.get("variants")) {
        (true, Some(v)) => variants(&place, v, own, quantiles)?,
        (false, None) => Variants::new(),
        _ => fail!("{place}: variants must be present exactly on a leaf"),
    };
    let circuit = match m.get("circuit") {
        None => None,
        Some(c) => {
            let c = obj(&field("circuit"), c)?;
            keys(&field("circuit"), c, &["group", "rule", "plan_digest"], &[])?;
            Some(Join {
                group: count(&field("circuit.group"), &c["group"])?,
                rule: text(&field("circuit.rule"), &c["rule"])?,
                plan_digest: text(&field("circuit.plan_digest"), &c["plan_digest"])?,
            })
        }
    };
    let members = match m.get("members") {
        None => Vec::new(),
        Some(x) => strings(&field("members"), x)?,
    };
    let baseline = if with_base {
        opt_ns(&field("baseline_time"), &m["baseline_time"])?
    } else {
        None
    };
    let n = Node {
        fuses: strings(&field("fuses"), &m["fuses"])?,
        members,
        circuit,
        own,
        variants,
        quantiles,
        baseline,
        slope,
        kernels,
    };
    let t = n.total();
    check_quantiles(&place, n.quantiles, t.n, Some(t))?;
    let stated = (
        total,
        launches,
        opt_ns("", &m["min_us"])?,
        opt_ns("", &m["max_us"])?,
    );
    if stated != (t.sum, t.n, t.min, t.max) {
        fail!(
            "{place}: total_time, launches, min_us and max_us are not what its kernels add up to"
        );
    }
    let mean = m["mean_us"].as_f64();
    let want = (t.n > 0).then(|| t.sum as f64 / t.n as f64 / 1000.0);
    let mean_ok = match (mean, want) {
        (None, None) => m["mean_us"].is_null(),
        (Some(a), Some(b)) => (a - b).abs() <= MEAN_TOL_US * (1.0 + b.abs()),
        _ => false,
    };
    if !mean_ok {
        fail!(
            "{place}: mean_us {} is not total_time / launches",
            m["mean_us"]
        );
    }
    if with_base {
        let delta = opt_ns(&field("delta"), &m["delta"])?;
        if delta != n.baseline.map(|b| t.sum - b) {
            fail!(
                "{place}: delta is not total_time - baseline_time ({})",
                us(t.sum)
            );
        }
    }
    Ok(n)
}
