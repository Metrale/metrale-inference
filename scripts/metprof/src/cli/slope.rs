// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `slope`: per-row cost. The same decode step captured at three
//! or more widths (`--at ROWS=CAPTURE`, any order), each cut to its pinned
//! decode steps (`segment decode`), each a family tree (`tree`), joined into
//! one kernel tree whose every node carries its cost per step at each width,
//! its marginal ms per added row, and its class (FALLING / FLAT / RISING
//! against `[slope] flat_band_ms_per_row`). A family node also carries its
//! declared `[slope.cost_driver]`, or `undeclared`.

use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::busy::DEVICE;
use crate::cli::tree::{digest, overlay};
use crate::cli::{Command, Done, Io, Run, load};
use crate::core::error::{Error, Result};
use crate::core::families::{classify, table_from_toml};
use crate::core::provenance::{parse_provenance, render, sha256_hex};
use crate::core::segment::decode_steps;
use crate::core::slope::{order_widths, slope_config};
use crate::core::tree::config::tree_config;
use crate::core::tree::families::families_tree;
use crate::core::tree::slope::{NodeSlope, Width, slope_tree};
use crate::core::tree::{Tree, span};

/// The segment every width is cut to.
const SEGMENT: &str = "decode_steps";

pub const COMMANDS: &[&Command] = &[&Command {
    spec: Spec {
        name: "slope",
        about: "per-row cost: the kernel tree of the pinned decode step at three or more widths",
        positionals: &[],
        opts: &[
            Opt {
                name: "at",
                kind: Kind::RowsFile,
                required: true,
                metavar: "ROWS=SQLITE",
                help: "rows per step and the nsys sqlite export at that width; three or more",
            },
            DEVICE,
            Opt {
                name: "families",
                kind: Kind::File,
                required: true,
                metavar: "TOML",
                help: "family table with its [tree] and [slope] sections",
            },
            Opt {
                name: "min-graph-share",
                kind: Kind::Float,
                required: true,
                metavar: "SHARE",
                help: "refuse a width whose pinned graph covers less of its busy time",
            },
            Opt {
                name: "provenance",
                kind: Kind::File,
                required: true,
                metavar: "TOML",
                help: "what the captures were measured on",
            },
            Opt {
                name: "model",
                kind: Kind::Str,
                required: false,
                metavar: "NAME",
                help: "family-table overlay [tree.model.NAME]",
            },
            Opt {
                name: "out",
                kind: Kind::File,
                required: true,
                metavar: "JSON",
                help: "where the kernel tree with per-node slope is written",
            },
        ],
    },
    run: Run::Files(slope),
}];

/// `name@rows` for every point of `s`, valued by `f`.
fn per_width(
    row: &mut Map<String, Value>,
    name: &str,
    s: &NodeSlope,
    f: fn(&NodeSlope, usize) -> Value,
) {
    for (i, p) in s.points.iter().enumerate() {
        row.insert(format!("{name}@{}", p.rows), f(s, i));
    }
}

fn summary_row(row: &mut Map<String, Value>, s: &NodeSlope) {
    row.insert("class".into(), json!(s.analysis.class.as_str()));
    per_width(row, "ms_per_step", s, |s, i| json!(s.points[i].ms_per_step));
    for m in &s.analysis.marginals {
        row.insert(
            format!("ms_per_row@{}-{}", m.from_rows, m.to_rows),
            json!(m.ms_per_row),
        );
    }
    per_width(row, "launches_per_row", s, |s, i| {
        json!(s.points[i].launches_per_row())
    });
}

fn slope(a: &Args, io: &Io) -> Result<Done> {
    let at = a.rows_files("at")?;
    order_widths(&at.iter().map(|(r, _)| *r).collect::<Vec<_>>())?;
    let config = a.str("families")?;
    let doc = load::read_toml(io, config)?;
    let table = table_from_toml(&doc).map_err(|e| e.context(config))?;
    let tc = tree_config(&doc).map_err(|e| e.context(config))?;
    let cfg = slope_config(&doc, &table).map_err(|e| e.context(config))?;
    let resolved = overlay(a, &tc, &table)?;
    let prov_path = a.str("provenance")?;
    let spec =
        parse_provenance(&load::read_toml(io, prov_path)?).map_err(|e| e.context(prov_path))?;
    let binary = digest(io, &spec.binary, "provenance binary")?;
    let env = digest(io, &spec.env, "provenance env")?;
    let device = a.int("device")?;
    let min_share = a.float("min-graph-share")?;
    let mut widths = Vec::with_capacity(at.len());
    // The widest width's capture digest and segment window name the tree.
    let mut widest: Option<(u64, String, Value)> = None;
    for (rows, path) in &at {
        let place = format!("--at {rows}={path}");
        let one = || -> Result<(Width, Value)> {
            let kernels = load::kernels(io, path, device)?;
            let (ps, share) = decode_steps(&kernels, min_share)?;
            let seg: Vec<usize> = ps.steps.iter().flatten().copied().collect();
            let labels = classify(&kernels, &table);
            let tree: Tree = families_tree(&kernels, &labels, &seg, &table, &resolved, Map::new())?;
            tree.gate(tc.coverage_min)?;
            let window = span(&kernels, &seg)
                .map_or(Value::Null, |(s, e)| json!({"start_ns": s, "end_ns": e}));
            let width = Width {
                rows: *rows,
                steps: ps.steps.len(),
                graph_share: share,
                capture_sha256: sha256_hex(&load::read_bytes(io, path)?),
                tree,
            };
            Ok((width, window))
        };
        let (w, window) = one().map_err(|e| e.context(&place))?;
        if widest.as_ref().is_none_or(|(r, _, _)| *r < w.rows) {
            widest = Some((w.rows, w.capture_sha256.clone(), window));
        }
        widths.push(w);
    }
    let Some((_, capture_sha, window)) = widest else {
        return Err(Error("slope: no widths".into()));
    };
    let mut tree = slope_tree(widths, &cfg, &table)?;
    tree.provenance = render(&spec, &capture_sha, &binary, &env, device, SEGMENT, None);
    tree.provenance.insert("window".into(), window);
    let Some(root) = tree.slope.clone() else {
        return Err(Error("slope: the joined tree has no slope".into()));
    };
    let mut families: Vec<(f64, String, Map<String, Value>)> = Vec::new();
    for (path, n) in tree.paths() {
        let Some(s) = &n.slope else { continue };
        let Some(driver) = &s.cost_driver else {
            continue;
        };
        let family = path.rsplit('/').next().unwrap_or(&path).to_string();
        let mut row = Map::new();
        row.insert("family".into(), json!(family));
        row.insert("path".into(), json!(path));
        row.insert("cost_driver".into(), json!(driver));
        summary_row(&mut row, s);
        row.insert("absent_at".into(), json!(s.analysis.absent_at));
        let widest_ms = s.points.last().map_or(0.0, |p| p.ms_per_step);
        families.push((widest_ms, path, row));
    }
    families.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
    let width_rows: Vec<Value> = root
        .widths
        .iter()
        .zip(&root.step.points)
        .map(|(w, p)| {
            json!({
                "rows": w.rows,
                "steps": w.steps,
                "graph_share": w.graph_share,
                "ms_per_step": p.ms_per_step,
                "launches_per_step": p.launches_per_step,
            })
        })
        .collect();
    let mut step = Map::new();
    step.insert("family".into(), json!("(whole step)"));
    summary_row(&mut step, &root.step);
    let out = a.str("out")?;
    let mut bytes =
        serde_json::to_vec_pretty(&tree.to_json()?).map_err(|e| Error(e.to_string()))?;
    bytes.push(b'\n');
    let mut result = Map::new();
    result.insert("out".into(), json!(out));
    result.insert("segment".into(), json!(SEGMENT));
    result.insert("flat_band_ms_per_row".into(), json!(root.band_ms_per_row));
    result.insert(
        "step_class".into(),
        json!(root.step.analysis.class.as_str()),
    );
    result.insert("nodes".into(), json!(tree.paths().len()));
    result.insert("widths".into(), Value::Array(width_rows));
    result.insert("step".into(), Value::Array(vec![Value::Object(step)]));
    result.insert(
        "families".into(),
        Value::Array(
            families
                .into_iter()
                .map(|(_, _, r)| Value::Object(r))
                .collect(),
        ),
    );
    Ok(Done {
        result,
        files: vec![(out.to_string(), bytes)],
    })
}
