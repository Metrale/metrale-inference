// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `tree` and `tree-diff`: the per-kernel tree of one segment, as
//! a JSON file, and the comparison of two such files.
//!
//! The tree comes from the circuit's fusion plan when `--circuit` is given
//! (with `--mode`, `--rows`, `--policy` and `--kernels-dir`, all or none), and
//! from the family table (with an optional `--model` overlay) otherwise. The
//! coverage gate is `[tree] coverage_min` of the config; it has no default.

use serde_json::{Map, Value, json};

use crate::adapters::circuit::{KernelsDir, plan_texts};
use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::busy::DEVICE;
use crate::cli::{Command, Done, Io, Run, load};
use crate::core::error::{Error, Result};
use crate::core::families::{classify, table_from_toml};
use crate::core::model::{Kernel, ms_to_ns};
use crate::core::provenance::{Digest, UNKNOWN, parse_provenance, render, sha256_hex};
use crate::core::segment::split_requests;
use crate::core::tree::circuit::{circuit_tree, plan_provenance, plan_view};
use crate::core::tree::config::{Resolved, TreeConfig, tree_config};
use crate::core::tree::diff::{apply_baseline, tree_diff};
use crate::core::tree::families::families_tree;
use crate::core::tree::json::tree_from_json;
use crate::core::tree::{Tree, span, us};
use crate::fail;

const fn opt(
    name: &'static str,
    kind: Kind,
    required: bool,
    metavar: &'static str,
    help: &'static str,
) -> Opt {
    Opt {
        name,
        kind,
        required,
        metavar,
        help,
    }
}

const CIRCUIT_FLAGS: [&str; 5] = ["circuit", "mode", "rows", "policy", "kernels-dir"];

pub const COMMANDS: &[&Command] = &[
    &Command {
        spec: Spec {
            name: "tree",
            about: "per-kernel time tree of one segment (circuit plan or family table) as JSON",
            positionals: &[],
            opts: &[
                opt("capture", Kind::File, true, "SQLITE", "nsys sqlite export"),
                DEVICE,
                opt(
                    "segment",
                    Kind::Str,
                    true,
                    "all|N",
                    "the whole window, or request N (0-based) of --gap-ms/--expect",
                ),
                opt(
                    "gap-ms",
                    Kind::Float,
                    false,
                    "MS",
                    "request boundary gap (with a request --segment only)",
                ),
                opt(
                    "expect",
                    Kind::Int,
                    false,
                    "N",
                    "requests in the window (with a request --segment only)",
                ),
                opt(
                    "config",
                    Kind::File,
                    true,
                    "TOML",
                    "family table with its [tree] section",
                ),
                opt(
                    "provenance",
                    Kind::File,
                    true,
                    "TOML",
                    "what the capture was measured on",
                ),
                opt(
                    "model",
                    Kind::Str,
                    false,
                    "NAME",
                    "family-table overlay [tree.model.NAME] (families only)",
                ),
                opt(
                    "circuit",
                    Kind::Str,
                    false,
                    "ARCH",
                    "plan with this circuit arch",
                ),
                opt(
                    "mode",
                    Kind::Str,
                    false,
                    "MODE",
                    "circuit mode (e.g. decode)",
                ),
                opt("rows", Kind::Int, false, "R", "rows the plan is fused for"),
                opt(
                    "policy",
                    Kind::Str,
                    false,
                    "RECIPE",
                    "the instance (recipe id) whose policy and precision apply",
                ),
                opt(
                    "kernels-dir",
                    Kind::Dir,
                    false,
                    "DIR",
                    "the engine's kernels/ directory",
                ),
                opt(
                    "baseline",
                    Kind::File,
                    false,
                    "JSON",
                    "an older tree: add baseline_time and delta per node",
                ),
                opt("out", Kind::File, true, "JSON", "where the tree is written"),
            ],
        },
        run: Run::Files(tree),
    },
    &Command {
        spec: Spec {
            name: "tree-diff",
            about: "node-by-node delta of two kernel trees, largest first, with fused-away and new nodes",
            positionals: &[("a", "older tree JSON"), ("b", "newer tree JSON")],
            opts: &[opt(
                "out",
                Kind::File,
                false,
                "JSON",
                "also write the full diff here",
            )],
        },
        run: Run::Files(diff),
    },
];

fn segment(a: &Args, kernels: &[Kernel]) -> Result<(String, Vec<usize>)> {
    let seg = a.str("segment")?;
    let (gap, expect) = (a.opt_float("gap-ms")?, a.opt_int("expect")?);
    if seg == "all" {
        if gap.is_some() || expect.is_some() {
            fail!("argument --segment all takes no --gap-ms or --expect");
        }
        return Ok((seg.to_string(), (0..kernels.len()).collect()));
    }
    let Ok(n) = seg.parse::<usize>() else {
        fail!("argument --segment: expected \"all\" or a request index, got {seg:?}");
    };
    let (Some(gap), Some(expect)) = (gap, expect) else {
        fail!("argument --segment {n} requires --gap-ms and --expect");
    };
    let segs = split_requests(kernels, ms_to_ns(gap), expect)?;
    match segs.get(n) {
        Some(s) => Ok((format!("request.{n}"), s.kernels.clone())),
        None => fail!(
            "argument --segment {n}: the window has {} request(s)",
            segs.len()
        ),
    }
}

pub(super) fn digest(io: &Io, d: &Digest, what: &str) -> Result<String> {
    Ok(match d {
        Digest::Path(p) => sha256_hex(&load::read_bytes(io, p).map_err(|e| e.context(what))?),
        Digest::Sha256(h) => h.clone(),
        Digest::Unknown => UNKNOWN.to_string(),
    })
}

fn circuit_flags(a: &Args) -> Result<bool> {
    let given: Vec<&str> = CIRCUIT_FLAGS
        .iter()
        .copied()
        .filter(|f| a.opt_str(f).is_some())
        .collect();
    if given.is_empty() {
        return Ok(false);
    }
    if given.len() != CIRCUIT_FLAGS.len() {
        let missing: Vec<String> = CIRCUIT_FLAGS
            .iter()
            .filter(|f| !given.contains(f))
            .map(|f| format!("--{f}"))
            .collect();
        fail!(
            "a circuit tree needs --circuit, --mode, --rows, --policy and --kernels-dir; missing {}",
            missing.join(", ")
        );
    }
    if a.opt_str("model").is_some() {
        fail!(
            "argument --model is a family-table overlay; a circuit tree takes its structure from the plan"
        );
    }
    Ok(true)
}

fn rows(a: &Args) -> Result<u64> {
    let r = a.int("rows")?;
    match u64::try_from(r) {
        Ok(r) if r > 0 => Ok(r),
        _ => fail!("argument --rows: must be at least 1, got {r}"),
    }
}

pub(super) fn overlay(
    a: &Args,
    tc: &TreeConfig,
    table: &crate::core::families::FamilyTable,
) -> Result<Resolved> {
    match a.opt_str("model") {
        None => Ok(Resolved::default()),
        Some(m) => match tc.models.get(m) {
            Some(o) => o.resolve(table),
            None => {
                let known: Vec<&str> = tc.models.keys().map(String::as_str).collect();
                fail!(
                    "argument --model: no [tree.model.{m}] in the config (declared: {})",
                    known.join(", ")
                )
            }
        },
    }
}

fn tree(a: &Args, io: &Io) -> Result<Done> {
    let config = a.str("config")?;
    let doc = load::read_toml(io, config)?;
    let table = table_from_toml(&doc).map_err(|e| e.context(config))?;
    let tc = tree_config(&doc).map_err(|e| e.context(config))?;
    let prov_path = a.str("provenance")?;
    let spec =
        parse_provenance(&load::read_toml(io, prov_path)?).map_err(|e| e.context(prov_path))?;
    let circuit = circuit_flags(a)?;
    let capture = a.str("capture")?;
    let device = a.int("device")?;
    let kernels = load::kernels(io, capture, device)?;
    let (seg_name, seg) = segment(a, &kernels)?;
    let capture_sha = sha256_hex(&load::read_bytes(io, capture)?);
    let binary = digest(io, &spec.binary, "provenance binary")?;
    let env = digest(io, &spec.env, "provenance env")?;
    let prov = |c: Option<Value>| render(&spec, &capture_sha, &binary, &env, device, &seg_name, c);
    let mut summary = Map::new();
    let mut aligned = None;
    let mut tree: Tree = if circuit {
        let dir = KernelsDir {
            files: io.files,
            root: a.str("kernels-dir")?,
        };
        let (recipe, arch) = (a.str("policy")?, a.str("circuit")?);
        let texts = plan_texts(&dir, recipe, arch)?;
        let plan = plan_view(&texts, recipe, arch, a.str("mode")?, rows(a)?)?;
        let ks: Vec<Kernel> = seg.iter().map(|&i| kernels[i].clone()).collect();
        let (mut t, al) = circuit_tree(&plan, &tc.circuit_names, &ks, Map::new())?;
        aligned = Some(format!(
            "aligned_steps {}, matched {}, missed {}, unmatched {}, outside_steps {}",
            al.steps, al.matched, al.missed, al.unmatched, al.outside_steps
        ));
        let block = plan_provenance(&plan, &al);
        summary.insert("alignment".into(), block.clone());
        t.provenance = prov(Some(block));
        t
    } else {
        let resolved = overlay(a, &tc, &table)?;
        let labels = classify(&kernels, &table);
        families_tree(&kernels, &labels, &seg, &table, &resolved, prov(None))?
    };
    // The segment's window on the capture clock (ns): first launch start to
    // last launch end, `null` for an empty segment.
    let window =
        span(&kernels, &seg).map_or(Value::Null, |(s, e)| json!({"start_ns": s, "end_ns": e}));
    tree.provenance.insert("window".into(), window);
    let coverage = tree.gate(tc.coverage_min).map_err(|e| match &aligned {
        Some(s) => Error(format!(
            "{}; circuit alignment: {s} (launches outside the pinned steps are unattributed)",
            e.0
        )),
        None => e,
    })?;
    if let Some(b) = a.opt_str("baseline") {
        let base = read_tree(io, b)?;
        apply_baseline(&mut tree, &base).map_err(|e| e.context(b))?;
        summary.insert("baseline_time".into(), json!(us(base.total_ns())));
        summary.insert("delta".into(), json!(us(tree.total_ns() - base.total_ns())));
    }
    let out = a.str("out")?;
    let mut bytes =
        serde_json::to_vec_pretty(&tree.to_json()?).map_err(|e| Error(e.to_string()))?;
    bytes.push(b'\n');
    let mut result = Map::new();
    result.insert("out".into(), json!(out));
    result.insert("source".into(), json!(tree.source.as_str()));
    result.insert("segment".into(), json!(seg_name));
    result.insert("launches".into(), json!(seg.len()));
    result.insert("total_time".into(), json!(us(tree.total_ns())));
    result.insert("coverage".into(), json!(coverage));
    result.insert("unattributed_names".into(), json!(tree.unattributed.len()));
    result.insert("nodes".into(), json!(tree.paths().len()));
    result.extend(summary);
    Ok(Done {
        result,
        files: vec![(out.to_string(), bytes)],
    })
}

fn read_tree(io: &Io, path: &str) -> Result<Tree> {
    let bytes = load::read_bytes(io, path)?;
    let v: Value =
        serde_json::from_slice(&bytes).map_err(|e| Error(format!("{path}: not JSON: {e}")))?;
    tree_from_json(&v).map_err(|e| e.context(path))
}

fn diff(a: &Args, io: &Io) -> Result<Done> {
    let (pa, pb) = (a.pos(0)?, a.pos(1)?);
    let d = tree_diff(&read_tree(io, pa)?, &read_tree(io, pb)?)?;
    let mut files = Vec::new();
    if let Some(out) = a.opt_str("out") {
        let mut bytes = serde_json::to_vec_pretty(&Value::Object(d.clone()))
            .map_err(|e| Error(e.to_string()))?;
        bytes.push(b'\n');
        files.push((out.to_string(), bytes));
    }
    Ok(Done { result: d, files })
}
