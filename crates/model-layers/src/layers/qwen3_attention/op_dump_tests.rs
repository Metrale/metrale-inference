// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `op_dump_dir` against the environment reader it replaced, frozen here
//! (`old_op_dump_dir`, unchanged but for taking the three raw values as parameters), over every
//! combination of the three variables.
//!
//! Owner: model-layers (attention).
//! Invariants: the three text levers read exactly as before; no combination differs.

use super::*;
use crate::lever_equivalence::{Old, check, settings_from};

fn old_parse_csv(v: Option<&str>) -> Vec<String> {
    v.map(|s| {
        s.split(',')
            .filter(|p| !p.trim().is_empty())
            .map(|p| p.trim().to_string())
            .collect()
    })
    .unwrap_or_default()
}

fn old_parse_csv_usize(v: Option<&str>) -> Vec<usize> {
    v.map(|s| s.split(',').filter_map(|p| p.trim().parse().ok()).collect())
        .unwrap_or_default()
}

/// 2026-10-17: The old `op_dump_dir`, its three `std::env::var` reads made parameters.
fn old_op_dump_dir(
    dir: Option<&str>,
    layers: Option<&str>,
    ops: Option<&str>,
    layer_idx: usize,
    op: &str,
) -> Option<String> {
    let dir = dir?.to_string();
    if dir.is_empty() {
        return None;
    }
    let layers = old_parse_csv_usize(layers);
    if !layers.is_empty() && !layers.contains(&layer_idx) {
        return None;
    }
    let ops = old_parse_csv(ops);
    if !ops.is_empty() && !ops.iter().any(|o| o == op) {
        return None;
    }
    Some(dir)
}

const DIR: &str = "METRALE_OP_DUMP";
const LAYERS: &str = "METRALE_OP_DUMP_LAYERS";
const OPS: &str = "METRALE_OP_DUMP_OPS";

/// 2026-10-17: Each variable alone over the standard probes; a list is probed with the directory
/// set to `d` on both sides, since without one neither reader dumps.
#[test]
fn each_variable_alone_reads_as_before() {
    for env in [DIR, LAYERS, OPS] {
        let changes = check(
            env,
            Old::Verbatim,
            |p| match env {
                DIR => old_op_dump_dir(p, None, None, 1, "q_proj"),
                LAYERS => old_op_dump_dir(Some("d"), p, None, 1, "q_proj"),
                _ => old_op_dump_dir(Some("d"), None, p, 1, "q_proj"),
            },
            |s| {
                let mut t = s.telemetry.clone();
                if env != DIR {
                    t.op_dump = Some("d".to_string());
                }
                op_dump_dir(&t, 1, "q_proj").map(str::to_string)
            },
        );
        assert!(changes.is_empty(), "{changes:?}");
    }
}

/// 2026-10-17: Every combination of a directory, a layer list and an op list, at two layers
/// and two ops, including empty values, padded entries and entries that do not parse.
#[test]
fn every_combination_reads_as_before() {
    let dirs = [None, Some(""), Some("out"), Some(" ")];
    let layers = [
        None,
        Some(""),
        Some("1"),
        Some(" 1 , 3"),
        Some("x,3"),
        Some(","),
        Some("2"),
    ];
    let ops = [
        None,
        Some(""),
        Some("q_proj"),
        Some(" k_proj , q_proj"),
        Some(","),
        Some("o"),
    ];
    for d in dirs {
        for l in layers {
            for o in ops {
                let pairs: Vec<(&str, &str)> = [(DIR, d), (LAYERS, l), (OPS, o)]
                    .into_iter()
                    .filter_map(|(env, v)| v.map(|v| (env, v)))
                    .collect();
                let s = settings_from(&pairs).expect("text levers take any value");
                for layer_idx in [1, 3] {
                    for op in ["q_proj", "k_proj"] {
                        assert_eq!(
                            op_dump_dir(&s.telemetry, layer_idx, op).map(str::to_string),
                            old_op_dump_dir(d, l, o, layer_idx, op),
                            "{d:?} {l:?} {o:?} L{layer_idx} {op}"
                        );
                    }
                }
            }
        }
    }
}
