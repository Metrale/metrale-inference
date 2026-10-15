// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `families`: the per-family and per-group table of one device's
//! window.

use serde_json::{Map, Value, json};

use crate::cli::args::{Args, Kind, Opt, Spec};
use crate::cli::busy::{CAPTURE, DEVICE, FAMILIES};
use crate::cli::{Command, Io, Run, load};
use crate::core::error::Result;
use crate::core::families::{classify, coverage_gate};
use crate::core::model::ms;
use crate::core::tables::{UNCLASSIFIED, family_tables, into_map, parse_grid_filters};
use crate::core::union::union_ns;

pub const COMMANDS: &[&Command] = &[&Command {
    spec: Spec {
        name: "families",
        about: "union-attributed time, kernel sum, launches and medians per family and group",
        positionals: &[CAPTURE],
        opts: &[
            DEVICE,
            FAMILIES,
            Opt {
                name: "grid-filter",
                kind: Kind::Append,
                required: false,
                metavar: "FAMILY:DIM>=MIN",
                help: "median over the family's launches whose grid dim (x, y or z) is at least MIN",
            },
        ],
    },
    run: Run::Report(families),
}];

fn families(a: &Args, io: &Io) -> Result<Map<String, Value>> {
    let table = load::load_families(io, a.str("families")?)?;
    let filters = parse_grid_filters(a.list("grid-filter"), &table)?;
    let device = a.int("device")?;
    let kernels = load::kernels(io, a.pos(0)?, device)?;
    let fams = classify(&kernels, &table);
    let u_all = union_ns(kernels.iter().map(|k| (k.start, k.end)))?;
    let classified = union_ns(
        kernels
            .iter()
            .zip(&fams)
            .filter(|(_, f)| f.is_some())
            .map(|(k, _)| (k.start, k.end)),
    )?;
    let coverage = coverage_gate(classified as f64, u_all as f64, &table)?;
    let labels: Vec<&str> = fams.iter().map(|f| f.unwrap_or(UNCLASSIFIED)).collect();
    let (rows, groups) = family_tables(&kernels, &labels, &table, &filters, u_all)?;
    let unclassified = rows
        .iter()
        .find(|r| r["family"] == UNCLASSIFIED)
        .and_then(|r| r["union_ms"].as_f64())
        .unwrap_or(0.0);
    Ok(into_map(json!({
        "device": device,
        "kernels": kernels.len(),
        "u_all_ms": ms(u_all as f64),
        "coverage": coverage,
        "unclassified_ms": unclassified,
        "groups": groups,
        "families": rows,
    })))
}
