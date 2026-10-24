// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `headroom/v1` reports (`kernels/circuits/plans/headroom/*.json`) → `HeadroomSite`
//! nodes: the predicted floors and measured exclusive time per plan site of one capture, with the
//! kernels each site runs (`RUNS_KERNEL`) and the circuit op it is (`SITE_OF`). Ledger rows whose
//! prediction cites `headroom <capture>` and names a site get `PREDICTED_BY` (ontology §3).
//!
//! Owner: metrale-latlas-pack (the metrale pack; was metrale-kg-derive).
//! Invariants: a site id is `hsite:<instance>/<capture>/<run>/<site>`; the instance comes from the
//! report's own `model` field (`recipe <id>`), never from the file name.

use crate::prelude::*;
use crate::vocab::*;
use latlas_core::bundle::Builder;
use latlas_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: The capture tag of a report file (`...--f13-p1k.json` → `f13-p1k`).
pub fn capture_of(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".json").unwrap_or(name);
    stem.rsplit_once("--").map_or(stem, |(_, c)| c).to_string()
}

/// 2026-10-23: One headroom report. `instances` maps a recipe id to (instance id, arch).
/// Returns the site ids with their capture and site names, for `PREDICTED_BY`.
pub fn report(
    b: &mut Builder,
    path: &str,
    text: &str,
    instances: &BTreeMap<String, (String, String)>,
) -> Vec<(NodeId, String, String)> {
    let v: Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            b.error(
                &error!(ErrorCode::LatlasBadRow, path.to_string(), e.to_string()),
                path,
            );
            return Vec::new();
        }
    };
    let recipe = v["model"]
        .as_str()
        .unwrap_or("")
        .trim_start_matches("recipe ")
        .to_string();
    let Some((inst, arch)) = instances.get(&recipe).cloned() else {
        let e = error!(
            ErrorCode::LatlasDanglingRef,
            path.to_string(),
            format!("instance:<recipe {recipe}>")
        );
        b.error(&e, path);
        return Vec::new();
    };
    let cap = capture_of(path);
    let mut out = Vec::new();
    for (i, s) in v["sites"].as_array().into_iter().flatten().enumerate() {
        let (run, site) = (
            s["run"].as_str().unwrap_or(""),
            s["site"].as_str().unwrap_or(""),
        );
        let at = format!("{path}:site{}", i + 1);
        let Ok(id) = NodeId::new(labels::HeadroomSite, format!("{inst}/{cap}/{run}/{site}")) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("instance".into(), Value::String(inst.clone()));
        p.insert("capture".into(), Value::String(cap.clone()));
        p.insert("run".into(), Value::String(run.to_string()));
        p.insert("site".into(), Value::String(site.to_string()));
        p.insert("rank".into(), Value::from(i + 1));
        for k in [
            "floor_us",
            "pct_of_floor",
            "headroom_us",
            "bound",
            "kernels",
            "rules",
        ] {
            p.insert(k.to_string(), s[k].clone());
        }
        p.insert("exclusive_us".into(), s["measured"]["exclusive_us"].clone());
        p.insert("drift".into(), s["measured"]["drift"].clone());
        p.insert("upside_us".into(), s["upside_target"]["saved_us"].clone());
        b.define(id.clone(), format!("{path}:1"), &s.to_string(), p);
        for k in s["kernels"]
            .as_str()
            .unwrap_or("")
            .split([',', '+', ' '])
            .filter(|k| k.contains("::"))
        {
            if let Ok(kid) = NodeId::new(labels::Kernel, k.trim()) {
                b.edge(&id, rels::RunsKernel, &kid, 1.0, &at);
            }
        }
        if let Ok(op) = NodeId::new(labels::Op, format!("{arch}/{site}")) {
            b.edge(&id, rels::SiteOf, &op, 1.0, &at);
        }
        out.push((id, cap.clone(), site.to_string()));
    }
    out
}

/// 2026-10-23: `PREDICTED_BY` from each ledger row whose `predicted_us_saved` cites
/// `headroom <tag>` to the sites of that capture it names.
pub fn predicted_by(
    b: &mut Builder,
    rows: &[(NodeId, String, String)],
    sites: &[(NodeId, String, String)],
) {
    for (row, at, pred) in rows {
        let Some(i) = pred.find("headroom ") else {
            continue;
        };
        let tail: Vec<&str> = pred[i + 9..].split_whitespace().take(2).collect();
        let tag = tail.join("-");
        for (sid, cap, site) in sites {
            if cap == &tag && pred.contains(site.as_str()) {
                b.edge(row, rels::PredictedBy, sid, 1.0, at);
            }
        }
    }
}
