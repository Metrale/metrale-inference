// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The metrale pack's `la_status` sections: `progress`, `blocked`, `tainted`, `claims`
//! (the campaign ledger), `owed_lkb` (Q3 for one instance) and `calibration` (predicted savings
//! against their verdicts). The core answers `dangling`, `freshness`, `schema` and `files`.
//!
//! Owner: metrale-latlas-pack (cut from metrale-kg-derive `status.rs`, 2026-10-23 K0/K1).
//! Invariants: tables, not prose; a section that cannot be computed is an error the core turns
//! into the declared `{unavailable, code, reason}`.

use crate::Metrale;
use crate::ledger;
use crate::prelude::*;
use crate::queries;
use crate::vocab::*;
use latlas_core::bundle::Bundle;
use latlas_core::pack::SectionCtx;
use latlas_core::status::ids;
use latlas_error::{ErrorCode, Result, error};
use serde_json::{Map, Value};

/// 2026-10-23: One pack section.
pub fn section(ctx: &SectionCtx<'_, Metrale>, name: &str) -> Result<Value> {
    let (b, campaign, instance) = (ctx.bundle, ctx.scope, ctx.instance);
    let c = ctx.cfg.sources.campaign(campaign)?;
    Ok(match name {
        "progress" => {
            let mut o = Map::new();
            for (phase, v) in queries::progress(b, campaign) {
                let m: Map<String, Value> =
                    v.into_iter().map(|(k, n)| (k, Value::from(n))).collect();
                o.insert(phase, Value::Object(m));
            }
            Value::Object(o)
        }
        "blocked" => Value::from(
            queries::blocked_by_gate(b)
                .into_iter()
                .map(|(id, phase, blockers, levers)| {
                    let mut o = Map::new();
                    o.insert("id".into(), Value::String(id.to_string()));
                    o.insert("phase".into(), Value::String(phase));
                    let kinds: Vec<String> = blockers
                        .iter()
                        .map(|x| x.label().name().to_string())
                        .collect();
                    o.insert("blocker_kinds".into(), Value::from(kinds));
                    o.insert("blockers".into(), ids(&blockers));
                    o.insert("levers".into(), ids(&levers));
                    Value::Object(o)
                })
                .collect::<Vec<_>>(),
        ),
        "tainted" => {
            queries::ledger_valid(b, &c.ledger)?;
            let t = ledger::tainted(&queries::ledger_links(b, campaign));
            Value::from(
                t.into_iter()
                    .map(|(id, why)| Value::from(vec![Value::String(id), Value::String(why)]))
                    .collect::<Vec<_>>(),
            )
        }
        "claims" => Value::from(
            queries::claims_without_gate(b)
                .into_iter()
                .map(|(id, v, strict)| {
                    let mut o = Map::new();
                    o.insert("id".into(), Value::String(id.to_string()));
                    o.insert("verdict".into(), Value::String(v));
                    o.insert("strict".into(), Value::Bool(strict));
                    Value::Object(o)
                })
                .collect::<Vec<_>>(),
        ),
        "owed_lkb" => owed_section(b, instance)?,
        "calibration" => calibration_section(b, campaign),
        other => {
            return Err(error!(
                ErrorCode::LatlasToolArgument,
                "sections",
                format!("unknown section {other}")
            ));
        }
    })
}

/// 2026-10-23: Q3 for one instance: every kernel that owes an LKB facet, landed ones first,
/// with the measurements of it that predate a change of its file (Q4's `stale_since`).
fn owed_section(b: &Bundle, instance: Option<&str>) -> Result<Value> {
    let instances: Vec<String> = b
        .of(labels::Instance)
        .map(|n| n.id.key().to_string())
        .collect();
    if instances.is_empty() {
        return Err(error!(
            ErrorCode::LatlasSource,
            "sources.lkb", "is not configured: the graph holds no LKB"
        ));
    }
    if let Some(f) = b.errors.iter().find(|f| f.code == "LatlasInventoryRefused") {
        return Err(error!(ErrorCode::LatlasSource, f.at.clone(), f.msg.clone()));
    }
    let Some(inst) = instance else {
        return Err(error!(
            ErrorCode::LatlasToolArgument,
            "instance",
            format!("is required; instances: {}", instances.join(", "))
        ));
    };
    if !instances.iter().any(|i| i == inst) {
        return Err(error!(
            ErrorCode::LatlasToolArgument,
            "instance",
            format!("no instance `{inst}`; instances: {}", instances.join(", "))
        ));
    }
    let rows = crate::owed::owed_lkb_rows(b, inst)?;
    let landed = rows.iter().filter(|r| r.landed).count();
    let out: Vec<Value> = rows
        .iter()
        .map(|r| {
            let mut o = Map::new();
            o.insert("kernel".into(), Value::String(r.kernel.to_string()));
            o.insert("landed".into(), Value::Bool(r.landed));
            o.insert("has_point".into(), Value::Bool(r.has_point));
            o.insert("classified".into(), Value::Bool(r.classified));
            o.insert("measured".into(), Value::Bool(r.measured));
            o.insert("evidenced".into(), Value::Bool(r.evidenced));
            let stale: Vec<Value> = b
                .into(&r.kernel, rels::Measures)
                .filter_map(|e| {
                    b.field(&e.from, "stale_since")
                        .map(|c| Value::String(format!("{} (file changed at {c})", e.from)))
                })
                .collect();
            o.insert("stale_since".into(), Value::from(stale));
            Value::Object(o)
        })
        .collect();
    let mut o = Map::new();
    o.insert("instance".into(), Value::String(inst.to_string()));
    o.insert("owed".into(), Value::from(rows.len()));
    o.insert("owed_landed".into(), Value::from(landed));
    o.insert("rows".into(), Value::from(out));
    Ok(Value::Object(o))
}

/// 2026-10-23: Predicted savings against their verdicts ([`queries::outcome`]): the counts,
/// `hit_rate` = hit / (hit + miss) (`null` while nothing is settled), and each row with its
/// measurement and `predicted_over_upside` per cited headroom site.
fn calibration_section(b: &Bundle, campaign: &str) -> Value {
    let preds = queries::calibration(b, campaign);
    let mut counts: BTreeMap<&str, u64> = [("hit", 0), ("miss", 0), ("open", 0)].into();
    let rows: Vec<Value> = preds
        .iter()
        .map(|p| {
            let out = queries::outcome(&p.verdict);
            *counts.entry(out).or_insert(0) += 1;
            let sites: Vec<Value> = p
                .sites
                .iter()
                .map(|(sid, up)| {
                    let mut s = Map::new();
                    s.insert("id".into(), Value::String(sid.to_string()));
                    s.insert("upside_us".into(), up.map_or(Value::Null, Value::from));
                    let ratio = up.filter(|u| *u > 0.0).map(|u| p.predicted_us / u);
                    s.insert(
                        "predicted_over_upside".into(),
                        ratio.map_or(Value::Null, Value::from),
                    );
                    Value::Object(s)
                })
                .collect();
            let mut o = Map::new();
            o.insert("id".into(), Value::String(p.id.to_string()));
            o.insert("verdict".into(), Value::String(p.verdict.clone()));
            o.insert("outcome".into(), Value::String(out.into()));
            o.insert("predicted_us".into(), Value::from(p.predicted_us));
            o.insert("measured".into(), p.measured.clone().unwrap_or(Value::Null));
            o.insert("sites".into(), Value::from(sites));
            Value::Object(o)
        })
        .collect();
    let settled = counts["hit"] + counts["miss"];
    let mut o = Map::new();
    o.insert("predicted".into(), Value::from(preds.len()));
    for (k, v) in &counts {
        o.insert((*k).into(), Value::from(*v));
    }
    o.insert(
        "hit_rate".into(),
        if settled == 0 {
            Value::Null
        } else {
            Value::from(counts["hit"] as f64 / settled as f64)
        },
    );
    o.insert("rows".into(), Value::from(rows));
    Value::Object(o)
}
