// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: FUSIONS.toml through `metrale_circuit::runtime::parse_rule_set` → `FusionRule` and
//! `RuntimeRoute` nodes with their lowering, emitter, guard and microtest edges (ontology §3), and
//! measurements.toml through `venn::parse_measurement_rows` → `Measurement` nodes.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - A measurement's kernel is `module::func` with the model prefix of its `kernel` field replaced
//!   by its `file`'s stem (the key KERNEL-PERF.md writes is `<model>::<func>`).
//! - A measurement whose `source` has no `@ <sha>` is marked `dangling_commit`, never guessed.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::lkb::line_of;
use crate::prelude::*;
use crate::text::{commit_shas, design_refs};
use metrale_circuit::critical_report::Allowed;
use metrale_circuit::rules::{Numerics, Rule};
use metrale_circuit::runtime::RuntimeRoute;
use metrale_circuit::venn::MeasurementRow;
use serde_json::Value;

fn modes(m: &BTreeSet<metrale_circuit::rules::Mode>) -> Value {
    Value::from(m.iter().map(|x| x.name().to_string()).collect::<Vec<_>>())
}

fn map(m: &BTreeMap<String, String>) -> Value {
    Value::Object(
        m.iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect(),
    )
}

/// 2026-10-23: The rules and routes of one FUSIONS.toml.
pub fn rules(
    b: &mut Builder,
    path: &str,
    text: &str,
    rules: &[Rule],
    routes: &[RuntimeRoute],
    families: &BTreeMap<String, Vec<String>>,
) {
    let mut cursor = 1;
    for r in rules {
        let line = line_of(text, &format!("id = \"{}\"", r.id), cursor);
        cursor = line;
        let at = format!("{path}:{line}");
        let Ok(id) = KgId::new(Label::FusionRule, r.id.clone()) else {
            continue;
        };
        let ops: Vec<String> = r
            .pattern
            .iter()
            .map(|p| p.op.base_name().to_string())
            .collect();
        let mut p = BTreeMap::new();
        p.insert("pattern".into(), Value::from(ops.clone()));
        p.insert("priority".into(), Value::from(r.priority));
        p.insert(
            "numerics".into(),
            Value::String(r.numerics.class().to_string()),
        );
        p.insert("emitter".into(), Value::String(r.emitter.clone()));
        p.insert("rows".into(), Value::from(vec![r.rows.0, r.rows.1]));
        p.insert("modes".into(), modes(&r.modes));
        p.insert("when".into(), map(&r.when));
        p.insert(
            "dims".into(),
            Value::from(r.dims.keys().cloned().collect::<Vec<_>>()),
        );
        p.insert("repeat".into(), Value::String(r.repeat.name()));
        p.insert("cite".into(), Value::String(r.cite.clone()));
        let (microtest, lever) = match &r.numerics {
            Numerics::BitIdentical { microtest } => (Some(microtest.clone()), None),
            Numerics::Differs { lever } => (None, Some(lever.clone())),
            Numerics::Reference => (None, None),
        };
        p.insert(
            "microtest".into(),
            microtest.clone().map_or(Value::Null, Value::String),
        );
        p.insert(
            "lever".into(),
            lever.clone().map_or(Value::Null, Value::String),
        );
        b.define(id.clone(), at.clone(), &format!("{r:?}"), p);
        b.text(&id, "cite", &r.cite);
        for k in &r.kernels {
            if let Ok(kid) = KgId::new(Label::Kernel, format!("{}::{}", k.module, k.func)) {
                b.edge(&id, Rel::LowersTo, &kid, 1.0, &at);
            }
        }
        for fam in families.get(&r.emitter).into_iter().flatten() {
            if let Ok(f) = KgId::new(Label::KernelFamily, fam.clone()) {
                b.edge(&id, Rel::EmitsVia, &f, 1.0, &at);
            }
        }
        for key in r.when.keys().cloned().chain(lever) {
            if let Ok(l) = KgId::new(Label::Lever, key) {
                b.edge(&id, Rel::GuardedBy, &l, 1.0, &at);
            }
        }
        if let Some(m) = microtest.and_then(|m| KgId::new(Label::Gate, format!("test/{m}")).ok()) {
            b.edge(&id, Rel::MicrotestedBy, &m, 1.0, &at);
        }
        for op in &ops {
            if let Ok(o) = KgId::new(Label::Op, format!("*/{op}")) {
                b.edge(&id, Rel::CoversOp, &o, 1.0, &at);
            }
        }
        for d in design_refs(&r.cite) {
            if let Ok(t) = KgId::new(Label::Design, d) {
                b.edge(&id, Rel::Cites, &t, 1.0, &at);
            }
        }
    }
    for rt in routes {
        let line = line_of(text, &format!("id = \"{}\"", rt.id), 1);
        let at = format!("{path}:{line}");
        let Ok(id) = KgId::new(Label::RuntimeRoute, rt.id.clone()) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("when".into(), map(&rt.when));
        p.insert("modes".into(), modes(&rt.modes));
        p.insert("rows".into(), Value::from(vec![rt.rows.0, rt.rows.1]));
        p.insert("plans_as".into(), map(&rt.plans_as));
        p.insert("why".into(), Value::String(rt.why.clone()));
        p.insert("cite".into(), Value::String(rt.cite.clone()));
        b.define(id.clone(), at.clone(), &format!("{rt:?}"), p);
        b.text(&id, "why", &rt.why);
        for key in rt.when.keys() {
            if let Ok(l) = KgId::new(Label::Lever, key.clone()) {
                b.edge(&id, Rel::GuardedBy, &l, 1.0, &at);
            }
        }
    }
}

/// 2026-10-23: The `module::func` a measurement row measures.
pub fn measured_kernel(row: &MeasurementRow) -> String {
    let func = row
        .kernel
        .rsplit_once("::")
        .map_or(row.kernel.as_str(), |(_, f)| f);
    let stem = row.file.rsplit('/').next().unwrap_or("");
    let module = stem.rsplit_once('.').map_or(stem, |(s, _)| s);
    format!("{module}::{func}")
}

/// 2026-10-23: The commit a `source` cites as `@ <sha>`.
pub fn source_commit(source: &str) -> Option<String> {
    let (_, after) = source.split_once("@ ")?;
    commit_shas(after.split_whitespace().next()?)
        .into_iter()
        .next()
}

/// 2026-10-23: measurements.toml rows; `arch_of` maps a checkpoint id to its circuit arch.
pub fn measurements(
    b: &mut Builder,
    path: &str,
    text: &str,
    rows: &[MeasurementRow],
    arch_of: &BTreeMap<String, String>,
) {
    let mut cursor = 1;
    for r in rows {
        let line = line_of(text, &format!("kernel = \"{}\"", r.kernel), cursor);
        let line = if text
            .lines()
            .nth(line.saturating_sub(1))
            .is_some_and(|l| l.contains(&r.kernel))
        {
            line
        } else {
            line_of(text, &format!("kernel = \"{}\"", r.kernel), 1)
        };
        cursor = line + 1;
        let at = format!("{path}:{line}");
        let key = format!("{} @ {}", r.kernel, r.regime);
        let Ok(id) = KgId::new(Label::Measurement, key) else {
            continue;
        };
        let commit = source_commit(&r.source);
        let mut p = BTreeMap::new();
        for (k, v) in [
            ("kernel", &r.kernel),
            ("regime", &r.regime),
            ("file", &r.file),
            ("hardware", &r.hardware),
            ("model", &r.model),
            ("bound", &r.bound),
            ("source", &r.source),
            ("notes", &r.notes),
        ] {
            p.insert(k.to_string(), Value::String(v.clone()));
        }
        for (k, v) in [
            ("time_us", r.time_us),
            ("bytes", r.bytes),
            ("flops", r.flops),
            ("floor_us", r.floor_us),
            ("pct_of_floor", r.pct_of_floor),
        ] {
            p.insert(k.to_string(), Value::from(v));
        }
        for (k, v) in &r.extra {
            let j = serde_json::to_value(v).unwrap_or(Value::Null);
            p.insert(k.clone(), j);
        }
        p.insert(
            "source_commit".into(),
            commit.clone().map_or(Value::Null, Value::String),
        );
        p.insert("dangling_commit".into(), Value::Bool(commit.is_none()));
        let kernel = measured_kernel(r);
        p.insert("measures".into(), Value::String(kernel.clone()));
        b.define(id.clone(), at.clone(), &format!("{r:?}"), p);
        b.text(&id, "notes", &r.notes);
        if let Ok(k) = KgId::new(Label::Kernel, kernel) {
            b.edge(&id, Rel::Measures, &k, 1.0, &at);
        }
        if let Ok(h) = KgId::new(Label::Hardware, r.hardware.clone()) {
            b.edge(&id, Rel::Targets, &h, 1.0, &at);
        }
        let model = arch_of
            .get(&r.model)
            .cloned()
            .unwrap_or_else(|| r.model.clone());
        if let Ok(m) = KgId::new(Label::Model, model) {
            b.edge(&id, Rel::ForModel, &m, 1.0, &at);
        }
        if let Some(c) = commit.and_then(|c| KgId::new(Label::Commit, c).ok()) {
            b.edge(&id, Rel::AtCommit, &c, 1.0, &at);
        }
    }
}

/// 2026-10-23: `CRITICAL_PAIRS.toml` (P4; GRAPH-THEORY-MAP experiment #3) through
/// `metrale_circuit::critical_report::parse_allowlist` → one `CriticalPair` per entry
/// (`cpair:<a>~<b>`, ids sorted), `OVERLAPS` both rules. Every entry is a pair the plan resolves by
/// rule priority alone and that the shrink-only gate allowlists, so `resolved_by = "priority"`
/// and `allowlisted = true` are what the file states, not defaults.
pub fn critical_pairs(b: &mut Builder, path: &str, text: &str, pairs: &[Allowed]) {
    let mut cursor = 1;
    for a in pairs {
        let (x, y) = (&a.rules.0, &a.rules.1);
        let line = line_of(text, &format!("\"{x}\""), cursor);
        cursor = line;
        let at = format!("{path}:{line}");
        let Ok(id) = KgId::new(Label::CriticalPair, format!("{x}~{y}")) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("rules".into(), Value::from(vec![x.clone(), y.clone()]));
        p.insert("why".into(), Value::String(a.why.clone()));
        p.insert("resolved_by".into(), Value::String("priority".into()));
        p.insert("allowlisted".into(), Value::Bool(true));
        b.define(id.clone(), at.clone(), &format!("{a:?}"), p);
        b.text(&id, "why", &a.why);
        for r in [x, y] {
            if let Ok(rid) = KgId::new(Label::FusionRule, r.clone()) {
                b.edge(&id, Rel::Overlaps, &rid, 1.0, &at);
            }
        }
    }
}
