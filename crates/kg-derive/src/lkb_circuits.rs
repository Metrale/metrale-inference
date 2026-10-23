// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: INSTANCES.toml (through `metrale_circuit::parse_instances`), the circuits' top level
//! (`arch`, `description`, `[parallel.<dim>]`) and the golden plans → `Instance`, `Model`,
//! `ParallelDim` and `Plan` nodes with `PLANS`, `TARGETS`, `SPLITS`, `LAUNCHES`, `USES_RULE`.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - An instance id is `<recipe basename>--<target hardware>`
//!   (`glm-5.3-flash-nvfp4-tp3-512k--gb10`), the key `kg_status owed_lkb` takes.
//! - A plan file is read only for its header lines and its group lines' `module::func` and
//!   `rule=<id>` tokens; the plan's meaning stays metrale-circuit's.

use crate::bundle::Builder;
use crate::id::{KgId, Label, Rel};
use crate::lkb::line_of;
use crate::prelude::*;
use metrale_circuit::Instance;
use serde_json::Value;

/// 2026-10-23: The instance id of `i`.
pub fn instance_id(i: &Instance) -> String {
    let base = i.recipe.rsplit('/').next().unwrap_or(&i.recipe);
    let hw = i.target.split('/').next().unwrap_or("");
    format!("{base}--{hw}")
}

/// 2026-10-23: The instances, their models' hardware edges and golden plan ids.
pub fn instances(b: &mut Builder, path: &str, text: &str, all: &[Instance]) {
    for i in all {
        let line = line_of(text, &format!("\"{}\"", i.recipe), 1);
        let at = format!("{path}:{line}");
        let Ok(id) = KgId::new(Label::Instance, instance_id(i)) else {
            continue;
        };
        let mut p = BTreeMap::new();
        p.insert("recipe".into(), Value::String(i.recipe.clone()));
        p.insert("checkpoint".into(), Value::String(i.checkpoint.clone()));
        p.insert("arch".into(), Value::String(i.arch.clone()));
        p.insert("target".into(), Value::String(i.target.clone()));
        p.insert("golden".into(), Value::Bool(i.golden));
        p.insert(
            "ranks".into(),
            Value::from(i.parallel.as_ref().map_or(1, |d| d.ranks)),
        );
        let plans: serde_json::Map<String, Value> = i
            .plans
            .iter()
            .map(|(m, rows)| (m.name().to_string(), Value::from(rows.clone())))
            .collect();
        p.insert("plans".into(), Value::Object(plans));
        b.define(id.clone(), at.clone(), &format!("{i:?}"), p);
        if let Ok(h) = KgId::new(
            Label::Hardware,
            i.target.split('/').next().unwrap_or("").to_string(),
        ) {
            b.edge(&id, Rel::Targets, &h, 1.0, &at);
        }
        if !i.golden {
            continue;
        }
        for (mode, rows) in &i.plans {
            for n in rows {
                if let Ok(pl) = KgId::new(
                    Label::Plan,
                    format!("{}/{}/n{n}", i.plan_stem(), mode.name()),
                ) {
                    b.edge(&id, Rel::Plans, &pl, 1.0, &at);
                }
            }
        }
    }
}

/// 2026-10-23: One circuit TOML's top level: the `Model` and its `[parallel.<dim>]` parts.
pub fn circuit(b: &mut Builder, path: &str, text: &str) {
    let Ok(t) = toml::from_str::<toml::Table>(text) else {
        return;
    };
    let Some(arch) = t.get("arch").and_then(toml::Value::as_str) else {
        return;
    };
    let Ok(id) = KgId::new(Label::Model, arch) else {
        return;
    };
    let desc = t
        .get("description")
        .and_then(toml::Value::as_str)
        .unwrap_or("");
    let mut p = BTreeMap::new();
    p.insert("arch".into(), Value::String(arch.to_string()));
    p.insert("description".into(), Value::String(desc.to_string()));
    let at = format!("{path}:{}", line_of(text, "arch", 1));
    b.define(id.clone(), at, &format!("{arch}\n{desc}"), p);
    for (dim, spec) in t
        .get("parallel")
        .and_then(toml::Value::as_table)
        .into_iter()
        .flatten()
    {
        let Ok(pd) = KgId::new(Label::ParallelDim, format!("{arch}/{dim}")) else {
            continue;
        };
        let pat = format!("{path}:{}", line_of(text, &format!("[parallel.{dim}]"), 1));
        let j = serde_json::to_value(spec).unwrap_or(Value::Null);
        let mut p = BTreeMap::new();
        p.insert("spec".into(), j);
        b.define(pd.clone(), pat.clone(), &format!("{dim} {spec}"), p);
        b.edge(&pd, Rel::Splits, &id, 1.0, &pat);
    }
}

/// 2026-10-23: `(stem, mode, rows)` of a golden plan file name `<stem>-<mode>-n<rows>.txt`.
pub fn plan_name(file: &str) -> Option<(String, String, u64)> {
    let name = file.rsplit('/').next()?.strip_suffix(".txt")?;
    let (head, rows) = name.rsplit_once("-n")?;
    let rows: u64 = rows.parse().ok()?;
    let (stem, mode) = head.rsplit_once('-')?;
    Some((stem.to_string(), mode.to_string(), rows))
}

/// 2026-10-23: One golden plan file.
pub fn plan(b: &mut Builder, path: &str, text: &str) {
    let Some((stem, mode, rows)) = plan_name(path) else {
        return;
    };
    let Ok(id) = KgId::new(Label::Plan, format!("{stem}/{mode}/n{rows}")) else {
        return;
    };
    let mut p = BTreeMap::new();
    for l in text.lines().take(12) {
        for (tag, key) in [
            ("# recipe: ", "recipe"),
            ("# checkpoint: ", "checkpoint"),
            ("# target: ", "target"),
            ("# settings: ", "settings"),
            ("digest: ", "digest"),
        ] {
            if let Some(v) = l.strip_prefix(tag) {
                p.insert(key.to_string(), Value::String(v.trim().to_string()));
            }
        }
    }
    let arch = stem.split('.').next().unwrap_or(&stem).to_string();
    let at = format!("{path}:1");
    let digest = p
        .get("digest")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    b.define(
        id.clone(),
        at.clone(),
        &format!("{stem}/{mode}/{rows}/{digest}"),
        p,
    );
    if let Ok(m) = KgId::new(Label::Model, arch) {
        b.edge(&id, Rel::Plans, &m, 1.0, &at);
    }
    for (i, l) in text.lines().enumerate() {
        if !l.starts_with('g') || !l.chars().nth(1).is_some_and(|c| c.is_ascii_digit()) {
            continue;
        }
        let gat = format!("{path}:{}", i + 1);
        for tok in l.split_whitespace() {
            if let Some(rule) = tok.strip_prefix("rule=") {
                if let Ok(r) = KgId::new(Label::FusionRule, rule) {
                    b.edge(&id, Rel::UsesRule, &r, 1.0, &gat);
                }
            } else if tok.contains("::") && !tok.contains('=') {
                for k in tok.split('+') {
                    if let Ok(kid) = KgId::new(Label::Kernel, k) {
                        b.edge(&id, Rel::Launches, &kid, 1.0, &gat);
                    }
                }
            }
        }
    }
}
