// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The LKB's kernel half (ontology §2): KERNEL_FAMILIES.toml through
//! `metrale_circuit::venn::parse_families` (families, params, points, evidence, the roofline as the
//! `Hardware` node), the kernel inventory `scripts/kernel_perf.py --json` (one `Kernel` node per
//! entry point of the hardware: file, line, taxonomy component, targets) and the shrink-only
//! FUSED_HANDWRITTEN.toml allowlist.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - Nothing here re-parses a schema metrale-circuit or kernel_perf.py owns: families come from
//!   the circuit's reader, the taxonomy join from kernel_perf.py's own JSON.
//! - A kernel id is `module::func`; several inventory entries of one id (a common and a model-dir
//!   copy) make one node listing every file.

use crate::bundle::Builder;
use crate::hash::record_sha;
use crate::id::{KgId, Label, Rel};
use crate::prelude::*;
use metrale_circuit::venn::Families;
use metrale_error::{ErrorCode, error};
use serde_json::Value;

/// 2026-10-23: The 1-based line of the first line at or after `from` that contains `needle`.
pub fn line_of(text: &str, needle: &str, from: usize) -> usize {
    text.lines()
        .enumerate()
        .skip(from.saturating_sub(1))
        .find(|(_, l)| l.contains(needle))
        .map_or(from.max(1), |(i, _)| i + 1)
}

fn short(s: &str) -> String {
    record_sha(s).hex().chars().take(8).collect()
}

fn values_text(v: &BTreeMap<String, String>) -> String {
    let parts: Vec<String> = v.iter().map(|(k, x)| format!("{k}={x}")).collect();
    parts.join(",")
}

/// 2026-10-23: One kernel of the inventory (the fields of a `kernel_perf.py --json` entry the graph
/// reads).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InvKernel {
    pub files: Vec<String>,
    pub line: u64,
    pub component: Option<String>,
    pub kind: Option<String>,
    pub targets: Vec<String>,
    pub call_sites: Vec<String>,
}

/// 2026-10-23: The inventory of `kernel_perf.py --json --partial`: the kernels of `hardware` by id,
/// and every problem the script reported other than an unclassified kernel (those are already
/// the kernels whose `component` is null), as `kind: text` lines.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Inventory {
    pub kernels: BTreeMap<String, InvKernel>,
    pub problems: Vec<String>,
}

/// 2026-10-23: Parse the `{kernels, problems}` object; an `unclassified` problem must name a
/// kernel of the inventory whose component is null, or the two halves disagree and it is refused.
pub fn inventory(json: &str, hardware: &str) -> Result<Inventory, String> {
    let v: Value = serde_json::from_str(json).map_err(|e| format!("kernel inventory: {e}"))?;
    let arr = v
        .get("kernels")
        .and_then(Value::as_array)
        .ok_or("kernel inventory: no `kernels` array (run with --json --partial)")?;
    let probs = v
        .get("problems")
        .and_then(Value::as_array)
        .ok_or("kernel inventory: no `problems` array (run with --json --partial)")?;
    let kernels = group(arr, hardware);
    let nulls: BTreeSet<String> = arr
        .iter()
        .filter(|e| e.get("component").is_none_or(Value::is_null))
        .filter_map(|e| {
            let f = e.get("file").and_then(Value::as_str)?;
            let k = e.get("kernel").and_then(Value::as_str)?;
            Some(format!("{f}::{}", k.rsplit("::").next().unwrap_or(k)))
        })
        .collect();
    let mut problems = Vec::new();
    for p in probs {
        let kind = p.get("kind").and_then(Value::as_str).unwrap_or("");
        match (kind, p.get("kernel").and_then(Value::as_str)) {
            ("unclassified", Some(k)) if nulls.contains(k) => {}
            ("unclassified", k) => {
                return Err(format!(
                    "kernel inventory: problem names {k:?} as unclassified but no kernel row has a null component there"
                ));
            }
            (_, _) => {
                let text = p.get("text").and_then(Value::as_str).unwrap_or("");
                problems.push(format!("{kind}: {text}"));
            }
        }
    }
    Ok(Inventory { kernels, problems })
}

fn group(arr: &[Value], hardware: &str) -> BTreeMap<String, InvKernel> {
    let strs = |x: &Value, k: &str| -> Vec<String> {
        x.get(k)
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|s| s.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let mut out: BTreeMap<String, InvKernel> = BTreeMap::new();
    for e in arr {
        if !strs(e, "hardware").iter().any(|h| h == hardware) {
            continue;
        }
        let Some(k) = e.get("kernel").and_then(Value::as_str) else {
            continue;
        };
        let file = e
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let ent = out.entry(k.to_string()).or_default();
        if ent.files.is_empty() || file < ent.files[0] {
            ent.line = e.get("line").and_then(Value::as_u64).unwrap_or(1);
        }
        ent.files.push(file);
        ent.files.sort();
        ent.files.dedup();
        let comp = e
            .get("component")
            .and_then(Value::as_str)
            .filter(|c| !c.is_empty());
        if ent.component.is_none() {
            ent.component = comp.map(str::to_string);
        }
        if ent.kind.is_none() {
            ent.kind = e.get("kind").and_then(Value::as_str).map(str::to_string);
        }
        ent.targets.extend(strs(e, "targets"));
        ent.targets.sort();
        ent.targets.dedup();
        ent.call_sites.extend(strs(e, "engine_call_sites"));
        ent.call_sites.sort();
        ent.call_sites.dedup();
    }
    out
}

/// 2026-10-23: `Kernel` nodes from the inventory.
pub fn kernels(b: &mut Builder, repo: &str, inv: &BTreeMap<String, InvKernel>) {
    for (k, e) in inv {
        let Ok(id) = KgId::new(Label::Kernel, k.clone()) else {
            continue;
        };
        let file = e.files.first().cloned().unwrap_or_default();
        let mut p = BTreeMap::new();
        p.insert("files".into(), Value::from(e.files.clone()));
        p.insert("file".into(), Value::String(file.clone()));
        p.insert(
            "taxonomy_component".into(),
            e.component.clone().map_or(Value::Null, Value::String),
        );
        p.insert(
            "kind".into(),
            e.kind.clone().map_or(Value::Null, Value::String),
        );
        p.insert("targets".into(), Value::from(e.targets.clone()));
        p.insert(
            "engine_call_sites".into(),
            Value::from(e.call_sites.clone()),
        );
        let canon = format!(
            "{k}\n{}\n{:?}\n{:?}",
            e.files.join(","),
            e.component,
            e.targets
        );
        b.define(id, format!("{repo}/{file}:{}", e.line), &canon, p);
    }
}

/// 2026-10-23: The families manifest.
pub fn families(
    b: &mut Builder,
    path: &str,
    text: &str,
    fams: &Families,
    inv: &BTreeMap<String, InvKernel>,
) {
    let r = &fams.roofline;
    if let Ok(hw) = KgId::new(Label::Hardware, fams.hardware.clone()) {
        let mut p = BTreeMap::new();
        p.insert("dram_gbps".into(), Value::from(r.dram_gbps));
        p.insert("bf16_tflops".into(), Value::from(r.bf16_tflops));
        p.insert("fp8_tflops".into(), Value::from(r.fp8_tflops));
        p.insert("nvfp4_tflops".into(), Value::from(r.nvfp4_tflops));
        p.insert("context_tokens".into(), Value::from(r.context_tokens));
        if let Some(en) = &fams.energy {
            p.insert("dram_pj_per_byte".into(), Value::from(en.dram_pj_per_byte));
            p.insert("static_w".into(), Value::from(en.static_w));
            p.insert("energy_label".into(), Value::String(en.label.clone()));
        }
        let at = format!("{path}:{}", line_of(text, "[roofline]", 1));
        b.define(hw, at, &format!("{r:?}"), p);
    }
    let mut cursor = 1;
    for f in &fams.families {
        let fline = line_of(text, &format!("id = \"{}\"", f.id), cursor);
        cursor = fline;
        let at = format!("{path}:{fline}");
        let Ok(fid) = KgId::new(Label::KernelFamily, f.id.clone()) else {
            continue;
        };
        let kernels: Vec<String> = f
            .kernels
            .iter()
            .map(|k| format!("{}::{}", k.module, k.func))
            .collect();
        let ops: Vec<String> = f.ops.iter().map(|o| o.op.clone()).collect();
        let mut p = BTreeMap::new();
        p.insert("description".into(), Value::String(f.description.clone()));
        p.insert("rows".into(), Value::from(vec![f.rows.0, f.rows.1]));
        p.insert("ops".into(), Value::from(ops.clone()));
        p.insert("emitters".into(), Value::from(f.emitters.clone()));
        p.insert("kernels".into(), Value::from(kernels.clone()));
        b.define(fid.clone(), at.clone(), &format!("{f:?}"), p);
        b.text(&fid, "description", &f.description);
        for op in &ops {
            if let Ok(o) = KgId::new(Label::Op, format!("*/{op}")) {
                b.edge(&fid, Rel::CoversOp, &o, 1.0, &at);
            }
        }
        for prm in &f.params {
            let Ok(pid) = KgId::new(Label::FamilyParam, format!("{}/{}", f.id, prm.name)) else {
                continue;
            };
            let mut p = BTreeMap::new();
            p.insert("kind".into(), Value::String(prm.kind.name().to_string()));
            p.insert(
                "from".into(),
                Value::from(prm.from.keys().cloned().collect::<Vec<_>>()),
            );
            p.insert(
                "absent".into(),
                prm.absent.clone().map_or(Value::Null, Value::String),
            );
            let pat = format!(
                "{path}:{}",
                line_of(text, &format!("name = \"{}\"", prm.name), fline)
            );
            b.define(pid.clone(), pat, &format!("{prm:?}"), p);
            b.edge(&fid, Rel::HasParam, &pid, 1.0, &at);
        }
        let mut pline = fline;
        for pt in &f.points {
            pline = line_of(text, "[[family.point]]", pline + 1);
            let vt = values_text(&pt.values);
            let Ok(pid) = KgId::new(Label::FamilyPoint, format!("{}/{}", f.id, short(&vt))) else {
                continue;
            };
            let mut p = BTreeMap::new();
            let vals: serde_json::Map<String, Value> = pt
                .values
                .iter()
                .map(|(k, v)| (k.clone(), Value::String(v.clone())))
                .collect();
            p.insert("values".into(), Value::Object(vals));
            p.insert("how".into(), Value::String(pt.how.name().to_string()));
            p.insert("files".into(), Value::from(pt.files.clone()));
            let pat = format!("{path}:{pline}");
            b.define(pid.clone(), pat.clone(), &vt, p);
            b.edge(&pid, Rel::Instantiates, &fid, 1.0, &pat);
            for k in &kernels {
                let in_point = inv
                    .get(k)
                    .is_some_and(|e| e.files.iter().any(|f| pt.files.contains(f)));
                if let (true, Ok(kid)) = (in_point, KgId::new(Label::Kernel, k.clone())) {
                    b.edge(&kid, Rel::Instantiates, &pid, 1.0, &pat);
                }
            }
        }
        for k in &kernels {
            if let Ok(kid) = KgId::new(Label::Kernel, k.clone()) {
                b.edge(&kid, Rel::Instantiates, &fid, 1.0, &at);
            }
        }
        let mut eline = fline;
        for ev in &f.evidence {
            eline = line_of(text, "[[family.evidence]]", eline + 1);
            let (key, src) = match &ev.source {
                metrale_circuit::venn::families::EvidenceSource::Measurement(m) => {
                    (Some(m.clone()), m.clone())
                }
                metrale_circuit::venn::families::EvidenceSource::Microbench(t) => (None, t.clone()),
            };
            let rows: Vec<u64> = ev.rows.iter().copied().collect();
            let canon = format!("{}|{rows:?}|{src}", values_text(&ev.point));
            let Ok(eid) = KgId::new(Label::Evidence, format!("{}/{}", f.id, short(&canon))) else {
                continue;
            };
            let mut p = BTreeMap::new();
            p.insert("rows".into(), Value::from(rows));
            p.insert("point".into(), Value::String(values_text(&ev.point)));
            p.insert(
                "measurement".into(),
                key.clone().map_or(Value::Null, Value::String),
            );
            p.insert(
                "microbench".into(),
                if key.is_none() {
                    Value::String(src.clone())
                } else {
                    Value::Null
                },
            );
            let eat = format!("{path}:{eline}");
            b.define(eid.clone(), eat.clone(), &canon, p);
            b.edge(&eid, Rel::EvidenceFor, &fid, 1.0, &eat);
            if let Some(k) = key.and_then(|k| KgId::new(Label::Measurement, k).ok()) {
                b.edge(&eid, Rel::KeyedTo, &k, 1.0, &eat);
            }
        }
    }
}

/// 2026-10-23: FUSED_HANDWRITTEN.toml: every listed kernel `ALLOWLISTED` by the shrink-only gate.
pub fn handwritten(b: &mut Builder, path: &str, text: &str) {
    let v: toml::Table = match toml::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            b.error(
                &error!(ErrorCode::KgBadRow, path.to_string(), e.to_string()),
                path,
            );
            return;
        }
    };
    let Ok(gate) = KgId::new(Label::Gate, "test/the_allowlist_only_shrinks") else {
        return;
    };
    for k in v
        .get("handwritten")
        .and_then(toml::Value::as_array)
        .into_iter()
        .flatten()
    {
        let Some(k) = k.as_str() else { continue };
        let at = format!("{path}:{}", line_of(text, &format!("\"{k}\""), 1));
        if let Ok(kid) = KgId::new(Label::Kernel, k) {
            b.edge(&kid, Rel::Allowlisted, &gate, 1.0, &at);
        }
    }
}
