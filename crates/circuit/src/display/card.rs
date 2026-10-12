// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The header card (model, shape, precision, plan totals) and the footer (legend
//! and per-step totals) of `met circuit display`.
//!
//! Owner: metrale-circuit.
//! Invariants: every line is at most the given width; long fact lists wrap between facts.

use std::collections::BTreeMap;

use super::glyphs::Set;
use super::{Document, Line, Style, clip};
use crate::format::Format;
use crate::fuser::{EdgeState, FusionPlan, section_of};
use crate::ir::{Circuit, OpKind};

/// 2026-09-28: Facts joined by the separator, wrapped into lines of at most `max` columns.
pub(super) fn wrap(facts: &[(String, Style)], max: usize, g: &Set) -> Vec<Line> {
    let mut lines = vec![Line::default()];
    for (text, style) in facts {
        let text = clip(text, max, g);
        let cur = lines.last_mut().expect("one line");
        let sep = if cur.spans.is_empty() {
            0
        } else {
            g.dot.chars().count()
        };
        if cur.width() + sep + text.chars().count() > max && !cur.spans.is_empty() {
            lines.push(Line::default());
        }
        let cur = lines.last_mut().expect("one line");
        if !cur.spans.is_empty() {
            cur.push(g.dot, Style::Dim);
        }
        cur.push(text, *style);
    }
    lines
}

fn card_row(doc: &mut Document, inner: Line, width: usize, g: &Set) {
    let mut l = Line::default();
    l.push(format!("{} ", g.light.v), Style::Border);
    for s in inner.spans {
        l.push(s.text, s.style);
    }
    l.pad_to(width - 2);
    l.push(format!(" {}", g.light.v), Style::Border);
    doc.lines.push(l);
}

fn bits(f: Format) -> u32 {
    match f {
        Format::Bf16 => 16,
        Format::F32 | Format::I32 => 32,
        Format::Fp8E4m3 { .. } => 8,
        Format::Nvfp4 { .. } | Format::Mxfp4 => 4,
    }
}

fn family(f: Format) -> &'static str {
    match f {
        Format::Bf16 => "bf16",
        Format::F32 => "f32",
        Format::I32 => "i32",
        Format::Fp8E4m3 { .. } => "fp8",
        Format::Nvfp4 { .. } => "nvfp4",
        Format::Mxfp4 => "mxfp4",
    }
}

fn role(op: &OpKind) -> String {
    match op {
        OpKind::Linear(r) => r.name().to_string(),
        OpKind::ExpertGateUp => "experts.gate_up".into(),
        OpKind::ExpertDown => "experts.down".into(),
        other => other.base_name().to_string(),
    }
}

/// 2026-09-28: `role W4A16 nvfp4` facts: what each weight-reading role runs at in the plan's
/// section, roles with one format merged. `A` counts the bits of the activation the kernel
/// reads when it is a 16-bit or quantized format; a projection that reads an FP32 activation
/// (the MoE down projections read the FP32 SiLU product) says so instead of claiming an A32
/// tier: `W8 fp8 · f32 act in`.
pub(super) fn precision(circuit: &Circuit, plan: &FusionPlan, g: &Set) -> Vec<(String, Style)> {
    let section = section_of(plan.mode);
    let mut by_format: BTreeMap<(u32, u32, &'static str), Vec<String>> = BTreeMap::new();
    let mut seen: BTreeMap<String, (u32, u32, &'static str)> = BTreeMap::new();
    for b in circuit.blocks.iter().filter(|b| b.section == section) {
        for n in &circuit.nodes[b.first..b.end] {
            let (Some(w), Some(&input)) = (n.weight, n.inputs.first()) else {
                continue;
            };
            let a = circuit.edges[input].format;
            let key = (bits(w), bits(a), family(w));
            let r = role(&n.op);
            if seen.insert(r.clone(), key).is_none() {
                by_format.entry(key).or_default().push(r);
            }
        }
    }
    let mut out: Vec<(Vec<String>, (u32, u32, &'static str))> =
        by_format.into_iter().map(|(k, v)| (v, k)).collect();
    out.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));
    out.into_iter()
        .map(|(roles, (w, a, fam))| {
            let label = if a == 32 {
                format!("{} W{w} {fam}{}f32 act in", roles.join("/"), g.dot)
            } else {
                format!("{} W{w}A{a} {fam}", roles.join("/"))
            };
            (label, Style::Format)
        })
        .collect()
}

fn dims_facts(circuit: &Circuit, g: &Set) -> Vec<(String, Style)> {
    let d = |k: &str| circuit.dims.get(k).copied();
    let mut f = vec![(
        format!("{} layers", circuit.layer_kinds.len()),
        Style::Accent,
    )];
    if let Some(h) = d("hidden") {
        f.push((format!("hidden {h}"), Style::Plain));
    }
    if let Some(i) = d("inter") {
        f.push((format!("ffn {i}"), Style::Plain));
    }
    if let (Some(q), Some(kv), Some(hd)) = (d("q_heads"), d("kv_heads"), d("head_dim")) {
        f.push((format!("attn {q}q/{kv}kv{}{hd}", g.times), Style::Plain));
    }
    if let (Some(k), Some(v), Some(kd)) = (d("lin_k_heads"), d("lin_v_heads"), d("lin_k_dim")) {
        f.push((format!("GDN {k}k/{v}v{}{kd}", g.times), Style::Plain));
    }
    if let (Some(e), Some(k), Some(m)) = (d("experts"), d("top_k"), d("moe_inter")) {
        f.push((format!("{e} experts top-{k} (ffn {m})"), Style::Plain));
    }
    if let Some(v) = d("vocab") {
        f.push((format!("vocab {v}"), Style::Plain));
    }
    f
}

pub(super) fn edge_counts(plan: &FusionPlan) -> (usize, usize) {
    let fused = plan
        .edge_states
        .iter()
        .filter(|s| matches!(s, Some(EdgeState::Fused(_))))
        .count();
    let mat = plan
        .edge_states
        .iter()
        .filter(|s| **s == Some(EdgeState::Materialized))
        .count();
    (fused, mat)
}

/// 2026-09-28: The rounded header card.
pub(super) fn header(
    doc: &mut Document,
    circuit: &Circuit,
    plan: &FusionPlan,
    info: &super::DisplayInfo,
    width: usize,
    g: &Set,
) {
    let inner = width - 4;
    let title = clip(&info.checkpoint, width.saturating_sub(8), g);
    let mut top = Line::default();
    top.push(format!("{}{} ", g.light.tl, g.light.h), Style::Border);
    top.push(title, Style::Heading);
    top.push(" ", Style::Plain);
    let fill = width - 1 - top.width();
    top.push(
        format!("{}{}", g.light.h.to_string().repeat(fill), g.light.tr),
        Style::Border,
    );
    doc.lines.push(top);

    let (fused, mat) = edge_counts(plan);
    let groups = [
        vec![
            (format!("arch {}", circuit.arch), Style::Accent),
            (format!("recipe {}", info.recipe), Style::Plain),
        ],
        vec![(circuit.description.clone(), Style::Dim)],
        dims_facts(circuit, g),
        precision(circuit, plan, g),
        vec![
            (format!("mode {}", plan.mode.name()), Style::Accent),
            (
                format!("{} row{}", plan.rows, if plan.rows == 1 { "" } else { "s" }),
                Style::Accent,
            ),
            (format!("{} launches/step", plan.launches()), Style::Accent),
            (format!("{} kernel groups", plan.groups.len()), Style::Plain),
        ],
        vec![
            (format!("{fused} fused edges"), Style::EdgeFused),
            (format!("{mat} materialized"), Style::EdgeMaterialized),
            (
                format!("plan {}", &plan.digest[..plan.digest.len().min(12)]),
                Style::Dim,
            ),
        ],
    ];
    for facts in groups {
        for l in wrap(&facts, inner, g) {
            card_row(doc, l, width, g);
        }
    }
    let mut bottom = Line::default();
    bottom.push(
        format!(
            "{}{}{}",
            g.light.bl,
            g.light.h.to_string().repeat(width - 2),
            g.light.br
        ),
        Style::Border,
    );
    doc.lines.push(bottom);
}

fn mib(b: u64) -> String {
    format!("{:.1} MiB", b as f64 / (1024.0 * 1024.0))
}

/// 2026-09-28: Legend and totals.
pub(super) fn footer(
    doc: &mut Document,
    plan: &FusionPlan,
    info: &super::DisplayInfo,
    width: usize,
    g: &Set,
) {
    doc.lines.push(Line::default());
    let mut rule = Line::default();
    rule.push(g.light.h.to_string().repeat(width), Style::Border);
    doc.lines.push(rule);
    let (fused, mat) = edge_counts(plan);
    let mut totals = vec![
        (format!("{} launches/step", plan.launches()), Style::Accent),
        (format!("{} groups", plan.groups.len()), Style::Plain),
        (format!("{fused} fused"), Style::EdgeFused),
        (format!("{mat} materialized edges"), Style::EdgeMaterialized),
    ];
    if let Some((m, arena)) = info.bytes {
        totals.push((format!("{} written/step", mib(m)), Style::EdgeMaterialized));
        totals.push((format!("arena {}", mib(arena)), Style::Dim));
    }
    doc.lines.extend(wrap(&totals, width, g));
    let (h, l, f) = (g.heavy, g.light, g.fused);
    let legend = [
        (format!("{}{}{} heavy op", h.tl, h.h, h.tr), Style::OpHeavy),
        (format!("{}{}{} light op", l.tl, l.h, l.tr), Style::OpLight),
        (
            format!("{}{}{} fused kernel", f.tl, f.h, f.tr),
            Style::FusedFrame,
        ),
        (
            format!("{}{} DRAM", g.wire, g.arrow),
            Style::EdgeMaterialized,
        ),
        (format!("{}{} on-chip", g.dotted, g.arrow), Style::EdgeFused),
        (g.badge[0].to_string(), Style::NumericsBitIdentical),
        (g.badge[1].to_string(), Style::NumericsReference),
        (g.badge[2].to_string(), Style::NumericsDiffers),
    ];
    doc.lines.extend(wrap(&legend, width, g));
}
