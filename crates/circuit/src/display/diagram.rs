// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Block diagrams: one per distinct layer plan (or per layer when expanded), each a
//! vertical flow of op boxes joined by labelled edges, with fused groups framed and titled by
//! their kernels.
//!
//! Owner: metrale-circuit.
//! Invariants: see [`super`]. Two layers share a diagram only when every node, rule and kernel
//! of their plans match.

use std::collections::BTreeMap;

use super::draw::Pen;
use super::glyphs::Set;
use super::labels::Geo;
use super::rows::{Row, frames, rows};
use super::{DisplayOpts, Document, Expand, Line, Style, clip};
use crate::fuser::{EdgeState, FusionPlan, section_of};
use crate::ir::{Circuit, EdgeIdx, LayerKind, NodeIdx};

/// 2026-09-28: Layer indices with nodes in the plan, ascending.
pub(super) fn layers_in_plan(c: &Circuit, plan: &FusionPlan) -> Vec<usize> {
    let mut out: Vec<usize> = plan
        .groups
        .iter()
        .flat_map(|g| g.nodes.iter().filter_map(|&n| c.nodes[n].layer))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// 2026-09-28: A run of nodes drawn as one diagram: a layer, or consecutive blocks outside the
/// layers (the embedding, the head, the draft head).
struct Segment {
    title: String,
    layer: Option<usize>,
    nodes: Vec<NodeIdx>,
    stream_in: Option<EdgeIdx>,
    stream_out: Option<EdgeIdx>,
}

fn segments(c: &Circuit, plan: &FusionPlan) -> Vec<Segment> {
    let section = section_of(plan.mode);
    let mut out: Vec<Segment> = Vec::new();
    for b in c.blocks.iter().filter(|b| b.section == section) {
        let nodes = b.first..b.end;
        match out.last_mut() {
            Some(s) if s.layer == b.layer => {
                if b.layer.is_none() {
                    s.title = format!("{} / {}", s.title, b.template);
                }
                s.nodes.extend(nodes);
                s.stream_out = b.stream_out.or(s.stream_out);
            }
            _ => out.push(Segment {
                title: b.template.clone(),
                layer: b.layer,
                nodes: nodes.collect(),
                stream_in: b.stream_in,
                stream_out: b.stream_out,
            }),
        }
    }
    out
}

fn signature(c: &Circuit, plan: &FusionPlan, group_of: &[Option<usize>], s: &Segment) -> String {
    let mut sig = String::new();
    for &n in &s.nodes {
        let node = &c.nodes[n];
        let g = group_of[n].map(|g| &plan.groups[g]);
        let rule = g.map_or("-", |g| g.rule.as_str());
        let pos = g
            .and_then(|g| g.nodes.iter().position(|&m| m == n))
            .unwrap_or(0);
        sig.push_str(&format!("{}.{}:{rule}:{pos};", node.block, node.local));
    }
    sig
}

fn ranges(layers: &[usize]) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut i = 0;
    while i < layers.len() {
        let mut j = i;
        while j + 1 < layers.len() && layers[j + 1] == layers[j] + 1 {
            j += 1;
        }
        parts.push(if i == j {
            layers[i].to_string()
        } else {
            format!("{}-{}", layers[i], layers[j])
        });
        i = j + 1;
    }
    parts.join(", ")
}

/// 2026-09-28: Draw every diagram the options ask for.
pub(super) fn diagrams(
    doc: &mut Document,
    c: &Circuit,
    plan: &FusionPlan,
    opts: &DisplayOpts,
    g: &Set,
) {
    let mut group_of = vec![None; c.nodes.len()];
    for (gi, grp) in plan.groups.iter().enumerate() {
        for &n in &grp.nodes {
            group_of[n] = Some(gi);
        }
    }
    let segs = segments(c, plan);
    let geo = Geo::new(opts.width);
    let mut order: Vec<String> = Vec::new();
    let mut by_sig: BTreeMap<String, (usize, Vec<usize>)> = BTreeMap::new();
    for (i, s) in segs.iter().enumerate() {
        let sig = signature(c, plan, &group_of, s);
        let entry = by_sig.entry(sig.clone()).or_insert_with(|| {
            order.push(sig);
            (i, Vec::new())
        });
        if let Some(l) = s.layer {
            entry.1.push(l);
        }
    }
    let pen = Pen {
        c,
        plan,
        g,
        geo: &geo,
        group_of: &group_of,
        bindings: false,
    };
    match opts.expand {
        Expand::Summary => {
            for sig in &order {
                let (i, layers) = &by_sig[sig];
                draw_segment(doc, &pen, &segs[*i], layers);
            }
        }
        Expand::AllLayers => {
            for s in &segs {
                draw_segment(doc, &pen, s, &s.layer.into_iter().collect::<Vec<_>>());
            }
        }
        Expand::Layer(n) => {
            let pen = Pen {
                bindings: true,
                ..pen
            };
            if let Some(s) = segs.iter().find(|s| s.layer == Some(n)) {
                draw_segment(doc, &pen, s, &[n]);
            }
        }
    }
}

fn draw_segment(doc: &mut Document, pen: &Pen<'_>, s: &Segment, layers: &[usize]) {
    let (c, g, geo) = (pen.c, pen.g, pen.geo);
    doc.lines.push(Line::default());
    let mut title = Line::default();
    match s.layer {
        Some(l) => {
            let (glyph, style, name) = match c.layer_kinds[l] {
                LayerKind::LinearAttention => (g.layer[0], Style::LayerGdn, "GatedDeltaNet layer"),
                LayerKind::FullAttention => (g.layer[1], Style::LayerAttn, "Full-attention layer"),
            };
            title.push(format!("{glyph} "), style);
            title.push(name, Style::Heading);
            if layers.len() > 1 {
                title.push(
                    format!("  {} {} layers", g.times, layers.len()),
                    Style::Accent,
                );
            }
            let room = geo.width.saturating_sub(title.width() + 2);
            let list = if layers.len() > 1 {
                ranges(layers)
            } else {
                format!("layer {l}")
            };
            title.push("  ", Style::Plain);
            title.push(clip(&list, room, g), Style::Dim);
        }
        None => {
            title.push(format!("{} ", g.layer[2]), Style::Dim);
            title.push(s.title.clone(), Style::Heading);
        }
    }
    doc.lines.push(title);
    let rows = rows(c, &s.nodes);
    let frame_of = frames(&rows, pen.plan, pen.group_of);
    if let Some(e) = s.stream_in {
        pen.connector(doc, None, e, "stream in");
    }
    let mut prev: Option<&Row> = None;
    for (i, row) in rows.iter().enumerate() {
        let frame = frame_of[i];
        let opens = frame.is_some() && (i == 0 || frame_of[i - 1] != frame);
        if let Some(p) = prev {
            let inside = frame.is_some() && frame_of[i - 1] == frame;
            pen.link(doc, p, row, if inside { frame } else { None });
        }
        if opens {
            pen.frame_top(doc, frame.unwrap_or_default());
        }
        pen.row(doc, row, frame);
        let closes = frame.is_some() && frame_of.get(i + 1).copied().flatten() != frame;
        if closes {
            pen.frame_bottom(doc, frame.unwrap_or_default());
        }
        prev = Some(row);
    }
    if let Some(e) = s.stream_out {
        let label = match pen.plan.edge_states[e] {
            Some(EdgeState::Fused(_)) => "stream out, fused into the next layer",
            _ => "stream out",
        };
        pen.connector(doc, None, e, label);
    }
}
