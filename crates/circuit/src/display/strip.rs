// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The layer strip: one glyph per layer for its mixer and one for its FFN, a
//! ruler numbering every 8th layer, and a run-length summary such as
//! `(3× GDN → 1× Attention) × 16`.
//!
//! Owner: metrale-circuit.
//! Invariants: the strip wraps at a multiple of 8 layers so the ruler stays aligned.

use super::glyphs::Set;
use super::{Document, Line, Style, clip};
use crate::ir::{Circuit, LayerKind, OpKind};

const INDENT: &str = "  ";

fn ffn_is_moe(circuit: &Circuit, layer: usize) -> bool {
    circuit
        .nodes
        .iter()
        .any(|n| n.layer == Some(layer) && n.op == OpKind::Router)
}

fn kind_name(k: LayerKind) -> &'static str {
    match k {
        LayerKind::LinearAttention => "GDN",
        LayerKind::FullAttention => "Attention",
        LayerKind::Mamba => "Mamba2",
        LayerKind::Moe => "MoE",
        LayerKind::SparseAttention => "DSA",
        LayerKind::SlidingAttention => "SWA",
        LayerKind::CompressedSparseAttention => "CSA",
        LayerKind::HeavilyCompressedAttention => "HCA",
    }
}

/// 2026-09-28: `(3× GDN → 1× Attention) × 16`, or the plain run list when the kinds do not
/// repeat.
pub(super) fn run_length(kinds: &[LayerKind], g: &Set) -> String {
    let n = kinds.len();
    let period = (1..=n)
        .find(|&p| n.is_multiple_of(p) && kinds.iter().enumerate().all(|(i, k)| *k == kinds[i % p]))
        .unwrap_or(n);
    let mut runs: Vec<(usize, LayerKind)> = Vec::new();
    for &k in &kinds[..period] {
        match runs.last_mut() {
            Some((c, last)) if *last == k => *c += 1,
            _ => runs.push((1, k)),
        }
    }
    let body = runs
        .iter()
        .map(|(c, k)| format!("{c}{} {}", g.times, kind_name(*k)))
        .collect::<Vec<_>>()
        .join(g.then);
    if n / period > 1 {
        format!("({body}) {} {}", g.times, n / period)
    } else {
        body
    }
}

/// 2026-09-28: Draw the strip for `layers` (the layer indices the plan covers).
pub(super) fn strip(
    doc: &mut Document,
    circuit: &Circuit,
    layers: &[usize],
    width: usize,
    g: &Set,
) {
    doc.lines.push(Line::default());
    let mut title = Line::default();
    title.push("Layers", Style::Heading);
    doc.lines.push(title);
    let per_line = ((width - INDENT.len()) / 8).max(1) * 8;
    for chunk in layers.chunks(per_line) {
        let mut ruler = Line::default();
        ruler.push(INDENT, Style::Plain);
        let mut col = 0;
        for (j, &i) in chunk.iter().enumerate() {
            if i % 8 == 0 && j >= col {
                let label = i.to_string();
                ruler.pad_to(INDENT.len() + j);
                ruler.push(label.clone(), Style::Dim);
                col = j + label.len() + 1;
            }
        }
        doc.lines.push(ruler);
        let mut mixer = Line::default();
        let mut ffn = Line::default();
        mixer.push(INDENT, Style::Plain);
        ffn.push(INDENT, Style::Plain);
        for &i in chunk {
            let (glyph, style) = match circuit.layer_kinds[i] {
                // 2026-09-29: A Mamba2 mixer draws as the recurrent-mixer glyph; a MoE-only
                // layer has no mixer, so its mixer cell draws as its FFN.
                LayerKind::LinearAttention | LayerKind::Mamba => (g.layer[0], Style::LayerGdn),
                LayerKind::FullAttention
                | LayerKind::SparseAttention
                | LayerKind::SlidingAttention
                | LayerKind::CompressedSparseAttention
                | LayerKind::HeavilyCompressedAttention => (g.layer[1], Style::LayerAttn),
                LayerKind::Moe => (g.layer[3], Style::LayerMoe),
            };
            mixer.push(glyph.to_string(), style);
            let (glyph, style) = if ffn_is_moe(circuit, i) {
                (g.layer[3], Style::LayerMoe)
            } else {
                (g.layer[2], Style::LayerDense)
            };
            ffn.push(glyph.to_string(), style);
        }
        doc.lines.push(mixer);
        doc.lines.push(ffn);
    }
    let kinds: Vec<LayerKind> = layers.iter().map(|&i| circuit.layer_kinds[i]).collect();
    let moe = layers.iter().filter(|&&i| ffn_is_moe(circuit, i)).count();
    let ffn = match moe {
        0 => "dense FFN on every layer".to_string(),
        m if m == layers.len() => "MoE FFN on every layer".to_string(),
        m => format!("MoE FFN on {m} of {} layers", layers.len()),
    };
    let mut summary = Line::default();
    summary.push(INDENT, Style::Plain);
    let text = format!("{}{}{ffn}", run_length(&kinds, g), g.dot);
    summary.push(clip(&text, width - INDENT.len(), g), Style::Accent);
    doc.lines.push(summary);
    let mut key = Line::default();
    key.push(INDENT, Style::Plain);
    key.push(format!("{} GDN ", g.layer[0]), Style::LayerGdn);
    key.push(format!("{} attention ", g.layer[1]), Style::LayerAttn);
    key.push(format!("{} dense FFN ", g.layer[2]), Style::LayerDense);
    key.push(format!("{} MoE FFN", g.layer[3]), Style::LayerMoe);
    doc.lines.push(key);
}
