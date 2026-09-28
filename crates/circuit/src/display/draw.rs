// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Drawing primitives of a diagram: op boxes, edge connectors, fused frames and the
//! MoE fan-out, all laid out on a fixed geometry derived from the width.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Every line spans `INDENT + 2 + content + 2` columns at most: the two-column gutters hold
//!   a frame's sides, or blanks outside a frame, so boxes line up either way.
//! - Inside a frame a connector is always drawn on-chip; outside it uses the edge's state.

use super::glyphs::{Frame, Set};
use super::labels::{Geo, INDENT, WIRE, badge, edge_label, format_label};
use super::rows::Row;
use super::{Document, Line, Style, clip};
use crate::fuser::{EdgeState, FusionPlan};
use crate::ir::{Circuit, EdgeIdx, NodeIdx};

/// 2026-09-28: Everything a diagram is drawn from.
pub(super) struct Pen<'a> {
    pub c: &'a Circuit,
    pub plan: &'a FusionPlan,
    pub g: &'a Set,
    pub geo: &'a Geo,
    pub group_of: &'a [Option<usize>],
    pub bindings: bool,
}

impl Pen<'_> {
    fn open(&self, frame: Option<usize>) -> Line {
        let mut l = Line::default();
        l.push(" ".repeat(INDENT), Style::Plain);
        match frame {
            Some(_) => l.push(format!("{} ", self.g.fused.v), Style::FusedFrame),
            None => l.push("  ", Style::Plain),
        };
        l
    }

    fn close(&self, doc: &mut Document, mut l: Line, frame: Option<usize>) {
        if frame.is_some() {
            l.pad_to(INDENT + 2 + self.geo.content);
            l.push(format!(" {}", self.g.fused.v), Style::FusedFrame);
        }
        doc.lines.push(l);
    }

    fn kernels(&self, grp: usize) -> String {
        let group = &self.plan.groups[grp];
        let mut s = if group.kernels.is_empty() {
            "d2d copy".to_string()
        } else {
            group
                .kernels
                .iter()
                .map(|k| k.to_string())
                .collect::<Vec<_>>()
                .join(self.g.then)
        };
        let reps = group.repeat.count(self.plan.rows);
        if reps > 1 {
            s.push_str(&format!(" {}{reps}", self.g.times));
        }
        s
    }

    /// 2026-09-28: A wire segment carrying `e` down the main path, with its label.
    pub(super) fn connector(
        &self,
        doc: &mut Document,
        frame: Option<usize>,
        e: EdgeIdx,
        note: &str,
    ) {
        let on_chip =
            frame.is_some() || matches!(self.plan.edge_states[e], Some(EdgeState::Fused(_)));
        let (stroke, style, place) = if on_chip {
            (self.g.dotted, Style::EdgeFused, "on-chip")
        } else {
            (self.g.wire, Style::EdgeMaterialized, "DRAM")
        };
        let mut l = self.open(frame);
        l.push(" ".repeat(WIRE), Style::Plain);
        l.push(format!("{stroke} {} ", self.g.arrow), style);
        let label = edge_label(self.c, self.plan, e, self.g);
        let room = self.geo.content.saturating_sub(WIRE + 4);
        let text = if note.is_empty() {
            format!("{label}  {place}")
        } else {
            format!("{label}  {place}  {note}")
        };
        l.push(clip(&text, room, self.g), style);
        self.close(doc, l, frame);
    }

    fn wires(&self, doc: &mut Document, frame: Option<usize>, cols: &[(usize, Option<String>)]) {
        let stroke = if frame.is_some() {
            self.g.dotted
        } else {
            self.g.wire
        };
        let style = if frame.is_some() {
            Style::EdgeFused
        } else {
            Style::EdgeMaterialized
        };
        let mut l = self.open(frame);
        let base = l.width();
        for (col, label) in cols {
            l.pad_to(base + col);
            let place = if frame.is_some() { "on-chip" } else { "DRAM" };
            let text = match label {
                Some(t) => format!("{stroke} {} {t}  {place}", self.g.arrow),
                None => format!("{} ", self.g.arrow),
            };
            let room = self.geo.pair_w.saturating_sub(WIRE);
            l.push(clip(&text, room, self.g), style);
        }
        self.close(doc, l, frame);
    }

    /// 2026-09-28: The connector between two rows.
    pub(super) fn link(&self, doc: &mut Document, prev: &Row, row: &Row, frame: Option<usize>) {
        let from = prev.nodes();
        let feeds = |n: NodeIdx| -> Option<EdgeIdx> {
            self.c.nodes[n]
                .inputs
                .iter()
                .copied()
                .find(|&e| self.c.edges[e].producer.is_some_and(|p| from.contains(&p)))
        };
        match (prev, row) {
            (_, Row::FanOut) => {
                let to_experts = from
                    .iter()
                    .flat_map(|&p| self.c.nodes[p].outputs.iter().copied())
                    .find(|&e| {
                        self.c.edges[e]
                            .consumers
                            .iter()
                            .any(|&r| self.c.nodes[r].op == crate::ir::OpKind::ExpertGateUp)
                    });
                if let Some(e) = to_experts {
                    self.connector(doc, frame, e, "");
                }
            }
            (Row::Pair(a, b), Row::FanIn) => {
                // 2026-09-28: Both paths carry their result down into the join.
                let mut cols = Vec::new();
                for (col, n) in [(WIRE, a), (self.geo.right_col() + WIRE, b)] {
                    if let Some(&e) = n.and_then(|n| self.c.nodes[n].outputs.first()) {
                        cols.push((col, Some(edge_label(self.c, self.plan, e, self.g))));
                    }
                }
                self.wires(doc, frame, &cols);
            }
            (Row::FanOut, _) | (_, Row::FanIn) | (Row::FanIn, _) => {}
            (_, Row::Single(n)) => match feeds(*n) {
                Some(e) => self.connector(doc, frame, e, ""),
                None => {
                    let l = self.open(frame);
                    self.close(doc, l, frame);
                }
            },
            (_, Row::Pair(a, b)) => {
                let mut cols = Vec::new();
                for (col, n) in [(WIRE, a), (self.geo.right_col() + WIRE, b)] {
                    if let Some(e) = n.and_then(feeds) {
                        cols.push((col, Some(edge_label(self.c, self.plan, e, self.g))));
                    }
                }
                self.wires(doc, frame, &cols);
            }
        }
    }

    /// 2026-09-28: A fused group's top edge, titled with its kernels and numerics badge.
    pub(super) fn frame_top(&self, doc: &mut Document, grp: usize) {
        let f = self.g.fused;
        let (b, bstyle) = badge(&self.plan.groups[grp].numerics, self.g);
        let span = self.geo.width - INDENT;
        let room = span.saturating_sub(b.chars().count() + 9);
        let title = clip(&self.kernels(grp), room, self.g);
        let mut l = Line::default();
        l.push(" ".repeat(INDENT), Style::Plain);
        l.push(format!("{}{} ", f.tl, f.h), Style::FusedFrame);
        l.push(title, Style::Accent);
        l.push(" ", Style::Plain);
        let fill = span.saturating_sub(l.width() - INDENT + b.chars().count() + 4);
        l.push(f.h.to_string().repeat(fill), Style::FusedFrame);
        l.push(" ", Style::Plain);
        l.push(b, bstyle);
        l.push(format!(" {}{}", f.h, f.tr), Style::FusedFrame);
        doc.lines.push(l);
    }

    /// 2026-09-28: A fused group's bottom edge, naming what the kernel writes to memory.
    pub(super) fn frame_bottom(&self, doc: &mut Document, grp: usize) {
        let f = self.g.fused;
        let span = self.geo.width - INDENT;
        let mut written = Vec::new();
        for &n in &self.plan.groups[grp].nodes {
            for &e in &self.c.nodes[n].outputs {
                if self.plan.edge_states[e] == Some(EdgeState::Materialized) {
                    written.push(
                        self.c.edges[e]
                            .id
                            .rsplit('.')
                            .next()
                            .unwrap_or("")
                            .to_string(),
                    );
                }
            }
        }
        let mut l = Line::default();
        l.push(" ".repeat(INDENT), Style::Plain);
        l.push(format!("{}{} ", f.bl, f.h), Style::FusedFrame);
        let note = format!("writes {}", written.join(", "));
        l.push(clip(&note, span.saturating_sub(8), self.g), Style::Dim);
        l.push(" ", Style::Plain);
        let fill = self.geo.width.saturating_sub(l.width() + 1);
        l.push(
            format!("{}{}", f.h.to_string().repeat(fill), f.br),
            Style::FusedFrame,
        );
        doc.lines.push(l);
    }

    fn label(&self, n: NodeIdx) -> String {
        let node = &self.c.nodes[n];
        let base = match node.op {
            crate::ir::OpKind::Linear(_) => "linear",
            crate::ir::OpKind::ExpertGateUp | crate::ir::OpKind::ExpertDown => "experts",
            ref op => op.base_name(),
        };
        if node.local == base || node.local.starts_with(base) {
            node.local.clone()
        } else {
            format!("{}{}{base}", node.local, self.g.dot)
        }
    }

    fn boxed(&self, n: NodeIdx, w: usize) -> ([Line; 3], Style) {
        let heavy = self.c.nodes[n].op.is_heavy();
        let (fr, style): (Frame, Style) = if heavy {
            (self.g.heavy, Style::OpHeavy)
        } else {
            (self.g.light, Style::OpLight)
        };
        let inner = w - 4;
        let text = clip(&self.label(n), inner, self.g);
        let mut top = Line::default();
        top.push(
            format!("{}{}{}", fr.tl, fr.h.to_string().repeat(w - 2), fr.tr),
            style,
        );
        let mut mid = Line::default();
        mid.push(format!("{} ", fr.v), style);
        mid.push(text, Style::Plain);
        mid.pad_to(w - 2);
        mid.push(format!(" {}", fr.v), style);
        let mut bot = Line::default();
        bot.push(
            format!("{}{}{}", fr.bl, fr.h.to_string().repeat(w - 2), fr.br),
            style,
        );
        ([top, mid, bot], style)
    }

    /// 2026-09-28: Annotations for a node's box: the top line (inputs from anywhere but the
    /// node drawn above, and the weight), the middle line (kernels and badge outside a frame,
    /// the weight inside one) and the bottom line (module bindings under `--layer`).
    fn notes(&self, n: NodeIdx, framed: bool) -> [Vec<(String, Style)>; 3] {
        let node = &self.c.nodes[n];
        let mut notes: [Vec<(String, Style)>; 3] = Default::default();
        let from_elsewhere: Vec<&str> = node
            .inputs
            .iter()
            .filter(|&&e| self.c.edges[e].producer.is_some_and(|p| p + 1 != n))
            .map(|&e| {
                let p = self.c.edges[e].producer.map(|p| &self.c.nodes[p]);
                if p.is_some_and(|p| {
                    p.layer != node.layer || p.block != node.block && p.layer.is_none()
                }) {
                    "stream in"
                } else {
                    self.c.edges[e].id.rsplit('.').next().unwrap_or("")
                }
            })
            .collect();
        if !from_elsewhere.is_empty() {
            notes[0].push((
                format!("{}{}", self.g.reads, from_elsewhere.join(", ")),
                Style::Dim,
            ));
        }
        let weight = node
            .weight
            .map(|w| (format!("weight {}", format_label(w, self.g)), Style::Format));
        match (framed, self.group_of[n]) {
            (false, Some(grp)) => {
                notes[1].push((self.kernels(grp), Style::Accent));
                notes[1].push(badge(&self.plan.groups[grp].numerics, self.g));
                notes[0].extend(weight);
            }
            _ => notes[1].extend(weight),
        }
        if self.bindings && !node.binding.is_empty() {
            notes[2].push((node.binding.join(", "), Style::Dim));
        }
        notes
    }

    /// 2026-09-28: Draw one row.
    pub(super) fn row(&self, doc: &mut Document, row: &Row, frame: Option<usize>) {
        match row {
            Row::Single(n) => self.single(doc, *n, frame),
            Row::Pair(a, b) => self.pair(doc, *a, *b, frame),
            Row::FanOut => self.fan(doc, frame, true),
            Row::FanIn => self.fan(doc, frame, false),
        }
    }

    fn single(&self, doc: &mut Document, n: NodeIdx, frame: Option<usize>) {
        let (lines, _) = self.boxed(n, self.geo.box_w);
        let notes = self.notes(n, frame.is_some());
        let side = self.geo.content.saturating_sub(self.geo.box_w + 2);
        // 2026-09-28: Each box line carries its own facts; what does not fit beside the middle
        // line moves to the bottom line when that is free.
        let mut side_lines: [Option<Line>; 3] = Default::default();
        for (i, facts) in notes.iter().enumerate() {
            if facts.is_empty() {
                continue;
            }
            let mut wrapped = super::card::wrap(facts, side, self.g).into_iter();
            side_lines[i] = wrapped.next();
            if i == 1 && notes[2].is_empty() {
                side_lines[2] = wrapped.next();
            }
        }
        for (b, side_line) in lines.into_iter().zip(side_lines) {
            let mut l = self.open(frame);
            l.spans.extend(b.spans);
            if let (false, Some(s)) = (self.geo.compact, side_line) {
                l.push("  ", Style::Plain);
                l.spans.extend(s.spans);
            }
            self.close(doc, l, frame);
        }
        if self.geo.compact {
            for (text, style) in notes.into_iter().flatten() {
                let mut l = self.open(frame);
                l.push("  ", Style::Plain);
                l.push(clip(&text, self.geo.content - 2, self.g), style);
                self.close(doc, l, frame);
            }
        }
    }

    fn pair(
        &self,
        doc: &mut Document,
        a: Option<NodeIdx>,
        b: Option<NodeIdx>,
        frame: Option<usize>,
    ) {
        let w = self.geo.pair_w.min(self.geo.box_w.max(24));
        let boxes = [a, b].map(|n| n.map(|n| self.boxed(n, w).0));
        for i in 0..3 {
            let mut l = self.open(frame);
            let base = l.width();
            for (col, bx) in [(0, &boxes[0]), (self.geo.right_col(), &boxes[1])] {
                if let Some(lines) = bx {
                    l.pad_to(base + col);
                    l.spans.extend(lines[i].spans.clone());
                } else if i == 1 && col == 0 {
                    let (stroke, style) = if frame.is_some() {
                        (self.g.dotted, Style::EdgeFused)
                    } else {
                        (self.g.wire, Style::EdgeMaterialized)
                    };
                    l.pad_to(base + WIRE);
                    l.push(stroke.to_string(), style);
                }
            }
            self.close(doc, l, frame);
        }
        if frame.is_none() {
            let mut l = self.open(frame);
            let base = l.width();
            for (col, n) in [(0, a), (self.geo.right_col(), b)] {
                if let Some(grp) = n.and_then(|n| self.group_of[n]) {
                    l.pad_to(base + col);
                    l.push(
                        clip(&self.kernels(grp), self.geo.pair_w - 1, self.g),
                        Style::Accent,
                    );
                }
            }
            self.close(doc, l, frame);
        }
    }

    fn fan(&self, doc: &mut Document, frame: Option<usize>, out: bool) {
        let style = if frame.is_some() {
            Style::EdgeFused
        } else {
            Style::EdgeMaterialized
        };
        let right = self.geo.right_col() + WIRE;
        let mut l = self.open(frame);
        let base = l.width();
        l.pad_to(base + WIRE);
        let corner = if out { self.g.fan[1] } else { self.g.fan[2] };
        let run = right - WIRE - 1;
        l.push(
            format!(
                "{}{}{corner}",
                self.g.fan[0],
                self.g.light.h.to_string().repeat(run)
            ),
            style,
        );
        self.close(doc, l, frame);
        if out {
            let d = &self.c.dims;
            let routed = match (d.get("top_k"), d.get("experts")) {
                (Some(k), Some(e)) => format!("routed: top-{k} of {e} experts"),
                _ => "routed experts".to_string(),
            };
            self.wires(
                doc,
                frame,
                &[(WIRE, Some(routed)), (right, Some("shared expert".into()))],
            );
        }
    }
}
