// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Diagram geometry and the text of edge, format and numerics labels.
//!
//! Owner: metrale-circuit.
//! Invariants: every label is built from the glyph set, so the ASCII render stays 7-bit.

use super::COMPACT_BELOW;
use super::Style;
use super::glyphs::Set;
use crate::format::{Format, Scale};
use crate::fuser::FusionPlan;
use crate::ir::{Circuit, EdgeIdx};
use crate::rules::Numerics;

pub(super) const INDENT: usize = 2;
/// 2026-09-28: Column of a path's wire inside its box's span.
pub(super) const WIRE: usize = 3;

/// 2026-09-28: Column geometry for one width.
pub(super) struct Geo {
    pub width: usize,
    pub content: usize,
    pub box_w: usize,
    pub pair_w: usize,
    pub compact: bool,
}

impl Geo {
    pub(super) fn new(width: usize) -> Self {
        let content = width - INDENT - 4;
        let compact = width < COMPACT_BELOW;
        let box_w = if compact {
            content.min(30)
        } else {
            (content * 2 / 5).clamp(22, 34)
        };
        Geo {
            width,
            content,
            box_w,
            pair_w: (content - 3) / 2,
            compact,
        }
    }

    pub(super) fn right_col(&self) -> usize {
        self.pair_w + 3
    }
}

/// 2026-09-28: `bf16`, `fp8 e4m3/token`, `nvfp4 g16`, `fp8 e4m3/128×128 blk`.
pub(super) fn format_label(f: Format, g: &Set) -> String {
    match f {
        Format::Fp8E4m3 { scale } => match scale {
            Scale::PerTensor => "fp8 e4m3/tensor".into(),
            Scale::PerToken => "fp8 e4m3/token".into(),
            Scale::PerChannel => "fp8 e4m3/channel".into(),
            Scale::Group(n) => format!("fp8 e4m3/g{n}"),
            Scale::Block(r, c) => format!("fp8 e4m3/{r}{}{c} blk", g.times),
        },
        Format::Nvfp4 { group } => format!("nvfp4 g{group}"),
        other => other.name(),
    }
}

/// 2026-09-28: `bf16 · n×5120` for an edge as this plan stores it.
pub(super) fn edge_label(c: &Circuit, plan: &FusionPlan, e: EdgeIdx, g: &Set) -> String {
    let edge = &c.edges[e];
    let mut dims = c.dims.clone();
    dims.insert("n".into(), 1);
    let rows = match edge.rows.eval(&dims) {
        Ok(1) => "n".to_string(),
        Ok(k) => format!("{k}n"),
        Err(_) => edge.rows.text().to_string(),
    };
    format!(
        "{}{}{rows}{}{}",
        format_label(plan.edge_formats[e], g),
        g.dot,
        g.times,
        edge.dim_value
    )
}

pub(super) fn badge(n: &Numerics, g: &Set) -> (String, Style) {
    match n {
        Numerics::BitIdentical { .. } => (g.badge[0].to_string(), Style::NumericsBitIdentical),
        Numerics::Reference => (g.badge[1].to_string(), Style::NumericsReference),
        Numerics::Differs { lever } => {
            (format!("{} ({lever})", g.badge[2]), Style::NumericsDiffers)
        }
    }
}
