// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit display`: the human view of one fusion plan. It builds a styled
//! document (lines of spans, each span a semantic [`Style`]) from the same circuit and plan
//! `met circuit show` prints; the CLI maps styles to colours.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - Pure: no environment, no TTY probe, no ANSI. Width and glyph set are inputs.
//! - No line is wider than [`DisplayOpts::width`]; below [`COMPACT_BELOW`] columns the layout
//!   drops the side annotation column instead of wrapping boxes.
//! - A materialised edge is never drawn inside a fused frame: inside a frame every hand-off is
//!   on-chip, and what the kernel also writes is named on the frame's bottom edge.

use crate::fuser::FusionPlan;
use crate::ir::Circuit;

mod card;
mod diagram;
mod draw;
pub mod glyphs;
mod labels;
mod rows;
mod strip;

pub use glyphs::Glyphs;

/// 2026-09-28: Below this width the diagrams move annotations onto their own lines.
pub const COMPACT_BELOW: usize = 72;
/// 2026-09-28: The narrowest width the renderer lays out.
pub const MIN_WIDTH: usize = 40;

/// 2026-09-28: What a span means; the CLI picks its colour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Style {
    /// 2026-09-28: Body text.
    Plain,
    /// 2026-09-28: Titles.
    Heading,
    /// 2026-09-28: Emphasised values (counts, the digest).
    Accent,
    /// 2026-09-28: Secondary text.
    Dim,
    /// 2026-09-28: Card and light-op borders.
    Border,
    /// 2026-09-28: A heavy (opaque) op: projections, attention, the recurrence, experts.
    OpHeavy,
    /// 2026-09-28: A light op: norms, adds, activations, quantization.
    OpLight,
    /// 2026-09-28: A materialised edge (a DRAM round trip).
    EdgeMaterialized,
    /// 2026-09-28: A fused edge (stays on chip).
    EdgeFused,
    /// 2026-09-28: A fused group's frame and title.
    FusedFrame,
    /// 2026-09-28: A tensor format or shape.
    Format,
    /// 2026-09-28: A bit-identical numerics badge.
    NumericsBitIdentical,
    /// 2026-09-28: A reference numerics badge.
    NumericsReference,
    /// 2026-09-28: A differs numerics badge.
    NumericsDiffers,
    /// 2026-09-28: A GatedDeltaNet layer glyph.
    LayerGdn,
    /// 2026-09-28: An attention layer glyph.
    LayerAttn,
    /// 2026-09-28: A dense FFN glyph.
    LayerDense,
    /// 2026-09-28: A MoE FFN glyph.
    LayerMoe,
}

/// 2026-09-28: A run of text in one style.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Span {
    /// 2026-09-28: The text; never contains a newline.
    pub text: String,
    /// 2026-09-28: Its meaning.
    pub style: Style,
}

/// 2026-09-28: One output line.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Line {
    /// 2026-09-28: Spans, left to right.
    pub spans: Vec<Span>,
}

impl Line {
    /// 2026-09-28: Terminal columns: every glyph the renderer emits is one column wide.
    pub fn width(&self) -> usize {
        self.spans.iter().map(|s| s.text.chars().count()).sum()
    }

    /// 2026-09-28: The text without styles.
    pub fn plain(&self) -> String {
        self.spans.iter().map(|s| s.text.as_str()).collect()
    }

    pub(crate) fn push(&mut self, text: impl Into<String>, style: Style) -> &mut Self {
        let text = text.into();
        if !text.is_empty() {
            self.spans.push(Span { text, style });
        }
        self
    }

    pub(crate) fn pad_to(&mut self, width: usize) -> &mut Self {
        let w = self.width();
        if w < width {
            self.push(" ".repeat(width - w), Style::Plain);
        }
        self
    }
}

/// 2026-09-28: A rendered document.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Document {
    /// 2026-09-28: Lines, top to bottom.
    pub lines: Vec<Line>,
}

impl Document {
    /// 2026-09-28: The uncoloured text, trailing spaces trimmed, one `\n` per line.
    pub fn plain(&self) -> String {
        let mut out = String::new();
        for l in &self.lines {
            out.push_str(l.plain().trim_end());
            out.push('\n');
        }
        out
    }
}

/// 2026-09-28: How much of the model to draw.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Expand {
    /// 2026-09-28: One diagram per distinct layer plan, with a `× N layers` badge.
    Summary,
    /// 2026-09-28: One concrete layer, with its module bindings.
    Layer(usize),
    /// 2026-09-28: Every layer.
    AllLayers,
}

/// 2026-09-28: Rendering options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisplayOpts {
    /// 2026-09-28: Columns available.
    pub width: usize,
    /// 2026-09-28: Glyph set.
    pub glyphs: Glyphs,
    /// 2026-09-28: How much to draw.
    pub expand: Expand,
}

/// 2026-09-28: Facts about the instance the plan came from, for the header card.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisplayInfo {
    /// 2026-09-28: Checkpoint id.
    pub checkpoint: String,
    /// 2026-09-28: Recipe id.
    pub recipe: String,
    /// 2026-09-28: Bytes of materialised edges per step and the arena they fit in, when the
    /// buffer planner laid the plan out.
    pub bytes: Option<(u64, u64)>,
}

/// 2026-09-28: Why a document was not produced.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DisplayError {
    /// 2026-09-28: `--layer` past the model's layers.
    #[error("layer {layer} is out of range: this model has layers 0..={last}")]
    LayerOutOfRange {
        /// 2026-09-28: Requested layer.
        layer: usize,
        /// 2026-09-28: Last valid layer.
        last: usize,
    },
    /// 2026-09-28: `--layer` for a plan whose section has no layers (the draft head).
    #[error("the {0} plan has no numbered layers; drop --layer")]
    NoLayers(&'static str),
    /// 2026-09-28: Narrower than [`MIN_WIDTH`].
    #[error("width {0} is below the {MIN_WIDTH}-column minimum")]
    TooNarrow(usize),
}

/// 2026-09-28: Render `plan` of `circuit`.
pub fn display(
    circuit: &Circuit,
    plan: &FusionPlan,
    info: &DisplayInfo,
    opts: &DisplayOpts,
) -> Result<Document, DisplayError> {
    if opts.width < MIN_WIDTH {
        return Err(DisplayError::TooNarrow(opts.width));
    }
    let layers = diagram::layers_in_plan(circuit, plan);
    if let Expand::Layer(n) = opts.expand {
        let Some(&last) = layers.last() else {
            return Err(DisplayError::NoLayers(plan.mode.name()));
        };
        if !layers.contains(&n) {
            return Err(DisplayError::LayerOutOfRange { layer: n, last });
        }
    }
    let g = opts.glyphs.set();
    let mut doc = Document::default();
    card::header(&mut doc, circuit, plan, info, opts.width, &g);
    if !layers.is_empty() {
        strip::strip(&mut doc, circuit, &layers, opts.width, &g);
    }
    diagram::diagrams(&mut doc, circuit, plan, opts, &g);
    card::footer(&mut doc, plan, info, opts.width, &g);
    for l in &mut doc.lines {
        fit(l, opts.width, &g);
    }
    Ok(doc)
}

/// 2026-09-28: Truncate a line to `width` columns, ending in the ellipsis. The layout sizes
/// every line to fit; this is the backstop for a long module name.
pub(crate) fn fit(line: &mut Line, width: usize, g: &glyphs::Set) {
    if line.width() <= width {
        return;
    }
    let keep = width.saturating_sub(g.ellipsis.chars().count());
    let mut used = 0;
    let mut out = Vec::new();
    for s in line.spans.drain(..) {
        let n = s.text.chars().count();
        if used + n <= keep {
            used += n;
            out.push(s);
            continue;
        }
        let text: String = s.text.chars().take(keep - used).collect();
        out.push(Span {
            text,
            style: s.style,
        });
        out.push(Span {
            text: g.ellipsis.to_string(),
            style: Style::Dim,
        });
        break;
    }
    line.spans = out;
}

/// 2026-09-28: Truncate `text` to `max` columns with the ellipsis.
pub(crate) fn clip(text: &str, max: usize, g: &glyphs::Set) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let e = g.ellipsis.chars().count();
    let mut s: String = text.chars().take(max.saturating_sub(e)).collect();
    s.push_str(g.ellipsis);
    s.chars().take(max).collect()
}
