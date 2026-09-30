// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Colour for `met circuit display`: each semantic style of the circuit document
//! maps to a colour of the TUI palette (`tui::theme`), drawn at the depth `theme::depth_of`
//! resolves from `NO_COLOR` and `COLORTERM`.
//!
//! Owner: server CLI.
//! Invariants:
//! - Pure: the TTY answer and the two environment values are arguments, so the precedence is
//!   tested without touching the process environment.
//! - At `Depth::None` the output holds no escape byte at all: no colour and no bold.

use metrale_circuit::display::{Document, Style};

use super::ColorChoice;
use crate::tui::theme::{self, C, Depth};

/// 2026-09-28: The depth to paint at: `never` is none; `auto` is none off a terminal; otherwise
/// what `NO_COLOR` and `COLORTERM` allow, exactly as the TUI resolves them.
pub(crate) fn resolve_depth(
    choice: ColorChoice,
    is_tty: bool,
    no_color: Option<&str>,
    colorterm: Option<&str>,
) -> Depth {
    match (choice, is_tty) {
        (ColorChoice::Never, _) | (ColorChoice::Auto, false) => Depth::None,
        (ColorChoice::Auto, true) | (ColorChoice::Always, _) => {
            theme::depth_of(no_color, colorterm)
        }
    }
}

/// 2026-09-28: A style's palette colour and whether it is bold. `None` keeps the terminal's
/// default colour.
fn look(style: Style) -> (Option<C>, bool) {
    match style {
        Style::Plain => (None, false),
        Style::Heading => (Some(theme::PURPLE), true),
        Style::Accent => (Some(theme::CYAN), false),
        Style::Dim => (Some(theme::TEXT_DIM), false),
        Style::Border | Style::OpLight | Style::Format | Style::LayerDense => {
            (Some(theme::TEXT_2), false)
        }
        Style::OpHeavy | Style::LayerAttn => (Some(theme::PURPLE), false),
        Style::EdgeMaterialized | Style::LayerMoe => (Some(theme::WARN), false),
        Style::EdgeFused | Style::NumericsBitIdentical => (Some(theme::GREEN), false),
        Style::FusedFrame => (Some(theme::CYAN), true),
        Style::NumericsReference => (Some(theme::TEXT_2), false),
        Style::NumericsDiffers => (Some(theme::ERROR), true),
        Style::LayerGdn => (Some(theme::CYAN), false),
    }
}

fn open(style: Style, depth: Depth) -> String {
    let (colour, bold) = look(style);
    let mut codes: Vec<String> = Vec::new();
    if bold {
        codes.push("1".into());
    }
    match (colour, depth) {
        (_, Depth::None) => return String::new(),
        (Some(c), Depth::True) => codes.push(format!("38;2;{};{};{}", c.0, c.1, c.2)),
        (Some(c), Depth::Ansi256) => codes.push(format!("38;5;{}", c.3)),
        (None, _) => {}
    }
    if codes.is_empty() {
        String::new()
    } else {
        format!("\x1b[{}m", codes.join(";"))
    }
}

/// 2026-09-28: The document as terminal text at `depth`.
pub(crate) fn paint(doc: &Document, depth: Depth) -> String {
    let mut out = String::new();
    for line in &doc.lines {
        let mut text = String::new();
        for span in &line.spans {
            let start = open(span.style, depth);
            text.push_str(&start);
            text.push_str(&span.text);
            if !start.is_empty() {
                text.push_str("\x1b[0m");
            }
        }
        out.push_str(text.trim_end());
        out.push('\n');
    }
    out
}

#[cfg(test)]
#[path = "circuit_paint_tests.rs"]
mod circuit_paint_tests;
