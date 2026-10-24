// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The Benchmarks section in a binary built without the `bench` cargo feature.
//!
//! The tab stays in the sidebar, so the section numbering and `1-7 jump` keep their meaning in
//! every build; its pane says the harness is not compiled in and names the rebuild, the declared
//! mode PCND asks for in place of an empty or missing tab.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};

use super::panel;
use crate::cli::no_bench::REBUILD;
use crate::tui::app::App;
use crate::tui::theme;

/// 2026-10-17: The footer while the Benchmarks section is shown.
pub const HINTS: &str = "⇥ cycle · 1-7 jump · ? help · q quit";

pub fn draw(f: &mut Frame, _app: &App, area: Rect) {
    let text = vec![
        Line::from(Span::styled(
            "Benchmarks are not in this build: met was built without the `bench` feature.",
            theme::text(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("Rebuild with `{REBUILD}`."),
            theme::dim(),
        )),
    ];
    f.render_widget(
        Paragraph::new(text)
            .wrap(Wrap { trim: false })
            .block(panel("BENCHMARKS".to_string(), true)),
        area,
    );
}
