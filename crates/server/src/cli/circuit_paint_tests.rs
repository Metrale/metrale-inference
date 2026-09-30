// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Colour depth precedence and the escapes each depth emits, with the TTY answer
//! and environment values injected.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use metrale_circuit::display::{Document, Line, Span, Style};

use super::*;

fn doc() -> Document {
    let spans = [
        Style::Heading,
        Style::EdgeMaterialized,
        Style::EdgeFused,
        Style::FusedFrame,
        Style::NumericsDiffers,
        Style::Plain,
    ]
    .into_iter()
    .map(|style| Span {
        text: format!("{style:?} "),
        style,
    })
    .collect();
    Document {
        lines: vec![Line { spans }],
    }
}

#[test]
fn never_and_no_color_emit_no_escape_byte() {
    for (choice, tty) in [
        (ColorChoice::Never, true),
        (ColorChoice::Never, false),
        (ColorChoice::Auto, false),
    ] {
        let depth = resolve_depth(choice, tty, None, Some("truecolor"));
        assert_eq!(depth, Depth::None, "{choice:?} tty={tty}");
        assert!(!paint(&doc(), depth).contains('\x1b'));
    }
    for choice in [ColorChoice::Always, ColorChoice::Auto] {
        // 2026-09-28: Any non-empty NO_COLOR, `0` included, outranks --color always.
        for no_color in ["1", "0", "yes"] {
            let depth = resolve_depth(choice, true, Some(no_color), Some("truecolor"));
            assert_eq!(depth, Depth::None);
            assert!(!paint(&doc(), depth).contains('\x1b'));
        }
    }
    // 2026-09-28: An empty NO_COLOR is ignored, as in the TUI.
    assert_eq!(
        resolve_depth(ColorChoice::Always, false, Some(""), Some("24bit")),
        Depth::True
    );
}

#[test]
fn always_with_truecolor_emits_24_bit_sequences_from_the_palette() {
    let depth = resolve_depth(ColorChoice::Always, false, None, Some("truecolor"));
    assert_eq!(depth, Depth::True);
    let text = paint(&doc(), depth);
    let w = theme::WARN;
    assert!(
        text.contains(&format!("\x1b[38;2;{};{};{}m", w.0, w.1, w.2)),
        "{text:?}"
    );
    let g = theme::GREEN;
    assert!(text.contains(&format!("\x1b[38;2;{};{};{}m", g.0, g.1, g.2)));
    assert!(
        text.contains("\x1b[1;38;2;"),
        "bold styles keep their weight"
    );
    assert!(
        !text.contains("38;5;"),
        "no 256-colour index at true colour"
    );
    assert!(
        text.contains("Plain"),
        "the plain span is still printed, uncoloured"
    );
}

#[test]
fn without_truecolor_the_pinned_256_colour_indices_are_used() {
    let depth = resolve_depth(ColorChoice::Auto, true, None, None);
    assert_eq!(depth, Depth::Ansi256);
    let text = paint(&doc(), depth);
    assert!(text.contains(&format!("\x1b[38;5;{}m", theme::WARN.3)));
    assert!(!text.contains("38;2;"));
}
