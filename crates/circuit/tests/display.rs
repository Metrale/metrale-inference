// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `met circuit display` on the two checked-in models: the width and glyph
//! invariants at every width, and that frames, edges and layer counts say what the plan says.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use metrale_circuit::display::{DisplayError, DisplayOpts, Document, Expand, Glyphs, Style};
use metrale_circuit::{Instance, LoadError, Loaded, Mode, Section};
use unicode_width::UnicodeWidthStr;

fn golden() -> Vec<(Instance, Loaded)> {
    common::instances()
        .into_iter()
        .filter(|i| i.golden)
        .map(|i| {
            let l = common::load(&i);
            (i, l)
        })
        .collect()
}

fn draw(
    inst: &Instance,
    loaded: &Loaded,
    mode: Mode,
    rows: u64,
    opts: DisplayOpts,
) -> Result<Document, LoadError> {
    let avail = common::available(inst, &loaded.rules);
    metrale_circuit::display_plan(inst, loaded, &avail, mode, rows, &opts)
}

fn opts(width: usize, glyphs: Glyphs) -> DisplayOpts {
    DisplayOpts {
        width,
        glyphs,
        expand: Expand::Summary,
    }
}

#[test]
fn no_line_is_wider_than_the_terminal_at_any_width() {
    for (inst, loaded) in golden() {
        for glyphs in [Glyphs::Unicode, Glyphs::Ascii] {
            for width in 60..=200 {
                let doc = draw(&inst, &loaded, Mode::Decode, 1, opts(width, glyphs)).unwrap();
                for (i, line) in doc.plain().lines().enumerate() {
                    let cols = UnicodeWidthStr::width(line);
                    assert!(
                        cols <= width,
                        "{} {glyphs:?} width {width}, line {i} is {cols} columns: {line}",
                        inst.arch
                    );
                    assert_eq!(
                        cols,
                        line.chars().count(),
                        "a glyph wider than one column: {line}"
                    );
                }
            }
        }
    }
}

#[test]
fn each_glyph_set_stays_in_its_own_alphabet() {
    for (inst, loaded) in golden() {
        for mode in [Mode::Decode, Mode::Verify, Mode::Draft] {
            let rows = if mode == Mode::Verify { 2 } else { 1 };
            let uni = draw(&inst, &loaded, mode, rows, opts(120, Glyphs::Unicode))
                .unwrap()
                .plain();
            let bad: Vec<char> = uni
                .chars()
                .filter(|c| matches!(c, '+' | '#' | '='))
                .collect();
            assert!(
                bad.is_empty(),
                "{} {mode:?}: ASCII-fallback glyphs {bad:?} in the Unicode render",
                inst.arch
            );
            assert!(
                uni.contains('┏') && uni.contains('╭'),
                "the Unicode render drew no frames"
            );
            let ascii = draw(&inst, &loaded, mode, rows, opts(120, Glyphs::Ascii))
                .unwrap()
                .plain();
            let wide: Vec<char> = ascii.chars().filter(|c| !c.is_ascii()).collect();
            assert!(
                wide.is_empty(),
                "{} {mode:?}: non-ASCII {wide:?} in the ASCII render",
                inst.arch
            );
        }
    }
}

/// 2026-09-28: The lines between a frame's top and bottom edges, keyed by the top edge.
fn frames(doc: &Document) -> Vec<(String, Vec<&metrale_circuit::display::Line>)> {
    let mut out = Vec::new();
    let mut open: Option<(String, Vec<&metrale_circuit::display::Line>)> = None;
    for line in &doc.lines {
        let edge = line
            .spans
            .iter()
            .find(|s| !s.text.trim().is_empty())
            .filter(|s| s.style == Style::FusedFrame)
            .map(|s| s.text.trim_start());
        match edge {
            Some(t) if t.starts_with('┏') => open = Some((line.plain(), Vec::new())),
            Some(t) if t.starts_with('┗') => out.extend(open.take()),
            _ => {
                if let Some((_, lines)) = open.as_mut() {
                    lines.push(line);
                }
            }
        }
    }
    assert!(
        open.is_none(),
        "a frame was never closed: {:?}",
        open.map(|o| o.0)
    );
    out
}

#[test]
fn a_fused_group_is_framed_with_its_kernel_and_members_and_no_dram_edge_inside() {
    for (inst, loaded) in golden() {
        for (mode, rows) in [(Mode::Decode, 1), (Mode::MultiSeq, 96), (Mode::Verify, 4)] {
            let doc = draw(&inst, &loaded, mode, rows, opts(140, Glyphs::Unicode)).unwrap();
            let frames = frames(&doc);
            assert!(!frames.is_empty(), "{} {mode:?}: no frames", inst.arch);
            for (title, inside) in &frames {
                for l in inside {
                    assert!(
                        l.spans.iter().all(|s| s.style != Style::EdgeMaterialized),
                        "{} {mode:?}: a DRAM edge inside the frame `{title}`: {}",
                        inst.arch,
                        l.plain()
                    );
                }
            }
            let add_norm = frames
                .iter()
                .find(|(t, _)| t.contains("norm::residual_add_rms_norm"))
                .expect("the residual add + norm frame");
            let body: String = add_norm.1.iter().map(|l| l.plain()).collect();
            assert!(
                body.contains("add · residual_add") && body.contains("post_norm · rms_norm"),
                "{body}"
            );
            assert!(
                body.contains("on-chip"),
                "the hand-off inside the frame is not drawn on-chip"
            );
        }
    }
}

#[test]
fn the_dedup_badge_counts_the_layers_of_each_kind() {
    for (inst, loaded) in golden() {
        let text = draw(&inst, &loaded, Mode::Decode, 1, opts(160, Glyphs::Unicode))
            .unwrap()
            .plain();
        let count = |t: &str| {
            loaded
                .circuit
                .blocks
                .iter()
                .filter(|b| b.template == t && b.section == Section::Main)
                .count()
        };
        let gdn = count("gdn");
        let attn = count("attn");
        assert!(
            text.contains(&format!("GatedDeltaNet layer  × {gdn} layers")),
            "{text}"
        );
        assert!(text.contains(&format!("Full-attention layer  × {attn} layers")));
        let titles = |t: &str| text.lines().filter(|l| l.starts_with(t)).count();
        assert_eq!(
            titles("▰ GatedDeltaNet layer"),
            1,
            "one diagram per distinct plan"
        );
        assert_eq!(
            titles("◆ Full-attention layer"),
            1,
            "one diagram per distinct plan"
        );
        let strip = text
            .lines()
            .find(|l| l.trim_start().starts_with('▰'))
            .unwrap();
        assert!(strip.contains("▰▰▰◆"));
        let rle = format!("(3× GDN → 1× Attention) × {}", attn);
        assert!(text.contains(&rle), "{rle}");
    }
}

#[test]
fn every_layer_expands_and_one_layer_shows_its_modules() {
    let (inst, loaded) = golden().into_iter().next().unwrap();
    let all = DisplayOpts {
        expand: Expand::AllLayers,
        ..opts(120, Glyphs::Unicode)
    };
    let text = draw(&inst, &loaded, Mode::Decode, 1, all).unwrap().plain();
    assert_eq!(
        text.lines()
            .filter(|l| l.starts_with("▰ GatedDeltaNet layer"))
            .count(),
        48
    );
    let one = DisplayOpts {
        expand: Expand::Layer(7),
        ..opts(160, Glyphs::Unicode)
    };
    let text = draw(&inst, &loaded, Mode::Decode, 1, one).unwrap().plain();
    assert!(
        text.contains("model.language_model.layers.7.self_attn.q_proj"),
        "{text}"
    );
    assert_eq!(
        text.lines()
            .filter(|l| l.starts_with("◆ Full-attention layer"))
            .count(),
        1
    );
}

#[test]
fn a_narrow_terminal_moves_annotations_below_the_boxes() {
    let (inst, loaded) = golden().into_iter().next().unwrap();
    let text = draw(&inst, &loaded, Mode::Decode, 1, opts(60, Glyphs::Unicode))
        .unwrap()
        .plain();
    let kernel = text
        .lines()
        .find(|l| l.contains("norm::rms_norm_residual"))
        .unwrap();
    assert!(
        !kernel.contains('│'),
        "annotation still beside a box at 60 columns: {kernel}"
    );
}

#[test]
fn bad_requests_are_typed_errors() {
    let (inst, loaded) = golden().into_iter().next().unwrap();
    let layer = |n| DisplayOpts {
        expand: Expand::Layer(n),
        ..opts(120, Glyphs::Unicode)
    };
    assert_eq!(
        draw(&inst, &loaded, Mode::Decode, 1, layer(64)).unwrap_err(),
        LoadError::Display(DisplayError::LayerOutOfRange {
            layer: 64,
            last: 63
        })
    );
    assert_eq!(
        draw(&inst, &loaded, Mode::Draft, 1, layer(0)).unwrap_err(),
        LoadError::Display(DisplayError::NoLayers("draft"))
    );
    assert_eq!(
        draw(&inst, &loaded, Mode::Decode, 1, opts(39, Glyphs::Ascii)).unwrap_err(),
        LoadError::Display(DisplayError::TooNarrow(39))
    );
}
