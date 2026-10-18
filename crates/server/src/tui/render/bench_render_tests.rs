// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Render tests for the Benchmarks section in a build with the `bench` feature: every
//! view at several sizes plus content checks, split from `render_tests.rs` (which keeps the
//! sections every build has) when the harness became optional.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::tests::{SIZES, app, render};
use crate::tui::app::{App, BenchSub, Section};
use crate::tui::bench_state::View;

#[test]
fn every_benchmarks_view_renders_at_every_size() {
    for sub in [BenchSub::Suite, BenchSub::History] {
        for view in [View::List, View::Params, View::Run] {
            for (w, h) in SIZES {
                let mut a = app();
                a.section = Section::Benchmarks;
                a.bench_sub = sub;
                a.bench.view = view;
                let out = render(&a, w, h);
                assert!(!out.is_empty(), "bench view at {w}x{h} drew nothing");
            }
        }
    }
}

#[test]
fn the_suite_list_shows_the_benchmarks_and_their_provenance() {
    let mut a = app();
    a.section = Section::Benchmarks;
    // 2026-09-26: The list scrolls with the cursor, so one frame need not
    // hold every name: select each entry in turn, which proves each one is
    // reachable.
    for (i, descriptor) in metrale_bench::registry::all().iter().enumerate() {
        a.bench.select(i);
        let out = render(&a, 160, 48);
        assert!(out.contains(descriptor.name), "missing {}", descriptor.name);
    }
    a.bench.select(0);
    let out = render(&a, 160, 48);
    assert!(out.contains("OFFICIAL"), "first-party badge is missing");
    assert!(out.contains("Metrale Engine"), "author is missing");
}

#[test]
fn the_parameter_form_shows_every_field_plus_the_endpoint() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::Params;
    let out = render(&a, 160, 48);
    for spec in &a.bench.specs {
        assert!(out.contains(spec.label), "missing field {}", spec.label);
    }
    assert!(out.contains("TARGET"));
    assert!(out.contains("START"), "the start key must be discoverable");
}

#[test]
fn the_confirmation_modal_says_what_it_will_do() {
    let mut a = app();
    a.section = Section::Benchmarks;
    let index = metrale_bench::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("one benchmark runs shell");
    a.bench.select(index);
    a.bench.view = View::Params;
    a.bench.confirm_open = true;
    let out = render(&a, 160, 48);
    assert!(out.contains("shell"), "the consent gate must name the risk");
    assert!(out.contains("sandbox"));
}

#[test]
fn the_glow_ring_is_titled_only_while_a_benchmark_runs() {
    let mut a = app();
    a.section = Section::Stats;
    assert!(
        !render(&a, 160, 48).contains("⏱"),
        "an idle ring carries no title"
    );
    a.bench.glow = true;
    let running = render(&a, 160, 48);
    assert!(
        running.contains("⏱"),
        "the run signal must follow you out of the Benchmarks section"
    );
    // 2026-09-26: The ring names the selected benchmark
    // (`BenchState::descriptor`).
    let name = a.bench.descriptor().expect("a benchmark is selected").name;
    assert!(running.contains(name), "ring title must carry {name:?}");
}

#[test]
fn the_history_pane_says_so_when_there_is_nothing_to_show() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench_sub = BenchSub::History;
    let out = render(&a, 160, 48);
    assert!(out.contains("No runs recorded yet"));
    assert!(out.contains(".metrale/runs"), "say where they will appear");
}

/// 2026-09-26: The Benchmarks pre-flight modal and the run log.
mod preflight {
    use super::*;
    use crate::tui::bench_preflight::Preflight;

    fn on_params() -> App {
        let mut app = app();
        app.section = Section::Benchmarks;
        app.bench_sub = BenchSub::Suite;
        app.bench.view = View::Params;
        app
    }

    #[test]
    fn the_checking_modal_shows_a_spinner_and_the_target() {
        let mut app = on_params();
        app.bench.preflight = Some(Preflight::pending());
        let out = render(&app, 200, 50);
        assert!(out.contains("CHECKING THE ENDPOINT"), "{out}");
        assert!(out.contains("known-answer"), "says what it is doing: {out}");
    }

    /// 2026-09-26: A long concern wraps; its tail stays on screen.
    #[test]
    fn a_concern_is_wrapped_not_truncated() {
        let mut app = on_params();
        let long = "http://127.0.0.1:8888 is serving \"nvidia/Qwen3.6-27B-NVFP4\", which did not \
                    answer as expected (recall answered nothing). This benchmark may be aimed at \
                    a different model, or the checkpoint may be a base (non-instruct) one — the \
                    run is still valid, but read the numbers with that in mind.";
        app.bench.preflight = Some(Preflight::with_concern(long.to_string()));
        let out = render(&app, 200, 50);
        assert!(out.contains("BEFORE YOU START"), "{out}");
        assert!(out.contains("run it anyway"), "offers to proceed: {out}");
        assert!(out.contains("back to the form"), "offers to go back: {out}");
        assert!(
            out.contains("with that in mind"),
            "the end of the reason was lost:\n{out}"
        );
    }

    /// 2026-09-26: A long log line wraps inside its panel.
    #[test]
    fn run_log_lines_wrap_inside_the_panel() {
        let mut app = on_params();
        app.bench.view = View::Run;
        let tail = "and this tail must still be on screen";
        app.bench.log.push_back(metrale_bench::LogLine {
            level: metrale_bench::LogLevel::Warn,
            text: format!(
                "http://127.0.0.1:8888 is serving a model that did not answer as expected, \
                 which usually means the benchmark is aimed somewhere else — {tail}"
            ),
        });
        let out = render(&app, 120, 40);
        assert!(
            out.contains(tail),
            "the line was truncated, not wrapped:\n{out}"
        );
    }
}

#[test]
fn the_benchmark_detail_pane_says_when_the_measurement_last_changed() {
    // 2026-09-26: The detail pane shows the descriptor's `updated` date.
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::List;
    let out = render(&a, 200, 50);
    assert!(out.contains("Updated"), "the row is drawn:\n{out}");
    let d = a
        .bench
        .descriptor()
        .expect("a benchmark is selected")
        .updated;
    assert!(out.contains(d), "and carries the date {d}:\n{out}");
}
