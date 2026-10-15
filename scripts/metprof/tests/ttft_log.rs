// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the serve-log parser, the cell selection and statistics, the
//! run comparison and the ttft config, over synthetic logs in the engine's
//! exact line formats.

mod common;

use common::ttft::*;
use common::{assert_close, assert_err};
use metprof::core::ttft::cells::{Class, Mode, cell_rows, compare, select, summary};
use metprof::core::ttft::log::{parse_serve_log, strip_ansi};
use serde_json::json;

#[test]
fn a_solo_request_carries_its_hit_first_token_and_done_ttft() {
    let text = join_lines(&[req(100, 8, true, 9, 91.26), req(200, 8, false, 7, 451.04)]);
    let log = parse_serve_log(&text).unwrap();
    assert_eq!(log.requests.len(), 2);
    let (w, c) = (&log.requests[0], &log.requests[1]);
    assert_eq!((w.line, w.prompt_tokens, w.max_tokens), (1, 100, 8));
    let hit = w.hit.as_ref().unwrap();
    assert_eq!(
        (hit.skipped, hit.blocks, hit.snapshot, hit.replay),
        (99, 4, 1, None)
    );
    assert_eq!(w.first_token, Some(9));
    assert_close(w.ttft_ms, 91.3);
    assert_eq!((w.done_tokens, w.reason.as_str()), (8, "length"));
    assert_eq!(w.time.as_deref(), Some("2026-01-01T00:00:00.000000Z"));
    assert_eq!((c.line, c.hit.is_none(), c.first_token), (5, true, Some(7)));
    assert_close(c.ttft_ms, 451.0);
}

#[test]
fn the_prefix_lookup_hit_format_carries_its_replay() {
    let text = join_lines(&[vec![
        start(100, 8),
        hit_lookup(96, 100),
        first(3),
        done(8, 80.0),
    ]]);
    let hit = parse_serve_log(&text).unwrap().requests[0]
        .hit
        .clone()
        .unwrap();
    assert_eq!((hit.skipped, hit.replay), (96, Some((4, 100))));
}

#[test]
fn a_cache_hit_line_in_an_unknown_format_is_an_error_not_a_cold_request() {
    let bad = "x: Marconi SSM cache hit: 96 tokens skipped, snapshot 2".to_string();
    let text = join_lines(&[vec![start(100, 8), bad, first(3), done(8, 80.0)]]);
    assert_err(
        parse_serve_log(&text),
        "^line 2: a cache-hit line in neither known format",
    );
}

#[test]
fn ansi_colour_is_stripped_before_matching() {
    let plain = join_lines(&[req(100, 8, true, 9, 91.2)]);
    let coloured: String = plain
        .lines()
        .map(|l| {
            let (ts, rest) = l.split_once("  ").unwrap();
            format!("\x1b[2m{ts}\x1b[0m \x1b[32m INFO\x1b[0m \x1b[2m{rest}\x1b[0m\n")
        })
        .collect();
    assert_eq!(
        parse_serve_log(&coloured).unwrap().requests,
        parse_serve_log(&plain).unwrap().requests
    );
    assert_eq!(strip_ansi("a\x1b[1;32mb\x1b[0m"), "ab");
    assert_eq!(strip_ansi("a\x1b]x"), "a\x1b]x", "a non-SGR escape is kept");
    assert_eq!(
        strip_ansi("a\x1b[12"),
        "a\x1b[12",
        "an unterminated SGR is kept"
    );
}

#[test]
fn a_done_line_with_no_request_open_is_an_error() {
    let text = join_lines(&[vec![done(8, 10.0)]]);
    assert_err(
        parse_serve_log(&text),
        "^line 1: a Done line with no request open",
    );
}

#[test]
fn a_done_line_in_an_unknown_format_is_an_error() {
    let bad = "Done: 8 tokens (length) 3.0 tok/s TTFT=1.0ms".to_string();
    let text = join_lines(&[vec![start(100, 8), bad]]);
    assert_err(
        parse_serve_log(&text),
        "^line 2: a Done line in an unknown format",
    );
}

#[test]
fn hit_and_first_token_lines_outside_or_twice_in_a_request_are_errors() {
    assert_err(
        parse_serve_log(&join_lines(&[vec![first(3)]])),
        "first-token line outside any request",
    );
    assert_err(
        parse_serve_log(&join_lines(&[vec![hit(3)]])),
        "cache-hit line outside any request",
    );
    let twice = join_lines(&[vec![start(100, 8), first(3), first(4), done(8, 1.0)]]);
    assert_err(
        parse_serve_log(&twice),
        "^line 3: a second first-token line in the request of line 1",
    );
    let twice = join_lines(&[vec![start(100, 8), hit(3), hit(3), done(8, 1.0)]]);
    assert_err(parse_serve_log(&twice), "^line 3: a second cache-hit line");
}

#[test]
fn overlapping_requests_are_concurrent_and_attribution_resumes_after() {
    let text = join_lines(&[
        vec![
            start(100, 8),
            start(200, 128),
            first(5),
            done(8, 50.0),
            hit(3),
            done(128, 60.0),
        ],
        req(200, 8, false, 7, 400.0),
        vec![start(100, 8)],
    ]);
    let log = parse_serve_log(&text).unwrap();
    let lines: Vec<(usize, u64)> = log
        .concurrent
        .iter()
        .map(|u| (u.line, u.prompt_tokens))
        .collect();
    assert_eq!(lines, vec![(1, 100), (2, 200)]);
    assert_eq!(log.requests.len(), 1);
    assert_eq!(
        (log.requests[0].line, log.requests[0].first_token),
        (7, Some(7))
    );
    assert_eq!(
        log.unfinished.iter().map(|u| u.line).collect::<Vec<_>>(),
        vec![10]
    );
}

#[test]
fn selection_measures_the_cells_at_the_max_tokens_and_counts_the_rest() {
    let cfg = parse_config(CONFIG).unwrap();
    let text = join_lines(&[
        req(100, 32, false, 7, 900.0),
        req(100, 8, false, 7, 450.0),
        req(100, 8, true, 9, 90.0),
        req(150, 8, false, 9, 500.0),
        req(200, 8, false, 9, 700.0),
    ]);
    let log = parse_serve_log(&text).unwrap();
    let sel = select(&log, &cfg).unwrap();
    assert_eq!((sel.other, sel.unassigned), (1, 1));
    let got: Vec<(&str, Mode, Class)> = sel
        .measured
        .iter()
        .map(|m| (m.cell.as_str(), m.mode, m.class))
        .collect();
    assert_eq!(
        got,
        vec![
            ("a", Mode::Cold, Class::Step),
            ("a", Mode::Warm, Class::Other),
            ("b", Mode::Cold, Class::Other)
        ]
    );
}

#[test]
fn a_measured_request_that_cannot_be_attributed_is_an_error() {
    let cfg = parse_config(CONFIG).unwrap();
    let conc = join_lines(&[vec![
        start(100, 8),
        start(300, 8),
        done(8, 1.0),
        done(8, 1.0),
    ]]);
    let log = parse_serve_log(&conc).unwrap();
    assert_err(
        select(&log, &cfg),
        "line 1 .100 prompt tokens, cell a. is concurrent",
    );
    let other = join_lines(&[vec![
        start(100, 128),
        start(300, 8),
        done(8, 1.0),
        done(8, 1.0),
    ]]);
    assert!(
        select(&parse_serve_log(&other).unwrap(), &cfg).is_ok(),
        "no cell was concurrent"
    );
    let open = join_lines(&[req(100, 8, false, 7, 1.0), vec![start(200, 8)]]);
    assert_err(
        select(&parse_serve_log(&open).unwrap(), &cfg),
        "cell b. is unfinished",
    );
    let nofirst = join_lines(&[vec![start(200, 8), done(8, 1.0)]]);
    assert_err(
        select(&parse_serve_log(&nofirst).unwrap(), &cfg),
        "line 1 .cell b. has no `Prefill first token`",
    );
}

#[test]
fn summary_takes_the_true_median_and_p90_only_from_ten_values() {
    let s = summary(&[4.0, 1.0, 3.0, 2.0]);
    assert_eq!(
        s,
        json!({"n": 4, "median": 2.5, "min": 1.0, "max": 4.0, "p90_interpolated": null})
    );
    let nine: Vec<f64> = (1..=9).map(f64::from).collect();
    assert_eq!(summary(&nine)["p90_interpolated"], json!(null));
    let ten: Vec<f64> = (1..=10).map(f64::from).collect();
    assert_close(summary(&ten)["p90_interpolated"].as_f64().unwrap(), 9.1);
    assert_eq!(summary(&[])["median"], json!(null));
}

#[test]
fn cell_rows_split_each_mode_by_first_token_class() {
    let cfg = parse_config(CONFIG).unwrap();
    let text = join_lines(&[
        req(100, 8, false, 7, 440.0),
        req(100, 8, false, 9, 450.0),
        req(100, 8, false, 9, 470.0),
        req(100, 8, true, 9, 90.0),
    ]);
    let log = parse_serve_log(&text).unwrap();
    let rows = cell_rows(&select(&log, &cfg).unwrap(), &cfg, |m| Some(m.req.ttft_ms));
    assert_eq!(rows.len(), 2, "cell b has no requests and gets no rows");
    assert_eq!(
        (&rows[0]["cell"], &rows[0]["mode"], &rows[0]["n"]),
        (&json!("a"), &json!("cold"), &json!(3))
    );
    assert_eq!(rows[0]["median"], json!(450.0));
    assert_eq!(rows[0]["step"], json!({"n": 1, "median": 440.0}));
    assert_eq!(rows[0]["other"], json!({"n": 2, "median": 460.0}));
    assert_eq!(
        (&rows[1]["mode"], &rows[1]["n"]),
        (&json!("warm"), &json!(1))
    );
}

fn rows(cell: &str, values: &[f64]) -> serde_json::Value {
    let mut r = summary(values);
    r["cell"] = json!(cell);
    r["mode"] = json!("cold");
    r
}

#[test]
fn compare_reports_delta_and_a_range_separation_verdict() {
    let a = vec![
        rows("w", &[10.0, 11.0, 12.0]),
        rows("l", &[10.0, 11.0, 12.0]),
        rows("o", &[10.0, 11.0, 12.0]),
        rows("f", &[10.0, 11.0]),
        rows("only_a", &[1.0, 1.0, 1.0]),
    ];
    let b = vec![
        rows("w", &[7.0, 8.0, 9.5]),
        rows("l", &[12.5, 13.0, 14.0]),
        rows("o", &[9.0, 10.0, 10.5]),
        rows("f", &[1.0, 2.0, 3.0]),
    ];
    let out = compare(&a, &b, 3).unwrap();
    let got: Vec<(&str, &str)> = out
        .iter()
        .map(|r| (r["cell"].as_str().unwrap(), r["verdict"].as_str().unwrap()))
        .collect();
    assert_eq!(
        got,
        vec![
            ("w", "win"),
            ("l", "loss"),
            ("o", "overlap"),
            ("f", "too-few")
        ]
    );
    assert_eq!(out[0]["delta"], json!(-3.0));
    assert_eq!(out[1]["delta"], json!(2.0));
    assert_eq!(
        (&out[0]["range_a"], &out[0]["range_b"]),
        (&json!([10.0, 12.0]), &json!([7.0, 9.5]))
    );
    let touch = compare(&[rows("t", &[10.0, 11.0])], &[rows("t", &[9.0, 10.0])], 2).unwrap();
    assert_eq!(
        touch[0]["verdict"],
        json!("overlap"),
        "a shared endpoint is not separation"
    );
    assert_err(compare(&a, &[rows("zz", &[1.0])], 1), "share no");
}

#[test]
fn the_config_requires_every_key_and_refuses_unknown_and_ambiguous_ones() {
    let cfg = parse_config(&config_with_client()).unwrap();
    assert_eq!(
        (cfg.max_tokens, cfg.step_token_id, cfg.cells.len()),
        (8, 7, 2)
    );
    let c = cfg.client().unwrap();
    assert_eq!(
        (c.salt_digits, c.warmup_requests, c.warmup_max_tokens),
        (4, 1, 32)
    );
    assert_err(
        parse_config(CONFIG).unwrap().client(),
        "no \\[client\\] table",
    );
    assert_err(
        cfg.cell("zz"),
        "no cell 'zz' in the config \\(cells: a, b\\)",
    );
    let cases = [
        ("max_tokens = 8\n", "", "\\[select\\]: missing"),
        (
            "step_token_id = 7\n",
            "step_token_id = 7\nx = 1\n",
            "\\[class\\]: unknown",
        ),
        (
            "prompt_tokens = 200\n",
            "prompt_tokens = 100\n",
            "cells 'a' and 'b' both claim prompt_tokens 100",
        ),
        (
            "name = \"b\"\n",
            "name = \"a\"\n",
            "name 'a' is declared twice",
        ),
        ("schema = 1\n", "schema = 2\n", "schema must be 1"),
    ];
    for (from, to, err) in cases {
        assert_err(parse_config(&CONFIG.replacen(from, to, 1)), err);
    }
    let client_cases = [
        (
            "salt_digits = 4\n",
            "salt_digits = 0\n",
            "salt_digits: must be 1..=19",
        ),
        (
            "salt_digits = 4\n",
            "salt_digits = 20\n",
            "salt_digits: must be 1..=19",
        ),
        (
            "timeout_s = 30\n",
            "timeout_s = 0\n",
            "timeout_s: must be positive",
        ),
        (
            "warmup_max_tokens = 32\n",
            "warmup_max_tokens = 8\n",
            "must differ from \\[select\\].max_tokens",
        ),
        ("seed = 42\n", "", "\\[client\\]: missing"),
        (
            "model = \"m\"\n",
            "model = \"\"\n",
            "model: must be a non-empty string",
        ),
    ];
    for (from, to, err) in client_cases {
        assert_err(
            parse_config(&config_with_client().replacen(from, to, 1)),
            err,
        );
    }
}

#[test]
fn the_shipped_example_config_parses_with_its_client_table() {
    let cfg = parse_config(include_str!("../config/ttft.example.toml")).unwrap();
    let names: Vec<&str> = cfg.cells.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, ["c256", "c1k"]);
    let client = cfg.client().unwrap();
    assert_ne!(client.warmup_max_tokens, cfg.max_tokens);
}
