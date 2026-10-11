// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: SCHEDULES.toml renders byte-stably whatever the input order, reads back to the
//! same value and the same bytes, and refuses every schema break.

use super::*;

fn entry(n: u64, rows: [u64; 2], kernel: &str, numerics: Numerics) -> Schedule {
    let new = numerics == Numerics::New;
    Schedule {
        op: "linear".into(),
        weight: "nvfp4/g16".into(),
        activation: "bf16".into(),
        k: 5120,
        n,
        rows,
        kernel: kernel.into(),
        family: "w4a16_tc".into(),
        default: if new {
            String::new()
        } else {
            "w4a16_gemv::w4a16_gemv_sw".into()
        },
        numerics,
        enabled: Enabled::of(numerics),
        median_us: 12.3,
        default_us: if new { 0.0 } else { 15.0 },
        floor_us: 1e-7,
        measured: "dgx1 2026-10-11T03:00:00Z".into(),
    }
}

fn file() -> Schedules {
    Schedules {
        schema: SCHEMA,
        hardware: "gb10".into(),
        generated_by: "met envelope schedules --records \"a b.jsonl\" \\ x".into(),
        sources: [
            (
                "w4a16_tc".to_string(),
                Source {
                    files: vec!["kernels/gb10/common/w4a16_gemv_tc.cu".into()],
                    sha256: "0123456789abcdef".repeat(4),
                },
            ),
            (
                "odd.family".to_string(),
                Source {
                    files: vec!["kernels/a.cu".into(), "kernels/b.cuh".into()],
                    sha256: "f".repeat(64),
                },
            ),
        ]
        .into(),
        // Deliberately out of order.
        schedule: vec![
            entry(
                17408,
                [9, 16],
                "w4a16_gemv_tc::w4a16_gemv_tc8",
                Numerics::Differs,
            ),
            entry(4096, [1, 1], "w4a16_gemv::w4a16_gemv_sw", Numerics::Same),
            entry(
                17408,
                [5, 8],
                "w4a16_gemv_tc::w4a16_gemv_tc8",
                Numerics::BitIdentical,
            ),
            entry(
                17408,
                [200, 256],
                "w4a16_gemv_tc::w4a16_gemv_tc8",
                Numerics::New,
            ),
        ],
    }
}

#[test]
fn render_parse_round_trips_byte_identically_in_sorted_order() {
    let s = file();
    let text = render(&s);
    let back = parse(&text).unwrap();
    assert_eq!(render(&back), text);
    let mut sorted = s.clone();
    sorted.schedule.sort_by_key(|e| (e.n, e.rows[0]));
    assert_eq!(back, sorted, "parse keeps file order, which render sorted");
    let mut shuffled = s.clone();
    shuffled.schedule.reverse();
    assert_eq!(
        render(&shuffled),
        text,
        "input order does not reach the bytes"
    );
    assert!(text.contains("[sources.\"odd.family\"]\n"));
    assert!(text.contains("default_us = 15.0\n") && text.contains("floor_us = 1e-7\n"));
}

fn refused(edit: impl Fn(&mut Schedules)) -> SchedulesError {
    let mut s = file();
    edit(&mut s);
    parse(&render(&s)).unwrap_err()
}

#[test]
fn schema_breaks_are_refused() {
    let text = render(&file());
    assert!(matches!(
        parse(&text.replace("schema = 1", "schema = 2")),
        Err(SchedulesError::Parse(_))
    ));
    assert!(matches!(
        parse(&text.replace("measured = ", "note = \"x\"\nmeasured = ")),
        Err(SchedulesError::Parse(_))
    ));
    assert!(matches!(
        parse(&text.replacen("floor_us = 1e-7\n", "", 1)),
        Err(SchedulesError::Parse(_))
    ));
    let cases: Vec<(&str, Box<dyn Fn(&mut Schedules)>)> = vec![
        (
            "differs enabled by default",
            Box::new(|s| s.schedule[0].enabled = Enabled::Default),
        ),
        (
            "overlapping rows",
            Box::new(|s| s.schedule[2].rows = [5, 9]),
        ),
        ("lo above hi", Box::new(|s| s.schedule[2].rows = [8, 5])),
        (
            "unknown family",
            Box::new(|s| s.schedule[0].family = "nope".into()),
        ),
        (
            "same with another kernel",
            Box::new(|s| s.schedule[1].kernel = "x::y".into()),
        ),
        (
            "bit_identical is the default",
            Box::new(|s| s.schedule[2].kernel = s.schedule[2].default.clone()),
        ),
        (
            "new with a default",
            Box::new(|s| s.schedule[3].default = "x::y".into()),
        ),
        (
            "served without default_us",
            Box::new(|s| s.schedule[0].default_us = 0.0),
        ),
        (
            "bad digest",
            Box::new(|s| s.sources.get_mut("w4a16_tc").unwrap().sha256 = "ABC".into()),
        ),
        (
            "unsorted files",
            Box::new(|s| s.sources.get_mut("odd.family").unwrap().files.reverse()),
        ),
        (
            "not a kernel source",
            Box::new(|s| s.sources.get_mut("w4a16_tc").unwrap().files = vec!["crates/x.rs".into()]),
        ),
    ];
    for (why, edit) in cases {
        assert!(
            matches!(refused(edit), SchedulesError::Invalid { .. }),
            "{why} must be refused"
        );
    }
}
