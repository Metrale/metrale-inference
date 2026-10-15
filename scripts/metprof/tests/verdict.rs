// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the A/B verdict rules, their precedence, the band edge, and
//! the record contract (every field required and typed).

mod common;

use common::{Mem, assert_close, assert_err, f};
use metprof::core::verdict::decide;
use toml::{Table, Value};

fn base() -> Table {
    let mut t = Table::new();
    for (k, v) in [
        ("metric", Value::from("ttft_ms")),
        ("lower_is_better", Value::from(true)),
        ("control", Value::from(100.0)),
        ("treatment", Value::from(90.0)),
        ("band", Value::from(5.0)),
        ("armed", Value::from(true)),
        ("engaged", Value::from(true)),
        ("lever_parity", Value::from("bit-identical")),
        ("bits", Value::from("identical")),
    ] {
        t.insert(k.to_string(), v);
    }
    t
}

fn with(over: &[(&str, Value)]) -> Table {
    let mut t = base();
    for (k, v) in over {
        t.insert(k.to_string(), v.clone());
    }
    t
}

fn v(over: &[(&str, Value)]) -> String {
    decide(&with(over)).unwrap()["verdict"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn each_rule() {
    let t = |x: f64| ("treatment", Value::from(x));
    let no = |k: &'static str| (k, Value::from(false));
    let s = |k: &'static str, x: &str| (k, Value::from(x));
    let cases: Vec<(Vec<(&str, Value)>, &str)> = vec![
        (vec![], "win"),
        (vec![no("engaged")], "unproven"),
        (vec![t(110.0)], "loss"),
        (vec![t(110.0), no("engaged")], "loss"),
        (vec![t(97.0)], "neutral"),
        (vec![t(97.0), no("engaged")], "unproven"),
        (vec![no("armed")], "invalid"),
        (vec![s("bits", "differ")], "parity-broken"),
        (
            vec![s("lever_parity", "bit-changing"), s("bits", "identical")],
            "unproven",
        ),
        (
            vec![s("lever_parity", "bit-changing"), s("bits", "differ")],
            "win",
        ),
    ];
    for (over, want) in cases {
        assert_eq!(v(&over), want, "{over:?}");
    }
}

#[test]
fn precedence() {
    assert_eq!(
        v(&[("armed", false.into()), ("bits", "differ".into())]),
        "invalid"
    );
    assert_eq!(
        v(&[("bits", "differ".into()), ("treatment", 50.0.into())]),
        "parity-broken"
    );
}

#[test]
fn the_band_edge_is_neutral_and_direction_follows_the_metric() {
    assert_eq!(v(&[("treatment", 95.0.into())]), "neutral");
    assert_eq!(v(&[("treatment", 94.999.into())]), "win");
    assert_eq!(v(&[("lower_is_better", false.into())]), "loss");
    assert_eq!(
        v(&[
            ("lower_is_better", false.into()),
            ("treatment", 110.0.into())
        ]),
        "win"
    );
}

#[test]
fn reported_numbers() {
    let r = decide(&with(&[
        ("control", 12314.into()),
        ("treatment", 12047.into()),
    ]))
    .unwrap();
    assert_eq!((f(&r["delta"]), f(&r["improvement"])), (-267.0, 267.0));
    assert_close(f(&r["ratio"]), 12047.0 / 12314.0);
}

#[test]
fn every_field_is_required_and_typed() {
    for key in base().keys() {
        let mut rec = base();
        rec.remove(key);
        assert_err(decide(&rec), "missing fields");
    }
    let bad: Vec<(&str, Value)> = vec![
        ("extra", 1.into()),
        ("band", 0.into()),
        ("band", (-1.0).into()),
        ("control", true.into()),
        ("armed", "yes".into()),
        ("lever_parity", "identical".into()),
        ("bits", "same".into()),
        ("treatment", f64::NAN.into()),
    ];
    for over in bad {
        assert!(
            decide(&with(std::slice::from_ref(&over))).is_err(),
            "{over:?}"
        );
    }
    assert_eq!(v(&[("notes", "band = control spread".into())]), "win");
}

#[test]
fn record_file() {
    let rec = [
        "metric = \"e2e_ms\"",
        "lower_is_better = true",
        "control = 12314",
        "treatment = 12016",
        "band = 159",
        "armed = true",
        "engaged = true",
        "lever_parity = \"bit-identical\"",
        "bits = \"differ\"",
    ]
    .join("\n");
    let m = Mem::new()
        .file("r.toml", &rec)
        .file("bad.toml", "metric = 1\n");
    assert_eq!(
        m.json(&["verdict", "--record", "r.toml"])["verdict"],
        "parity-broken"
    );
    let err = m.refused(&["verdict", "--record", "bad.toml"]);
    assert!(err.contains("missing fields"), "{err}");
}
