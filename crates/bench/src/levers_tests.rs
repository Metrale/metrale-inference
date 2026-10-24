// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Each harness lever resolved through [`BenchLevers`] against its frozen old reader,
//! over the same values; every difference must be listed in the lever's `changes` table.

use std::sync::Arc;

use metrale_io_host_mem::MapEnv;

use super::{BenchLevers, MODEL_CARD_SAMPLING, hw_switch};

/// 2026-10-17: The levers of an empty environment, for the crate's tests that build a handle or
/// an executor and read no lever.
pub(crate) fn unset() -> Arc<BenchLevers> {
    Arc::new(
        BenchLevers::resolve(&MapEnv::new(std::iter::empty::<(&str, &str)>()))
            .expect("an empty environment resolves"),
    )
}

const VALUES: [Option<&str>; 15] = [
    None,
    Some("1"),
    Some("true"),
    Some("TRUE"),
    Some("yes"),
    Some("on"),
    Some("0"),
    Some("false"),
    Some(""),
    Some(" 1 "),
    Some("garbage"),
    Some("-1"),
    Some("model-card"),
    Some(" model-card"),
    Some("/opt/py/bin/python3"),
];

/// 2026-10-17: A lever value as the harness used it; `Err` stops the process.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Out {
    B(bool),
    S(Option<String>),
    Err,
}

fn resolve(var: &str, value: Option<&str>) -> Result<BenchLevers, anyhow::Error> {
    BenchLevers::resolve(&MapEnv::new(value.map(|v| (var, v))))
}

/// 2026-10-17: Resolve `var` over [`VALUES`]; the outcomes that differ from `old` must be
/// exactly `changes`.
fn assert_old_reader_holds(
    var: &str,
    old: impl Fn(Option<&str>) -> Out,
    new: impl Fn(&BenchLevers) -> Out,
    changes: &[(&str, Out)],
) {
    let actual: Vec<(&str, Out)> = VALUES
        .iter()
        .filter_map(|&v| {
            let got = resolve(var, v).map_or(Out::Err, |l| new(&l));
            (got != old(v)).then(|| (v.unwrap_or("<unset>"), got))
        })
        .collect();
    let want: Vec<(&str, Out)> = changes.to_vec();
    assert_eq!(
        actual, want,
        "{var}: the changed values differ from the changes table"
    );
}

fn verbatim(v: Option<&str>) -> Out {
    Out::S(v.map(str::to_string))
}

#[test]
fn the_text_and_path_levers_keep_their_value_verbatim() {
    type Get = fn(&BenchLevers) -> Out;
    let rows: [(&str, Get); 5] = [
        ("METRALE_PYTHON", |l| Out::S(l.python.clone())),
        ("METRALE_GATE_SHA", |l| Out::S(l.gate_sha.clone())),
        ("METRALE_HOME", |l| Out::S(l.home.clone())),
        ("METRALE_WARM_TARGET_DIR", |l| {
            Out::S(l.warm_target_dir.clone())
        }),
        ("METRALE_WARM_TEMPLATE_DIR", |l| {
            Out::S(l.warm_template_dir.clone())
        }),
    ];
    for (var, get) in rows {
        assert_old_reader_holds(var, verbatim, get, &[]);
    }
}

#[test]
fn preserve_thinking_takes_the_switch_grammar() {
    // 2026-10-17: The old reader was `== "1"`, read once per process.
    let old = |v: Option<&str>| Out::B(v == Some("1"));
    assert_old_reader_holds(
        "METRALE_AGENTIC_PRESERVE_THINKING",
        old,
        |l| Out::B(l.preserve_thinking),
        &[
            ("true", Out::B(true)),
            ("TRUE", Out::B(true)),
            ("yes", Out::B(true)),
            ("on", Out::B(true)),
            ("", Out::Err),
            (" 1 ", Out::B(true)),
            ("garbage", Out::Err),
            ("-1", Out::Err),
            ("model-card", Out::Err),
            (" model-card", Out::Err),
            ("/opt/py/bin/python3", Out::Err),
        ],
    );
}

#[test]
fn model_card_sampling_is_the_exact_word() {
    // 2026-10-17: The old reader pinned greedy sampling unless the value was exactly the word.
    let old = |v: Option<&str>| Out::B(v == Some(MODEL_CARD_SAMPLING));
    assert_old_reader_holds(
        "METRALE_AGENTIC_SAMPLING",
        old,
        |l| Out::B(l.model_card_sampling),
        &[],
    );
}

#[test]
fn the_hardware_switches_keep_their_trimmed_word_grammar() {
    // 2026-10-17: The old `policy::flag`: `1`/`true`/`yes`, trimmed, any case.
    let old = |v: Option<&str>| {
        Out::B(v.is_some_and(|v| {
            matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes")
        }))
    };
    assert_old_reader_holds(
        "METRALE_NO_HW_PRECHECK",
        old,
        |l| Out::B(l.hw_kill_switch),
        &[],
    );
    assert_old_reader_holds("METRALE_HW_TEMP_GATE", old, |l| Out::B(l.hw_temp_gate), &[]);
    assert!(hw_switch(Some(" Yes ")) && !hw_switch(Some("on")) && !hw_switch(None));
}

#[test]
fn home_is_read_through_the_port_verbatim() {
    let l = BenchLevers::resolve(&MapEnv::new([("HOME", "/srv/u")])).expect("resolves");
    assert_eq!(l.user_home.as_deref(), Some("/srv/u"));
    let unset = BenchLevers::resolve(&MapEnv::new(std::iter::empty::<(&str, &str)>()))
        .expect("an empty environment resolves");
    assert_eq!(unset.user_home, None);
}

#[test]
fn a_malformed_serve_lever_stops_a_harness_process() {
    // 2026-10-17: The harness resolves every runtime group once, so a malformed lever it never
    // reads still stops it, as it stops a serve.
    let bad = MapEnv::new([("METRALE_NO_TUI", "maybe")]);
    assert!(BenchLevers::resolve(&bad).is_err());
}
