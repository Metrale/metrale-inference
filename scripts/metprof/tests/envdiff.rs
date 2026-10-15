// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: environment diff: both file forms, malformed entries, exactly
//! one lever, and the `envdiff` command.

mod common;

use common::{Mem, assert_err};
use metprof::core::envdiff::{Env, EnvChange, env_diff, one_lever, parse_env};
use regex::Regex;

fn env(pairs: &[(&str, &str)]) -> Env {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn change(key: &str, a: Option<&str>, b: Option<&str>) -> EnvChange {
    EnvChange {
        key: key.to_string(),
        a: a.map(str::to_string),
        b: b.map(str::to_string),
    }
}

#[test]
fn proc_environ_and_text_forms_parse_the_same() {
    let expect = env(&[("A", "1"), ("B", "x=y"), ("EMPTY", "")]);
    assert_eq!(parse_env(b"A=1\0B=x=y\0EMPTY=\0", "p").unwrap(), expect);
    assert_eq!(
        parse_env(b"# arm B\n\nA=1\nB=x=y\nEMPTY=\n", "t").unwrap(),
        expect
    );
}

#[test]
fn malformed_entries_are_errors() {
    for raw in [
        &b"export A=1\n"[..],
        b"JUSTAKEY\n",
        b"A=1\nA=2\n",
        b"1A=2\n",
    ] {
        assert!(
            parse_env(raw, "f").is_err(),
            "{:?}",
            String::from_utf8_lossy(raw)
        );
    }
    assert_eq!(parse_env(b"A=1\nA=1\n", "f").unwrap(), env(&[("A", "1")]));
}

#[test]
fn changed_added_removed_and_ignored() {
    let a = env(&[("L", "0"), ("GONE", "1"), ("PID", "10"), ("SAME", "s")]);
    let b = env(&[("L", "1"), ("NEW", ""), ("PID", "11"), ("SAME", "s")]);
    let d = env_diff(&a, &b, &[Regex::new("^PID$").unwrap()]);
    assert_eq!(
        d,
        vec![
            change("GONE", Some("1"), None),
            change("L", Some("0"), Some("1")),
            change("NEW", None, Some(""))
        ]
    );
}

#[test]
fn exactly_one_lever() {
    assert_eq!(
        one_lever(&[change("L", Some("0"), Some("1"))]).unwrap().key,
        "L"
    );
    assert_err(one_lever(&[]), "found 0 differing keys");
    assert_err(
        one_lever(&[
            change("X", Some("0"), Some("1")),
            change("Y", None, Some("1")),
        ]),
        "found 2 differing keys .*X.*Y",
    );
}

#[test]
fn proc_environ_against_a_text_file() {
    let m = Mem::new()
        .bytes("environ", b"LEVER=0\0HOSTPID=5\0")
        .file("b.env", "LEVER=1\nHOSTPID=6\n");
    let r = m.json(&["envdiff", "environ", "b.env", "--ignore", "PID$"]);
    assert_eq!(r["lever"]["key"], "LEVER");
    let err = m.refused(&["envdiff", "environ", "missing.env"]);
    assert!(err.contains("cannot read"), "{err}");
}
