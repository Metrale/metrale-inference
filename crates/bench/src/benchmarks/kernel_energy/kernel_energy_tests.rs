// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The descriptor refuses every unstated parameter and passes the stated ones to the
//! harness under its flag names.

use super::*;
use crate::params::{ParamValue, ParamValues};

fn values(set: &[(&str, &str)]) -> ParamValues {
    let b = KernelEnergy::default();
    let mut v = ParamValues::defaults(&b.parameters());
    for (k, x) in set {
        v.set(*k, ParamValue::Text((*x).to_string()));
    }
    v
}

const STATED: [(&str, &str); 7] = [
    ("trace", "/tmp/t.csv"),
    ("harness", "/bin/true"),
    ("kernel", "w4a16_gemv_pipe"),
    ("shape", "kda q"),
    ("platform", "gb10"),
    ("fingerprint", "fp"),
    ("energy_source", "src"),
];

#[test]
fn every_unstated_parameter_is_refused() {
    for (skip, (name, _)) in STATED.iter().enumerate() {
        let set: Vec<(&str, &str)> = STATED
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != skip)
            .map(|(_, p)| *p)
            .collect();
        let mut b = KernelEnergy::default();
        let e = b.configure(&values(&set)).unwrap_err();
        assert!(e.to_string().contains("no default"), "{name}: {e}");
    }
}

#[test]
fn stated_parameters_reach_the_harness_under_its_flags() {
    let mut b = KernelEnergy::default();
    b.configure(&values(&STATED)).unwrap();
    let flags: Vec<&str> = b.args.iter().map(|(k, _)| k.as_str()).collect();
    for f in [
        "kernel",
        "shape",
        "idle-s",
        "load-s",
        "sample-ms",
        "bootstrap-reps",
        "ci-level",
        "energy-source",
        "out",
    ] {
        assert!(flags.contains(&f), "{f} missing from {flags:?}");
    }
}

#[test]
fn recompute_of_a_missing_trace_fails() {
    assert!(recompute("/nonexistent/trace.csv").is_err());
}
