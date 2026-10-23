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

/// 2026-10-22: HPAR on the first real row: the committed trace recomputes to exactly the energy
/// lines committed in measurements.toml (hash included), so the row is evidence, not a copy.
#[test]
fn the_committed_trace_recomputes_the_committed_row_bitwise() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let trace = format!(
        "{root}/docs/kernel-perf/energy/\
         glm-5.3-flash__w4a16_gemv_pipe_m1__kda-in-proj__r1__spark-28c2__2026-10-22.trace.csv"
    );
    let (sha, lines) = recompute(&trace).unwrap();
    let toml =
        std::fs::read_to_string(format!("{root}/docs/kernel-perf/measurements.toml")).unwrap();
    let block = toml
        .split("\n[[m]]\n")
        .find(|b| b.contains(&sha))
        .expect("a row names the trace's hash");
    for l in &lines {
        assert!(
            block.lines().any(|b| b == l),
            "the committed row lacks `{l}`"
        );
    }
    assert!(
        lines
            .iter()
            .any(|l| l == &format!("trace_sha256 = \"{sha}\""))
    );
}
