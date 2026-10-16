// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for BENCH.toml loading and baseline assembly (`bench.rs`), against the
//! committed tree and against temporary fixtures.
//!
//! Owner: bench gate.
//! Invariants: none beyond the types.

use super::*;

pub(super) fn repo_root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace layout")
        .to_path_buf()
}

fn tmp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("metrale-bench-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

pub(super) fn fixture(name: &str, bench_toml: &str) -> std::path::PathBuf {
    let root = tmp(name);
    let m = root.join("kernels/gb10/modelA");
    std::fs::create_dir_all(m.join("nvfp4")).unwrap();
    std::fs::write(
        root.join("kernels/gb10/HARDWARE.toml"),
        "[hardware]\nvendor = \"nvidia\"\n",
    )
    .unwrap();
    std::fs::write(m.join("MODEL.toml"), "[behavior]\n").unwrap();
    std::fs::write(m.join("BENCH.toml"), bench_toml).unwrap();
    root
}

/// 2026-09-26: Every required gate resolves a baseline from the committed tree, with a default
/// checkpoint on each hardware that is among that hardware's own models.
#[test]
fn every_required_gate_still_resolves_a_baseline() {
    let root = repo_root();
    for id in super::super::REQUIRED_GATES {
        let baseline = baseline_for(&root, id).unwrap_or_else(|e| panic!("{id}: {e}"));
        assert!(
            !baseline.hardware.is_empty(),
            "{id}: no hardware entries after the move"
        );
        for (hw, entry) in &baseline.hardware {
            assert!(
                !entry.default.is_empty(),
                "{id}/{hw}: no default checkpoint"
            );
            assert!(
                entry.models.contains_key(&entry.default),
                "{id}/{hw}: default {:?} is not among its own models",
                entry.default
            );
        }
    }
}

/// 2026-09-26: `BENCH.toml` is not a closure input, so ratcheting a bar does not invalidate the
/// record that justified it; `MODEL.toml` is one.
#[test]
fn bench_toml_is_not_a_closure_input() {
    let root = repo_root();
    let target = taxon::Target {
        hardware: "gb10".into(),
        model: "qwen3.6-27b".into(),
        quant: "nvfp4".into(),
    };
    let configs = taxon::configs(&root, &target).expect("configs");
    assert!(
        !configs.iter().any(|p| p.ends_with("BENCH.toml")),
        "BENCH.toml must not be hashed: a threshold ratchet would invalidate \
         the very record that justified it. Found: {configs:?}"
    );
    assert!(
        configs.iter().any(|p| p.ends_with("MODEL.toml")),
        "MODEL.toml IS compiled in and must stay hashed: {configs:?}"
    );
}

/// 2026-09-26: An unmeasured entry carrying thresholds is rejected at load, so an unmeasured bar
/// can never report PASS.
#[test]
fn an_unmeasured_entry_carrying_thresholds_is_rejected() {
    let root = fixture(
        "guessed",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    let err = load_all(&root).unwrap_err().to_string();
    assert_eq!(
        err,
        format!(
            "{}: bfcl-subset / org/M is unmeasured but carries thresholds. A guessed number a run can clear is worse than no number — it reports PASS for something nobody measured.",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// 2026-09-26: A measured entry with no metrics table, or an empty one, is rejected at load.
#[test]
fn a_measured_entry_without_thresholds_is_rejected() {
    for (name, metrics) in [("absent", ""), ("empty", "[benchmarks.metrics]")] {
        let root = fixture(
            name,
            &format!(
                r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "measured"
{metrics}
"#
            ),
        );
        let err = load_all(&root).unwrap_err().to_string();
        assert_eq!(
            err,
            format!(
                "{}: bfcl-subset / org/M claims to be measured but declares no metrics",
                root.join("kernels/gb10/modelA/BENCH.toml").display()
            ),
            "partition {name}"
        );
    }
}

#[test]
fn an_unknown_status_is_rejected_rather_than_treated_as_unmeasured() {
    let root = fixture(
        "bad-status",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "probably-fine"
"#,
    );
    assert_eq!(
        load_all(&root).unwrap_err().to_string(),
        format!(
            "{}: status must be \"measured\" or \"unmeasured\", got \"probably-fine\"",
            root.join("kernels/gb10/modelA/BENCH.toml").display()
        )
    );
}

/// 2026-09-26: An unmeasured entry is dropped from the baseline, which assembles empty, and
/// resolving against it fails with "no baseline for hardware".
#[test]
fn an_unmeasured_entry_produces_no_baseline_at_all() {
    let root = fixture(
        "unmeasured",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/M"
gate = "bfcl-subset"
status = "unmeasured"
"#,
    );
    let baseline = baseline_for(&root, "bfcl-subset").unwrap();
    assert_eq!(baseline.schema, 2);
    assert!(baseline.hardware.is_empty(), "{baseline:?}");
    let err = baseline.resolve("gb10", None).unwrap_err().to_string();
    assert_eq!(
        err,
        "no baseline for hardware \"gb10\"; this benchmark has entries for []"
    );
}

/// 2026-09-26: Two checkpoints claiming `default = true` on one hardware is an error: the gate
/// would have no single subject.
#[test]
fn two_checkpoints_claiming_default_is_an_error() {
    let root = fixture(
        "two-defaults",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/B"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 86.0
"#,
    );
    let err = baseline_for(&root, "bfcl-subset").unwrap_err().to_string();
    assert_eq!(
        err,
        "bfcl-subset: both org/A (in modelA) and org/B (in modelA) claim to be the default on gb10"
    );
}

/// 2026-09-26: A lone checkpoint must still set `default = true`, so adding a second one later
/// cannot silently change which one the gate scores.
#[test]
fn a_lone_checkpoint_must_still_declare_itself_default() {
    let root = fixture(
        "no-default",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: no checkpoint on gb10 sets `default = true`; one must, or the gate has no defined subject"
    );
}

#[test]
fn the_same_checkpoint_declared_twice_for_one_gate_is_an_error() {
    let root = fixture(
        "dupe",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0

[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 90.0
"#,
    );
    assert_eq!(
        baseline_for(&root, "bfcl-subset").unwrap_err().to_string(),
        "bfcl-subset: org/A is declared twice on gb10"
    );
}

/// 2026-09-26: BENCH.toml is per model, so a model with two quant dirs still yields each entry
/// once.
#[test]
fn entries_are_not_duplicated_across_a_models_quant_dirs() {
    let root = fixture(
        "multi-quant",
        r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "bfcl-subset"
default = true
status = "measured"
[benchmarks.metrics.overall_accuracy]
min = 85.0
"#,
    );
    std::fs::create_dir_all(root.join("kernels/gb10/modelA/fp8")).unwrap();
    let all = load_all(&root).unwrap();
    assert_eq!(all.len(), 1, "one entry, not one per quant dir: {all:?}");
}

fn with_target(name: &str, max: f64, target_key: &str, target: &str) -> std::path::PathBuf {
    fixture(
        name,
        &format!(
            r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "ttft-cold-gate"
default = true
status = "measured"
[benchmarks.metrics.median_ms]
max = {max:.1}
[benchmarks.targets]
{target_key} = {target}
"#
        ),
    )
}

fn verdict_at(root: &Path, value: f64) -> bool {
    let baseline = baseline_for(root, "ttft-cold-gate").unwrap();
    let bound = &baseline.hardware["gb10"].models["org/A"].metrics["median_ms"];
    matches!(
        super::super::scoring::compare("median_ms", value, bound),
        super::super::scoring::Comparison::Pass
    )
}

/// 2026-10-15: A target is a reference, never a bound, in either direction: a run between the
/// target and a looser ceiling passes, and a target looser than the ceiling does not let a run
/// above the ceiling through.
#[test]
fn a_target_never_moves_the_verdict() {
    let tight_target = with_target("target-tight", 500.0, "median_ms", "100.0");
    assert!(
        verdict_at(&tight_target, 300.0),
        "300 is under the 500 ceiling"
    );
    let loose_target = with_target("target-loose", 100.0, "median_ms", "500.0");
    assert!(
        !verdict_at(&loose_target, 300.0),
        "300 is over the 100 ceiling"
    );
    let entry = &load_all(&loose_target).unwrap()[0].1;
    assert_eq!(entry.targets.get("median_ms"), Some(&500.0));
}

/// 2026-10-15: A target on a metric the entry does not bound, or a non-finite one, is refused
/// at load; the same target on a declared metric loads.
#[test]
fn a_target_must_name_a_declared_metric_and_be_finite() {
    let undeclared = with_target("target-undeclared", 500.0, "p90_ms", "100.0");
    let err = format!("{:#}", load_all(&undeclared).unwrap_err());
    assert!(
        err.contains("[benchmarks.targets] names p90_ms, which is not a declared metric"),
        "{err}"
    );
    let nan = with_target("target-nan", 500.0, "median_ms", "nan");
    let err = format!("{:#}", load_all(&nan).unwrap_err());
    assert!(err.contains("target median_ms must be finite"), "{err}");
    let declared = with_target("target-declared", 500.0, "median_ms", "100.0");
    assert_eq!(load_all(&declared).unwrap().len(), 1);
}

/// 2026-10-16: An `unmeasured` entry has no metrics to name, so its targets load (a decode
/// ladder's reference figures precede its first measured floors); a non-finite one is still
/// refused there.
#[test]
fn an_unmeasured_entry_may_carry_targets() {
    let entry = |name: &str, target: &str| {
        fixture(
            name,
            &format!(
                r#"
[[benchmarks]]
quant = "nvfp4"
checkpoint = "org/A"
gate = "ttft-cold-gate"
default = true
status = "unmeasured"
[benchmarks.targets]
median_ms = {target}
"#
            ),
        )
    };
    let loaded = load_all(&entry("target-unmeasured", "100.0")).unwrap();
    assert_eq!(loaded[0].1.targets.get("median_ms"), Some(&100.0));
    let err = format!(
        "{:#}",
        load_all(&entry("target-unmeasured-nan", "inf")).unwrap_err()
    );
    assert!(err.contains("target median_ms must be finite"), "{err}");
}
