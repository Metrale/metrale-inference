// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The result record of a run, keyed by the closure hash of the kernel target that
//! ran, and the calibration rows a calibration run proposes. TOML, stable order.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - A record names what ran (closure, device, commit) and what it was judged by (SHA-256 of
//!   the contracts and of the family manifest), so two records compare only like with like.

use toml::{Table, Value};

use crate::check::{Arm, Outcome, Verdict};
use crate::jobs::Coverage;

/// 2026-10-09: A run's record.
#[derive(Debug, Clone)]
pub struct Record {
    /// 2026-10-09: Hardware class.
    pub hardware: String,
    /// 2026-10-09: The kernel target(s) and their closure hashes, `target=hash`.
    pub closures: Vec<String>,
    /// 2026-10-09: The device that ran.
    pub device: String,
    /// 2026-10-09: The engine commit.
    pub commit: String,
    /// 2026-10-09: SHA-256 of ACCURACY.toml.
    pub contracts_sha256: String,
    /// 2026-10-09: SHA-256 of KERNEL_FAMILIES.toml.
    pub families_sha256: String,
    /// 2026-10-09: `quick` or `full`.
    pub scope: String,
    /// 2026-10-09: The outcomes.
    pub outcomes: Vec<Outcome>,
    /// 2026-10-09: Coverage of the sweep.
    pub coverage: Coverage,
}

fn arm(a: &Arm) -> Value {
    let mut t = Table::new();
    t.insert("name".into(), Value::String(a.name.clone()));
    t.insert("emulated".into(), Value::Boolean(a.emulated));
    t.insert("ratio".into(), Value::Float(a.ratio));
    t.insert("max_err".into(), Value::Float(a.max_err));
    t.insert("compared".into(), Value::Integer(a.compared as i64));
    t.insert("misrounded".into(), Value::Float(a.misrounded));
    Value::Table(t)
}

/// 2026-10-09: The margin of a passing derived good arm: how far below the bound it stays
/// (`1 / ratio`; infinite at a zero ratio).
pub fn margin(ratio: f64) -> f64 {
    if ratio > 0.0 {
        1.0 / ratio
    } else {
        f64::INFINITY
    }
}

/// 2026-10-09: Render the record.
pub fn render(r: &Record) -> String {
    let mut top = Table::new();
    let mut head = Table::new();
    head.insert("hardware".into(), Value::String(r.hardware.clone()));
    head.insert(
        "closures".into(),
        Value::Array(r.closures.iter().cloned().map(Value::String).collect()),
    );
    head.insert("device".into(), Value::String(r.device.clone()));
    head.insert("commit".into(), Value::String(r.commit.clone()));
    head.insert(
        "contracts_sha256".into(),
        Value::String(r.contracts_sha256.clone()),
    );
    head.insert(
        "families_sha256".into(),
        Value::String(r.families_sha256.clone()),
    );
    head.insert("scope".into(), Value::String(r.scope.clone()));
    let pass = r
        .outcomes
        .iter()
        .filter(|o| o.verdict == Verdict::Pass)
        .count();
    head.insert("checks".into(), Value::Integer(r.outcomes.len() as i64));
    head.insert("passed".into(), Value::Integer(pass as i64));
    head.insert(
        "swept_points".into(),
        Value::Integer(r.coverage.swept_points as i64),
    );
    head.insert(
        "covered_points".into(),
        Value::Integer(r.coverage.covered_points as i64),
    );
    top.insert("record".into(), Value::Table(head));
    let results = r
        .outcomes
        .iter()
        .map(|o| {
            let mut t = Table::new();
            t.insert("family".into(), Value::String(o.family.clone()));
            t.insert("kernel".into(), Value::String(o.kernel.clone()));
            t.insert("point".into(), Value::String(o.key.clone()));
            t.insert("input".into(), Value::String(o.input.name().into()));
            t.insert("verdict".into(), Value::String(o.verdict.name()));
            if let Verdict::FailDrift { threshold } = o.verdict {
                t.insert("drift_threshold".into(), Value::Float(threshold));
            }
            if let Some(g) = &o.good {
                t.insert("good".into(), arm(g));
                t.insert("margin".into(), Value::Float(margin(g.ratio).min(f64::MAX)));
            }
            if let Some(f) = &o.floor {
                t.insert("floor".into(), arm(f));
            }
            t.insert(
                "output_sha256".into(),
                Value::String(o.output_sha256.clone()),
            );
            t.insert(
                "mutations".into(),
                Value::Array(o.mutations.iter().map(arm).collect()),
            );
            Value::Table(t)
        })
        .collect();
    top.insert("result".into(), Value::Array(results));
    let uncovered = r
        .coverage
        .uncovered
        .iter()
        .map(|((f, k, op), why)| {
            let mut t = Table::new();
            t.insert("family".into(), Value::String(f.clone()));
            t.insert("kernels".into(), Value::String(k.clone()));
            t.insert("op".into(), Value::String(op.clone()));
            t.insert("why".into(), Value::String(why.clone()));
            Value::Table(t)
        })
        .collect();
    top.insert("uncovered".into(), Value::Array(uncovered));
    let unused = r
        .coverage
        .unused
        .iter()
        .map(|(f, k)| Value::String(format!("{f} {k}")))
        .collect();
    top.insert("contracted_but_unswept".into(), Value::Array(unused));
    toml::to_string(&top).unwrap_or_else(|e| format!("# record could not render: {e}\n"))
}

/// 2026-10-09: The `[[contract.calibration]]` rows a calibration run proposes for `outcomes`
/// (passing derived checks only), keyed for `closure`.
pub fn calibration_rows(outcomes: &[Outcome], closure: &str) -> String {
    let mut s = String::new();
    for o in outcomes.iter().filter(|o| o.verdict == Verdict::Pass) {
        let (Some(g), Some(f)) = (&o.good, &o.floor) else {
            continue;
        };
        let min_m = o
            .mutations
            .iter()
            .map(|m| m.misrounded)
            .fold(f64::INFINITY, f64::min);
        if !min_m.is_finite() {
            continue;
        }
        s.push_str(&format!(
            "# {} {}\n[[contract.calibration]]\npoint = {:?}\ninput = {:?}\nratio = {:e}\nmisrounded = {:e}\nfloor_misrounded = {:e}\nmutation_min_misrounded = {:e}\nclosure = {:?}\n\n",
            o.family,
            o.kernel,
            o.key,
            o.input.name(),
            g.ratio,
            g.misrounded,
            f.misrounded,
            min_m,
            closure
        ));
    }
    s
}
