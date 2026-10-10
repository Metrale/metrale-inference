// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The union Venn over every instance in kernels/circuits/INSTANCES.toml (each model
//! the target of a Venn report against the golden instances, at C1, C16 and C128), checked in as
//! kernels/circuits/venn/UNION.md. Regenerate after an intended change with
//! `cargo test -p metrale-circuit --test venn_union -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;
mod venn_common;

use metrale_circuit::Mode;
use metrale_circuit::venn::union::UnionReport;
use metrale_circuit::venn::union_repo::union_of_repo;
use metrale_circuit::venn::{Class, render_union};
use venn_common::*;

const UNION: &str = "kernels/circuits/venn/UNION.md";
const COMMAND: &str = "cargo test -p metrale-circuit --test venn_union -- --ignored regenerate";

fn union_text() -> (UnionReport, String) {
    // 2026-10-10: Before any envelope sweep (`met accuracy envelope union` regenerates the
    // post-sweep report from SCHEDULES.toml).
    let u = union_of_repo(&Tree, "gb10", &|_| false).expect("union");
    let text = render_union(&u, COMMAND);
    (u, text)
}

#[test]
fn the_checked_in_union_report_is_current() {
    let (_, text) = union_text();
    assert!(
        text == common::read(UNION),
        "{UNION} is stale; regenerate with `{COMMAND}` and review the diff"
    );
}

#[test]
fn every_instance_is_in_the_union_and_the_golden_models_are_in_their_own_envelope() {
    let (u, _) = union_text();
    assert_eq!(u.models.len(), common::instances().len());
    // 2026-10-10: A golden model's evidence was measured on its own kernel target, so some of its
    // C1 step is in envelope; a model no golden one resembles has none of it measured at its
    // shapes unless a record names them.
    let dense = u
        .models
        .iter()
        .position(|m| m.recipe == DENSE)
        .expect("the dense golden model");
    let c1 = |m: usize, env: metrale_circuit::venn::union::Envelope| -> f64 {
        u.uses
            .iter()
            .filter(|x| x.model == m && x.run.mode == Mode::Decode && x.envelope == env)
            .map(|x| x.share)
            .sum()
    };
    assert!(c1(dense, metrale_circuit::venn::union::Envelope::Shape) > 0.5);
    // 2026-10-10: Every share is a Venn share: per model and run they sum to 1.
    for (i, _) in u.models.iter().enumerate() {
        for run in &u.runs {
            let total: f64 = u
                .uses
                .iter()
                .filter(|x| x.model == i && x.run == *run)
                .map(|x| x.share)
                .sum();
            assert!((total - 1.0).abs() < 1e-9, "model {i} {run:?}: {total}");
        }
    }
    // 2026-10-10: Novel sites carry no family.
    assert!(u.uses.iter().filter(|x| x.class == Class::Novel).all(
        |x| x.family.is_none() || x.envelope == metrale_circuit::venn::union::Envelope::Outside
    ));
}

#[test]
#[ignore]
fn regenerate() {
    let (_, text) = union_text();
    std::fs::write(common::root().join(UNION), text).expect("write");
}
