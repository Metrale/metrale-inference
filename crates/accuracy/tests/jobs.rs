// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The plan over the checked-in contracts and instances: nothing swept is silently
//! dropped, every check names a kernel its point's plan group launches, the quick scope is a
//! subset of the full one that still runs every contracted class and every distinct shape, and
//! the contracts fit the families.

mod common;

use std::collections::BTreeSet;

use metrale_accuracy::contract::parse_contracts;
use metrale_accuracy::jobs::{Scope, plan, validate};
use metrale_accuracy::points::sweep;
use metrale_circuit::venn::Repo;

#[test]
fn the_plan_accounts_for_every_swept_point() {
    let s = sweep(&common::Tree, "gb10").unwrap();
    let contracts = parse_contracts(
        &common::Tree
            .read("kernels/gb10/common/ACCURACY.toml")
            .unwrap(),
    )
    .unwrap();
    let fams = common::families();
    assert_eq!(validate(&contracts, &fams), Vec::<String>::new());
    let (full, cov) = plan(&s, &contracts, &fams, Scope::Full, None, None);
    let (quick, _) = plan(&s, &contracts, &fams, Scope::Quick, None, None);
    assert_eq!(cov.swept_points, s.points.len());
    let uncovered_points = s
        .points
        .iter()
        .filter(|p| {
            cov.uncovered
                .contains_key(&(p.family.clone(), p.kernels.clone(), p.shape.op.clone()))
        })
        .count();
    assert_eq!(
        cov.covered_points + uncovered_points,
        cov.swept_points,
        "a swept point is neither planned nor reported"
    );
    assert!(!full.is_empty() && quick.len() < full.len());
    let key = |j: &metrale_accuracy::jobs::Planned<'_>| {
        format!(
            "{} {:?} {:?} {}",
            j.kernel,
            j.point,
            j.shape,
            j.input.name()
        )
    };
    let fullset: BTreeSet<String> = full.iter().map(key).collect();
    for j in &quick {
        assert!(
            fullset.contains(&key(j)),
            "quick has a check full lacks: {}",
            key(j)
        );
        assert!(j.contract.kernels.contains(&j.kernel));
    }
    // 2026-10-09: Quick keeps every contracted (kernel, class) and every distinct (kernel, K, N).
    let classes = |v: &[metrale_accuracy::jobs::Planned<'_>]| -> BTreeSet<(String, String)> {
        v.iter()
            .map(|j| (j.kernel.clone(), j.input.name().to_string()))
            .collect()
    };
    assert_eq!(classes(&quick), classes(&full));
    let shapes = |v: &[metrale_accuracy::jobs::Planned<'_>]| -> BTreeSet<(String, u64, u64)> {
        v.iter()
            .map(|j| (j.kernel.clone(), j.shape.in_dim, j.shape.out_dim))
            .collect()
    };
    assert_eq!(shapes(&quick), shapes(&full));
}

#[test]
fn a_contract_naming_a_foreign_kernel_is_refused() {
    let text = common::Tree
        .read("kernels/gb10/common/ACCURACY.toml")
        .unwrap();
    let bad = text.replacen(
        "\"w4a16_gemv::w4a16_gemv_sw\"]",
        "\"w4a16_gemv::not_a_kernel\"]",
        1,
    );
    assert_ne!(bad, text);
    let problems = validate(&parse_contracts(&bad).unwrap(), &common::families());
    assert!(
        problems.iter().any(|p| p.contains("not_a_kernel")),
        "{problems:?}"
    );
}
