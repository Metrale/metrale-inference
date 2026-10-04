// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A tensor-parallel rank's plans (tests/common/parallel.rs) equal their goldens, and
//! every row-parallel output is reduced before anything reads it. Regenerate the goldens with
//! `cargo test -p metrale-circuit --test parallel -- --ignored regenerate`.
//!
//! Owner: metrale-circuit tests (FEATURES workstream).
//! Invariants: none beyond the types.

mod common;

use std::collections::BTreeSet;

use metrale_circuit::{LinearRole, OpKind, fuse};

#[test]
fn the_tp_plans_equal_their_goldens() {
    let dir = common::plans_dir();
    let want = common::parallel::plans();
    let mut problems = Vec::new();
    for (name, text) in &want {
        match std::fs::read_to_string(dir.join(name)) {
            Ok(on_disk) if on_disk == *text => {}
            Ok(_) => problems.push(format!("{name}: stale")),
            Err(_) => problems.push(format!("{name}: missing")),
        }
    }
    let expected: BTreeSet<&str> = want.iter().map(|(n, _)| n.as_str()).collect();
    for entry in std::fs::read_dir(dir.join("tp"))
        .expect("plans/tp")
        .flatten()
    {
        let name = format!("tp/{}", entry.file_name().to_string_lossy());
        if !expected.contains(name.as_str()) {
            problems.push(format!("{name}: no TP shape produces it"));
        }
    }
    assert!(
        problems.is_empty(),
        "TP goldens: regenerate with `cargo test -p metrale-circuit --test parallel -- --ignored \
         regenerate` and review the diff:\n{}",
        problems.join("\n")
    );
}

#[test]
#[ignore = "writes kernels/circuits/plans/tp/; run explicitly to regenerate"]
fn regenerate() {
    let dir = common::plans_dir().join("tp");
    std::fs::create_dir_all(&dir).expect("plans/tp");
    for entry in std::fs::read_dir(&dir).expect("plans/tp").flatten() {
        std::fs::remove_file(entry.path()).expect("remove stale TP plan");
    }
    for (name, text) in common::parallel::plans() {
        std::fs::write(common::plans_dir().join(name), text).expect("write TP plan");
    }
}

#[test]
fn every_row_parallel_output_is_reduced_in_its_own_group_before_it_is_read() {
    let (inst, loaded) = common::parallel::rank();
    let c = &loaded.circuit;
    let avail = common::available(&inst, &loaded.rules);
    for (mode, rows) in common::parallel::SHAPES {
        let p = fuse(c, &loaded.rules, &avail, &inst.policy, mode, rows).unwrap();
        let group_of = |n: usize| p.groups.iter().position(|g| g.nodes.contains(&n)).unwrap();
        let mut seen = 0;
        for (i, n) in c.nodes.iter().enumerate() {
            let OpKind::Linear(r @ (LinearRole::O | LinearRole::GdnOut)) = n.op else {
                continue;
            };
            let y = n.outputs[0];
            let readers = &c.edges[y].consumers;
            assert_eq!(
                readers.len(),
                1,
                "{r:?}: only the reduce reads the partial sum"
            );
            let reduce = readers[0];
            assert_eq!(c.nodes[reduce].op, OpKind::AllReduce);
            let g = &p.groups[group_of(reduce)];
            assert_eq!((g.rule.as_str(), g.nodes.len()), ("tp_all_reduce", 1));
            assert!(
                group_of(reduce) > group_of(i),
                "{mode:?}: the reduce follows {}",
                n.id
            );
            for &later in &c.edges[c.nodes[reduce].outputs[0]].consumers {
                assert!(
                    group_of(later) > group_of(reduce),
                    "{mode:?}: {} reads before the sum",
                    c.nodes[later].id
                );
            }
            seen += 1;
        }
        assert_eq!(seen, c.layer_kinds.len(), "{mode:?}");
    }
}
