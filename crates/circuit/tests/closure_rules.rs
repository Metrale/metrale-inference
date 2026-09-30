// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The closure hash and the planner name the same FUSIONS.toml files for every
//! class. The planner resolves a class's rules along FUSIONS `inherits` and, for a class
//! without rules, along HARDWARE.toml `inherits` (`class_rules`); the closure hashes
//! `metrale_closure::fusions_chain`. If the two drift, an edit to a file the planner reads
//! could leave a record's closure unchanged, which is the defect this pins (hopper and b300
//! hashed only their own FUSIONS.toml, not gb10's; b200 hashed none).
//!
//! Owner: metrale-circuit tests.
//! Invariants: runs on the checked-out tree, every class under `kernels/` with a HARDWARE.toml.

mod common;
mod venn_common;

use metrale_circuit::hardware::class::{ClassRules, class_rules};
use venn_common::Tree;

fn classes() -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(common::root().join("kernels"))
        .expect("kernels/")
        .flatten()
        .filter(|e| e.path().join("HARDWARE.toml").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

#[test]
fn the_closure_hashes_every_fusions_file_the_planner_reads() {
    let root = common::root();
    let mut compared = Vec::new();
    for class in classes() {
        // 2026-09-30: Every class's chain resolves: the kernels build script refuses to
        // attest a target whose chain does not, so one bad class would stop every build.
        let chain = metrale_closure::fusions_chain(&root, &class)
            .unwrap_or_else(|e| panic!("{class}: the rule chain does not resolve: {e}"));
        let planner: Vec<String> = match class_rules(&Tree, &class) {
            Ok(ClassRules::Rules { files, .. }) => files,
            Ok(ClassRules::None) => Vec::new(),
            // 2026-09-30: A class the planner cannot resolve plans nothing; the closure side is
            // checked for its own refusals in metrale-closure.
            Err(_) => continue,
        };
        let closure: Vec<String> = chain
            .iter()
            .map(|p| p.strip_prefix(&root).unwrap().display().to_string())
            .collect();
        assert_eq!(
            closure, planner,
            "{class}: closure and planner name different rules"
        );
        compared.push((class, planner));
    }
    // 2026-09-30: Not vacuous: both inheritance routes are on the tree and were compared.
    let chain_of = |c: &str| {
        compared
            .iter()
            .find(|(n, _)| n == c)
            .map(|(_, f)| f.clone())
    };
    let gb10 = "kernels/gb10/common/FUSIONS.toml".to_string();
    assert_eq!(chain_of("gb10"), Some(vec![gb10.clone()]));
    for child in ["hopper", "b300"] {
        assert_eq!(
            chain_of(child),
            Some(vec![
                gb10.clone(),
                format!("kernels/{child}/common/FUSIONS.toml")
            ]),
            "{child} inherits gb10's rules through FUSIONS.toml"
        );
    }
    assert_eq!(
        chain_of("b200"),
        Some(vec![gb10]),
        "b200 takes gb10's rules through HARDWARE.toml"
    );
}
