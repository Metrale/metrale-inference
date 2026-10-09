// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: GB10's tensor-core policy over its whole rule set: every FUSIONS.toml rule that runs
//! a policy-covered op off tensor cores is listed by an exemption for its every mode, row and
//! kernel, and every kernel any rule names declares its compute unit. A new rule or kernel that
//! would run a matmul-class op on CUDA cores fails here until it is moved to tensor cores or
//! exempted with its reason (kernels/gb10/HARDWARE.toml `[tensor_core_policy]`).
//!
//! 2026-10-05: Hopper too: its rule set (gb10's through `inherits`, less its `remove` list)
//! against its own policy (kernels/hopper/HARDWARE.toml), with the manifest it inherits.
//!
//! Owner: metrale-circuit tests.
//! Invariants: none beyond the types.

mod common;

use metrale_circuit::hardware::class::{ClassRules, class_rules};
use metrale_circuit::hardware::tc_policy::{lint_rules, parse_policy};
use metrale_circuit::venn::Repo;

/// 2026-10-05: The checked-out tree, for the class overlay resolution.
struct Tree;

impl Repo for Tree {
    fn read(&self, rel: &str) -> Result<String, String> {
        std::fs::read_to_string(common::root().join(rel)).map_err(|e| format!("{rel}: {e}"))
    }

    fn list(&self, rel: &str) -> Result<Vec<String>, String> {
        Err(format!(
            "{rel}: listing is not needed to resolve a class's rules"
        ))
    }
}

/// 2026-10-05: Hopper's resolved rules (gb10's, less `remove`) against Hopper's own policy, with
/// the gb10 manifest it inherits: every rule off tensor cores is exempted for its whole range.
#[test]
fn every_hopper_rule_complies_with_hoppers_tensor_core_policy() {
    let hw: toml::Table = toml::from_str(&common::read("kernels/hopper/HARDWARE.toml")).unwrap();
    let policy = parse_policy(&hw)
        .unwrap()
        .expect("hopper states a tensor-core policy");
    let families = metrale_circuit::venn::parse_families(&common::read(
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ))
    .unwrap();
    let ClassRules::Rules { rules, files, .. } = class_rules(&Tree, "hopper").unwrap() else {
        panic!("hopper resolves no rules");
    };
    assert!(
        files.iter().any(|f| f.starts_with("kernels/hopper/")),
        "{files:?}"
    );
    let found = lint_rules(&rules, &families, &policy);
    assert!(
        found.is_empty(),
        "hopper tensor-core policy:\n  {}",
        found.join("\n  ")
    );
}

#[test]
fn every_gb10_rule_complies_with_the_tensor_core_policy() {
    let hw: toml::Table = toml::from_str(&common::read("kernels/gb10/HARDWARE.toml")).unwrap();
    let policy = parse_policy(&hw)
        .unwrap()
        .expect("gb10 states a tensor-core policy");
    let families = metrale_circuit::venn::parse_families(&common::read(
        "kernels/gb10/common/KERNEL_FAMILIES.toml",
    ))
    .unwrap();
    let rules =
        metrale_circuit::runtime::parse_rule_set(&common::read("kernels/gb10/common/FUSIONS.toml"))
            .unwrap()
            .rules;
    let found = lint_rules(&rules, &families, &policy);
    assert!(
        found.is_empty(),
        "tensor-core policy:\n  {}",
        found.join("\n  ")
    );
    for r in &rules {
        for k in &r.kernels {
            assert!(
                families.compute_of(k).is_some(),
                "rule `{}`: `{k}` declares no compute unit",
                r.id
            );
        }
    }
}
