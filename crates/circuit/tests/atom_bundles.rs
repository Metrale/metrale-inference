// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The atom bundles a manifest declares list only classes that realize them: the
//! manifest's own class or a class whose HARDWARE.toml inherits it.
//!
//! Owner: metrale-circuit tests.
//! Invariants: reads the repository; writes nothing.

mod common;

// 2026-10-05: Mutation: listing a class that neither is gb10 nor inherits it (b300 compiles a
// subset by `[sources] use`, strix is another vendor) fails here.
#[test]
fn every_listed_class_is_the_manifest_class_or_inherits_it() {
    let root = common::root();
    let text = std::fs::read_to_string(root.join("kernels/gb10/common/KERNEL_FAMILIES.toml"))
        .expect("gb10 manifest");
    let f = metrale_circuit::venn::parse_families(&text).expect("parses");
    assert!(!f.bundles.is_empty(), "gb10 declares its bundles");
    for b in f.bundles.values() {
        for class in &b.classes {
            if class == "gb10" {
                continue;
            }
            let hw = std::fs::read_to_string(root.join(format!("kernels/{class}/HARDWARE.toml")))
                .unwrap_or_else(|_| panic!("bundle `{}` lists unknown class `{class}`", b.id));
            assert!(
                hw.lines().any(|l| l.trim() == "inherits = \"gb10\""),
                "bundle `{}` lists `{class}`, which does not inherit gb10",
                b.id
            );
        }
    }
}
