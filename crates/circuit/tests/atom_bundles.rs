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

// 2026-10-05: The ldmatrix bundle's swizzle, as the layout algebra states it, is the closed form
// gemm_mainloop.cuh's `SwizzledRows<64, 2, 4, 3>` evaluates (64-byte rows: e4m3_mma_pipe.cuh's BK),
// and every ldmatrix phase (8 rows at one 16-byte chunk) hits 8 distinct bank groups. Mutation:
// declaring another swizzle, or a closed form with another row shift, fails one of the two.
#[test]
fn the_ldmatrix_bundle_swizzle_is_conflict_free_and_matches_the_device_form() {
    let root = common::root();
    let text = std::fs::read_to_string(root.join("kernels/gb10/common/KERNEL_FAMILIES.toml"))
        .expect("gb10 manifest");
    let f = metrale_circuit::venn::parse_families(&text).expect("parses");
    let b = &f.bundles["mma16832_e4m3_cpasync_ldsm"];
    for operand in ["a", "b"] {
        let p: Vec<u32> = b.swizzle[operand]
            .split(',')
            .map(|x| x.trim().parse().unwrap())
            .collect();
        let s = metrale_layout::Swizzle::new(p[0], p[1], p[2]).unwrap();
        const ROW_BYTES: u64 = 64;
        let row_shift = u64::from(p[1] + p[2]) - ROW_BYTES.trailing_zeros() as u64;
        let mask = (1u64 << p[0]) - 1;
        for row in 0..256u64 {
            for ch in 0..4u64 {
                let device = row * ROW_BYTES + ((ch ^ ((row >> row_shift) & mask)) << p[1]);
                assert_eq!(
                    s.apply(row * ROW_BYTES + ch * 16),
                    device,
                    "row {row} chunk {ch}"
                );
            }
        }
        for first in (0..256u64).step_by(8) {
            for ch in 0..4u64 {
                let lanes: Vec<u64> = (first..first + 8)
                    .map(|r| s.apply(r * ROW_BYTES + ch * 16))
                    .collect();
                assert_eq!(
                    metrale_layout::bank_conflicts(&lanes, 16).ways,
                    1,
                    "rows {first}.. chunk {ch}"
                );
            }
        }
    }
}
