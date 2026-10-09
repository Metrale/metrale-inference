// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Reduction depths from declared levels, and the manifest's pipeline precedence
//! (kernel over point over family) on the real gb10 manifest.

use super::*;
use crate::contract::Level;

fn lv(level: &str, width: &str, order: &str) -> Level {
    Level {
        level: level.into(),
        width: width.into(),
        order: order.into(),
    }
}

#[test]
fn depths_follow_the_levels() {
    assert_eq!(level_width("k/2048", 5120), Some(3));
    assert_eq!(level_width("32", 5120), Some(32));
    assert_eq!(level_width("k/0", 5120), None);
    assert_eq!(level_width("x", 5120), None);
    let w4a16_sw = [
        lv("thread", "k/2048", "sequential"),
        lv("pair", "2", "tree"),
        lv("warp", "32", "tree"),
        lv("pair", "2", "tree"),
    ];
    assert_eq!(tree_depth(&w4a16_sw, 5120), Some(3 + 1 + 5 + 1));
    assert_eq!(tree_depth(&[lv("warp", "33", "tree")], 1), Some(6));
    assert_eq!(tree_depth(&[lv("t", "1", "tree")], 1), Some(0));
    assert_eq!(tree_depth(&[lv("t", "4", "zigzag")], 1), None);
}

#[test]
fn kernel_declarations_override_the_family() {
    let text = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/common/KERNEL_FAMILIES.toml"
    ))
    .unwrap();
    let fams = metrale_circuit::venn::parse_families(&text).unwrap();
    let f = fams.families.iter().find(|f| f.id == "w4a16_gemm").unwrap();
    let none = Values::new();
    let fam = declared(f, "w4a16::w4a16_gemm_t", "linear", &none).unwrap();
    let p3 = declared(f, "w4a16::w4a16_gemm_t_p3", "linear", &none).unwrap();
    assert_ne!(fam, p3, "the _p3 kernel declares an fp8 activation");
    assert!(p3.steps.iter().any(|s| s.value.name() == "e4m3*e4m3"));
    assert!(declared(f, "w4a16::w4a16_gemm_t", "rope", &none).is_err());
    assert!(declared(f, "not_a_kernel", "linear", &none).is_err());
}
