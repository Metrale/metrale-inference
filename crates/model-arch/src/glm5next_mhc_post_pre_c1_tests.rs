// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: `kernel_for`'s choice: the fused launch replaces exactly the pair
//! `glm5next_hc_post_rank_sum` + `glm5next_hc_pre_norm_c1`, never another pre-norm.
//!
//! Owner: model-arch (GLM-5.3).

use super::*;
use crate::glm5next_mhc::mhc_shape_tests::test_kernels;

fn pick(k: &Glm5NextMhcKernels, flags: [bool; 3], t: u32, h: u32, hc: u32) -> Option<u64> {
    kernel_for(k, flags, [t, h, hc]).unwrap().map(|kh| kh.0)
}

/// 2026-10-24: Lever on, partials exchanged, the fused pre-norm at 1 or 2 rows of GLM-5.3's shape:
/// the fused kernel (positive).
#[test]
fn the_pair_at_decode_widths_takes_the_fused_kernel() {
    let mut k = test_kernels();
    k.hc_post_pre_norm_c1 = KernelHandle(21);
    assert!(
        k.levers.pre_c1 && !k.levers.pre_splitk,
        "declared levers take c1"
    );
    for t in [1, 2] {
        assert_eq!(pick(&k, [true; 3], t, 4096, 4), Some(21), "{t} rows");
    }
}

/// 2026-10-24: Every case where the pair would not be post-sum + c1 keeps the pair: lever off, no
/// exchange (plain all-reduce, or one rank), the unfused pre-norm, three rows and up (c1 stops at
/// two), another shape, split-k ahead of c1, c1 switched off (negative).
#[test]
fn any_other_pair_keeps_its_two_launches() {
    let mut k = test_kernels();
    k.hc_post_pre_norm_c1 = KernelHandle(21);
    assert_eq!(pick(&k, [false, true, true], 2, 4096, 4), None);
    assert_eq!(pick(&k, [true, false, true], 2, 4096, 4), None);
    assert_eq!(pick(&k, [true, true, false], 2, 4096, 4), None);
    for t in [0, 3, 64] {
        assert_eq!(pick(&k, [true; 3], t, 4096, 4), None, "{t} rows");
    }
    assert_eq!(pick(&k, [true; 3], 2, 5120, 4), None);
    assert_eq!(pick(&k, [true; 3], 2, 4096, 3), None);
    let mut sk = k.clone();
    sk.levers.pre_splitk = true;
    assert_eq!(pick(&sk, [true; 3], 2, 4096, 4), None);
    let mut no_c1 = k.clone();
    no_c1.levers.pre_c1 = false;
    assert_eq!(pick(&no_c1, [true; 3], 2, 4096, 4), None);
}

/// 2026-10-24: The lever on where the pair would be fused, without the kernel in the build: an
/// error naming the flag, never the pair in its place.
#[test]
fn the_lever_without_the_kernel_is_an_error() {
    let mut k = test_kernels();
    k.hc_post_pre_norm_c1 = KernelHandle(0);
    let e = kernel_for(&k, [true; 3], [2, 4096, 4])
        .unwrap_err()
        .to_string();
    assert!(e.contains("--glm-mhc-post-pre"), "{e}");
    assert_eq!(pick(&k, [false, true, true], 2, 4096, 4), None);
}
