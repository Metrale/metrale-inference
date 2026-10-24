// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `c1_kernel`'s choice.
//!
//! Owner: model-arch (GLM-5.3).

use super::*;
use crate::glm5next_mhc::mhc_shape_tests::test_kernels;

#[test]
fn the_decode_pre_norm_needs_the_lever_one_or_two_tokens_the_shape_and_the_kernel() {
    let k = test_kernels();
    let pick = |k: &Glm5NextMhcKernels, t: u32, h: u32, hc: u32, on: bool| {
        c1_kernel(k, t, h, hc, on).map(|kh| kh.0)
    };
    assert_eq!(pick(&k, 1, 4096, 4, true), Some(k.hc_pre_norm_c1.0));
    assert_eq!(pick(&k, 2, 4096, 4, true), Some(k.hc_pre_norm_c1.0));
    assert_eq!(pick(&k, 3, 4096, 4, true), None);
    assert_eq!(pick(&k, 0, 4096, 4, true), None);
    assert_eq!(pick(&k, 2, 4096, 4, false), None);
    assert_eq!(pick(&k, 2, 5120, 4, true), None);
    assert_eq!(pick(&k, 2, 4096, 3, true), None);
    let mut absent = test_kernels();
    absent.hc_pre_norm_c1 = KernelHandle(0);
    assert_eq!(pick(&absent, 2, 4096, 4, true), None);
}
