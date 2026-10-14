// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Which shared expert the grouped NVFP4 decode runs, from what the layer holds:
//! every combination of an admissible FP8, an NVFP4 and an installed BF16 shared expert.

use super::{Nvfp4GroupedShared as S, grouped_shared_form};

#[test]
fn each_lone_shared_expert_is_admitted_in_its_own_form() {
    assert_eq!(grouped_shared_form(true, false, false), Some(S::Fp8));
    assert_eq!(grouped_shared_form(false, true, false), Some(S::Nvfp4));
    assert_eq!(grouped_shared_form(false, false, true), Some(S::Bf16));
}

#[test]
fn fp8_wins_over_nvfp4_as_before_the_bf16_form() {
    assert_eq!(grouped_shared_form(true, true, false), Some(S::Fp8));
}

#[test]
fn a_bf16_shared_expert_beside_a_quantized_one_or_none_at_all_is_refused() {
    assert_eq!(grouped_shared_form(true, false, true), None);
    assert_eq!(grouped_shared_form(false, true, true), None);
    assert_eq!(grouped_shared_form(true, true, true), None);
    assert_eq!(grouped_shared_form(false, false, false), None);
}
