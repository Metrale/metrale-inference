// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests pinning the two multi-sequence FFN ladders
//! (`ffn_rows_select.rs`): each arm is reachable, and each ladder's order is the
//! one its `if` chain had. A swap of two arms in either selector fails a test here.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use super::ffn_rows::FfnArm;
use super::ffn_rows_select::{LadderCaps, select_attn_arm, select_ssm_arm};

fn caps(n: usize) -> LadderCaps {
    LadderCaps {
        n,
        ..LadderCaps::default()
    }
}

/// 2026-10-13: With no capability on, both ladders run one row at a time, except
/// that the fused two- and three-row kernels need no capability on either.
#[test]
fn with_nothing_available_both_ladders_run_per_row() {
    for n in [4, 7, 16, 128] {
        assert_eq!(select_ssm_arm(&caps(n)), FfnArm::PerRow, "ssm n={n}");
        assert_eq!(select_attn_arm(&caps(n)), FfnArm::PerRow, "attn n={n}");
    }
    assert_eq!(select_ssm_arm(&caps(2)), FfnArm::K2);
    assert_eq!(select_attn_arm(&caps(2)), FfnArm::K2);
    assert_eq!(select_ssm_arm(&caps(3)), FfnArm::K3);
    assert_eq!(select_attn_arm(&caps(3)), FfnArm::K3);
}

/// 2026-10-13: The GDN ladder, arm by arm, from the top: each row enables one
/// more capability below the previous winner and expects the previous winner to
/// stay, then disables it and expects the new one.
#[test]
fn ssm_ladder_order() {
    let mut c = caps(16);
    c.batched = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Batched);
    c.token_major = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::TokenMajor);
    c.moe_grouped = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Prefill);
    c.fp8_grouped = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Fp8Grouped);
    c.km_chunk = Some(8);
    assert_eq!(select_ssm_arm(&c), FfnArm::Km { chunk: 8 });
    c.ssm_dense_prefill = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Prefill);
    c.fp8_row_invariant = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Fp8RowInvariant);
    c.nvfp4_grouped = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Nvfp4Grouped);
    c.dense_fixed = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::DenseFixed);
}

#[test]
fn ssm_ladder_small_widths() {
    // 2026-10-13: n = 2 and 3 take the fused kernels ahead of every lower arm,
    // and behind the three tier arms.
    let mut c = caps(2);
    c.km_chunk = Some(8);
    c.fp8_grouped = true;
    c.token_major = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::K2);
    c.n = 3;
    assert_eq!(select_ssm_arm(&c), FfnArm::K3);
    c.fp8_row_invariant = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Fp8RowInvariant);
    // The Km arm needs n >= 4 even when the chunk is offered.
    let mut c = caps(3);
    c.km_chunk = Some(8);
    assert_eq!(select_ssm_arm(&c), FfnArm::K3);
    c.n = 4;
    assert_eq!(select_ssm_arm(&c), FfnArm::Km { chunk: 8 });
    // The atomic C4 arm sits between the grouped MoE and token-major arms.
    let mut c = caps(4);
    c.atomic_c4 = true;
    c.token_major = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::AtomicC4);
    c.moe_grouped = true;
    assert_eq!(select_ssm_arm(&c), FfnArm::Prefill);
}

#[test]
fn attn_ladder_order() {
    let mut c = caps(16);
    c.batched = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Batched);
    c.pairwise = true;
    assert_eq!(select_attn_arm(&c), FfnArm::PairwiseK2);
    c.use_grouped = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Prefill);
    c.use_grouped = false;
    c.moe_grouped = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Prefill);
    c.fp8_grouped = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Fp8Grouped);
    c.km_chunk = Some(16);
    assert_eq!(select_attn_arm(&c), FfnArm::Km { chunk: 16 });
    c.fp8_row_invariant = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Fp8RowInvariant);
    c.nvfp4_grouped = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Nvfp4Grouped);
    c.dense_fixed = true;
    assert_eq!(select_attn_arm(&c), FfnArm::DenseFixed);
}

#[test]
fn attn_ladder_small_widths_and_the_grouped_opt_in() {
    let mut c = caps(3);
    c.km_chunk = Some(3);
    c.fp8_grouped = true;
    assert_eq!(select_attn_arm(&c), FfnArm::K3);
    c.n = 2;
    assert_eq!(select_attn_arm(&c), FfnArm::K2);
    // The grouped routed opt-in skips the fused kernels and, below the km, FP8
    // and prefill arms, runs `forward_prefill`.
    c.use_grouped = true;
    c.km_chunk = None;
    c.fp8_grouped = false;
    assert_eq!(select_attn_arm(&c), FfnArm::Prefill);
    // A dense FFN takes `forward_prefill` whatever `moe_grouped` says.
    let mut c = caps(6);
    c.is_dense = true;
    c.pairwise = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Prefill);
    // Pairwise needs an even n above 2.
    let mut c = caps(5);
    c.pairwise = true;
    assert_eq!(select_attn_arm(&c), FfnArm::PerRow);
    c.n = 6;
    assert_eq!(select_attn_arm(&c), FfnArm::PairwiseK2);
    c.n = 2;
    assert_eq!(select_attn_arm(&c), FfnArm::K2);
}

/// 2026-10-13: An MLA layer skips every batched arm; only the `batched` opt-in
/// of the final branch survives it, as before.
#[test]
fn attn_force_seq_leaves_only_the_final_branch() {
    let mut c = caps(16);
    c.force_seq = true;
    c.dense_fixed = true;
    c.nvfp4_grouped = true;
    c.fp8_row_invariant = true;
    c.km_chunk = Some(16);
    c.fp8_grouped = true;
    c.is_dense = true;
    c.moe_grouped = true;
    c.use_grouped = true;
    c.pairwise = true;
    assert_eq!(select_attn_arm(&c), FfnArm::PerRow);
    c.batched = true;
    assert_eq!(select_attn_arm(&c), FfnArm::Batched);
}
