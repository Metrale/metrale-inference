// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for the pure admission helpers of the grouped FP8 MoE decode path.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::{
    FP8_GROUPED_DECODE_MAX_ROWS, FP8_GROUPED_DECODE_TC_MAX_ROWS, fp8_grouped_decode_rows_ok,
    fp8_grouped_decode_shape_ok, grouped_decode_buffer_need,
};

/// 2026-09-25: Qwen3.6-35B-A3B-FP8: hidden 2048, moe_intermediate 512, 256 experts, top-8.
const H: u32 = 2048;
const INTER: u32 = 512;

#[test]
fn row_envelope_is_2_to_max() {
    assert!(!fp8_grouped_decode_shape_ok(0, H, INTER));
    assert!(
        !fp8_grouped_decode_shape_ok(1, H, INTER),
        "M=1 is the single-token path"
    );
    for m in [2usize, 3, 4, 8, 16, 24, 32, FP8_GROUPED_DECODE_MAX_ROWS] {
        assert!(fp8_grouped_decode_shape_ok(m, H, INTER), "m={m}");
    }
    assert!(!fp8_grouped_decode_shape_ok(
        FP8_GROUPED_DECODE_MAX_ROWS + 1,
        H,
        INTER
    ));
}

/// 2026-09-29: The tensor-core kernels take one row and every width up to one 128-sequence
/// verify at one draft; the scalar kernels keep 2..=64, so the kill switch restores the old
/// envelope exactly.
#[test]
fn tensor_core_rows_reach_a_128_sequence_verify() {
    assert_eq!(FP8_GROUPED_DECODE_TC_MAX_ROWS, 2 * 128);
    for m in [1usize, 2, 64, 65, 96, 128, 129, 192, 256] {
        assert!(fp8_grouped_decode_rows_ok(m, true), "tc m={m}");
    }
    assert!(!fp8_grouped_decode_rows_ok(0, true));
    assert!(!fp8_grouped_decode_rows_ok(257, true));
    for m in [1usize, 65, 128, 256] {
        assert!(!fp8_grouped_decode_rows_ok(m, false), "scalar m={m}");
    }
    for m in 2..=FP8_GROUPED_DECODE_MAX_ROWS {
        assert_eq!(
            fp8_grouped_decode_rows_ok(m, false),
            fp8_grouped_decode_shape_ok(m, H, INTER),
            "m={m}"
        );
    }
}

#[test]
fn hidden_must_be_a_multiple_of_16_for_the_uint4_pair_loads() {
    assert!(fp8_grouped_decode_shape_ok(4, 16, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 2040, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 8, INTER));
    assert!(!fp8_grouped_decode_shape_ok(4, 0, INTER));
}

#[test]
fn intermediate_must_be_a_multiple_of_8() {
    assert!(fp8_grouped_decode_shape_ok(4, H, 768));
    assert!(fp8_grouped_decode_shape_ok(4, H, 1024));
    assert!(!fp8_grouped_decode_shape_ok(4, H, 516));
    assert!(!fp8_grouped_decode_shape_ok(4, H, 0));
    // 2026-09-26: The down kernel reads the SiLU product from global memory, so
    // no shared-memory pass bounds the width any more.
    assert!(fp8_grouped_decode_shape_ok(4, H, 1512));
    assert!(fp8_grouped_decode_shape_ok(4, H, 4096));
}

#[test]
fn buffer_need_matches_the_launch_layout() {
    let n = grouped_decode_buffer_need(16, 2048, 512, 256, 8);
    let te = 16 * 8;
    assert_eq!(n.scratch, 2 * te * 4);
    // 2026-09-25: sort scratch 3*te*4 + (E+1)*4 + (min(te,E)+1)*4 = 3080 < 16x256 BF16 = 8192
    assert_eq!(
        n.gate_logits,
        (16 * 256 * 2).max(3 * te * 4 + 257 * 4 + 129 * 4)
    );
    assert_eq!(n.gate_logits, 8192);
    assert_eq!(n.expert_gate_out, te * 512 * 4);
    assert_eq!(n.expert_down_out, te * 2048 * 2);
    assert_eq!(n.shared_act, 16 * 512 * 4);
    assert_eq!(n.row_hidden, 16 * 2048 * 2);
    // 2026-09-25: At M=2 the sort scratch dominates the logits extent (cap = 16 < E).
    let n2 = grouped_decode_buffer_need(2, 2048, 256, 256, 8);
    assert_eq!(n2.gate_logits, 3 * 16 * 4 + 257 * 4 + 17 * 4);
    // 2026-09-25: Wide batch: cap saturates at E; the [64, 256] BF16 logits (32768 B)
    // still dominate the sort scratch (3*512*4 + 257*4 + 257*4 = 8200 B).
    let n64 = grouped_decode_buffer_need(64, 2048, 512, 256, 8);
    assert_eq!(n64.gate_logits, 32768);
    // 2026-09-29: A 256-row verify: te = 2048, the routed FP32 SiLU rows are 4 MiB and the
    // sorted down rows 8 MiB, inside the arena's `max_batch_tokens`-row expert buffers.
    let n256 = grouped_decode_buffer_need(256, 2048, 512, 256, 8);
    assert_eq!(n256.expert_gate_out, 2048 * 512 * 4);
    assert_eq!(n256.expert_down_out, 2048 * 2048 * 2);
    assert_eq!(n256.gate_logits, 256 * 256 * 2);
}
