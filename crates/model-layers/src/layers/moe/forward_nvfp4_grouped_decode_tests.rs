// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: Shape admission of the grouped NVFP4 MoE decode (`forward_nvfp4_grouped_decode.rs`).
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-27: The admitted widths and the shape terms, each refused alone.
#[test]
fn shape_admission() {
    let max = NVFP4_GROUPED_DECODE_MAX_ROWS;
    assert!(nvfp4_grouped_decode_shape_ok(1, max, 2048, 512, 512));
    assert!(nvfp4_grouped_decode_shape_ok(max, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(0, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(max + 1, max, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(4, max, 2048 + 16, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(
        4,
        max,
        2048,
        512 + 8,
        512 + 8
    ));
    assert!(!nvfp4_grouped_decode_shape_ok(4, max, 2048, 512, 1024));
    // 2026-10-02: The tensor-core kernels' envelope: 256 rows, and the 35B's gate+up
    // (512 x 2048) and down (2048 x 512) fit their tiles; a K that is not a whole load
    // group of 256 does not.
    let tc = NVFP4_GROUPED_DECODE_TC_MAX_ROWS;
    assert_eq!(tc, 256);
    assert!(nvfp4_grouped_decode_shape_ok(tc, tc, 2048, 512, 512));
    assert!(!nvfp4_grouped_decode_shape_ok(tc + 1, tc, 2048, 512, 512));
    assert!(ops::nvfp4_grouped_tc_shape_ok(
        512,
        2048,
        ops::NVFP4_GROUPED_GATE_UP_TC
    ));
    assert!(ops::nvfp4_grouped_tc_shape_ok(
        2048,
        512,
        ops::NVFP4_GROUPED_DOWN_TC
    ));
    assert!(!ops::nvfp4_grouped_tc_shape_ok(
        2048,
        384,
        ops::NVFP4_GROUPED_DOWN_TC
    ));
    assert!(!ops::nvfp4_grouped_tc_shape_ok(
        96,
        2048,
        ops::NVFP4_GROUPED_GATE_UP_TC
    ));
}

/// 2026-10-05: The tensor-core pair writes each routed SiLU product as a BF16 hi + lo pair, and
/// the grouped FP8 down kernel reads FP32 rows, so a decode whose routed downs run that kernel
/// (`nvfp4-gate-up`) takes the CUDA-core gate+up; otherwise the tensor-core pair, as before.
/// Mutation: selecting the tensor-core pair under an FP8 down (the defect: `because because ...`
/// on the gate-up recipe) fails the second assertion.
#[test]
fn an_fp8_down_takes_the_gate_up_that_writes_fp32_products() {
    let k = |h: u64| KernelHandle(h);
    let pair = Nvfp4GroupedKernels {
        gate_up: k(1),
        down: k(2),
        gate_up_tc: k(3),
        down_tc: k(4),
        declared_experts: false,
        gate_up_tc_lean: k(0),
        down_tc_lean: k(0),
        lean_repack: k(0),
        lean: false,
        bf16_gate_up_tc: k(0),
        bf16_down_tc: k(0),
    };
    if !nvfp4_grouped_tc_enabled() {
        return;
    }
    let nvfp4_down = pair.select(2048, 512, false);
    assert_eq!(
        (nvfp4_down.gate_up.0, nvfp4_down.down.0, nvfp4_down.max_rows),
        (3, 4, NVFP4_GROUPED_DECODE_TC_MAX_ROWS)
    );
    let fp8_down = pair.select(2048, 512, true);
    assert_eq!(
        (fp8_down.gate_up.0, fp8_down.max_rows),
        (1, NVFP4_GROUPED_DECODE_MAX_ROWS)
    );
}
