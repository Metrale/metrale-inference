// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Tests of the W4A4 weight side: NVFP4 row and column slicing for a TP rank, the
//! K contract, and the uniform activation scales.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::TpSlice;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::*;

/// 2026-10-08: An `[n, k]` NVFP4 tensor whose packed byte `(r, j)` is `r * 64 + j` (mod 256)
/// and whose scale byte `(r, b)` is `r * 8 + b`, so every byte names its position.
fn tagged(n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
    let p = (0..n * k / 2)
        .map(|i| ((i / (k / 2)) * 64 + i % (k / 2)) as u8)
        .collect();
    let s = (0..n * k / 16)
        .map(|i| ((i / (k / 16)) * 8 + i % (k / 16)) as u8)
        .collect();
    (p, s)
}

fn sl(start: usize, len: usize) -> TpSlice {
    TpSlice { start, len }
}

/// 2026-10-08: Three ranks' column slices of a `[2, 96]` weight rejoin, row by row, to the
/// whole tensor; each rank's bytes are its own 32 columns (16 bytes, 2 scales per row).
#[test]
fn column_slices_rejoin_to_the_whole_weight() {
    let (n, k) = (2, 96);
    let (p, s) = tagged(n, k);
    let parts: Vec<_> = (0..3)
        .map(|r| slice_nvfp4_cols(&p, &s, n, k, sl(r * 32, 32)).unwrap())
        .collect();
    for row in 0..n {
        let joined: Vec<u8> = parts
            .iter()
            .flat_map(|(pp, _)| pp[row * 16..row * 16 + 16].to_vec())
            .collect();
        assert_eq!(joined, p[row * 48..row * 48 + 48], "packed row {row}");
        let joined_s: Vec<u8> = parts
            .iter()
            .flat_map(|(_, ss)| ss[row * 2..row * 2 + 2].to_vec())
            .collect();
        assert_eq!(joined_s, s[row * 6..row * 6 + 6], "scale row {row}");
    }
    // 2026-10-08: Rank 1, row 1 starts at packed byte 16 of that row and scale block 2.
    assert_eq!(parts[1].0[16], (64 + 16) as u8);
    assert_eq!(parts[1].1[2], (8 + 2) as u8);
}

/// 2026-10-08: A row slice keeps whole rows, codes and scales together.
#[test]
fn row_slices_keep_whole_rows() {
    let (n, k) = (6, 32);
    let (p, s) = tagged(n, k);
    let (pp, ss) = slice_nvfp4_rows(&p, &s, n, k, sl(2, 3)).unwrap();
    assert_eq!(pp, p[2 * 16..5 * 16]);
    assert_eq!(ss, s[2 * 2..5 * 2]);
}

/// 2026-10-08: A column slice that splits a scale block, or runs past the width, and a tensor
/// of the wrong size are refused.
#[test]
fn bad_slices_and_shapes_are_refused() {
    let (n, k) = (2, 96);
    let (p, s) = tagged(n, k);
    let e = slice_nvfp4_cols(&p, &s, n, k, sl(8, 32))
        .unwrap_err()
        .to_string();
    assert!(e.contains("16-column"), "{e}");
    let e = slice_nvfp4_cols(&p, &s, n, k, sl(64, 48))
        .unwrap_err()
        .to_string();
    assert!(e.contains("does not fit"), "{e}");
    let e = slice_nvfp4_rows(&p[1..], &s, n, k, sl(0, 1))
        .unwrap_err()
        .to_string();
    assert!(e.contains("packed"), "{e}");
}

/// 2026-10-08: The mx kernels' K contract: GLM's widths pass (hidden 4096, expert 2048, the
/// TP=3 dense share 4096); a width that is not whole k128 chunks is refused.
#[test]
fn k_must_be_whole_k128_chunks() {
    for k in [4096, 2048, 128] {
        check_w4a4_k(k, "ok").unwrap();
    }
    for k in [0, 64, 4160, 32768 + 128] {
        assert!(check_w4a4_k(k, "bad").is_err(), "{k}");
    }
}

fn proj(input_scale: Option<f32>) -> Nvfp4Proj {
    Nvfp4Proj {
        packed: DevicePtr::NULL,
        scale: DevicePtr::NULL,
        scale_2: 1.0,
        input_scale,
    }
}

fn expert(gu: Option<f32>, up: Option<f32>, down: Option<f32>) -> Glm5NextExpertWeights {
    Glm5NextExpertWeights {
        gate_proj: proj(gu),
        up_proj: proj(up),
        down_proj: proj(down),
    }
}

/// 2026-10-08: Experts that share one pair of scales give that pair; a missing scale, a gate/up
/// mismatch or a second distinct pair is refused.
#[test]
fn routed_experts_must_share_one_pair_of_scales() {
    let a = expert(Some(0.0007), Some(0.0007), Some(0.0142));
    let got = expert_act_scales(&[a, a, a]).unwrap();
    assert_eq!(
        got,
        W4a4ActScales {
            gate_up: 0.0007,
            down: 0.0142
        }
    );
    assert!(experts_have_scales(&[a, a]));

    let missing = expert(Some(0.0007), Some(0.0007), None);
    assert!(!experts_have_scales(&[a, missing]));
    let e = expert_act_scales(&[a, missing]).unwrap_err().to_string();
    assert!(e.contains("no input_scale"), "{e}");

    let crossed = expert(Some(0.0007), Some(0.0008), Some(0.0142));
    let e = expert_act_scales(&[crossed]).unwrap_err().to_string();
    assert!(e.contains("must be equal"), "{e}");

    let other = expert(Some(0.0007), Some(0.0007), Some(0.02));
    let e = expert_act_scales(&[a, other]).unwrap_err().to_string();
    assert!(e.contains("per-expert"), "{e}");

    assert!(expert_act_scales(&[]).is_err());
}
