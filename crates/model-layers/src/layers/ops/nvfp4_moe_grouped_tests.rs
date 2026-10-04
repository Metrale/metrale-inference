// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-04: Host checks of the lean NVFP4 point (`Nvfp4G16Lean`, tc_weight_formats.cuh): the
//! bit layout its repack writes and its fragments read, and the repack's shape admission.
//!
//! Owner: model-layers ops.
//! Invariants: the layout is read from the kernel header, not restated.

use super::{NVFP4_LEAN_REPACK_SMEM_PER_K, nvfp4_lean_repack_shape_ok};

const FORMATS_CUH: &str = include_str!("../../../../../kernels/gb10/common/tc_weight_formats.cuh");

/// 2026-10-04: The four values `ntc_lean_<which>_rot(p)` returns for p = 0..3, read from the
/// header's `return p == 0 ? a : p == 1 ? b : p == 2 ? c : d;`.
fn rotations(which: &str) -> [u32; 4] {
    let head = format!("constexpr int ntc_lean_{which}_rot(int p) {{ return ");
    let line = FORMATS_CUH
        .lines()
        .find(|l| l.contains(&head))
        .unwrap_or_else(|| panic!("no ntc_lean_{which}_rot in tc_weight_formats.cuh"));
    let body = &line[line.find(&head).expect("found") + head.len()..];
    let v: Vec<u32> = body
        .split(['?', ':', ';'])
        .map(str::trim)
        .filter_map(|t| t.parse().ok())
        .collect();
    v.try_into()
        .unwrap_or_else(|v| panic!("ntc_lean_{which}_rot: {v:?}"))
}

/// 2026-10-04: The repack's placement of word `w`'s eight E2M1 values (element e in bits 4e..4e+3):
/// pair p's magnitudes at 6 + i + mag_rot(p) and 22 + i + mag_rot(p), its signs at 15 + sign_rot(p)
/// and 31 + sign_rot(p) (mod 32), as `ntc_lean_word` (moe_nvfp4_grouped_tc.cu) writes them.
fn place(w: u32, mag: [u32; 4], sign: [u32; 4]) -> u32 {
    let mut o = 0u32;
    for p in 0..4 {
        let (x, y) = ((w >> (8 * p)) & 0xF, (w >> (8 * p + 4)) & 0xF);
        for i in 0..3 {
            o |= ((x >> i) & 1) << ((6 + i + mag[p]) % 32);
            o |= ((y >> i) & 1) << ((22 + i + mag[p]) % 32);
        }
        o |= (x >> 3) << ((15 + sign[p]) % 32);
        o |= (y >> 3) << ((31 + sign[p]) % 32);
    }
    o
}

fn bf16_value(b: u32) -> f64 {
    let (s, e, m) = ((b >> 15) & 1, (b >> 7) & 0xFF, b & 0x7F);
    let v = if e == 0 {
        m as f64 / 128.0 * 2f64.powi(-126)
    } else {
        (1.0 + m as f64 / 128.0) * 2f64.powi(e as i32 - 127)
    };
    if s == 1 { -v } else { v }
}

/// 2026-10-04: Every bit of a word lands on one distinct bit, and pair p read back by the
/// fragment's rotate-and-mask (`ntc_lean_pair<P>`) is the BF16 pair E2M1 * 2^-126, sign bit
/// included, for all 256 values of the pair (the other six elements random). A layout edit that
/// overlaps two fields or misplaces one fails here before any GPU run.
#[test]
fn the_lean_layout_places_every_bit_once_and_reads_back_e2m1() {
    const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let (mag, sign) = (rotations("mag"), rotations("sign"));
    let mut seen = 0u64;
    for bit in 0..32 {
        let o = place(1 << bit, mag, sign);
        assert_eq!(o.count_ones(), 1, "input bit {bit} lands on {o:#010x}");
        seen |= u64::from(o);
    }
    assert_eq!(seen, 0xFFFF_FFFF, "the layout leaves bits unwritten");
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    for p in 0..4usize {
        for pair in 0..256u32 {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let w = ((rng >> 32) as u32 & !(0xFF << (8 * p))) | (pair << (8 * p));
            let o = place(w, mag, sign);
            let read =
                (o.rotate_right(mag[p]) & 0x01C0_01C0) | (o.rotate_right(sign[p]) & 0x8000_8000);
            for (h, nib) in [(0, pair & 0xF), (1, pair >> 4)] {
                let half = (read >> (16 * h)) & 0xFFFF;
                let want = E2M1[(nib & 7) as usize] * 2f64.powi(-126);
                assert_eq!(
                    bf16_value(half).abs(),
                    want,
                    "pair {p} value {pair:#04x} half {h}"
                );
                assert_eq!(
                    half >> 15,
                    nib >> 3,
                    "pair {p} value {pair:#04x} half {h}: sign"
                );
            }
        }
    }
}

/// 2026-10-04: The repack admits the Qwen3.6-35B-A3B expert shapes and refuses partial tiles,
/// partial 128-K chunks and a tile that overflows 48 KiB of shared memory.
#[test]
fn the_lean_repack_admits_whole_tiles_within_shared_memory() {
    assert!(nvfp4_lean_repack_shape_ok(512, 2048));
    assert!(nvfp4_lean_repack_shape_ok(2048, 512));
    assert!(!nvfp4_lean_repack_shape_ok(0, 2048));
    assert!(!nvfp4_lean_repack_shape_ok(520, 2048));
    assert!(!nvfp4_lean_repack_shape_ok(512, 2000));
    let widest = 48 * 1024 / NVFP4_LEAN_REPACK_SMEM_PER_K / 128 * 128;
    assert!(nvfp4_lean_repack_shape_ok(16, widest));
    assert!(!nvfp4_lean_repack_shape_ok(16, widest + 128));
}
