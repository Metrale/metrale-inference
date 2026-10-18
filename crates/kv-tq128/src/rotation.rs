// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The 128-group rotation: `x * signs1`, a 128-point fast Walsh-Hadamard transform,
//! then `* (1/sqrt(128)) * signs2` (arXiv:2504.19874 — TurboQuant rotation). It is orthogonal,
//! so the inverse is the same steps with the sign tables swapped. The sign tables are the
//! engine's `kernels/gb10/common/tq_plus_signs.cuh` (`TQP_SIGNS{1,2}_128`), which equal
//! spiritbuun/buun-llama-cpp's seed-42 tables (MIT; credit spiritbuun) entry for entry; a test
//! parses the header and compares.
//!
//! A wider head (256, 512) is rotated 128 elements at a time with the same tables: the
//! codebooks were trained on the coordinate distribution of exactly this 128-point transform,
//! and per-group independence is what lets a kernel read any group of any token on its own.
//! docs/kv-cache-codecs.md gives the full argument.
//!
//! Owner: kv-tq128.
//! Invariants: the butterfly order (h = 1, 2, .., 64; `v[j] = a + b`, `v[j + h] = a - b`) is the
//! order the kernels' warp transform computes, so results agree bit for bit.

/// 2026-10-18: `1/sqrt(128)` rounded to f32 (0x3DB504F3; the kernels write 0.08838834764831845f).
pub const INV_SQRT_128: f32 = f32::from_bits(0x3DB5_04F3);

/// 2026-10-18: The first sign table (applied before the transform when rotating).
pub const SIGNS1: [i8; 128] = [
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1,
    1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, 1, 1, 1, -1, -1, -1, -1,
    -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, 1, -1,
    -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, 1, 1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
];

/// 2026-10-18: The second sign table (applied after the transform when rotating).
pub const SIGNS2: [i8; 128] = [
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1,
    1, -1, 1, 1, 1, 1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, -1,
    -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
];

/// 2026-10-18: The unnormalized 128-point Walsh-Hadamard transform, in place.
pub fn fwht128(v: &mut [f32; 128]) {
    let mut h = 1;
    while h < 128 {
        for base in (0..128).step_by(2 * h) {
            for j in base..base + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

fn apply(v: &mut [f32; 128], first: &[i8; 128], last: &[i8; 128]) {
    for (x, s) in v.iter_mut().zip(first) {
        *x *= f32::from(*s);
    }
    fwht128(v);
    for (x, s) in v.iter_mut().zip(last) {
        *x *= INV_SQRT_128 * f32::from(*s);
    }
}

/// 2026-10-18: Rotate a group into the codebook's domain.
pub fn rotate(v: &mut [f32; 128]) {
    apply(v, &SIGNS1, &SIGNS2);
}

/// 2026-10-18: Undo [`rotate`] (up to rounding).
pub fn unrotate(v: &mut [f32; 128]) {
    apply(v, &SIGNS2, &SIGNS1);
}
