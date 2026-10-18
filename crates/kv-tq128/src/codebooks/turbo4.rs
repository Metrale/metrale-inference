// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The turbo4_n128 scalar codebook: 16 Lloyd-Max levels for a unit-norm 128-vector
//! after the rotation (coordinates near N(0, 1/128)), and the 15 decision midpoints. One book
//! serves K and V. Copied verbatim from spiritbuun/buun-llama-cpp (MIT, Copyright (c) 2023-2026
//! The ggml authors), `ggml/src/ggml-cuda/turbo-quant-cuda.cuh` at 34ef0ef7c6ea:
//! `d_turbo_centroids_4bit` and `d_turbo_mid_4bit` (upstream ships the midpoints as literals,
//! not as computed means, so they are copied too). Credit spiritbuun.
//! Rotation and norm: arXiv:2504.19874 (TurboQuant).
//!
//! Owner: kv-tq128.
//! Invariants: ascending; `TURBO4_MID[i]` lies between `TURBO4_LEVELS[i]` and `[i + 1]`.

/// 2026-10-18: The 16 reconstruction levels, ascending.
pub const TURBO4_LEVELS: [f32; 16] = [
    -0.241556, -0.182907, -0.143047, -0.111065, -0.083317, -0.058069, -0.034311, -0.011353,
    0.011353, 0.034311, 0.058069, 0.083317, 0.111065, 0.143047, 0.182907, 0.241556,
];

/// 2026-10-18: The 15 decision midpoints: a value at or above `TURBO4_MID[i]` maps above level `i`.
pub const TURBO4_MID: [f32; 15] = [
    -0.212232, -0.162977, -0.127056, -0.097191, -0.070693, -0.046190, -0.022832, 0.000000,
    0.022832, 0.046190, 0.070693, 0.097191, 0.127056, 0.162977, 0.212232,
];
