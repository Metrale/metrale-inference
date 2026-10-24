// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: Pick the r-th of sixteen scalar kernel arguments, for the "rows" kernels that take
// one state pointer and one workspace row per row of a launch as separate arguments
// (`causal_conv1d_update_l2norm_rows`, `kda_recurrent_decode_bf16_smem_rows`,
// `kda_recurrent_decode_bf16_rows_reg`).
//
// Owner: gb10 kernels.
// Invariants:
// - Returns a<r> for r < 16 and a15 otherwise; callers check r < 16 first.
// - Compiles to a chain of selects over the parameter bank. Indexing a local array built from
//   the arguments (`const T a[16] = {a0, ...}; a[r]`) instead makes every thread spill all
//   sixteen to local memory first (a 192-byte stack frame and 12 stores per thread for the
//   pointer and row arrays together), which measured slower than one launch per row for the
//   small conv rows kernel. Only addresses are picked, so no row's arithmetic changes.

#pragma once

template <typename T>
__device__ __forceinline__ T rows_pick16(
    unsigned int r,
    T a0, T a1, T a2, T a3, T a4, T a5, T a6, T a7,
    T a8, T a9, T a10, T a11, T a12, T a13, T a14, T a15
) {
    T v = a0;
    v = (r == 1u) ? a1 : v;
    v = (r == 2u) ? a2 : v;
    v = (r == 3u) ? a3 : v;
    v = (r == 4u) ? a4 : v;
    v = (r == 5u) ? a5 : v;
    v = (r == 6u) ? a6 : v;
    v = (r == 7u) ? a7 : v;
    v = (r == 8u) ? a8 : v;
    v = (r == 9u) ? a9 : v;
    v = (r == 10u) ? a10 : v;
    v = (r == 11u) ? a11 : v;
    v = (r == 12u) ? a12 : v;
    v = (r == 13u) ? a13 : v;
    v = (r == 14u) ? a14 : v;
    v = (r >= 15u) ? a15 : v;
    return v;
}
