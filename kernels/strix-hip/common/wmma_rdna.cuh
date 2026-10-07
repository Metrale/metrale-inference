// SPDX-License-Identifier: MIT OR Apache-2.0
// 2026-10-07: One WMMA 16x16x16 (wave32) interface for RDNA3.5 (gfx1151) and RDNA4
// (gfx1200 / gfx1201), so a strix-hip source compiles and runs on both.
//
// The two generations hold different fragment slices per lane (lane l, 0..31):
//   gfx11: A row (l & 15), K 0..15, the two half-waves duplicated; B column (l & 15), K 0..15;
//          accumulator element e -> row 2e + (l >> 4), column (l & 15).
//   gfx12: A row (l & 15), K 8(l >> 4) .. 8(l >> 4) + 7; B column (l & 15), same K slice;
//          accumulator element e -> row e + 8(l >> 4), column (l & 15).
// Verified on gfx1201 (RX 9070 XT) against a CPU reference: the gfx12 mapping matches 256/256,
// the gfx11 accumulator mapping on gfx12 mismatches 220/256.
//
// A kernel fills WMMA_FRAG_K elements starting at K = wmma_k0(lane) and stores accumulator
// element e at row wmma_acc_row(lane, e). On gfx11 these reduce to the original expressions
// (k0 = 0, 16 elements, row 2e + (l >> 4)), so the gfx1151 fragments are unchanged.
//
// Owner: strix-hip kernels.
#pragma once

typedef float wmma_v8f __attribute__((ext_vector_type(8)));

#if defined(__gfx1200__) || defined(__gfx1201__)
#define WMMA_FRAG_K 8
typedef __bf16 wmma_bf16x __attribute__((ext_vector_type(8)));
__device__ __forceinline__ int wmma_k0(int lane) {
    return 8 * (lane >> 4);
}
__device__ __forceinline__ int wmma_acc_row(int lane, int e) {
    return e + 8 * (lane >> 4);
}
__device__ __forceinline__ wmma_v8f wmma_bf16(wmma_bf16x a, wmma_bf16x b, wmma_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
#else
#define WMMA_FRAG_K 16
typedef __bf16 wmma_bf16x __attribute__((ext_vector_type(16)));
__device__ __forceinline__ int wmma_k0(int) {
    return 0;
}
__device__ __forceinline__ int wmma_acc_row(int lane, int e) {
    return 2 * e + (lane >> 4);
}
__device__ __forceinline__ wmma_v8f wmma_bf16(wmma_bf16x a, wmma_bf16x b, wmma_v8f c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#endif
