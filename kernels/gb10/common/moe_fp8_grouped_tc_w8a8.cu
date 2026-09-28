// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: W8A8 twin of moe_fp8_grouped_tc.cu, the checkpoint's declared scheme for a
// block-FP8 checkpoint (activation_scheme "dynamic"): activations are quantized to E4M3 per
// (row, 128-K group) and the FP8 weights go into mma.sync m16n8k32 E4M3 tiles with no decode.
// The SiLU product is re-quantized per (row, 128 intermediate columns) before the down
// projection. Each 128-K block's FP32 partial sum is scaled by weight_scale * act_scale.
//
// The weight-quantization policy selects it (the MoE layer's published expert activation
// format, moe/fp8_grouped_tc_w8a8.rs); with nothing published the experts stay W8A16.
// Standalone at the Qwen3.6-35B-A3B shape (dgx3): 7-10% fewer GPU-rail joules per layer than
// the W8A16 tensor-core kernels at the same bytes per second; relative L2 error on random data
// 4e-2 (the E4M3 activation rounding) vs 2e-3 for W8A16.
//
// Owner: gb10 kernels.
// Invariants:
// - Weights: row-major [N, K] E4M3 with FP32 block scales [N / 128, K / 128]; K and N are
//   multiples of 128 (the host checks). moe_act_quant_e4m3 writes Xq [T, K] E4M3 and Xs
//   [T, K / 128] FP32: s = max(amax / 448, 1e-12), q = E4M3(clamp(x / s)).
// - Inside a 64-wide K chunk lane t = lane & 3 holds K = 16t .. 16t + 15 of its weight and
//   activation rows (one 16-byte load each); MMA j (0, 1) takes K = 16t + 8j .. + 3 as slots
//   4t .. 4t + 3 and K = 16t + 8j + 4 .. + 7 as slots 4t + 16 .. 4t + 19.
// - A row's arithmetic, including its activation quantization, reads only that row, so its
//   output bits do not depend on the other rows (moe_fp8_grouped_tc_rows.cuh block rows).
// - The gate+up block covers TC8_GU_COLS = 128 columns, one quantization group of the down
//   projection's K, and writes act_q [pos, N] E4M3 and act_s [pos, N / 128].
// - Grids: quant (T), block 128; gate+up (N / TC8_GU_COLS, cap + S), down
//   (N / TC8_DOWN_COLS, cap + S), block TC8_THREADS. TC8_* must equal FP8_GROUPED_TC_W8A8_*
//   in fp8_moe_grouped_tc_w8a8.rs.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "moe_fp8_grouped_tc_rows.cuh"

#define TC8_WARPS 4
#define TC8_THREADS (TC8_WARPS * 32)
#define TC8_GU_MT 2
#define TC8_DOWN_MT 4
#define TC8_G 2
#define TC8_GU_COLS (TC8_WARPS * 16 * TC8_GU_MT)
#define TC8_DOWN_COLS (TC8_WARPS * 16 * TC8_DOWN_MT)

__device__ __forceinline__ void tc8_mma_e4m3(float* c, unsigned int a0, unsigned int a1, unsigned int a2,
                                             unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2026-09-28: Per (row, 128-K group) E4M3 quantization; one block per row, warp w takes groups
// w, w + 4, ...
extern "C" __global__ void __launch_bounds__(128) moe_act_quant_e4m3(
    const __nv_bfloat16* __restrict__ X, unsigned char* __restrict__ Q, float* __restrict__ S, unsigned int K
) {
    const unsigned int row = blockIdx.x, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (unsigned int gi = warp; gi < K / 128; gi += 4) {
        const __nv_bfloat16* p = X + (unsigned long long)row * K + gi * 128 + lane * 4;
        float v[4];
        #pragma unroll
        for (int i = 0; i < 4; i++) v[i] = __bfloat162float(p[i]);
        float am = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        const float s = fmaxf(am / 448.0f, 1e-12f);
        const __nv_fp8x2_storage_t lo =
            __nv_cvt_float2_to_fp8x2(make_float2(v[0] / s, v[1] / s), __NV_SATFINITE, __NV_E4M3);
        const __nv_fp8x2_storage_t hi =
            __nv_cvt_float2_to_fp8x2(make_float2(v[2] / s, v[3] / s), __NV_SATFINITE, __NV_E4M3);
        *(unsigned int*)(Q + (unsigned long long)row * K + gi * 128 + lane * 4) = (unsigned int)lo | ((unsigned int)hi << 16);
        if (lane == 0) S[(unsigned long long)row * (K / 128) + gi] = s;
    }
}

// 2026-09-28: One warp's MT output tiles (GATE_UP: MT gate then MT up tiles of the same columns)
// over rows [begin, end), W8A8. GATE_UP writes E4M3(SiLU(bf16(g)) * bf16(u) / s) with s from the
// block's 128 columns; otherwise the BF16 projection.
template <bool GATE_UP, int MT>
__device__ __forceinline__ void tc8_warp(
    const unsigned char* __restrict__ Xq, const float* __restrict__ Xs,
    const int* __restrict__ sorted_token_ids, bool by_pos, unsigned int begin, unsigned int end,
    const unsigned char* __restrict__ W0, const float* __restrict__ S0,
    const unsigned char* __restrict__ W1, const float* __restrict__ S1,
    __nv_bfloat16* __restrict__ out, unsigned char* __restrict__ out_q, float* __restrict__ out_s,
    unsigned int N, unsigned int K, unsigned int f0
) {
    constexpr int TILES = GATE_UP ? 2 * MT : MT;
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int kblocks = K / 128, ngroups = (K / 64) / TC8_G;
    __shared__ float s_amax[TC8_WARPS][TC_ROWS];
    const unsigned char* wr[TILES][2];
    const float* sr[TILES];
    #pragma unroll
    for (int m = 0; m < TILES; m++) {
        const bool second = GATE_UP && m >= MT;
        const unsigned int col = f0 + 16 * (m % MT);
        const unsigned char* W = second ? W1 : W0;
        wr[m][0] = W + (unsigned long long)(col + g) * K + t * 16;
        wr[m][1] = W + (unsigned long long)(col + g + 8) * K + t * 16;
        sr[m] = (second ? S1 : S0) + (col / 128) * kblocks;
    }

    for (unsigned int row0 = begin; row0 < end; row0 += TC_ROWS) {
        const unsigned int cnt = min((unsigned int)TC_ROWS, end - row0);
        const bool live = g < cnt;
        auto row_of = [&](unsigned int r) {
            return by_pos ? row0 + r : (unsigned int)sorted_token_ids[row0 + r];
        };
        const unsigned int xrow = live ? row_of(g) : 0u;
        const unsigned int r0 = 2 * t, r1 = 2 * t + 1;
        const unsigned int srow0 = r0 < cnt ? row_of(r0) : 0u, srow1 = r1 < cnt ? row_of(r1) : 0u;
        const uint4* xp = (const uint4*)(Xq + (unsigned long long)xrow * K + t * 16);
        float acc[TILES][4], tmp[TILES][4];
        #pragma unroll
        for (int m = 0; m < TILES; m++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[m][e] = tmp[m][e] = 0.f;
        uint4 wn[TC8_G][TILES][2];
        #pragma unroll
        for (int c = 0; c < TC8_G; c++)
            #pragma unroll
            for (int m = 0; m < TILES; m++) {
                wn[c][m][0] = *(const uint4*)(wr[m][0] + c * 64);
                wn[c][m][1] = *(const uint4*)(wr[m][1] + c * 64);
            }
        for (unsigned int gi = 0; gi < ngroups; gi++) {
            uint4 w[TC8_G][TILES][2];
            #pragma unroll
            for (int c = 0; c < TC8_G; c++)
                #pragma unroll
                for (int m = 0; m < TILES; m++) { w[c][m][0] = wn[c][m][0]; w[c][m][1] = wn[c][m][1]; }
            if (gi + 1 < ngroups) {
                #pragma unroll
                for (int c = 0; c < TC8_G; c++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        wn[c][m][0] = *(const uint4*)(wr[m][0] + ((gi + 1) * TC8_G + c) * 64);
                        wn[c][m][1] = *(const uint4*)(wr[m][1] + ((gi + 1) * TC8_G + c) * 64);
                    }
            }
            #pragma unroll
            for (int c = 0; c < TC8_G; c++) {
                const unsigned int chunk = gi * TC8_G + c;
                const uint4 xa = live ? xp[chunk * 4] : make_uint4(0u, 0u, 0u, 0u);
                #pragma unroll
                for (int m = 0; m < TILES; m++) {
                    tc8_mma_e4m3(tmp[m], w[c][m][0].x, w[c][m][1].x, w[c][m][0].y, w[c][m][1].y, xa.x, xa.y);
                    tc8_mma_e4m3(tmp[m], w[c][m][0].z, w[c][m][1].z, w[c][m][0].w, w[c][m][1].w, xa.z, xa.w);
                }
                // 2026-09-28: Chunks 2kb and 2kb + 1 make 128-K block kb.
                if (chunk & 1) {
                    const unsigned int kb = chunk >> 1;
                    const float as0 = r0 < cnt ? Xs[(unsigned long long)srow0 * kblocks + kb] : 0.f;
                    const float as1 = r1 < cnt ? Xs[(unsigned long long)srow1 * kblocks + kb] : 0.f;
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        const float s = sr[m][kb];
                        acc[m][0] += tmp[m][0] * (s * as0);
                        acc[m][1] += tmp[m][1] * (s * as1);
                        acc[m][2] += tmp[m][2] * (s * as0);
                        acc[m][3] += tmp[m][3] * (s * as1);
                        #pragma unroll
                        for (int e = 0; e < 4; e++) tmp[m][e] = 0.f;
                    }
                }
            }
        }
        // 2026-09-28: acc[m][e]: column f0 + 16 (m % MT) + g (+ 8 for e >= 2) of row 2t + (e & 1).
        if (GATE_UP) {
            float h[MT][4];
            float am[2] = {0.f, 0.f};
            #pragma unroll
            for (int m = 0; m < MT; m++)
                #pragma unroll
                for (int e = 0; e < 4; e++) {
                    const float gv = __bfloat162float(__float2bfloat16(acc[m][e]));
                    const float uv = __bfloat162float(__float2bfloat16(acc[m + MT][e]));
                    h[m][e] = (gv / (1.0f + __expf(-gv))) * uv;
                    am[e & 1] = fmaxf(am[e & 1], fabsf(h[m][e]));
                }
            #pragma unroll
            for (int q = 0; q < 2; q++)
                #pragma unroll
                for (int o = 4; o < 32; o <<= 1) am[q] = fmaxf(am[q], __shfl_xor_sync(0xffffffffu, am[q], o));
            __syncthreads();
            if (g == 0) { s_amax[warp][r0] = am[0]; s_amax[warp][r1] = am[1]; }
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < 2; q++) {
                const unsigned int r = 2 * t + q;
                float bm = 0.f;
                #pragma unroll
                for (int w8 = 0; w8 < TC8_WARPS; w8++) bm = fmaxf(bm, s_amax[w8][r]);
                if (r >= cnt) continue;
                const unsigned long long pos = row0 + r;
                const float s = fmaxf(bm / 448.0f, 1e-12f);
                #pragma unroll
                for (int m = 0; m < MT; m++)
                    #pragma unroll
                    for (int hh = 0; hh < 2; hh++)
                        out_q[pos * N + f0 + 16 * m + g + 8 * hh] =
                            (unsigned char)__nv_cvt_float_to_fp8(h[m][2 * hh + q] / s, __NV_SATFINITE, __NV_E4M3);
                if (warp == 0 && g == 0) out_s[pos * (N / 128) + blockIdx.x] = s;
            }
        } else {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int r = 2 * t + (e & 1);
                if (r >= cnt) continue;
                #pragma unroll
                for (int m = 0; m < TILES; m++)
                    out[(unsigned long long)(row0 + r) * N + f0 + 16 * m + g + ((e >> 1) ? 8 : 0)] =
                        __float2bfloat16(acc[m][e]);
            }
        }
    }
}

// 2026-09-28: Gate+up and SiLU of the routed experts and the shared expert from the quantized
// layer input (moe_act_quant_e4m3). act_q/act_s routed by sorted position, sh_q/sh_s shared by
// token. A null routed gate or up pointer makes that expert's act 0 (scale 1e-12).
extern "C" __global__ void __launch_bounds__(TC8_THREADS) moe_expert_gate_up_act_fp8_grouped_tc_w8a8(
    const unsigned char* __restrict__ Xq, const float* __restrict__ Xs,
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    unsigned char* __restrict__ act_q, float* __restrict__ act_s,
    const int* __restrict__ expert_offsets, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts, const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_weight, const float* __restrict__ sh_gate_block_scale,
    const unsigned char* __restrict__ sh_up_weight, const float* __restrict__ sh_up_block_scale,
    unsigned char* __restrict__ sh_q, float* __restrict__ sh_s,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC8_GU_COLS + (threadIdx.x >> 5) * 16 * TC8_GU_MT;
    if (is_shared) {
        tc8_warp<true, TC8_GU_MT>(Xq, Xs, sorted_token_ids, true, begin, end, sh_gate_weight,
                                  sh_gate_block_scale, sh_up_weight, sh_up_block_scale, nullptr, sh_q, sh_s,
                                  N, K, f0);
        return;
    }
    const unsigned char* Wg = (const unsigned char*)gate_weight_ptrs[expert];
    const unsigned char* Wu = (const unsigned char*)up_weight_ptrs[expert];
    if (Wg == 0 || Wu == 0) {
        for (unsigned int pos = begin; pos < end; pos++) {
            for (unsigned int i = threadIdx.x; i < TC8_GU_COLS; i += TC8_THREADS)
                act_q[(unsigned long long)pos * N + blockIdx.x * TC8_GU_COLS + i] = 0;
            if (threadIdx.x == 0) act_s[(unsigned long long)pos * (N / 128) + blockIdx.x] = 1e-12f;
        }
        return;
    }
    tc8_warp<true, TC8_GU_MT>(Xq, Xs, sorted_token_ids, false, begin, end, Wg,
                              (const float*)gate_block_scale_ptrs[expert], Wu,
                              (const float*)up_block_scale_ptrs[expert], nullptr, act_q, act_s, N, K, f0);
}

// 2026-09-28: Down projection of the quantized SiLU products into C [pos, N] and sh_down_out
// [token, N] BF16. A null routed down pointer zeroes that expert's rows.
extern "C" __global__ void __launch_bounds__(TC8_THREADS) moe_expert_down_act_fp8_grouped_tc_w8a8(
    const unsigned char* __restrict__ act_q, const float* __restrict__ act_s,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_q, const float* __restrict__ sh_s,
    const unsigned char* __restrict__ sh_down_weight, const float* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC8_DOWN_COLS + (threadIdx.x >> 5) * 16 * TC8_DOWN_MT;
    if (is_shared) {
        tc8_warp<false, TC8_DOWN_MT>(sh_q, sh_s, nullptr, true, begin, end, sh_down_weight, sh_down_block_scale,
                                     nullptr, nullptr, sh_down_out, nullptr, nullptr, N, K, f0);
        return;
    }
    const unsigned char* W = (const unsigned char*)weight_ptrs[expert];
    if (W == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < TC8_DOWN_COLS; i += TC8_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * TC8_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    tc8_warp<false, TC8_DOWN_MT>(act_q, act_s, nullptr, true, begin, end, W, (const float*)block_scale_ptrs[expert],
                                 nullptr, nullptr, C, nullptr, nullptr, N, K, f0);
}
