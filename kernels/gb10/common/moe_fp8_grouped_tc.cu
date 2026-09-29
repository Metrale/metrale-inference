// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: Tensor-core grouped FP8 MoE decode: the gate+up (SiLU product) and down
// projections of moe_shared_expert_fused_fp8_grouped.cu on mma.sync m16n8k16 BF16 tiles,
// FP32 accumulation. Same grid contract (one block row per active expert, the first
// ceil(num_tokens / TC_ROWS) block rows the shared expert), same weights and scales, same
// outputs by sorted position.
//
// Why: on GB10 the CUDA-core kernels spend their power on the per-byte E4M3 table lookup,
// the scale multiply and the per-row FP32 products; here a weight byte costs a byte permute
// and two logic ops, and the products run on the tensor cores. Standalone at the
// Qwen3.6-35B-A3B shape (256 experts, top-8, uniform routing, dgx3): the same bytes per
// second within 1-5% and 30-45% fewer GPU-rail joules per layer (M = 1..64).
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are row-major [N, K] FP8 E4M3 with FP32 block scales [N / 128, K / 128]; K and
//   N are multiples of 128 (the host checks `fp8_grouped_tc_shape_ok`).
// - A weight byte b becomes the BF16 whose bits are sign(b) | (b & 0x7F) << 4, which equals
//   E4M3(b) * 2^-120 exactly (normals and subnormals). Activations are multiplied by 2^60
//   (exact), and the 128-K block scale by 2^60, so every product and partial sum stays an
//   FP32 normal: products are E4M3 * A * 2^-60.
// - 2026-09-28: gate+up computes the FP32 SiLU product a and stores it as two BF16 terms of
//   a * 2^60, hi = BF16(a * 2^60) and lo = BF16(a * 2^60 - hi), in the SiLU buffer's FP32 row
//   space: row r holds N hi values then N lo values (2N BF16 = N FP32). The down projection
//   runs one MMA on each (hi first), so the product keeps about 16 bits of its mantissa where
//   one BF16 would keep 8, and the split is paid once per element, not once per down warp.
// - Inside a 64-wide K chunk, lane t = lane & 3 holds K = 16t .. 16t + 15 of its rows,
//   weights and activations alike (one 16-byte load each), and MMA j (0..3) takes
//   K = 16t + 4j + {0,1} as fragment slots 2t, 2t+1 and K = 16t + 4j + {2,3} as 2t+8, 2t+9.
//   The K order of a row's sum is therefore fixed by K alone.
// - Rows run TC_ROWS at a time as the MMA's N columns. Column r of an MMA depends only on
//   row r's activations, so a row's output bits do not depend on which rows share its
//   launch, its pass or its expert (padding rows are zero).
// - Grids: gate+up (N / TC_GU_COLS, cap + S), down (N / TC_DOWN_COLS, cap + S), block
//   TC_THREADS, S = ceil(num_tokens / TC_ROWS). TC_* must equal FP8_GROUPED_TC_* in
//   fp8_moe_grouped.rs.

#include <cuda_bf16.h>

#include "moe_fp8_grouped_tc_rows.cuh"

#define TC_WARPS 4
#define TC_THREADS (TC_WARPS * 32)
// 2026-09-28: m-tiles (16 output columns) per warp and 64-K chunks per load group. A warp
// keeps 2 groups in flight (prefetch); gate+up tiles are the gate and up rows of the same
// columns, so its 16 columns feed the SiLU product in registers.
#define TC_GU_MT 1
#define TC_DOWN_MT 2
#define TC_G 2
#define TC_GU_COLS (TC_WARPS * 16 * TC_GU_MT)
#define TC_DOWN_COLS (TC_WARPS * 16 * TC_DOWN_MT)

// 2026-09-28: E4M3 bytes 0,1 (sel 0x1404) or 2,3 (sel 0x3424) of w as a BF16 pair, each
// E4M3 * 2^-120.
__device__ __forceinline__ unsigned int tc_e4m3_pair_bf16(unsigned int w, unsigned int sel) {
    const unsigned int x = __byte_perm(w, 0u, sel);
    return (x & 0x80008000u) | ((x >> 4) & 0x07F007F0u);
}

__device__ __forceinline__ void tc_mma_bf16(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-09-28: One warp's MT output tiles (GATE_UP: MT gate tiles then MT up tiles of the same
// columns) over rows [begin, end). Input row r is X[row(r)], row(r) = sorted_token_ids[r]
// unless by_pos: BF16 [.., K] for gate+up, the hi|lo SiLU rows [.., 2K] for down. GATE_UP
// writes the hi|lo split of SiLU(g) * u * 2^60, g and u rounded to BF16 first (the FP8
// kernels' rounding); otherwise the BF16 projection.
// 2026-09-29: RG row groups of TC_ROWS share each decoded weight fragment (one pass reads
// the weights once for TC_ROWS * RG rows). Group q is its own MMA column block, so a row's
// products, their K order and its 128-block scaling are those of RG = 1: its bits do not
// depend on RG, on the other rows of its pass or on how many passes the expert takes.
template <bool GATE_UP, int MT, int RG>
__device__ __forceinline__ void tc_warp(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end,
    const unsigned char* __restrict__ W0, const float* __restrict__ S0,
    const unsigned char* __restrict__ W1, const float* __restrict__ S1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    constexpr int TILES = GATE_UP ? 2 * MT : MT;
    const unsigned int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int kblocks = K / 128, ngroups = (K / 64) / TC_G;
    const float two60 = 1152921504606846976.0f;
    const __nv_bfloat162 two60x2 = __floats2bfloat162_rn(two60, two60);
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

    for (unsigned int row0 = begin; row0 < end; row0 += TC_ROWS * RG) {
        unsigned int cnt[RG];
        bool live[RG];
        const uint4* xp[RG];
        const uint4* xh[RG];
        const uint4* xlo[RG];
        #pragma unroll
        for (int q = 0; q < RG; q++) {
            const unsigned int r0 = row0 + q * TC_ROWS;
            cnt[q] = r0 < end ? min((unsigned int)TC_ROWS, end - r0) : 0u;
            live[q] = g < cnt[q];
            const unsigned int xrow = live[q] ? (by_pos ? r0 + g : (unsigned int)sorted_token_ids[r0 + g]) : 0u;
            xp[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * K + t * 16);
            // 2026-09-28: The down input's hi and lo rows (already times 2^60).
            xh[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * 2 * K + t * 16);
            xlo[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * 2 * K + K + t * 16);
        }
        float acc[RG][TILES][4], tmp[RG][TILES][4];
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int m = 0; m < TILES; m++)
                #pragma unroll
                for (int e = 0; e < 4; e++) acc[q][m][e] = tmp[q][m][e] = 0.f;
        uint4 wn[TC_G][TILES][2];
        #pragma unroll
        for (int c = 0; c < TC_G; c++)
            #pragma unroll
            for (int m = 0; m < TILES; m++) {
                wn[c][m][0] = *(const uint4*)(wr[m][0] + c * 64);
                wn[c][m][1] = *(const uint4*)(wr[m][1] + c * 64);
            }
        for (unsigned int gi = 0; gi < ngroups; gi++) {
            uint4 w[TC_G][TILES][2];
            #pragma unroll
            for (int c = 0; c < TC_G; c++)
                #pragma unroll
                for (int m = 0; m < TILES; m++) { w[c][m][0] = wn[c][m][0]; w[c][m][1] = wn[c][m][1]; }
            if (gi + 1 < ngroups) {
                #pragma unroll
                for (int c = 0; c < TC_G; c++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        wn[c][m][0] = *(const uint4*)(wr[m][0] + ((gi + 1) * TC_G + c) * 64);
                        wn[c][m][1] = *(const uint4*)(wr[m][1] + ((gi + 1) * TC_G + c) * 64);
                    }
            }
            #pragma unroll
            for (int c = 0; c < TC_G; c++) {
                const unsigned int chunk = gi * TC_G + c;
                // 2026-09-28: xw word i holds K = 16t + 2i, 2i + 1 (times 2^60); for the down
                // input, xl holds the lo terms.
                unsigned int xw[RG][8], xl[RG][8];
                #pragma unroll
                for (int q = 0; q < RG; q++) {
                    if (GATE_UP) {
                        uint4 xa = make_uint4(0u, 0u, 0u, 0u), xb = make_uint4(0u, 0u, 0u, 0u);
                        if (live[q]) { xa = xp[q][chunk * 8]; xb = xp[q][chunk * 8 + 1]; }
                        const unsigned int raw[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
                        #pragma unroll
                        for (int i = 0; i < 8; i++) {
                            __nv_bfloat162 v = *(const __nv_bfloat162*)&raw[i];
                            v = __hmul2(v, two60x2);
                            xw[q][i] = *(unsigned int*)&v;
                        }
                    } else {
                        uint4 ha = make_uint4(0u, 0u, 0u, 0u), hb = ha, la = ha, lb = ha;
                        if (live[q]) { ha = xh[q][chunk * 8]; hb = xh[q][chunk * 8 + 1]; la = xlo[q][chunk * 8]; lb = xlo[q][chunk * 8 + 1]; }
                        const unsigned int h8[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
                        const unsigned int l8[8] = {la.x, la.y, la.z, la.w, lb.x, lb.y, lb.z, lb.w};
                        #pragma unroll
                        for (int i = 0; i < 8; i++) { xw[q][i] = h8[i]; xl[q][i] = l8[i]; }
                    }
                }
                #pragma unroll
                for (int j = 0; j < 4; j++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        const uint4& lo = w[c][m][0];
                        const uint4& hi = w[c][m][1];
                        const unsigned int wg = (j == 0) ? lo.x : (j == 1) ? lo.y : (j == 2) ? lo.z : lo.w;
                        const unsigned int wh = (j == 0) ? hi.x : (j == 1) ? hi.y : (j == 2) ? hi.z : hi.w;
                        unsigned int a[4];
                        a[0] = tc_e4m3_pair_bf16(wg, 0x1404u);
                        a[1] = tc_e4m3_pair_bf16(wh, 0x1404u);
                        a[2] = tc_e4m3_pair_bf16(wg, 0x3424u);
                        a[3] = tc_e4m3_pair_bf16(wh, 0x3424u);
                        #pragma unroll
                        for (int q = 0; q < RG; q++) {
                            tc_mma_bf16(tmp[q][m], a, xw[q][2 * j], xw[q][2 * j + 1]);
                            if (!GATE_UP) tc_mma_bf16(tmp[q][m], a, xl[q][2 * j], xl[q][2 * j + 1]);
                        }
                    }
                // 2026-09-28: Chunks 2kb and 2kb + 1 make 128-K block kb: scale it once.
                if (chunk & 1) {
                    const unsigned int kb = chunk >> 1;
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        const float s = sr[m][kb] * two60;
                        #pragma unroll
                        for (int q = 0; q < RG; q++)
                            #pragma unroll
                            for (int e = 0; e < 4; e++) { acc[q][m][e] += tmp[q][m][e] * s; tmp[q][m][e] = 0.f; }
                    }
                }
            }
        }
        // 2026-09-28: acc[q][m][e]: column f + g (+ 8 for e >= 2) of row 2t + (e & 1) of group q.
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int r = 2 * t + (e & 1);
                if (r >= cnt[q]) continue;
                const unsigned int row = row0 + q * TC_ROWS + r;
                const unsigned long long o = (unsigned long long)row * N + f0 + g + ((e >> 1) ? 8 : 0);
                #pragma unroll
                for (int m = 0; m < MT; m++) {
                    if (GATE_UP) {
                        const float gv = __bfloat162float(__float2bfloat16(acc[q][m][e]));
                        const float uv = __bfloat162float(__float2bfloat16(acc[q][m + MT][e]));
                        const float hv = (gv / (1.0f + __expf(-gv))) * uv * two60;
                        const __nv_bfloat16 hi = __float2bfloat16(hv);
                        const unsigned long long ro = (unsigned long long)row * 2 * N + f0 + g + ((e >> 1) ? 8 : 0);
                        ((__nv_bfloat16*)out)[ro + 16 * m] = hi;
                        ((__nv_bfloat16*)out)[ro + N + 16 * m] = __float2bfloat16(hv - __bfloat162float(hi));
                    } else {
                        ((__nv_bfloat16*)out)[o + 16 * m] = __float2bfloat16(acc[q][m][e]);
                    }
                }
            }
    }
}

// 2026-09-29: tc_warp for a routed expert's rows: RG = 2 (one weight pass per 16 rows) when the
// expert has more than TC_ROWS rows, else RG = 1. The shared expert's block rows hold at most
// TC_ROWS rows and keep RG = 1.
template <bool GATE_UP, int MT>
__device__ __forceinline__ void tc_warp_routed(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end,
    const unsigned char* __restrict__ W0, const float* __restrict__ S0,
    const unsigned char* __restrict__ W1, const float* __restrict__ S1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    if (end - begin > TC_ROWS)
        tc_warp<GATE_UP, MT, 2>(X, sorted_token_ids, by_pos, begin, end, W0, S0, W1, S1, out, N, K, f0);
    else
        tc_warp<GATE_UP, MT, 1>(X, sorted_token_ids, by_pos, begin, end, W0, S0, W1, S1, out, N, K, f0);
}

// 2026-09-28: Gate+up and SiLU of the routed experts and the shared expert. A: [num_tokens, K]
// BF16. act: routed hi|lo rows [pos, 2N] BF16 by sorted position; sh_act: shared hi|lo rows
// [token, 2N]; both in FP32-sized buffers.
// A null routed gate or up pointer makes that expert's act 0.
extern "C" __global__ void __launch_bounds__(TC_THREADS) moe_expert_gate_up_act_fp8_grouped_tc(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_weight,
    const float* __restrict__ sh_gate_block_scale,
    const unsigned char* __restrict__ sh_up_weight,
    const float* __restrict__ sh_up_block_scale,
    __nv_bfloat16* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC_GU_COLS + (threadIdx.x >> 5) * 16 * TC_GU_MT;
    if (is_shared) {
        tc_warp<true, TC_GU_MT, 1>(A, sorted_token_ids, true, begin, end, sh_gate_weight, sh_gate_block_scale,
                                sh_up_weight, sh_up_block_scale, sh_act, N, K, f0);
        return;
    }
    const unsigned char* Wg = (const unsigned char*)gate_weight_ptrs[expert];
    const unsigned char* Wu = (const unsigned char*)up_weight_ptrs[expert];
    if (Wg == 0 || Wu == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < TC_GU_COLS; i += TC_THREADS)
                for (unsigned int hl = 0; hl < 2; hl++)
                    act[(unsigned long long)pos * 2 * N + hl * N + blockIdx.x * TC_GU_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    tc_warp_routed<true, TC_GU_MT>(A, sorted_token_ids, false, begin, end, Wg,
                            (const float*)gate_block_scale_ptrs[expert], Wu,
                            (const float*)up_block_scale_ptrs[expert], act, N, K, f0);
}

// 2026-09-28: Down projection of the hi|lo SiLU rows (act routed by position, sh_act
// shared by token) into C [pos, N] and sh_down_out [token, N] BF16. A null routed down pointer
// zeroes that expert's rows.
extern "C" __global__ void __launch_bounds__(TC_THREADS) moe_expert_down_act_fp8_grouped_tc(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_weight,
    const float* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * TC_DOWN_COLS + (threadIdx.x >> 5) * 16 * TC_DOWN_MT;
    if (is_shared) {
        tc_warp<false, TC_DOWN_MT, 1>(sh_act, nullptr, true, begin, end, sh_down_weight, sh_down_block_scale,
                                   nullptr, nullptr, sh_down_out, N, K, f0);
        return;
    }
    const unsigned char* W = (const unsigned char*)weight_ptrs[expert];
    if (W == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < TC_DOWN_COLS; i += TC_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * TC_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    tc_warp_routed<false, TC_DOWN_MT>(act, nullptr, true, begin, end, W, (const float*)block_scale_ptrs[expert],
                               nullptr, nullptr, C, N, K, f0);
}
