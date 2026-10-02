// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: Tensor-core grouped NVFP4 MoE decode (W4A16): the gate+up (SiLU product) and down
// projections of moe_nvfp4_grouped.cu on mma.sync m16n8k16 BF16 tiles, FP32 accumulation. Same
// grid contract as moe_fp8_grouped_tc.cu (one block row per active expert, the first
// ceil(num_tokens / TC_ROWS) block rows the shared expert), same weights and kernel arguments
// as moe_nvfp4_grouped.cu.
//
// Why: the CUDA-core kernels decode E2M1 through a shared-memory table and run one FP32 FMA per
// weight per row, so at 8+ rows they are compute bound and draw CUDA-core power; here a weight
// costs a few byte permutes and one exact BF16 multiply, and the products run on the tensor
// cores (memory: bf16-mma-power-scales-with-live-rows, moe-energy-gap-vs-vllm).
//
// Owner: gb10 kernels.
// Invariants:
// - Weights are row-major NVFP4: packed E2M1 [N, K / 2] (element 2j in the low nibble of byte
//   j), E4M3 block scales [N, K / 16], a per-tensor FP32 scale s2 (per expert for the routed
//   tables). K % 256 == 0 and N % 16 == 0 (the host checks `nvfp4_grouped_tc_shape_ok`).
// - A weight enters the MMA as the BF16 E2M1 * E4M3: exact, since the product has at most five
//   significant bits and lies in [2^-10, 2688]. s2 multiplies the FP32 sum once per output.
//   The products and sums are therefore those of the declared W4A16 arithmetic up to FP32
//   summation order.
// - Inside a 128-wide K chunk lane t = lane & 3 holds K = 32t .. 32t + 31 of its rows (one
//   16-byte weight load per row, activations one uint4 per 8 K), and MMA i (0..7) takes
//   K = 32t + 4i + {0,1} as fragment slots 2t, 2t+1 and K = 32t + 4i + {2,3} as 2t+8, 2t+9.
//   The K order of a row's sum is fixed by K alone.
// - gate+up rounds gate and up to BF16 (after s2), forms the FP32 SiLU product
//   a = (g / (1 + exp(-g))) * u and stores it as two BF16 terms hi = BF16(a), lo = BF16(a - hi)
//   in the act buffer's FP32 row space: row r holds N hi values then N lo values. The down
//   projection runs one MMA on each (hi first), so the product keeps about 16 mantissa bits.
// - Rows run TC_ROWS at a time as the MMA's N columns; column r depends only on row r, so a
//   row's output bits do not depend on which rows share its launch, pass or expert (padding
//   rows are zero), nor on RG.
// - A null routed weight pointer zeroes that expert's act or output rows; the shared-expert
//   pointers are not checked.
// - Grids: gate+up (N / NTC_GU_COLS, cap + S), down (N / NTC_DOWN_COLS, cap + S), block
//   NTC_THREADS, S = ceil(num_tokens / TC_ROWS). NTC_* and TC_ROWS must equal
//   NVFP4_GROUPED_TC_* in nvfp4_moe_grouped.rs.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "moe_fp8_grouped_tc_rows.cuh"

#define NTC_WARPS 4
#define NTC_THREADS (NTC_WARPS * 32)
// 2026-10-02: m-tiles (16 output columns) per warp and 128-K chunks per load group (a warp
// keeps 2 groups in flight). gate+up tiles are the gate and up rows of the same columns.
#define NTC_GU_MT 1
#define NTC_DOWN_MT 2
#define NTC_G 2
#define NTC_GU_COLS (NTC_WARPS * 16 * NTC_GU_MT)
#define NTC_DOWN_COLS (NTC_WARPS * 16 * NTC_DOWN_MT)

// 2026-10-02: The eight E2M1 values of packed word w (element e in bits 4e .. 4e + 3) as four
// BF16 pairs: d[p] = elements 2p (low half) and 2p + 1 (high half). The magnitude picks the low
// and high BF16 bytes from two 8-entry byte tables; the sign bit is moved to bit 15 or 31.
__device__ __forceinline__ void ntc_e2m1x8_bf16(unsigned int w, unsigned int* d) {
    // Low bytes of |v| for magnitudes 0..7 (0, 0.5, 1, 1.5 | 2, 3, 4, 6) and high bytes.
    const unsigned int L0 = 0xC0800000u, L1 = 0xC0804000u;
    const unsigned int H0 = 0x3F3F3F00u, H1 = 0x40404040u;
    const unsigned int s0 = w & 0x7777u, s1 = (w >> 16) & 0x7777u;
    unsigned int lo = __byte_perm(L0, L1, s0), hi = __byte_perm(H0, H1, s0);
    d[0] = __byte_perm(lo, hi, 0x5140u) | ((w << 12) & 0x8000u) | ((w << 24) & 0x80000000u);
    d[1] = __byte_perm(lo, hi, 0x7362u) | ((w << 4) & 0x8000u) | ((w << 16) & 0x80000000u);
    lo = __byte_perm(L0, L1, s1);
    hi = __byte_perm(H0, H1, s1);
    d[2] = __byte_perm(lo, hi, 0x5140u) | ((w >> 4) & 0x8000u) | ((w << 8) & 0x80000000u);
    d[3] = __byte_perm(lo, hi, 0x7362u) | ((w >> 12) & 0x8000u) | (w & 0x80000000u);
}

// 2026-10-02: E4M3 byte b as a BF16 pair (b, b), exact.
__device__ __forceinline__ unsigned int ntc_e4m3_bf16x2(unsigned int b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b, __NV_E4M3);
    const __nv_bfloat16 v = __float2bfloat16_rn(__half2float(__half(h)));
    const unsigned short u = *(const unsigned short*)&v;
    return (unsigned int)u | ((unsigned int)u << 16);
}

__device__ __forceinline__ unsigned int ntc_hmul2(unsigned int a, unsigned int s) {
    __nv_bfloat162 r = __hmul2(*(const __nv_bfloat162*)&a, *(const __nv_bfloat162*)&s);
    return *(unsigned int*)&r;
}

__device__ __forceinline__ void ntc_mma(float* c, const unsigned int* a, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned int ntc_word(const uint4& v, int j) {
    return (j == 0) ? v.x : (j == 1) ? v.y : (j == 2) ? v.z : v.w;
}

// 2026-10-02: One warp's MT output tiles (GATE_UP: MT gate tiles then MT up tiles of the same
// columns) over rows [begin, end). Input row r is X[row(r)], row(r) = sorted_token_ids[r]
// unless by_pos: BF16 [.., K] for gate+up, the hi|lo SiLU rows [.., 2K] for down. RG row groups
// of TC_ROWS share each decoded weight fragment; group q is its own MMA column block.
template <bool GATE_UP, int MT, int RG>
__device__ __forceinline__ void ntc_warp(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0, float s2_0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1, float s2_1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    constexpr int TILES = GATE_UP ? 2 * MT : MT;
    const unsigned int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const unsigned int half_k = K / 2, sk = K / 16, ngroups = (K / 128) / NTC_G;
    const unsigned char* wr[TILES][2];
    const unsigned char* sr[TILES][2];
    #pragma unroll
    for (int m = 0; m < TILES; m++) {
        const bool second = GATE_UP && m >= MT;
        const unsigned int col = f0 + 16 * (m % MT);
        const unsigned char* W = second ? W1 : W0;
        const unsigned char* S = second ? S1 : S0;
        wr[m][0] = W + (unsigned long long)(col + g) * half_k + t * 16;
        wr[m][1] = W + (unsigned long long)(col + g + 8) * half_k + t * 16;
        sr[m][0] = S + (unsigned long long)(col + g) * sk + 2 * t;
        sr[m][1] = S + (unsigned long long)(col + g + 8) * sk + 2 * t;
    }

    for (unsigned int row0 = begin; row0 < end; row0 += TC_ROWS * RG) {
        unsigned int cnt[RG];
        bool live[RG];
        const uint4* xp[RG];
        const uint4* xlo[RG];
        #pragma unroll
        for (int q = 0; q < RG; q++) {
            const unsigned int r0 = row0 + q * TC_ROWS;
            cnt[q] = r0 < end ? min((unsigned int)TC_ROWS, end - r0) : 0u;
            live[q] = g < cnt[q];
            const unsigned int xrow = live[q] ? (by_pos ? r0 + g : (unsigned int)sorted_token_ids[r0 + g]) : 0u;
            // 2026-10-02: gate+up reads BF16 [.., K]; down reads the hi row then the lo row of
            // [.., 2K].
            const unsigned long long stride = GATE_UP ? K : 2ull * K;
            xp[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * stride + t * 32);
            xlo[q] = (const uint4*)((const __nv_bfloat16*)X + (unsigned long long)xrow * stride + K + t * 32);
        }
        float acc[RG][TILES][4];
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int m = 0; m < TILES; m++)
                #pragma unroll
                for (int e = 0; e < 4; e++) acc[q][m][e] = 0.f;
        uint4 wn[NTC_G][TILES][2];
        unsigned int sn[NTC_G][TILES][2];
        #pragma unroll
        for (int c = 0; c < NTC_G; c++)
            #pragma unroll
            for (int m = 0; m < TILES; m++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    wn[c][m][h] = *(const uint4*)(wr[m][h] + c * 64);
                    sn[c][m][h] = *(const unsigned short*)(sr[m][h] + c * 8);
                }
        for (unsigned int gi = 0; gi < ngroups; gi++) {
            uint4 w[NTC_G][TILES][2];
            unsigned int s[NTC_G][TILES][2];
            #pragma unroll
            for (int c = 0; c < NTC_G; c++)
                #pragma unroll
                for (int m = 0; m < TILES; m++)
                    #pragma unroll
                    for (int h = 0; h < 2; h++) { w[c][m][h] = wn[c][m][h]; s[c][m][h] = sn[c][m][h]; }
            if (gi + 1 < ngroups) {
                #pragma unroll
                for (int c = 0; c < NTC_G; c++)
                    #pragma unroll
                    for (int m = 0; m < TILES; m++)
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const unsigned int chunk = (gi + 1) * NTC_G + c;
                            wn[c][m][h] = *(const uint4*)(wr[m][h] + chunk * 64);
                            sn[c][m][h] = *(const unsigned short*)(sr[m][h] + chunk * 8);
                        }
            }
            #pragma unroll
            for (int c = 0; c < NTC_G; c++) {
                const unsigned int chunk = gi * NTC_G + c;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    // 2026-10-02: Activation words for K = 32t + 8j .. 8j + 7: word i holds
                    // K + 2i, 2i + 1. MMA 2j takes words 0, 1; MMA 2j + 1 words 2, 3.
                    unsigned int xw[RG][4], xl[RG][4];
                    #pragma unroll
                    for (int q = 0; q < RG; q++) {
                        uint4 a = make_uint4(0u, 0u, 0u, 0u), b = a;
                        if (live[q]) {
                            a = xp[q][chunk * 16 + j];
                            if (!GATE_UP) b = xlo[q][chunk * 16 + j];
                        }
                        xw[q][0] = a.x; xw[q][1] = a.y; xw[q][2] = a.z; xw[q][3] = a.w;
                        xl[q][0] = b.x; xl[q][1] = b.y; xl[q][2] = b.z; xl[q][3] = b.w;
                    }
                    #pragma unroll
                    for (int m = 0; m < TILES; m++) {
                        // 2026-10-02: Words 0, 1 lie in block 2t, words 2, 3 in block 2t + 1.
                        const unsigned int sg = ntc_e4m3_bf16x2((s[c][m][0] >> (j < 2 ? 0 : 8)) & 0xFFu);
                        const unsigned int sh = ntc_e4m3_bf16x2((s[c][m][1] >> (j < 2 ? 0 : 8)) & 0xFFu);
                        unsigned int dg[4], dh[4];
                        ntc_e2m1x8_bf16(ntc_word(w[c][m][0], j), dg);
                        ntc_e2m1x8_bf16(ntc_word(w[c][m][1], j), dh);
                        #pragma unroll
                        for (int p = 0; p < 4; p++) { dg[p] = ntc_hmul2(dg[p], sg); dh[p] = ntc_hmul2(dh[p], sh); }
                        const unsigned int a0[4] = {dg[0], dh[0], dg[1], dh[1]};
                        const unsigned int a1[4] = {dg[2], dh[2], dg[3], dh[3]};
                        #pragma unroll
                        for (int q = 0; q < RG; q++) {
                            ntc_mma(acc[q][m], a0, xw[q][0], xw[q][1]);
                            if (!GATE_UP) ntc_mma(acc[q][m], a0, xl[q][0], xl[q][1]);
                            ntc_mma(acc[q][m], a1, xw[q][2], xw[q][3]);
                            if (!GATE_UP) ntc_mma(acc[q][m], a1, xl[q][2], xl[q][3]);
                        }
                    }
                }
            }
        }
        // 2026-10-02: acc[q][m][e]: column f + g (+ 8 for e >= 2) of row 2t + (e & 1) of group q.
        #pragma unroll
        for (int q = 0; q < RG; q++)
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const unsigned int r = 2 * t + (e & 1);
                if (r >= cnt[q]) continue;
                const unsigned int row = row0 + q * TC_ROWS + r;
                const unsigned int colo = f0 + g + ((e >> 1) ? 8 : 0);
                #pragma unroll
                for (int m = 0; m < MT; m++) {
                    if (GATE_UP) {
                        const float gv = __bfloat162float(__float2bfloat16(acc[q][m][e] * s2_0));
                        const float uv = __bfloat162float(__float2bfloat16(acc[q][m + MT][e] * s2_1));
                        const float av = (gv / (1.0f + __expf(-gv))) * uv;
                        const __nv_bfloat16 hi = __float2bfloat16(av);
                        const unsigned long long ro = (unsigned long long)row * 2 * N + colo + 16 * m;
                        ((__nv_bfloat16*)out)[ro] = hi;
                        ((__nv_bfloat16*)out)[ro + N] = __float2bfloat16(av - __bfloat162float(hi));
                    } else {
                        ((__nv_bfloat16*)out)[(unsigned long long)row * N + colo + 16 * m] =
                            __float2bfloat16(acc[q][m][e] * s2_0);
                    }
                }
            }
    }
}

// 2026-10-02: ntc_warp for a routed expert's rows: RG = 2 (one weight pass per 16 rows) when the
// expert has more than TC_ROWS rows, else RG = 1. The shared expert's block rows hold at most
// TC_ROWS rows and keep RG = 1.
template <bool GATE_UP, int MT>
__device__ __forceinline__ void ntc_warp_routed(
    const void* __restrict__ X, const int* __restrict__ sorted_token_ids, bool by_pos,
    unsigned int begin, unsigned int end,
    const unsigned char* __restrict__ W0, const unsigned char* __restrict__ S0, float s2_0,
    const unsigned char* __restrict__ W1, const unsigned char* __restrict__ S1, float s2_1,
    void* __restrict__ out, unsigned int N, unsigned int K, unsigned int f0
) {
    if (end - begin > TC_ROWS)
        ntc_warp<GATE_UP, MT, 2>(X, sorted_token_ids, by_pos, begin, end, W0, S0, s2_0, W1, S1, s2_1, out, N, K, f0);
    else
        ntc_warp<GATE_UP, MT, 1>(X, sorted_token_ids, by_pos, begin, end, W0, S0, s2_0, W1, S1, s2_1, out, N, K, f0);
}

// 2026-10-02: Gate+up and SiLU of the routed experts and the shared expert. A: [num_tokens, K]
// BF16. act: routed hi|lo rows [pos, 2N] BF16 by sorted position; sh_act: shared hi|lo rows
// [token, 2N]; both in FP32-sized buffers. Arguments as moe_expert_gate_up_act_nvfp4_grouped.
extern "C" __global__ void __launch_bounds__(NTC_THREADS) moe_expert_gate_up_act_nvfp4_grouped_tc(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2,
    __nv_bfloat16* __restrict__ act,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_packed,
    const unsigned char* __restrict__ sh_gate_scale,
    float sh_gate_s2,
    const unsigned char* __restrict__ sh_up_packed,
    const unsigned char* __restrict__ sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * NTC_GU_COLS + (threadIdx.x >> 5) * 16 * NTC_GU_MT;
    if (is_shared) {
        ntc_warp<true, NTC_GU_MT, 1>(A, sorted_token_ids, true, begin, end, sh_gate_packed, sh_gate_scale,
                                     sh_gate_s2, sh_up_packed, sh_up_scale, sh_up_s2, sh_act, N, K, f0);
        return;
    }
    const unsigned char* Pg = (const unsigned char*)gate_packed_ptrs[expert];
    const unsigned char* Pu = (const unsigned char*)up_packed_ptrs[expert];
    if (Pg == 0 || Pu == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < NTC_GU_COLS; i += NTC_THREADS)
                for (unsigned int hl = 0; hl < 2; hl++)
                    act[(unsigned long long)pos * 2 * N + hl * N + blockIdx.x * NTC_GU_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    ntc_warp_routed<true, NTC_GU_MT>(A, sorted_token_ids, false, begin, end,
                                     Pg, (const unsigned char*)gate_scale_ptrs[expert], gate_scale2[expert],
                                     Pu, (const unsigned char*)up_scale_ptrs[expert], up_scale2[expert],
                                     act, N, K, f0);
}

// 2026-10-02: Down projection of the hi|lo SiLU rows (act routed by position, sh_act shared by
// token) into C [pos, N] and sh_down_out [token, N] BF16. Arguments as
// moe_expert_down_act_nvfp4_grouped.
extern "C" __global__ void __launch_bounds__(NTC_THREADS) moe_expert_down_act_nvfp4_grouped_tc(
    const __nv_bfloat16* __restrict__ act,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const __nv_bfloat16* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!tc_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                       &is_shared, &expert, &begin, &end)) return;
    const unsigned int f0 = blockIdx.x * NTC_DOWN_COLS + (threadIdx.x >> 5) * 16 * NTC_DOWN_MT;
    if (is_shared) {
        ntc_warp<false, NTC_DOWN_MT, 1>(sh_act, nullptr, true, begin, end, sh_down_packed, sh_down_scale,
                                        sh_down_s2, nullptr, nullptr, 0.f, sh_down_out, N, K, f0);
        return;
    }
    const unsigned char* P = (const unsigned char*)packed_ptrs[expert];
    if (P == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < NTC_DOWN_COLS; i += NTC_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * NTC_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    ntc_warp_routed<false, NTC_DOWN_MT>(act, nullptr, true, begin, end,
                                        P, (const unsigned char*)scale_ptrs[expert], scale2[expert],
                                        nullptr, nullptr, 0.f, C, N, K, f0);
}
