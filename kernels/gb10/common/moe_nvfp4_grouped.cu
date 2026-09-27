// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: Grouped NVFP4 MoE decode GEMVs (W4A16): one block per active expert, which reads
// each weight row once per pass and applies it to every row routed to that expert.
//
// Owner: gb10 kernels.
// Invariants:
// - Block rows, sorted positions and the shared expert follow moe_grouped_rows.cuh; act and the
//   routed C are indexed by sorted position, and moe_weighted_sum_blend_fp8_grouped
//   (moe_fp8_grouped_blend.cu) maps a token's slot back through token_to_perm.
// - Weights are row-major NVFP4: packed E2M1 [N, K / 2] (element 2j in the low nibble of byte
//   j), E4M3 block scales [N, K / 16], and a per-tensor scale s2 (per expert for the routed
//   tables). A null routed weight pointer zeroes that expert's act or output rows; the
//   shared-expert pointers are not checked.
// - A row's sums do not depend on the other rows, on how many rows share the launch, or on the
//   pass a row falls in: each lane walks its K chunks in a fixed order, sums a 16-element block
//   against the E2M1 values in fixed groups of four products, scales the block sum once, and the
//   warp reduces with a fixed shuffle tree. The same row therefore gets the same bits at every
//   row count.
// - gate+up rounds gate and up to BF16 and writes the FP32 product (g / (1 + __expf(-g))) * u,
//   as the grouped FP8 kernels do; down reads it and writes BF16.
// - Launch: gate+up grid (ceil(N / NG_GU_COLS_PER_CTA), cap + ceil(num_tokens / NG_GU_ROWS), 1),
//   block NG_GU_BLOCK, K % 32 == 0; down grid (ceil(N / NG_DOWN_COLS_PER_CTA), cap +
//   ceil(num_tokens / NG_DOWN_ROWS), 1), block NG_DOWN_BLOCK, K % 16 == 0. No dynamic shared
//   memory. The four NG_* launch constants must equal NVFP4_GROUPED_* in nvfp4_moe_grouped.rs.

#include <cuda_bf16.h>
#include "mx_block_scale.cuh"
#include "moe_grouped_rows.cuh"

#define NG_WARP 32
#define NG_GU_BLOCK 128
#define NG_GU_CGROUPS 2
#define NG_GU_COLS_PER_CTA (2 * NG_GU_CGROUPS * (NG_GU_BLOCK / NG_WARP))
#define NG_GU_ROWS 4
#define NG_DOWN_BLOCK 256
#define NG_DOWN_COLS_PER_WARP 4
#define NG_DOWN_CGROUPS 4
#define NG_DOWN_COLS_PER_CTA ((NG_DOWN_BLOCK / NG_WARP) * NG_DOWN_COLS_PER_WARP * NG_DOWN_CGROUPS)
#define NG_DOWN_ROWS 4

__device__ __constant__ float NG_E2M1[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// 2026-09-27: The eight E2M1 values of one packed word (element e in bits 4e .. 4e + 3).
__device__ __forceinline__ void ng_decode8(const float* s_lut, unsigned int w, float* v) {
    #pragma unroll
    for (int e = 0; e < 8; e++) v[e] = s_lut[(w >> (4 * e)) & 0xFu];
}

// 2026-09-27: The sum of eight products in two fixed groups of four.
__device__ __forceinline__ float ng_dot8(const float* a, const float* w) {
    float p = a[0] * w[0] + a[1] * w[1] + a[2] * w[2] + a[3] * w[3];
    p += a[4] * w[4] + a[5] * w[5] + a[6] * w[6] + a[7] * w[7];
    return p;
}

// 2026-09-27: Eight BF16 values of a uint4 as floats (exact).
__device__ __forceinline__ void ng_bf16x8(uint4 v, float* a) {
    a[0] = __uint_as_float(v.x << 16); a[1] = __uint_as_float(v.x & 0xFFFF0000u);
    a[2] = __uint_as_float(v.y << 16); a[3] = __uint_as_float(v.y & 0xFFFF0000u);
    a[4] = __uint_as_float(v.z << 16); a[5] = __uint_as_float(v.z & 0xFFFF0000u);
    a[6] = __uint_as_float(v.w << 16); a[7] = __uint_as_float(v.w & 0xFFFF0000u);
}

// 2026-09-27: Gate and up projections of columns n1 = 2 * (blockIdx.x * warps + warp) and
// n2 = n1 + 1 in one warp, then the SiLU product into act[pos * N + n]. A lane takes 32
// consecutive K elements per step (16 packed bytes and two scale bytes per projection column),
// loaded one step ahead.
extern "C" __global__ void __launch_bounds__(NG_GU_BLOCK) moe_expert_gate_up_act_nvfp4_grouped(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_packed_ptrs,
    const unsigned long long* __restrict__ gate_scale_ptrs,
    const float* __restrict__ gate_scale2,
    const unsigned long long* __restrict__ up_packed_ptrs,
    const unsigned long long* __restrict__ up_scale_ptrs,
    const float* __restrict__ up_scale2,
    float* __restrict__ act,
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
    float* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!moe_grouped_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                                NG_GU_ROWS, &is_shared, &expert, &begin, &end)) return;

    const unsigned char *Pg, *Sg, *Pu, *Su;
    float g2, u2;
    float* out;
    if (is_shared) {
        Pg = sh_gate_packed; Sg = sh_gate_scale; g2 = sh_gate_s2;
        Pu = sh_up_packed; Su = sh_up_scale; u2 = sh_up_s2; out = sh_act;
    } else {
        Pg = (const unsigned char*)gate_packed_ptrs[expert];
        Sg = (const unsigned char*)gate_scale_ptrs[expert];
        g2 = gate_scale2[expert];
        Pu = (const unsigned char*)up_packed_ptrs[expert];
        Su = (const unsigned char*)up_scale_ptrs[expert];
        u2 = up_scale2[expert];
        out = act;
        if (Pg == 0 || Pu == 0) {
            const unsigned int n_base = blockIdx.x * NG_GU_COLS_PER_CTA;
            for (unsigned int pos = begin; pos < end; pos++)
                for (unsigned int i = threadIdx.x; i < NG_GU_COLS_PER_CTA && n_base + i < N; i += NG_GU_BLOCK)
                    act[(unsigned long long)pos * N + n_base + i] = 0.0f;
            return;
        }
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = NG_E2M1[threadIdx.x];
    __syncthreads();

    const unsigned int warp = threadIdx.x / NG_WARP;
    const unsigned int lane = threadIdx.x % NG_WARP;
    const unsigned int n_warp = blockIdx.x * NG_GU_COLS_PER_CTA + warp * 2 * NG_GU_CGROUPS;
    if (n_warp >= N) return;
    const unsigned int half_K = K / 2;
    const unsigned int groups = K / 16;
    const unsigned int K32 = K / 32;
    const unsigned int steps = (K32 + NG_WARP - 1) / NG_WARP;
    const unsigned int total = NG_GU_CGROUPS * steps;
    const float s2v[4] = {g2, g2, u2, u2};

    // 2026-09-27: Step t covers column pair t / steps ([projection column]: g1, g2, u1, u2) and
    // 32-element chunk (t % steps) * 32 + lane; a lane past the last chunk loads nothing.
    auto load = [&](unsigned int t, uint4* w, unsigned int* sb) {
        const unsigned int n1 = n_warp + 2 * (t / steps);
        const unsigned int n2 = min(n1 + 1, N - 1);
        const unsigned int c = (t % steps) * NG_WARP + lane;
        const unsigned char* P[4] = {Pg, Pg, Pu, Pu};
        const unsigned char* S[4] = {Sg, Sg, Su, Su};
        const unsigned int n[4] = {n1, n2, n1, n2};
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            w[q] = make_uint4(0u, 0u, 0u, 0u); sb[q] = 0u;
            if (n1 < N && c < K32) {
                w[q] = *(const uint4*)(P[q] + (unsigned long long)n[q] * half_K + c * 16);
                sb[q] = *(const unsigned short*)(S[q] + (unsigned long long)n[q] * groups + c * 2);
            }
        }
    };

    for (unsigned int row0 = begin; row0 < end; row0 += NG_GU_ROWS) {
        const unsigned int cnt = min((unsigned int)NG_GU_ROWS, end - row0);
        const __nv_bfloat16* a_ptr[NG_GU_ROWS];
        #pragma unroll
        for (int r = 0; r < NG_GU_ROWS; r++) {
            const unsigned int row = (r < (int)cnt) ? moe_grouped_a_row(sorted_token_ids, is_shared, row0 + r) : 0u;
            a_ptr[r] = A + (unsigned long long)row * K;
        }
        float acc[4][NG_GU_ROWS];
        #pragma unroll
        for (int q = 0; q < 4; q++)
            #pragma unroll
            for (int r = 0; r < NG_GU_ROWS; r++) acc[q][r] = 0.0f;

        uint4 w[4];
        unsigned int sb[4];
        load(0, w, sb);

        for (unsigned int t = 0; t < total; t++) {
            unsigned int cw[4][4];
            unsigned int cs[4];
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                cw[q][0] = w[q].x; cw[q][1] = w[q].y; cw[q][2] = w[q].z; cw[q][3] = w[q].w;
                cs[q] = sb[q];
            }
            if (t + 1 < total) load(t + 1, w, sb);
            const unsigned int c = (t % steps) * NG_WARP + lane;
            if (c < K32) {
                #pragma unroll
                for (int g = 0; g < 2; g++) {
                    float part[4][NG_GU_ROWS];
                    #pragma unroll
                    for (int q = 0; q < 4; q++)
                        #pragma unroll
                        for (int r = 0; r < NG_GU_ROWS; r++) part[q][r] = 0.0f;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const int wi = 2 * g + h;
                        float wf[4][8];
                        #pragma unroll
                        for (int q = 0; q < 4; q++) ng_decode8(s_lut, cw[q][wi], wf[q]);
                        #pragma unroll
                        for (int r = 0; r < NG_GU_ROWS; r++) {
                            if (r < (int)cnt) {
                                float a[8];
                                ng_bf16x8(*(const uint4*)(a_ptr[r] + c * 32 + wi * 8), a);
                                #pragma unroll
                                for (int q = 0; q < 4; q++) part[q][r] += ng_dot8(a, wf[q]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int q = 0; q < 4; q++) {
                        const float sc = mx_block_scale<false>((unsigned char)(cs[q] >> (8 * g)), s2v[q]);
                        #pragma unroll
                        for (int r = 0; r < NG_GU_ROWS; r++) acc[q][r] += part[q][r] * sc;
                    }
                }
            }
            if ((t + 1) % steps == 0) {
                const unsigned int n1 = n_warp + 2 * (t / steps);
                const bool have_n2 = n1 + 1 < N;
                #pragma unroll
                for (int r = 0; r < NG_GU_ROWS; r++) {
                    if (r < (int)cnt) {
                        float v[4];
                        #pragma unroll
                        for (int q = 0; q < 4; q++) {
                            v[q] = acc[q][r];
                            acc[q][r] = 0.0f;
                            #pragma unroll
                            for (int offset = NG_WARP / 2; offset > 0; offset >>= 1)
                                v[q] += __shfl_down_sync(0xFFFFFFFF, v[q], offset);
                        }
                        if (lane == 0 && n1 < N) {
                            const unsigned long long base = (unsigned long long)(row0 + r) * N;
                            const float ga = __bfloat162float(__float2bfloat16(v[0]));
                            const float ua = __bfloat162float(__float2bfloat16(v[2]));
                            out[base + n1] = (ga / (1.0f + __expf(-ga))) * ua;
                            if (have_n2) {
                                const float gb = __bfloat162float(__float2bfloat16(v[1]));
                                const float ub = __bfloat162float(__float2bfloat16(v[3]));
                                out[base + n1 + 1] = (gb / (1.0f + __expf(-gb))) * ub;
                            }
                        }
                    }
                }
            }
        }
    }
}

// 2026-09-27: Down projection of the SiLU product act ([pos, K] FP32), NG_DOWN_COLS_PER_WARP
// output columns per warp. A lane takes one 16-element block per step (8 packed bytes and one
// scale byte per column), loaded one step ahead.
extern "C" __global__ void __launch_bounds__(NG_DOWN_BLOCK, 2) moe_expert_down_act_nvfp4_grouped(
    const float* __restrict__ act,
    const unsigned long long* __restrict__ packed_ptrs,
    const unsigned long long* __restrict__ scale_ptrs,
    const float* __restrict__ scale2,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const float* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_packed,
    const unsigned char* __restrict__ sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!moe_grouped_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                                NG_DOWN_ROWS, &is_shared, &expert, &begin, &end)) return;

    const unsigned char *Pw, *Sw;
    float s2;
    const float* a_base;
    __nv_bfloat16* out_base;
    if (is_shared) {
        Pw = sh_down_packed; Sw = sh_down_scale; s2 = sh_down_s2;
        a_base = sh_act; out_base = sh_down_out;
    } else {
        Pw = (const unsigned char*)packed_ptrs[expert];
        Sw = (const unsigned char*)scale_ptrs[expert];
        s2 = scale2[expert];
        a_base = act; out_base = C;
        if (Pw == 0) {
            const unsigned int n_base = blockIdx.x * NG_DOWN_COLS_PER_CTA;
            for (unsigned int pos = begin; pos < end; pos++)
                for (unsigned int i = threadIdx.x; i < NG_DOWN_COLS_PER_CTA && n_base + i < N; i += NG_DOWN_BLOCK)
                    C[(unsigned long long)pos * N + n_base + i] = __float2bfloat16(0.0f);
            return;
        }
    }

    __shared__ float s_lut[16];
    if (threadIdx.x < 16) s_lut[threadIdx.x] = NG_E2M1[threadIdx.x];
    __syncthreads();

    const unsigned int warp = threadIdx.x / NG_WARP;
    const unsigned int lane = threadIdx.x % NG_WARP;
    const unsigned int n_warp = blockIdx.x * NG_DOWN_COLS_PER_CTA + warp * NG_DOWN_COLS_PER_WARP * NG_DOWN_CGROUPS;
    if (n_warp >= N) return;
    const unsigned int half_K = K / 2;
    const unsigned int groups = K / 16;
    const unsigned int steps = (groups + NG_WARP - 1) / NG_WARP;
    const unsigned int total = NG_DOWN_CGROUPS * steps;

    // 2026-09-27: Step t covers column group t / steps and 16-element block (t % steps) * 32 +
    // lane; a lane past the last block loads nothing and adds nothing.
    auto load = [&](unsigned int t, uint2* wv, unsigned int* sb) {
        const unsigned int cg = t / steps;
        const unsigned int k16 = (t % steps) * NG_WARP + lane;
        #pragma unroll
        for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) {
            const unsigned int n = n_warp + cg * NG_DOWN_COLS_PER_WARP + c;
            wv[c] = make_uint2(0u, 0u); sb[c] = 0u;
            if (n < N && k16 < groups) {
                wv[c] = *(const uint2*)(Pw + (unsigned long long)n * half_K + k16 * 8);
                sb[c] = Sw[(unsigned long long)n * groups + k16];
            }
        }
    };

    for (unsigned int row0 = begin; row0 < end; row0 += NG_DOWN_ROWS) {
        const unsigned int cnt = min((unsigned int)NG_DOWN_ROWS, end - row0);
        float acc[NG_DOWN_ROWS][NG_DOWN_COLS_PER_WARP];
        #pragma unroll
        for (int r = 0; r < NG_DOWN_ROWS; r++)
            #pragma unroll
            for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) acc[r][c] = 0.0f;

        uint2 wv[NG_DOWN_COLS_PER_WARP];
        unsigned int sb[NG_DOWN_COLS_PER_WARP];
        load(0, wv, sb);

        for (unsigned int t = 0; t < total; t++) {
            unsigned int cw[NG_DOWN_COLS_PER_WARP][2];
            unsigned int cs[NG_DOWN_COLS_PER_WARP];
            #pragma unroll
            for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) { cw[c][0] = wv[c].x; cw[c][1] = wv[c].y; cs[c] = sb[c]; }
            if (t + 1 < total) load(t + 1, wv, sb);
            const unsigned int k16 = (t % steps) * NG_WARP + lane;
            if (k16 < groups) {
                // 2026-09-27: The block's weights are decoded once and applied to every row of the pass.
                float wf[NG_DOWN_COLS_PER_WARP][16];
                float sc[NG_DOWN_COLS_PER_WARP];
                #pragma unroll
                for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) {
                    ng_decode8(s_lut, cw[c][0], wf[c]);
                    ng_decode8(s_lut, cw[c][1], wf[c] + 8);
                    sc[c] = mx_block_scale<false>((unsigned char)cs[c], s2);
                }
                #pragma unroll
                for (int r = 0; r < NG_DOWN_ROWS; r++) {
                    if (r < (int)cnt) {
                        const float4* ar = (const float4*)(a_base + (unsigned long long)(row0 + r) * K + k16 * 16);
                        float a[16];
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            const float4 v = ar[j];
                            a[4 * j] = v.x; a[4 * j + 1] = v.y; a[4 * j + 2] = v.z; a[4 * j + 3] = v.w;
                        }
                        #pragma unroll
                        for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) {
                            float part = ng_dot8(a, wf[c]);
                            part += ng_dot8(a + 8, wf[c] + 8);
                            acc[r][c] += part * sc[c];
                        }
                    }
                }
            }
            if ((t + 1) % steps == 0) {
                const unsigned int n0 = n_warp + (t / steps) * NG_DOWN_COLS_PER_WARP;
                #pragma unroll
                for (int r = 0; r < NG_DOWN_ROWS; r++) {
                    if (r < (int)cnt) {
                        __nv_bfloat16* out = out_base + (unsigned long long)(row0 + r) * N;
                        #pragma unroll
                        for (int c = 0; c < NG_DOWN_COLS_PER_WARP; c++) {
                            float v = acc[r][c];
                            #pragma unroll
                            for (int offset = NG_WARP / 2; offset > 0; offset >>= 1)
                                v += __shfl_down_sync(0xFFFFFFFF, v, offset);
                            if (lane == 0 && n0 + c < N) out[n0 + c] = __float2bfloat16(v);
                            acc[r][c] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}
