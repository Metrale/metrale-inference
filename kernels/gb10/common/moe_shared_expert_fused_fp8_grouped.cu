// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Grouped FP8 MoE decode GEMVs: one block per active expert, which reads each
// weight row once per pass and applies it to every row routed to that expert.
//
// Owner: gb10 kernels.
// Invariants:
// - Rows are in sorted order (moe_sort_by_expert): expert e owns positions
//   [expert_offsets[e], expert_offsets[e + 1]), and position pos reads input row
//   sorted_token_ids[pos]. act and the routed C are indexed by position; the blend in
//   moe_fp8_grouped_blend.cu maps a token's slot back through token_to_perm.
// - 2026-09-26: blockIdx.y < S = ceil(num_tokens / rows per pass) is the shared expert,
//   rows [y * rows, (y + 1) * rows); blockIdx.y = S + i indexes active_experts[i] (from
//   moe_fp8_grouped_sort), and blocks with i at or past active_count[0] return. The host
//   sets cap = min(num_tokens * top_k, num_experts) (fp8_grouped_active_cap) and grid.y =
//   cap + S, so the grid does not depend on the routing.
// - 2026-09-26: Per row, the output equals moe_shared_expert_fused_fp8.cu's gate+up, SiLU
//   and down bit for bit. gate+up keeps its 16-input lane chunks, four-product sums and
//   shuffle reduction, rounds gate and up to BF16 as that kernel stores them, and writes
//   the FP32 product (g / (1 + __expf(-g))) * u that its down kernel stages; down keeps the
//   8-input chunks and four-product sums in the same order, reading that product from act.
//   A row's sums are independent of the other rows and of the order rows are visited in.
// - Weights are row-major [N, K] FP8 E4M3 with FP32 scales [ceil(N / 128), ceil(K / 128)].
//   A null routed gate or up pointer makes that expert's act 0, a null routed down pointer
//   zeroes its output rows; the shared-expert pointers are not checked.
// - Launch: gate+up grid (ceil(N / GU_COLS_PER_CTA), cap + S, 1),
//   block 128; down grid (ceil(N / DOWN_CTA_COLS), cap + S, 1), block 256, no dynamic
//   shared memory. GU_COLS_PER_CTA, GU_GROUP_ROWS, DOWN_CTA_COLS and GROUP_ROWS must
//   equal FP8_GROUPED_GATE_UP_COLS_PER_CTA, FP8_GROUPED_GATE_UP_ROWS_PER_PASS,
//   FP8_GROUPED_DOWN_COLS_PER_CTA and FP8_GROUPED_DOWN_ROWS_PER_PASS in fp8_moe_grouped.rs.

#include <cuda_bf16.h>

#define BLOCK_SIZE 128
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128
// 2026-09-26: Rows per down pass; an expert with more rows reads its weights once per pass.
#define GROUP_ROWS 4
// 2026-09-26: Rows per gate+up pass, and the columns of a gate+up CTA (one pair per warp).
#define GU_GROUP_ROWS 4
#define GU_COLS_PER_CTA (2 * N_PER_BLOCK)
#define DOWN_BLOCK 256
#define DOWN_COLS_PER_WARP 4
#define DOWN_COLS_PER_CTA ((DOWN_BLOCK / WARP_SIZE) * DOWN_COLS_PER_WARP)
// 2026-09-27: Column groups per down block: a block covers DOWN_CTA_COLS output columns.
#define DOWN_CG 4
#define DOWN_CTA_COLS (DOWN_COLS_PER_CTA * DOWN_CG)

__device__ __constant__ float E4M3_LUT_MOE_GROUPED[256] = {

    0.0f, 0.001953125f, 0.00390625f, 0.005859375f,
    0.0078125f, 0.009765625f, 0.01171875f, 0.013671875f,
    0.015625f, 0.017578125f, 0.01953125f, 0.021484375f,
    0.0234375f, 0.025390625f, 0.02734375f, 0.029296875f,
    0.03125f, 0.03515625f, 0.0390625f, 0.04296875f,
    0.046875f, 0.05078125f, 0.0546875f, 0.05859375f,
    0.0625f, 0.0703125f, 0.078125f, 0.0859375f,
    0.09375f, 0.1015625f, 0.109375f, 0.1171875f,
    0.125f, 0.140625f, 0.15625f, 0.171875f,
    0.1875f, 0.203125f, 0.21875f, 0.234375f,
    0.25f, 0.28125f, 0.3125f, 0.34375f,
    0.375f, 0.40625f, 0.4375f, 0.46875f,
    0.5f, 0.5625f, 0.625f, 0.6875f,
    0.75f, 0.8125f, 0.875f, 0.9375f,
    1.0f, 1.125f, 1.25f, 1.375f,
    1.5f, 1.625f, 1.75f, 1.875f,
    2.0f, 2.25f, 2.5f, 2.75f,
    3.0f, 3.25f, 3.5f, 3.75f,
    4.0f, 4.5f, 5.0f, 5.5f,
    6.0f, 6.5f, 7.0f, 7.5f,
    8.0f, 9.0f, 10.0f, 11.0f,
    12.0f, 13.0f, 14.0f, 15.0f,
    16.0f, 18.0f, 20.0f, 22.0f,
    24.0f, 26.0f, 28.0f, 30.0f,
    32.0f, 36.0f, 40.0f, 44.0f,
    48.0f, 52.0f, 56.0f, 60.0f,
    64.0f, 72.0f, 80.0f, 88.0f,
    96.0f, 104.0f, 112.0f, 120.0f,
    128.0f, 144.0f, 160.0f, 176.0f,
    192.0f, 208.0f, 224.0f, 240.0f,
    256.0f, 288.0f, 320.0f, 352.0f,
    384.0f, 416.0f, 448.0f, 0.0f,

    -0.0f, -0.001953125f, -0.00390625f, -0.005859375f,
    -0.0078125f, -0.009765625f, -0.01171875f, -0.013671875f,
    -0.015625f, -0.017578125f, -0.01953125f, -0.021484375f,
    -0.0234375f, -0.025390625f, -0.02734375f, -0.029296875f,
    -0.03125f, -0.03515625f, -0.0390625f, -0.04296875f,
    -0.046875f, -0.05078125f, -0.0546875f, -0.05859375f,
    -0.0625f, -0.0703125f, -0.078125f, -0.0859375f,
    -0.09375f, -0.1015625f, -0.109375f, -0.1171875f,
    -0.125f, -0.140625f, -0.15625f, -0.171875f,
    -0.1875f, -0.203125f, -0.21875f, -0.234375f,
    -0.25f, -0.28125f, -0.3125f, -0.34375f,
    -0.375f, -0.40625f, -0.4375f, -0.46875f,
    -0.5f, -0.5625f, -0.625f, -0.6875f,
    -0.75f, -0.8125f, -0.875f, -0.9375f,
    -1.0f, -1.125f, -1.25f, -1.375f,
    -1.5f, -1.625f, -1.75f, -1.875f,
    -2.0f, -2.25f, -2.5f, -2.75f,
    -3.0f, -3.25f, -3.5f, -3.75f,
    -4.0f, -4.5f, -5.0f, -5.5f,
    -6.0f, -6.5f, -7.0f, -7.5f,
    -8.0f, -9.0f, -10.0f, -11.0f,
    -12.0f, -13.0f, -14.0f, -15.0f,
    -16.0f, -18.0f, -20.0f, -22.0f,
    -24.0f, -26.0f, -28.0f, -30.0f,
    -32.0f, -36.0f, -40.0f, -44.0f,
    -48.0f, -52.0f, -56.0f, -60.0f,
    -64.0f, -72.0f, -80.0f, -88.0f,
    -96.0f, -104.0f, -112.0f, -120.0f,
    -128.0f, -144.0f, -160.0f, -176.0f,
    -192.0f, -208.0f, -224.0f, -240.0f,
    -256.0f, -288.0f, -320.0f, -352.0f,
    -384.0f, -416.0f, -448.0f, -0.0f,
};

// 2026-09-25: Input row of sorted position pos: pos itself for the shared expert, else sorted_token_ids[pos].
__device__ __forceinline__ unsigned int grouped_a_row(
    const int* __restrict__ sorted_token_ids, bool is_shared, unsigned int pos
) {
    return is_shared ? pos : (unsigned int)sorted_token_ids[pos];
}

// 2026-09-26: The row range of this block, and whether it runs at all. The first
// ceil(num_tokens / rows) block rows are the shared expert, `rows` tokens each (one pass),
// so its all-token work spreads over as many blocks as passes and is dispatched first,
// instead of trailing the grid; block row shared_slots + i is active expert i. Shared by the
// gate+up and down kernels.
__device__ __forceinline__ bool grouped_block_rows(
    const int* __restrict__ expert_offsets, const int* __restrict__ active_experts,
    const int* __restrict__ active_count, unsigned int num_tokens, unsigned int rows,
    bool* is_shared, unsigned int* expert, unsigned int* begin, unsigned int* end
) {
    const unsigned int y = blockIdx.y;
    const unsigned int shared_slots = (num_tokens + rows - 1) / rows;
    *is_shared = (y < shared_slots);
    if (*is_shared) {
        *expert = 0; *begin = y * rows; *end = min(num_tokens, (y + 1) * rows);
    } else {
        const unsigned int i = y - shared_slots;
        if ((int)i >= active_count[0]) return false;
        *expert = (unsigned int)active_experts[i];
        *begin = (unsigned int)expert_offsets[*expert];
        *end = (unsigned int)expert_offsets[*expert + 1];
    }
    return *begin < *end;
}

// 2026-09-26: Gate and up projections of columns n1 = 2 * (blockIdx.x * N_PER_BLOCK +
// warp) and n2 = n1 + 1 in one warp, then the SiLU product into act[pos * N + n].
extern "C" __global__ void moe_expert_gate_up_act_fp8_grouped(
    const __nv_bfloat16* __restrict__ A,
    const unsigned long long* __restrict__ gate_weight_ptrs,
    const unsigned long long* __restrict__ gate_block_scale_ptrs,
    const unsigned long long* __restrict__ up_weight_ptrs,
    const unsigned long long* __restrict__ up_block_scale_ptrs,
    float* __restrict__ act,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const unsigned char* __restrict__ sh_gate_weight,
    const float* __restrict__ sh_gate_block_scale,
    const unsigned char* __restrict__ sh_up_weight,
    const float* __restrict__ sh_up_block_scale,
    float* __restrict__ sh_act,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!grouped_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                            GU_GROUP_ROWS, &is_shared, &expert, &begin, &end)) return;

    const unsigned char* Wg;
    const float* Sg;
    const unsigned char* Wu;
    const float* Su;
    float* out;
    if (is_shared) {
        Wg = sh_gate_weight; Sg = sh_gate_block_scale;
        Wu = sh_up_weight; Su = sh_up_block_scale; out = sh_act;
    } else {
        Wg = (const unsigned char*)gate_weight_ptrs[expert];
        Sg = (const float*)gate_block_scale_ptrs[expert];
        Wu = (const unsigned char*)up_weight_ptrs[expert];
        Su = (const float*)up_block_scale_ptrs[expert];
        out = act;
        if (Wg == 0 || Wu == 0) {
            const unsigned int n_base = blockIdx.x * GU_COLS_PER_CTA;
            for (unsigned int pos = begin; pos < end; pos++) {
                for (unsigned int i = threadIdx.x; i < GU_COLS_PER_CTA && n_base + i < N; i += BLOCK_SIZE) {
                    act[(unsigned long long)pos * N + n_base + i] = 0.0f;
                }
            }
            return;
        }
    }

    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_MOE_GROUPED[threadIdx.x];
    s_lut[threadIdx.x + BLOCK_SIZE] = E4M3_LUT_MOE_GROUPED[threadIdx.x + BLOCK_SIZE];
    __syncthreads();

    const unsigned int n1 = blockIdx.x * GU_COLS_PER_CTA + local_out * 2;
    const unsigned int n2 = n1 + 1;
    if (n1 >= N) return;
    const bool have_n2 = (n2 < N);
    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int s1 = (n1 / FP8_BLOCK) * k_blocks;
    const unsigned int s2 = (n2 / FP8_BLOCK) * k_blocks;
    const unsigned long long r1 = (unsigned long long)n1 * K;
    const unsigned long long r2 = (unsigned long long)n2 * K;

    for (unsigned int row0 = begin; row0 < end; row0 += GU_GROUP_ROWS) {
        const unsigned int cnt = min((unsigned int)GU_GROUP_ROWS, end - row0);
        unsigned int a_row[GU_GROUP_ROWS];
        #pragma unroll
        for (int r = 0; r < GU_GROUP_ROWS; r++) {
            a_row[r] = (r < (int)cnt) ? grouped_a_row(sorted_token_ids, is_shared, row0 + r) : 0u;
        }
        // 2026-09-26: [projection][column]: g1, g2, u1, u2.
        float acc[4][GU_GROUP_ROWS];
        #pragma unroll
        for (int q = 0; q < 4; q++)
            #pragma unroll
            for (int r = 0; r < GU_GROUP_ROWS; r++) acc[q][r] = 0.0f;

        // 2026-09-26: The next chunk's weights and scales are loaded while the current one is
        // computed; each chunk is dequantized once per b for all the rows of the pass.
        uint4 w[4];
        float sc[4];
        #pragma unroll
        for (int q = 0; q < 4; q++) { w[q] = make_uint4(0u, 0u, 0u, 0u); sc[q] = 0.0f; }
        if (lane < K16) {
            const unsigned int kb = (lane * 16) / FP8_BLOCK;
            sc[0] = Sg[s1 + kb]; w[0] = *(const uint4*)(Wg + r1 + lane * 16);
            sc[2] = Su[s1 + kb]; w[2] = *(const uint4*)(Wu + r1 + lane * 16);
            if (have_n2) {
                sc[1] = Sg[s2 + kb]; w[1] = *(const uint4*)(Wg + r2 + lane * 16);
                sc[3] = Su[s2 + kb]; w[3] = *(const uint4*)(Wu + r2 + lane * 16);
            }
        }

        for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
            unsigned int cw[4][4];
            float cs[4];
            #pragma unroll
            for (int q = 0; q < 4; q++) {
                cw[q][0] = w[q].x; cw[q][1] = w[q].y; cw[q][2] = w[q].z; cw[q][3] = w[q].w;
                cs[q] = sc[q];
            }
            const unsigned int nk = k16 + threads_per_out;
            if (nk < K16) {
                const unsigned int kb = (nk * 16) / FP8_BLOCK;
                sc[0] = Sg[s1 + kb]; w[0] = *(const uint4*)(Wg + r1 + nk * 16);
                sc[2] = Su[s1 + kb]; w[2] = *(const uint4*)(Wu + r1 + nk * 16);
                if (have_n2) {
                    sc[1] = Sg[s2 + kb]; w[1] = *(const uint4*)(Wg + r2 + nk * 16);
                    sc[3] = Su[s2 + kb]; w[3] = *(const uint4*)(Wu + r2 + nk * 16);
                }
            }
            const unsigned int base_k = k16 * 16;
            #pragma unroll
            for (int b = 0; b < 4; b++) {
                float wf[4][4];
                #pragma unroll
                for (int q = 0; q < 4; q++)
                    #pragma unroll
                    for (int j = 0; j < 4; j++) wf[q][j] = s_lut[(cw[q][b] >> (8 * j)) & 0xFF] * cs[q];
                #pragma unroll
                for (int r = 0; r < GU_GROUP_ROWS; r++) {
                    if (r < (int)cnt) {
                        // 2026-09-26: Elements base_k + 4b .. + 3 of the row: the two words the
                        // per-row kernel takes from its uint4 loads for this b.
                        const uint2 a2 = *(const uint2*)(A + (unsigned long long)a_row[r] * K + base_k + 4 * b);
                        __nv_bfloat16 av0, av1, av2, av3;
                        *(unsigned short*)&av0 = (unsigned short)(a2.x & 0xFFFF);
                        *(unsigned short*)&av1 = (unsigned short)(a2.x >> 16);
                        *(unsigned short*)&av2 = (unsigned short)(a2.y & 0xFFFF);
                        *(unsigned short*)&av3 = (unsigned short)(a2.y >> 16);
                        const float af0 = __bfloat162float(av0), af1 = __bfloat162float(av1);
                        const float af2 = __bfloat162float(av2), af3 = __bfloat162float(av3);
                        #pragma unroll
                        for (int q = 0; q < 4; q++)
                            acc[q][r] += af0 * wf[q][0] + af1 * wf[q][1] + af2 * wf[q][2] + af3 * wf[q][3];
                    }
                }
            }
        }

        #pragma unroll
        for (int r = 0; r < GU_GROUP_ROWS; r++) {
            if (r < (int)cnt) {
                float v[4];
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    v[q] = acc[q][r];
                    #pragma unroll
                    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                        v[q] += __shfl_down_sync(0xFFFFFFFF, v[q], offset);
                }
                if (lane == 0) {
                    const unsigned long long base = (unsigned long long)(row0 + r) * N;
                    const float g1 = __bfloat162float(__float2bfloat16(v[0]));
                    const float u1 = __bfloat162float(__float2bfloat16(v[2]));
                    out[base + n1] = (g1 / (1.0f + __expf(-g1))) * u1;
                    if (have_n2) {
                        const float g2 = __bfloat162float(__float2bfloat16(v[1]));
                        const float u2 = __bfloat162float(__float2bfloat16(v[3]));
                        out[base + n2] = (g2 / (1.0f + __expf(-g2))) * u2;
                    }
                }
            }
        }
    }
}

// 2026-09-26: Down projection of the SiLU product act ([pos, K] FP32), DOWN_COLS_PER_WARP
// output columns per warp. A column group streams 16 KB of weight, so the SM keeps three blocks
// resident (at most 85 registers, 4 rows per pass) to overlap their loads with each other's
// arithmetic. 2026-09-27: A block runs DOWN_CG column groups in turn (DOWN_CTA_COLS columns),
// which took the M = 32..64 down launches 18% lower on GB10 than one group per block: the row
// lookups, pointer loads and table fill ahead of the first weight byte are paid once per 64 KB.
extern "C" __global__ void __launch_bounds__(DOWN_BLOCK, 3) moe_expert_down_act_fp8_grouped(
    const float* __restrict__ act,
    const unsigned long long* __restrict__ weight_ptrs,
    const unsigned long long* __restrict__ block_scale_ptrs,
    __nv_bfloat16* __restrict__ C,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ active_experts,
    const int* __restrict__ active_count,
    const float* __restrict__ sh_act,
    const unsigned char* __restrict__ sh_down_weight,
    const float* __restrict__ sh_down_block_scale,
    __nv_bfloat16* __restrict__ sh_down_out,
    unsigned int N, unsigned int K, unsigned int cap, unsigned int num_tokens
) {
    bool is_shared;
    unsigned int expert, begin, end;
    if (!grouped_block_rows(expert_offsets, active_experts, active_count, num_tokens,
                            GROUP_ROWS, &is_shared, &expert, &begin, &end)) return;

    const unsigned char* B_weight;
    const float* B_block_scale;
    const float* a_base;
    __nv_bfloat16* out_base;
    if (is_shared) {
        B_weight = sh_down_weight; B_block_scale = sh_down_block_scale;
        a_base = sh_act; out_base = sh_down_out;
    } else {
        B_weight = (const unsigned char*)weight_ptrs[expert];
        B_block_scale = (const float*)block_scale_ptrs[expert];
        a_base = act; out_base = C;
        if (B_weight == 0) {
            const unsigned int n_base = blockIdx.x * DOWN_CTA_COLS;
            for (unsigned int pos = begin; pos < end; pos++) {
                for (unsigned int i = threadIdx.x; i < DOWN_CTA_COLS && n_base + i < N; i += DOWN_BLOCK) {
                    C[(unsigned long long)pos * N + n_base + i] = __float2bfloat16(0.0f);
                }
            }
            return;
        }
    }

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_MOE_GROUPED[threadIdx.x];
    __syncthreads();

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    // 2026-09-27: DOWN_CG column groups of DOWN_COLS_PER_CTA per block, one after another,
    // so a block's row lookups and table fill serve DOWN_CG times the weight bytes.
    for (unsigned int cg = 0; cg < DOWN_CG; cg++) {
        const unsigned int n0 = (blockIdx.x * DOWN_CG + cg) * DOWN_COLS_PER_CTA + warp * DOWN_COLS_PER_WARP;
        if (n0 >= N) break;
        unsigned int ncol[DOWN_COLS_PER_WARP];
        bool have[DOWN_COLS_PER_WARP];
        #pragma unroll
        for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
            have[c] = (n0 + c < N);
            ncol[c] = have[c] ? n0 + c : 0;
        }
        const unsigned int K8 = K / 8;
        const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

        for (unsigned int row0 = begin; row0 < end; row0 += GROUP_ROWS) {
            const unsigned int cnt = min((unsigned int)GROUP_ROWS, end - row0);
            float acc[GROUP_ROWS][DOWN_COLS_PER_WARP];
            #pragma unroll
            for (int r = 0; r < GROUP_ROWS; r++)
                #pragma unroll
                for (int c = 0; c < DOWN_COLS_PER_WARP; c++) acc[r][c] = 0.0f;

            // 2026-09-26: The next chunk's weights and scales are loaded one chunk ahead.
            uint2 wv[DOWN_COLS_PER_WARP];
            float sc[DOWN_COLS_PER_WARP];
            #pragma unroll
            for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                wv[c] = make_uint2(0u, 0u); sc[c] = 0.0f;
                if (have[c] && lane < K8) {
                    sc[c] = B_block_scale[(ncol[c] / FP8_BLOCK) * k_blocks + (lane * 8) / FP8_BLOCK];
                    wv[c] = *(const uint2*)(B_weight + (unsigned long long)ncol[c] * K + lane * 8);
                }
            }

            for (unsigned int k8 = lane; k8 < K8; k8 += WARP_SIZE) {
                unsigned int wa[DOWN_COLS_PER_WARP], wb[DOWN_COLS_PER_WARP];
                float cs[DOWN_COLS_PER_WARP];
                #pragma unroll
                for (int c = 0; c < DOWN_COLS_PER_WARP; c++) { wa[c] = wv[c].x; wb[c] = wv[c].y; cs[c] = sc[c]; }
                const unsigned int nk = k8 + WARP_SIZE;
                if (nk < K8) {
                    #pragma unroll
                    for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                        if (have[c]) {
                            sc[c] = B_block_scale[(ncol[c] / FP8_BLOCK) * k_blocks + (nk * 8) / FP8_BLOCK];
                            wv[c] = *(const uint2*)(B_weight + (unsigned long long)ncol[c] * K + nk * 8);
                        }
                    }
                }
                const unsigned int base_k = k8 * 8;
                #pragma unroll
                for (int b = 0; b < 2; b++) {
                    float wf[DOWN_COLS_PER_WARP][4];
                    #pragma unroll
                    for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                        const unsigned int w32 = (b == 0) ? wa[c] : wb[c];
                        wf[c][0] = s_lut[(w32      ) & 0xFF] * cs[c];
                        wf[c][1] = s_lut[(w32 >>  8) & 0xFF] * cs[c];
                        wf[c][2] = s_lut[(w32 >> 16) & 0xFF] * cs[c];
                        wf[c][3] = s_lut[(w32 >> 24) & 0xFF] * cs[c];
                    }
                    #pragma unroll
                    for (int r = 0; r < GROUP_ROWS; r++) {
                        if (r < (int)cnt) {
                            const float4 al = *(const float4*)(a_base + (unsigned long long)(row0 + r) * K + base_k + b * 4);
                            #pragma unroll
                            for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                                acc[r][c] += al.x * wf[c][0] + al.y * wf[c][1] + al.z * wf[c][2] + al.w * wf[c][3];
                            }
                        }
                    }
                }
            }

            #pragma unroll
            for (int r = 0; r < GROUP_ROWS; r++) {
                if (r < (int)cnt) {
                    __nv_bfloat16* out = out_base + (unsigned long long)(row0 + r) * N;
                    #pragma unroll
                    for (int c = 0; c < DOWN_COLS_PER_WARP; c++) {
                        float v = acc[r][c];
                        #pragma unroll
                        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
                            v += __shfl_down_sync(0xFFFFFFFF, v, offset);
                        if (lane == 0 && have[c]) out[ncol[c]] = __float2bfloat16(v);
                    }
                }
            }
        }
    }
}
