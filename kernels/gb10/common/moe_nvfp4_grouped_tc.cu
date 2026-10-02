// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-02: Tensor-core grouped NVFP4 MoE decode (W4A16): the gate+up (SiLU product) and down
// projections of moe_nvfp4_grouped.cu on mma.sync m16n8k16 BF16 tiles, FP32 accumulation: the
// NVFP4 point (Nvfp4G16 below) of the weight-format-parameterized warp in moe_grouped_tc.cuh,
// whose FP8 point is moe_fp8_grouped_tc.cu. Same grid contract (one block row per active expert,
// the first ceil(num_tokens / TC_ROWS) block rows the shared expert), same weights and kernel
// arguments as moe_nvfp4_grouped.cu.
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
//   tables). K % 256 == 0 and N a multiple of the CTA's columns (the host checks
//   `nvfp4_grouped_tc_shape_ok`).
// - A weight enters the MMA as the BF16 E2M1 * E4M3: exact, since the product has at most five
//   significant bits and lies in [2^-10, 2688]. s2 multiplies the FP32 sum once per output.
//   The products and sums are therefore those of the declared W4A16 arithmetic up to FP32
//   summation order.
// - Inside a 128-wide K chunk lane t = lane & 3 holds K = 32t .. 32t + 31 of its rows (one
//   16-byte weight load per row), and MMA i (0..7) takes K = 32t + 4i + {0,1} as fragment slots
//   2t, 2t+1 and K = 32t + 4i + {2,3} as 2t+8, 2t+9. The K order of a row's sum is fixed by K
//   alone.
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

#include "moe_grouped_tc.cuh"
#include "tc_weight_formats.cuh"

#define NTC_WARPS 4
#define NTC_THREADS (NTC_WARPS * 32)
// 2026-10-02: m-tiles (16 output columns) per warp and 128-K chunks per load group (a warp
// keeps 2 groups in flight). gate+up tiles are the gate and up rows of the same columns.
// K must be a multiple of 128 * the group's chunk count (`nvfp4_grouped_tc_shape_ok`).
#define NTC_GU_MT 1
#define NTC_DOWN_MT 2
// 2026-10-02: Down keeps one chunk per group: with its hi + lo activation words (32 per row
// group) two chunks per group take 250 registers, one takes 162.
#define NTC_GU_G 2
#define NTC_DOWN_G 1
#define NTC_GU_COLS (NTC_WARPS * 16 * NTC_GU_MT)
#define NTC_DOWN_COLS (NTC_WARPS * 16 * NTC_DOWN_MT)

// 2026-10-02: The gate+up kernel body over a weight-format policy WF (Nvfp4G16 or Nvfp4G16R).
template <class WF>
__device__ __forceinline__ void ntc_gate_up(
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
        gtc_warp<WF, true, NTC_GU_MT, 1, NTC_GU_G>(
            A, sorted_token_ids, true, begin, end, {sh_gate_packed, sh_gate_scale, sh_gate_s2},
            {sh_up_packed, sh_up_scale, sh_up_s2}, sh_act, N, K, f0);
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
    gtc_warp_routed<WF, true, NTC_GU_MT, NTC_GU_G>(
        A, sorted_token_ids, false, begin, end,
        {Pg, (const unsigned char*)gate_scale_ptrs[expert], gate_scale2[expert]},
        {Pu, (const unsigned char*)up_scale_ptrs[expert], up_scale2[expert]}, act, N, K, f0);
}

// 2026-10-02: The down kernel body over a weight-format policy WF.
template <class WF>
__device__ __forceinline__ void ntc_down(
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
        gtc_warp<WF, false, NTC_DOWN_MT, 1, NTC_DOWN_G>(
            sh_act, nullptr, true, begin, end, {sh_down_packed, sh_down_scale, sh_down_s2},
            {nullptr, nullptr, 0.f}, sh_down_out, N, K, f0);
        return;
    }
    const unsigned char* P = (const unsigned char*)packed_ptrs[expert];
    if (P == 0) {
        for (unsigned int pos = begin; pos < end; pos++)
            for (unsigned int i = threadIdx.x; i < NTC_DOWN_COLS; i += NTC_THREADS)
                C[(unsigned long long)pos * N + blockIdx.x * NTC_DOWN_COLS + i] = __float2bfloat16(0.0f);
        return;
    }
    gtc_warp_routed<WF, false, NTC_DOWN_MT, NTC_DOWN_G>(
        act, nullptr, true, begin, end, {P, (const unsigned char*)scale_ptrs[expert], scale2[expert]},
        {nullptr, nullptr, 0.f}, C, N, K, f0);
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
    ntc_gate_up<Nvfp4G16>(A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2, up_packed_ptrs, up_scale_ptrs, up_scale2, act, expert_offsets, sorted_token_ids, active_experts, active_count, sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_up_packed, sh_up_scale, sh_up_s2, sh_act, N, K, cap, num_tokens);
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
    ntc_down<Nvfp4G16>(act, packed_ptrs, scale_ptrs, scale2, C, expert_offsets, active_experts, active_count, sh_act, sh_down_packed, sh_down_scale, sh_down_s2, sh_down_out, N, K, cap, num_tokens);
}

// 2026-10-02: The same two kernels on MMA-paired nibbles (Nvfp4G16R; the weights permuted by
// nvfp4_repack_mma_pairs). Arguments as above.
extern "C" __global__ void __launch_bounds__(NTC_THREADS) moe_expert_gate_up_act_nvfp4_grouped_tc_r(
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
    ntc_gate_up<Nvfp4G16R>(A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2, up_packed_ptrs, up_scale_ptrs, up_scale2, act, expert_offsets, sorted_token_ids, active_experts, active_count, sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_up_packed, sh_up_scale, sh_up_s2, sh_act, N, K, cap, num_tokens);
}

extern "C" __global__ void __launch_bounds__(NTC_THREADS) moe_expert_down_act_nvfp4_grouped_tc_r(
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
    ntc_down<Nvfp4G16R>(act, packed_ptrs, scale_ptrs, scale2, C, expert_offsets, active_experts, active_count, sh_act, sh_down_packed, sh_down_scale, sh_down_s2, sh_down_out, N, K, cap, num_tokens);
}

// 2026-10-02: Permute the nibbles of each 32-bit word of `n_words` packed E2M1 words in place into
// the Nvfp4G16R order: element 2p goes to nibble 3 - p, element 2p + 1 to nibble 7 - p (p = 0..3).
// Applied once at load, after every copy the other paths build from the row-major order.
extern "C" __global__ void nvfp4_repack_mma_pairs(unsigned int* __restrict__ w, unsigned long long n_words) {
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n_words;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        const unsigned int v = w[i];
        unsigned int r = 0;
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            r |= ((v >> (8 * p)) & 0xFu) << (4 * (3 - p));
            r |= ((v >> (8 * p + 4)) & 0xFu) << (4 * (7 - p));
        }
        w[i] = r;
    }
}
