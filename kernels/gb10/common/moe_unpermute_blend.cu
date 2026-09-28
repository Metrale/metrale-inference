// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: `moe_unpermute_blend`: `moe_unpermute_reduce_indexed` followed by `moe_batched_blend` (moe_permute.cu)
// in one pass per token, with the same bits:
//   routed[c] = bf16( sum_k topk_weights[t, k] * expert_output[token_to_perm[t, k], c] )   (k ascending, FP32)
//   output[c] = bf16( float(routed[c]) + g * shared_out[t, c] ),  g = sigmoid(dot(normed[t], gate_weight)) or 1
// The dot keeps the blend's reduction: thread i sums elements i, i + 256, ... in order, a shfl_down tree per warp,
// then thread 0 adds the 8 warp sums in order. The routed sum keeps each column's k order; a thread takes 8 adjacent
// columns so the expert rows are read as 16-byte vectors. Saves the routed output's store and reload and one launch.
//
// Owner: gb10 kernels.
// Invariants: block 256 (the dot's 8 warp slots); one block per token; hidden_size % 8 == 0 and 16-byte-aligned rows
// (the caller falls back to the two kernels otherwise).

#include <cuda_bf16.h>

extern "C" __global__ void __launch_bounds__(256) moe_unpermute_blend(
    const __nv_bfloat16* __restrict__ expert_output,
    __nv_bfloat16* __restrict__ output,
    const int* __restrict__ token_to_perm,
    const float* __restrict__ topk_weights,
    const __nv_bfloat16* __restrict__ shared_out,
    const __nv_bfloat16* __restrict__ normed,
    const __nv_bfloat16* __restrict__ gate_weight,
    unsigned int hidden_size,
    unsigned int num_tokens,
    unsigned int topk
) {
    __shared__ float s_dot_partial[8];
    const unsigned int token = blockIdx.x;
    if (token >= num_tokens) return;
    const unsigned int tid = threadIdx.x, warp_id = tid / 32, lane = tid % 32;
    const unsigned long long row = (unsigned long long)token * hidden_size;

    float local_dot = 0.0f;
    if (gate_weight != 0) {
        for (unsigned int i = tid; i < hidden_size; i += blockDim.x) {
            const float n = __bfloat162float(normed[row + i]);
            const float g = __bfloat162float(gate_weight[i]);
            local_dot += n * g;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) local_dot += __shfl_down_sync(0xFFFFFFFF, local_dot, offset);
    if (lane == 0) s_dot_partial[warp_id] = local_dot;
    __syncthreads();
    if (tid == 0) {
        float gate_scalar = 1.0f;
        if (gate_weight != 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < blockDim.x / 32; w++) total += s_dot_partial[w];
            gate_scalar = 1.0f / (1.0f + __expf(-total));
        }
        s_dot_partial[0] = gate_scalar;
    }
    __syncthreads();
    const float gate_scalar = s_dot_partial[0];

    for (unsigned int c0 = tid * 8; c0 < hidden_size; c0 += blockDim.x * 8) {
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (unsigned int k = 0; k < topk; k++) {
            const int perm_row = token_to_perm[token * topk + k];
            const float w = topk_weights[token * topk + k];
            const uint4 v = *(const uint4*)&expert_output[(unsigned long long)perm_row * hidden_size + c0];
            const __nv_bfloat16* e = (const __nv_bfloat16*)&v;
            #pragma unroll
            for (int j = 0; j < 8; j++) acc[j] += w * __bfloat162float(e[j]);
        }
        const uint4 sv = *(const uint4*)&shared_out[row + c0];
        const __nv_bfloat16* s = (const __nv_bfloat16*)&sv;
        uint4 ov;
        __nv_bfloat16* o = (__nv_bfloat16*)&ov;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const float routed = __bfloat162float(__float2bfloat16(acc[j]));
            o[j] = __float2bfloat16(routed + gate_scalar * __bfloat162float(s[j]));
        }
        *(uint4*)&output[row + c0] = ov;
    }
}
