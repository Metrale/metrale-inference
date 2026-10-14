// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: Kernels `moe_topk_softmax`, `moe_topk_softmax_rows`, `moe_topk_softmax_f32` and `moe_topk_softmax_batched`:
// softmax MoE routing; 2026-10-14: and `moe_topk_softmax_rows_warp`, `moe_topk_softmax_rows` from one warp per row.
// Each picks the top_k experts by gate logit, highest first, and returns their softmax weights.
//
// One 256-thread block per token. Only the first MAX_EXPERTS logits are read. The weight of a chosen expert is
// the softmax over all read logits taken at that expert; with `normalize` the top_k weights are then divided by
// their sum. When top_k exceeds num_experts, the output entries past num_experts are left unwritten.
//
// Owner: gb10 kernels.
// Invariants: none beyond the types. The kernels assume blockDim.x == 256 (BLOCK_SIZE; WARP_ROWS * 32 for
// `moe_topk_softmax_rows_warp`) and top_k <= MAX_TOP_K, and check neither.

#include <cuda_bf16.h>

#include "moe_topk_warp_tail.cuh"

#define BLOCK_SIZE 256
#define MAX_EXPERTS 512
#define MAX_TOP_K 32  // 2026-09-25: MoeLayer::new_with_hash refuses num_experts_per_tok above MOE_TOPK_SIGMOID_MAX_TOP_K (32)
#define WARP_SIZE 32

// 2026-09-25: `moe_topk_softmax`: one token, BF16 logits, grid (1, 1, 1). Experts are picked one per round by a
// block-wide argmax; on equal logits the lower expert index wins, in the thread, warp and cross-warp steps.
// 2026-09-26: The body is `topk_softmax_lowidx_block`, which `moe_topk_softmax_rows` also runs.







// 2026-10-14: A logit as FP32, so the BF16 and FP32 kernels share one body.
__device__ __forceinline__ float topk_logit_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float topk_logit_f32(float v) { return v; }

template <typename Logit>
__device__ __forceinline__ void topk_softmax_lowidx_block(
    const Logit* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;


    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = topk_logit_f32(gate_logits[i]);
    }

    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {

        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }




        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max || (other_val == local_max && other_idx < local_idx)) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }


        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();

        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val || (s_warp_val[w] == best_val && s_warp_idx[w] < best_idx)) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;

            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }

    // 2026-09-25: The softmax denominator runs over all read experts. Chosen experts are -1e30 in s_vals by now, so
    // they add zero to the parallel sum and thread 0 adds their terms back from s_top_vals. s_top_vals[0] is the
    // largest logit, the softmax max.








    float global_max = s_top_vals[0];



    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();


        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }

            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];


    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            expert_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;

            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                expert_weights[t] = softmax_weight;
            }
        }

        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                expert_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}

extern "C" __global__ void moe_topk_softmax(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    topk_softmax_lowidx_block(gate_logits, expert_indices, expert_weights, num_experts, top_k, normalize);
}

// 2026-09-26: `moe_topk_softmax_rows`: `moe_topk_softmax` for N tokens, one block per token, grid (N, 1, 1).
// Row t reads gate_logits[t * num_experts ..] and writes expert_indices / expert_weights[t * top_k ..]. It
// runs the same device function as `moe_topk_softmax`, so each row's indices, weights and lower-index
// tie-break are those of a `moe_topk_softmax` launch on that row (unlike `moe_topk_softmax_batched`).
extern "C" __global__ void moe_topk_softmax_rows(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    const unsigned int t = blockIdx.x;
    topk_softmax_lowidx_block(gate_logits + (unsigned long long)t * num_experts, expert_indices + t * top_k,
                              expert_weights + t * top_k, num_experts, top_k, normalize);
}

// 2026-10-14: `moe_topk_softmax_rows_warp`: `moe_topk_softmax_rows`' bytes from one warp per row, WARP_ROWS rows
// per block, grid (ceil(num_rows / WARP_ROWS), 1, 1), block WARP_ROWS * 32. Lane l holds experts l + 32 j in
// registers, so a pick needs no block barrier and no serial cross-warp step: each of the top_k rounds is a lane scan
// and a 5-step butterfly. A pick is the largest logit, the lower expert index on an equal one, whatever the tree.
// The softmax sum keeps the block body's tree: lane l plays thread 32 w + l of every warp w of the 256-thread
// block (its experts are slots w and w + 8), the per-warp shuffle-down steps run in the same order, and lane 0
// adds the 8 warp sums and then the picks' terms, in order, as thread 0 does (moe_topk_warp_tail.cuh, shared
// with the Hopper fused router). Assumes top_k >= 1, as the block body does.
#define WARP_ROWS 4
#define WARP_SLOTS (MAX_EXPERTS / WARP_SIZE)
#define BODY_WARPS (BLOCK_SIZE / WARP_SIZE)
static_assert(WARP_SLOTS == TOPK_WARP_SLOTS && BODY_WARPS == TOPK_WARP_BODY_WARPS,
              "moe_topk_warp_tail.cuh plays this file's block body");

extern "C" __global__ void __launch_bounds__(WARP_ROWS * WARP_SIZE) moe_topk_softmax_rows_warp(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize,
    unsigned int num_rows
) {
    __shared__ float s_top_vals[WARP_ROWS][MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[WARP_ROWS][MAX_TOP_K];

    const unsigned int warp = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;
    const unsigned int row = blockIdx.x * WARP_ROWS + warp;
    if (row >= num_rows) return;
    const __nv_bfloat16* g = gate_logits + (unsigned long long)row * num_experts;
    unsigned int* idx_out = expert_indices + row * top_k;
    float* w_out = expert_weights + row * top_k;
    float* top_vals = s_top_vals[warp];
    unsigned int* top_idxs = s_top_idxs[warp];

    const unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    const unsigned int rounds = top_k < actual_n ? top_k : actual_n;
    float v[WARP_SLOTS];
    #pragma unroll
    for (unsigned int j = 0; j < WARP_SLOTS; j++) {
        const unsigned int i = lane + WARP_SIZE * j;
        v[j] = i < actual_n ? __bfloat162float(g[i]) : -1e30f;
    }

    float global_max = 0.0f;
    for (unsigned int t = 0; t < rounds; t++) {
        float best_val = -1e30f;
        unsigned int best_idx = 0;
        #pragma unroll
        for (unsigned int j = 0; j < WARP_SLOTS; j++) {
            const unsigned int i = lane + WARP_SIZE * j;
            if (i < actual_n && v[j] > best_val) {
                best_val = v[j];
                best_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            const float other_val = __shfl_xor_sync(0xFFFFFFFF, best_val, offset);
            const unsigned int other_idx = __shfl_xor_sync(0xFFFFFFFF, best_idx, offset);
            if (other_val > best_val || (other_val == best_val && other_idx < best_idx)) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }
        if (t == 0) global_max = best_val;
        if (lane == 0) {
            top_vals[t] = best_val;
            top_idxs[t] = best_idx;
        }
        #pragma unroll
        for (unsigned int j = 0; j < WARP_SLOTS; j++) {
            if (lane + WARP_SIZE * j == best_idx) v[j] = -1e30f;
        }
    }

    topk_warp_softmax_tail(v, lane, actual_n, rounds, global_max, top_vals, top_idxs, normalize, idx_out, w_out);
}

// 2026-09-25: `moe_topk_softmax_f32`: `moe_topk_softmax` with F32 logits, same tie-break. The MoE layer uses it when
// the router logits are kept in F32 (the `fp32_gate` or `fp32_routing` lever, METRALE_FP32_GATE /
// METRALE_FP32_ROUTING).






extern "C" __global__ void moe_topk_softmax_f32(
    const float* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    topk_softmax_lowidx_block(gate_logits, expert_indices, expert_weights, num_experts, top_k, normalize);
}

// 2026-09-25: `moe_topk_softmax_batched`: BF16 logits for N tokens, one block per token, grid (N, 1, 1). Its warp
// and cross-warp steps keep the lower lane's or warp's candidate on an equal logit, which need not be the
// lower expert index that the single-token kernels pick.
// gate_logits [N, num_experts]; expert_indices and expert_weights [N, top_k].



extern "C" __global__ void moe_topk_softmax_batched(
    const __nv_bfloat16* __restrict__ gate_logits,
    unsigned int* __restrict__ expert_indices,
    float* __restrict__ expert_weights,
    unsigned int num_experts,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_vals[MAX_EXPERTS];
    __shared__ float s_top_vals[MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[MAX_TOP_K];
    __shared__ float s_warp_val[8];
    __shared__ unsigned int s_warp_idx[8];

    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = BLOCK_SIZE / 32;


    const __nv_bfloat16* my_gate = gate_logits + token * num_experts;
    unsigned int* my_indices = expert_indices + token * top_k;
    float* my_weights = expert_weights + token * top_k;

    unsigned int actual_n = num_experts < MAX_EXPERTS ? num_experts : MAX_EXPERTS;
    for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
        s_vals[i] = __bfloat162float(my_gate[i]);
    }
    for (unsigned int i = actual_n + tid; i < MAX_EXPERTS; i += BLOCK_SIZE) {
        s_vals[i] = -1e30f;
    }
    __syncthreads();


    for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
        float local_max = -1e30f;
        unsigned int local_idx = 0;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            float v = s_vals[i];
            if (v > local_max) {
                local_max = v;
                local_idx = i;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, local_max, offset);
            unsigned int other_idx = __shfl_down_sync(0xFFFFFFFF, local_idx, offset);
            if (other_val > local_max) {
                local_max = other_val;
                local_idx = other_idx;
            }
        }
        if (lane == 0) {
            s_warp_val[warp_id] = local_max;
            s_warp_idx[warp_id] = local_idx;
        }
        __syncthreads();
        if (tid == 0) {
            float best_val = s_warp_val[0];
            unsigned int best_idx = s_warp_idx[0];
            for (unsigned int w = 1; w < num_warps; w++) {
                if (s_warp_val[w] > best_val) {
                    best_val = s_warp_val[w];
                    best_idx = s_warp_idx[w];
                }
            }
            s_top_vals[t] = best_val;
            s_top_idxs[t] = best_idx;
            s_vals[best_idx] = -1e30f;
        }
        __syncthreads();
    }


    float global_max = s_top_vals[0];
    {
        float local_sum = 0.0f;
        for (unsigned int i = tid; i < actual_n; i += BLOCK_SIZE) {
            local_sum += __expf(s_vals[i] - global_max);
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        }
        if (lane == 0) s_warp_val[warp_id] = local_sum;
        __syncthreads();
        if (tid == 0) {
            float total = 0.0f;
            for (unsigned int w = 0; w < num_warps; w++) {
                total += s_warp_val[w];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                total += __expf(s_top_vals[t] - global_max);
            }
            s_warp_val[0] = total;
        }
        __syncthreads();
    }

    float exp_sum = s_warp_val[0];
    if (tid == 0) {
        for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
            my_indices[t] = s_top_idxs[t];
            float softmax_weight = __expf(s_top_vals[t] - global_max) / exp_sum;
            if (normalize) {
                s_top_vals[t] = softmax_weight;
            } else {
                my_weights[t] = softmax_weight;
            }
        }
        if (normalize) {
            float topk_sum = 0.0f;
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                topk_sum += s_top_vals[t];
            }
            for (unsigned int t = 0; t < top_k && t < actual_n; t++) {
                my_weights[t] = s_top_vals[t] / topk_sum;
            }
        }
    }
}
