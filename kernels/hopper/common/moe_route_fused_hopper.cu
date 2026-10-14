// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: `moe_route_fused_hopper`: the exact per-row MoE router of the grouped decode
// (`GroupedRouting::PerRow`: `dense_gemv_bf16_batchm` -> `moe_topk_softmax_rows_warp` ->
// `moe_fp8_grouped_sort`) in one launch for up to RF_MAX_ROWS rows. The three launches are
// latency-bound at decode (each waits on a few dependent memory round trips and a launch gap);
// here every block issues its whole weight slice at once, and the block that finishes last runs
// the top-k and the sort, so the step pays one launch and one weight round trip.
//
// Owner: hopper kernels.
// Invariants:
// - Launch: grid (ceil(N / RF_OUTS), 1, 1), block (RF_BLOCK, 1, 1); M <= RF_MAX_ROWS,
//   K % 8 == 0, K / 8 <= 64 * RF_KV_MAX, N <= TOPK_WARP_MAX_EXPERTS, 1 <= top_k <= min(N,
//   RF_MAX_TOP_K). The host (ops::moe_route_fused_shape_ok) refuses anything else.
// - `counter` is one u32, zero before the launch; the last block puts it back to zero, so
//   launches that use one counter must not overlap (one per MoE layer, on its decode stream).
// - Logits: row t, column n is dense_gemv_bf16's bits on that row (dense_gemv_bf16_batchm.cu):
//   the 64-lane group of output n takes kv = l + 64 j in ascending j, lo-then-hi per 32-bit
//   word, products and sums rounded separately (--fmad=false and __fmul_rn/__fadd_rn),
//   shfl_down 16..1, then warp 0 + warp 1. Holding the weight slice in registers and reading A
//   per row changes where operands come from, not their values or order; a row's bits do not
//   depend on M.
// - Picks and weights: moe_topk_softmax_rows_warp's bytes. A pick there is the largest logit
//   above -1e30 (NaN never), the lower expert index on an equal one (-0 equals +0); here each
//   expert carries a 25-bit key, its BF16 logit mapped to an order-preserving 16-bit code
//   (-0 folded onto +0) above 511 - index, 0 when it cannot be picked, and a round takes the
//   warp's largest key (redux.sync), which is that same expert. A round with no key left is
//   the warp kernel's (-1e30, expert 0) and clears expert 0's logit as it does. The softmax
//   tail is the warp kernel's (moe_topk_warp_tail.cuh).
// - Sort outputs: moe_fp8_grouped_sort's (moe_fp8_grouped_sort.cuh, run by the last block on the
//   picks kept in shared memory). They may alias `logits`: the last block has read every logit
//   it needs before the barrier that precedes the sort.

#include <cuda_bf16.h>

#include "moe_fp8_grouped_sort.cuh"
#include "moe_topk_warp_tail.cuh"

#define RF_BLOCK 256
#define RF_OUTS 4
#define RF_KV_MAX 8
#define RF_MAX_ROWS 16
#define RF_ROW_GROUP 4
#define RF_MAX_TOP_K 32
#define RF_WARPS (RF_BLOCK / 32)

static_assert(RF_BLOCK == PMS_BLOCK, "the last block runs the sort body");
static_assert(RF_BLOCK == 64 * RF_OUTS, "one 64-lane group per output");

// 2026-10-14: The pick key of expert `i` with BF16 logit bits `bits` (float value `f`).
__device__ __forceinline__ unsigned int rf_key(unsigned short bits, float f, unsigned int i) {
    if (!(f > -1e30f)) return 0u;
    const unsigned int b = bits == 0x8000u ? 0u : bits;
    const unsigned int ord = (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
    return (ord << 9) | (511u - i);
}

extern "C" __global__ void __launch_bounds__(RF_BLOCK) moe_route_fused_hopper(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* logits,
    unsigned int* idx_out,
    float* w_out,
    int* sorted_token_ids,
    int* sorted_expert_ids,
    int* expert_offsets,
    int* token_to_perm,
    int* active_experts,
    int* active_count,
    unsigned int* counter,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int top_k,
    unsigned int normalize
) {
    __shared__ float s_part[RF_MAX_ROWS][RF_OUTS][2];
    __shared__ unsigned int s_last;
    __shared__ float s_top_vals[RF_WARPS][RF_MAX_TOP_K];
    __shared__ unsigned int s_top_idxs[RF_WARPS][RF_MAX_TOP_K];
    __shared__ unsigned int s_ids[RF_MAX_ROWS * RF_MAX_TOP_K];

    // 2026-10-14: Phase 1, every block: RF_OUTS logits of every row.
    const unsigned int g = threadIdx.x / 64;
    const unsigned int lane = threadIdx.x % 64;
    const unsigned int n = blockIdx.x * RF_OUTS + g;
    const bool active = n < N;
    const unsigned int K_VEC = K / 8;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)(active ? n : 0) * K);
    uint4 b[RF_KV_MAX];
    #pragma unroll
    for (int j = 0; j < RF_KV_MAX; j++) {
        const unsigned int kv = lane + 64 * j;
        b[j] = kv < K_VEC ? __ldcs(B_vec + kv) : make_uint4(0u, 0u, 0u, 0u);
    }
    const uint4* A_vec = (const uint4*)A;
    for (unsigned int t0 = 0; t0 < M; t0 += RF_ROW_GROUP) {
        float acc[RF_ROW_GROUP];
        #pragma unroll
        for (int r = 0; r < RF_ROW_GROUP; r++) acc[r] = 0.0f;
        #pragma unroll
        for (int j = 0; j < RF_KV_MAX; j++) {
            const unsigned int kv = lane + 64 * j;
            if (kv < K_VEC) {
                const unsigned int bw[4] = {b[j].x, b[j].y, b[j].z, b[j].w};
                #pragma unroll
                for (int r = 0; r < RF_ROW_GROUP; r++) {
                    if (t0 + r < M) {
                        const uint4 a = __ldg(A_vec + (unsigned long long)(t0 + r) * K_VEC + kv);
                        const unsigned int aw[4] = {a.x, a.y, a.z, a.w};
                        float s = acc[r];
                        #pragma unroll
                        for (int i = 0; i < 4; i++) {
                            s = __fadd_rn(s, __fmul_rn(__uint_as_float(aw[i] << 16),
                                                       __uint_as_float(bw[i] << 16)));
                            s = __fadd_rn(s, __fmul_rn(__uint_as_float(aw[i] & 0xFFFF0000u),
                                                       __uint_as_float(bw[i] & 0xFFFF0000u)));
                        }
                        acc[r] = s;
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < RF_ROW_GROUP; r++) {
            float a = acc[r];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                a = __fadd_rn(a, __shfl_down_sync(0xFFFFFFFF, a, offset));
            }
            if (lane % 32 == 0 && t0 + r < M) s_part[t0 + r][g][lane / 32] = a;
        }
    }
    __syncthreads();
    if (active) {
        for (unsigned int t = lane; t < M; t += 64) {
            logits[(unsigned long long)t * N + n] = __float2bfloat16(__fadd_rn(s_part[t][g][0], s_part[t][g][1]));
        }
    }

    // 2026-10-14: The last block to finish phase 1 sees every logit (each writer fences before
    // the barrier that precedes the count) and runs phases 2 and 3.
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) s_last = atomicAdd(counter, 1u) == gridDim.x - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (threadIdx.x == 0) atomicExch(counter, 0u);

    // 2026-10-14: Phase 2, last block: one warp per row, rows w, w + RF_WARPS.
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int wl = threadIdx.x % 32;
    const unsigned int actual_n = N < TOPK_WARP_MAX_EXPERTS ? N : TOPK_WARP_MAX_EXPERTS;
    const unsigned int rounds = top_k < actual_n ? top_k : actual_n;
    float* top_vals = s_top_vals[warp];
    unsigned int* top_idxs = s_top_idxs[warp];
    for (unsigned int row = warp; row < M; row += RF_WARPS) {
        const unsigned short* lg = (const unsigned short*)logits + (unsigned long long)row * N;
        float v[TOPK_WARP_SLOTS];
        unsigned int key[TOPK_WARP_SLOTS];
        #pragma unroll
        for (unsigned int j = 0; j < TOPK_WARP_SLOTS; j++) {
            const unsigned int i = wl + 32 * j;
            if (i < actual_n) {
                const unsigned short bits = __ldcg(lg + i);
                v[j] = __bfloat162float(__ushort_as_bfloat16(bits));
                key[j] = rf_key(bits, v[j], i);
            } else {
                v[j] = -1e30f;
                key[j] = 0u;
            }
        }
        for (unsigned int t = 0; t < rounds; t++) {
            unsigned int m = 0u;
            #pragma unroll
            for (unsigned int j = 0; j < TOPK_WARP_SLOTS; j++) m = max(m, key[j]);
            m = __reduce_max_sync(0xFFFFFFFFu, m);
            if (m == 0u) {
                if (wl == 0) {
                    top_vals[t] = -1e30f;
                    top_idxs[t] = 0u;
                    v[0] = -1e30f;
                }
            } else {
                if (wl == 0) top_idxs[t] = 511u - (m & 511u);
                #pragma unroll
                for (unsigned int j = 0; j < TOPK_WARP_SLOTS; j++) {
                    if (key[j] == m) {
                        top_vals[t] = v[j];
                        v[j] = -1e30f;
                        key[j] = 0u;
                    }
                }
            }
        }
        __syncwarp();
        topk_warp_softmax_tail(v, wl, actual_n, rounds, top_vals[0], top_vals, top_idxs, normalize,
                               idx_out + row * top_k, w_out + row * top_k);
        for (unsigned int t = wl; t < rounds; t += 32) s_ids[row * top_k + t] = top_idxs[t];
        __syncwarp();
    }

    // 2026-10-14: Phase 3, last block: the slot sort over every row's picks.
    __syncthreads();
    pms_sort_block(s_ids, sorted_token_ids, sorted_expert_ids, expert_offsets, token_to_perm, active_experts,
                   active_count, M * top_k, N, top_k);
}
