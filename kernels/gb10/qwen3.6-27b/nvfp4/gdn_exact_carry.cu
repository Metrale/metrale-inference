// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: Carried-state exact MTP verify for K = 2, 3, 4 rows per sequence
// (gdn_exact_carry{2,3,4} and the _lazy forms), and the fold that matches it
// (gdn_exact_carry_flush): the per-token chain of this directory's
// gated_delta_rule_decode_f32_strided, K tokens in one launch, with the stash protocol of
// common/gated_delta_rule_carry.cu instead of h snapshots.
//
// Owner: gb10 kernels (qwen3.6-27b).
// Invariants:
// - Each token runs gdn_decode_f32_strided_body's expressions in its order (gated_delta_rule.cu
//   here, built under the same --fmad=false), including the per-head Frobenius clamp after the
//   update (SSM_STATE_MAX_NORM, the same value), with the state kept in registers between tokens:
//   an FP32 store and reload is exact, so the outputs are the bits of K strided decode launches.
// - Pending rows are applied first with the decode's update and clamp, as gdn_exact_carry_flush
//   applies them. common's gdn_carry_flush has no clamp, so a model served from this directory
//   folds through gdn_exact_carry_flush under the exact verify.
// - Write-back, stash and engaged words follow gated_delta_rule_carry.cu: the state (pending rows
//   folded) is written back when the eager form runs or pend + K > CARRY_CAP; engaged_flag[b] is
//   2 after a write-back, else 1.
// exact_carry_microtest (model-arch examples, TARGET=qwen3.8-27b) checks outputs and states over
// several verify rounds bitwise against the strided decode chain.
//
// Rows are sequence-major (row r = b * K + t): q and k at r * qk_stride (FP32), v at
// r * v_stride, gate and beta at r * gb_stride, the FP32 output at r * out_stride. Grid
// (num_v_heads, batch), block 128.

#include <cuda_bf16.h>
#include "../../common/gdn_carry_stash.cuh"

// 2026-10-01: gated_delta_rule.cu's clamp threshold; a different value here breaks the bit match.
#define EXACT_MAX_NORM 1000.0f

// 2026-10-01: The decode body's clamp: the block's sum of `norm_acc`, reduced in its order, and
// the rescale of the head's state when it exceeds the threshold. Every thread of the block calls
// it; `sums` is 4 floats of shared memory.
__device__ __forceinline__ void exact_clamp(float (&H_reg)[CARRY_KD], float norm_acc, float* sums,
                                            unsigned int tid) {
    float local_sq = norm_acc;
    for (int offset = 16; offset >= 1; offset >>= 1)
        local_sq += __shfl_down_sync(0xFFFFFFFF, local_sq, offset);
    if (tid % 32 == 0) sums[tid / 32] = local_sq;
    __syncthreads();
    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < 4; w++) total += sums[w];
        sums[0] = total;
    }
    __syncthreads();
    const float head_norm_sq = sums[0];
    // 2026-10-01: The next call's lane-0 writes stay behind every thread's read of sums[0].
    __syncthreads();
    if (head_norm_sq > EXACT_MAX_NORM * EXACT_MAX_NORM) {
        const float scale = EXACT_MAX_NORM * rsqrtf(head_norm_sq);
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j++) H_reg[j] *= scale;
    }
}

// 2026-10-01: One pending row applied to the column: the decode's update, then its clamp.
__device__ __forceinline__ void exact_fold_row(float (&H_reg)[CARRY_KD], const volatile float* k,
                                               float g, float vn, float* sums, unsigned int tid) {
    float norm_acc = 0.0f;
    #pragma unroll
    for (int j = 0; j < CARRY_KD; j += 4) {
        const float h0 = g * H_reg[j]     + k[j]     * vn;
        const float h1 = g * H_reg[j + 1] + k[j + 1] * vn;
        const float h2 = g * H_reg[j + 2] + k[j + 2] * vn;
        const float h3 = g * H_reg[j + 3] + k[j + 3] * vn;
        H_reg[j] = h0; H_reg[j + 1] = h1; H_reg[j + 2] = h2; H_reg[j + 3] = h3;
        norm_acc += h0 * h0;
        norm_acc += h1 * h1;
        norm_acc += h2 * h2;
        norm_acc += h3 * h3;
    }
    exact_clamp(H_reg, norm_acc, sums, tid);
}

template <int K, bool LAZY>
__device__ __forceinline__ void gdn_exact_carry_body(
    float* const* __restrict__ h_table,
    const float* __restrict__ query,
    const float* __restrict__ key,
    const float* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ output,
    float* __restrict__ carry_base,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int seq_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride,
    unsigned int out_stride,
    unsigned int* __restrict__ engaged_flag
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int head_repeat = num_v_heads / num_k_heads;
    const unsigned int kh = vh / head_repeat;
    const unsigned int v_dim = CARRY_VD;

    float* H_global = h_table[b] + (unsigned long long)vh * CARRY_KD * CARRY_VD;
    const unsigned int slot = slot_tab[b];
    float* S = carry_base + (unsigned long long)slot * seq_floats;
    const unsigned int np = pend[slot];
    const bool wb = !LAZY || np + K > CARRY_CAP;
    const unsigned int base = wb ? 0u : np;
    if (tid == 0 && vh == 0) engaged_flag[b] = wb ? 2u : 1u;

    __shared__ float sk[K][CARRY_KD], sq[K][CARRY_KD];
    __shared__ float pk[CARRY_CAP][CARRY_KD], pg[CARRY_CAP];
    __shared__ float sums[4];
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned long long row = (unsigned long long)b * K + t;
        sk[t][tid] = key[row * qk_stride + kh * k_dim + tid];
        sq[t][tid] = query[row * qk_stride + kh * k_dim + tid];
    }
    // 2026-10-01: Pending rows are read before the barrier: this block overwrites the same stash
    // rows below after a write-back.
    for (unsigned int t = 0; t < np; ++t) pk[t][tid] = CARRY_SK(S, t, vh)[tid];
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    __syncthreads();

    float H_reg[CARRY_KD];
    #pragma unroll
    for (int j = 0; j < CARRY_KD; j++) H_reg[j] = H_global[j * CARRY_VD + tid];
    for (unsigned int t = 0; t < np; ++t)
        exact_fold_row(H_reg, pk[t], pg[t], CARRY_VN(S, t, vh)[tid], sums, tid);
    if (wb && np > 0) {
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j++) H_global[j * CARRY_VD + tid] = H_reg[j];
    }

    float vn[K], gs[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned long long row = (unsigned long long)b * K + t;
        const float g = fminf(fmaxf(gate[row * gb_stride + vh], 1e-6f), 1.0f - 1e-6f);
        const float bt = beta[row * gb_stride + vh];
        const float v_i = value[row * v_stride + vh * v_dim + tid];
        const volatile float* smem_k = sk[t];
        const volatile float* smem_q = sq[t];

        float hk_dot = 0.0f;
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j += 4) {
            hk_dot += H_reg[j] * smem_k[j] + H_reg[j + 1] * smem_k[j + 1]
                    + H_reg[j + 2] * smem_k[j + 2] + H_reg[j + 3] * smem_k[j + 3];
        }
        const float v_new_i = (v_i - g * hk_dot) * bt;

        float q_dot = 0.0f;
        float norm_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j += 4) {
            const float h0 = g * H_reg[j]     + smem_k[j]     * v_new_i;
            const float h1 = g * H_reg[j + 1] + smem_k[j + 1] * v_new_i;
            const float h2 = g * H_reg[j + 2] + smem_k[j + 2] * v_new_i;
            const float h3 = g * H_reg[j + 3] + smem_k[j + 3] * v_new_i;
            H_reg[j] = h0; H_reg[j + 1] = h1; H_reg[j + 2] = h2; H_reg[j + 3] = h3;
            q_dot += h0 * smem_q[j] + h1 * smem_q[j + 1] + h2 * smem_q[j + 2] + h3 * smem_q[j + 3];
            norm_acc += h0 * h0;
            norm_acc += h1 * h1;
            norm_acc += h2 * h2;
            norm_acc += h3 * h3;
        }
        exact_clamp(H_reg, norm_acc, sums, tid);

        const float inv_sqrt_d = rsqrtf((float)k_dim);
        output[row * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
        vn[t] = v_new_i;
        gs[t] = g;
    }

    #pragma unroll
    for (int t = 0; t < K; ++t) {
        CARRY_VN(S, base + t, vh)[tid] = vn[t];
        CARRY_SK(S, base + t, vh)[tid] = sk[t][tid];
        if (tid == (unsigned int)t) *CARRY_G(S, base + t, vh) = gs[t];
    }
}

#define EXACT_CARRY_ENTRY(NAME, K, LAZY) \
extern "C" __global__ void __launch_bounds__(128, 1) NAME( \
    float* const* __restrict__ h_table, const float* __restrict__ query, \
    const float* __restrict__ key, const float* __restrict__ value, \
    const float* __restrict__ gate, const float* __restrict__ beta, \
    float* __restrict__ output, float* __restrict__ carry_base, \
    const unsigned int* __restrict__ slot_tab, const unsigned int* __restrict__ pend, \
    unsigned int seq_floats, unsigned int batch_size, unsigned int num_k_heads, \
    unsigned int num_v_heads, unsigned int k_dim, unsigned int qk_stride, \
    unsigned int v_stride, unsigned int gb_stride, unsigned int out_stride, \
    unsigned int* __restrict__ engaged_flag) { \
    gdn_exact_carry_body<K, LAZY>(h_table, query, key, value, gate, beta, output, carry_base, \
        slot_tab, pend, seq_floats, batch_size, num_k_heads, num_v_heads, k_dim, qk_stride, \
        v_stride, gb_stride, out_stride, engaged_flag); \
}
EXACT_CARRY_ENTRY(gdn_exact_carry2, 2, false)
EXACT_CARRY_ENTRY(gdn_exact_carry3, 3, false)
EXACT_CARRY_ENTRY(gdn_exact_carry4, 4, false)
EXACT_CARRY_ENTRY(gdn_exact_carry2_lazy, 2, true)
EXACT_CARRY_ENTRY(gdn_exact_carry3_lazy, 3, true)
EXACT_CARRY_ENTRY(gdn_exact_carry4_lazy, 4, true)

// 2026-10-01: gdn_carry_flush's twin with the clamp: apply pending rows without a verify and write
// H. Same arguments and grid as gdn_carry_flush (grid (num_v_heads, batch, layers), block 128).
extern "C" __global__ void __launch_bounds__(128, 1) gdn_exact_carry_flush(
    float* const* __restrict__ h_table,
    unsigned long long table_layer_entries,
    const float* __restrict__ carry_base,
    unsigned long long carry_layer_floats,
    const unsigned int* __restrict__ slot_tab,
    const unsigned int* __restrict__ pend,
    unsigned int pend_layer_entries,
    unsigned int seq_floats,
    unsigned int batch_size,
    unsigned int num_v_heads
) {
    const unsigned int vh = blockIdx.x;
    const unsigned int b = blockIdx.y;
    const unsigned int l = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    const unsigned int slot = slot_tab[b];
    const unsigned int np = pend[(unsigned long long)l * pend_layer_entries + slot];
    if (np == 0) return;
    const unsigned int tid = threadIdx.x;
    float* H = h_table[l * table_layer_entries + b] + (unsigned long long)vh * CARRY_KD * CARRY_VD;
    const float* S = carry_base + l * carry_layer_floats + (unsigned long long)slot * seq_floats;
    __shared__ float pk[CARRY_CAP][CARRY_KD], pg[CARRY_CAP];
    __shared__ float sums[4];
    for (unsigned int t = 0; t < np; ++t) pk[t][tid] = CARRY_SK(S, t, vh)[tid];
    if (tid < np) pg[tid] = *CARRY_G(S, tid, vh);
    __syncthreads();
    float H_reg[CARRY_KD];
    #pragma unroll
    for (int j = 0; j < CARRY_KD; j++) H_reg[j] = H[j * CARRY_VD + tid];
    for (unsigned int t = 0; t < np; ++t)
        exact_fold_row(H_reg, pk[t], pg[t], CARRY_VN(S, t, vh)[tid], sums, tid);
    #pragma unroll
    for (int j = 0; j < CARRY_KD; j++) H[j * CARRY_VD + tid] = H_reg[j];
}
