// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-01: Carried-state exact MTP verify for K = 2, 3, 4 rows per sequence
// (gdn_exact_carry{2,3,4} and the _lazy forms, with the fold gdn_exact_carry_flush): the
// per-token chain of this directory's gated_delta_rule_decode_f32_strided, K tokens in one
// launch, with the stash protocol of common/gated_delta_rule_carry.cu instead of h snapshots.
//
// Owner: gb10 kernels (qwen3.6-35b-a3b).
// Invariants:
// - Each token runs gdn_decode_f32_strided_body's expressions in its order (gated_delta_rule.cu
//   here, built under the same --fmad=false), with the state kept in registers between tokens:
//   an FP32 store and reload is exact, so the outputs are the bits of K strided decode launches.
// - The state in memory lags by `pend[slot]` accepted rows, held in the slot's stash; they are
//   applied first with the decode's update expression (g * h + k * vn), as gdn_carry_flush
//   applies them, so the register copy is what K decode launches would have left.
// - The state is written back (the pending rows folded in) when the eager form runs or the
//   verify's rows might not fit behind the pending ones (pend + K > CARRY_CAP); the verified
//   rows are stashed behind the pending ones, or from row 0 after a write-back. No other
//   h-state is written. engaged_flag[b] becomes 2 after a write-back, else 1.
// exact_carry_microtest (model-arch examples) checks outputs and states over several verify
// rounds bitwise against the strided decode chain.
//
// Rows are sequence-major (row r = b * K + t): q and k at r * qk_stride (FP32), v at
// r * v_stride, gate and beta at r * gb_stride, the FP32 output at r * out_stride. The h-state
// is a per-sequence pointer table (slab 0 of the verify WY tables) and slot_tab maps a batch
// position to its carry slot. Grid (num_v_heads, batch), block 128.

#include <cuda_bf16.h>
#include "../../common/gdn_carry_stash.cuh"

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
    for (int j = 0; j < CARRY_KD; j++) {
        H_reg[j] = H_global[j * CARRY_VD + tid];
    }
    for (unsigned int t = 0; t < np; ++t) {
        const float pv = CARRY_VN(S, t, vh)[tid];
        const float pgt = pg[t];
        const volatile float* vpk = pk[t];
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j++) H_reg[j] = pgt * H_reg[j] + vpk[j] * pv;
    }
    if (wb && np > 0) {
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j++) H_global[j * CARRY_VD + tid] = H_reg[j];
    }

    float vn[K], gs[K];
    #pragma unroll
    for (int t = 0; t < K; ++t) {
        const unsigned long long row = (unsigned long long)b * K + t;
        const float* smem_k = sk[t];
        const float v_i = value[row * v_stride + vh * v_dim + tid];
        const float g = fminf(
            fmaxf(gate[row * gb_stride + vh], 1e-6f),
            1.0f - 1e-6f
        );
        const float bt = beta[row * gb_stride + vh];

        float hk0 = 0.0f, hk1 = 0.0f, hk2 = 0.0f, hk3 = 0.0f;
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j += 4) {
            hk0 += H_reg[j]     * smem_k[j];
            hk1 += H_reg[j + 1] * smem_k[j + 1];
            hk2 += H_reg[j + 2] * smem_k[j + 2];
            hk3 += H_reg[j + 3] * smem_k[j + 3];
        }
        const float hk_dot = (hk0 + hk1) + (hk2 + hk3);

        const float v_new = (v_i - g * hk_dot) * bt;

        float qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f, qd3 = 0.0f;
        // 2026-10-01: Volatile shared reads, as the decode body does, so the 128 k values do not
        // stay live next to H_reg.
        const volatile float* vk = sk[t];
        const volatile float* vq = sq[t];
        #pragma unroll
        for (int j = 0; j < CARRY_KD; j += 4) {
            float h0 = g * H_reg[j]     + vk[j]     * v_new;
            float h1 = g * H_reg[j + 1] + vk[j + 1] * v_new;
            float h2 = g * H_reg[j + 2] + vk[j + 2] * v_new;
            float h3 = g * H_reg[j + 3] + vk[j + 3] * v_new;
            H_reg[j]     = h0;
            H_reg[j + 1] = h1;
            H_reg[j + 2] = h2;
            H_reg[j + 3] = h3;
            qd0 += h0 * vq[j];
            qd1 += h1 * vq[j + 1];
            qd2 += h2 * vq[j + 2];
            qd3 += h3 * vq[j + 3];
        }
        const float q_dot = (qd0 + qd1) + (qd2 + qd3);

        const float inv_sqrt_d = rsqrtf((float)k_dim);
        output[row * out_stride + vh * v_dim + tid] = q_dot * inv_sqrt_d;
        vn[t] = v_new;
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

// 2026-10-01: The fold under the exact verify (carry_flush_kernel looks it up in this module). This
// directory's strided decode has no state clamp, so it is gdn_carry_flush's body; the 27B directory
// defines its own with the clamp.
extern "C" __global__ void gdn_exact_carry_flush(
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
    carry_flush_body(h_table, table_layer_entries, carry_base, carry_layer_floats, slot_tab, pend,
                     pend_layer_entries, seq_floats, batch_size, num_v_heads);
}
