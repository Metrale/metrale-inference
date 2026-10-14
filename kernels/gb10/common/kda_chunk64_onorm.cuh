// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The sigmoid-gated RMSNorm of the KDA output (kda_o_norm_gated_bf16 in
// kda_layer_ops.cu) inside the 64-token scan (`METRALE_GLM_KDA_ONORM_IN_SCAN`,
// kda_chunk64_scan_onorm in kda_chunk64.cu), so the FP32 core rows are normalised while they are
// in L2 instead of one more launch re-reading every row from DRAM.
//
// kda_o_norm_gated_bf16 runs one block of head_dim = 128 threads per (token, head) row: thread i
// squares column i, warp w sums its 32 columns by a __shfl_down_sync tree (16, 8, 4, 2, 1), and
// warp 0 folds the four warp sums r_w (lanes 4.. holding 0) as ((r0 + r2) + (r1 + r3)); then
// inv = rsqrtf(sum / head_dim + eps) and out_i = bf16(x_i * inv * w_i * sigmoid(g_i)). A scan
// block owns 64 value columns of one head, and its warp w owns rows 16 (w & 3) .. x the 32
// columns 32 (w >> 2) .. of them: exactly one o_norm warp's columns. So:
// - k64_onorm_part: each scan warp forms its rows' warp sums by o_norm's tree, pair for pair,
//   from the accumulator layout (column 8 nt + 2 tig + e: the 16-step pairs nt with nt + 2, the
//   8-step nt 0 with 1, the 4- and 2-steps are lane shuffles inside the quad, the 1-step pairs
//   e 0 with 1) and stores them to `part`;
// - k64_onorm_tail: the last of the head's D / KDA64_VB blocks to arrive at a chunk (a counter,
//   no waiting) folds the four sums as o_norm does and writes the chunk's 64 x 128 BF16 outputs
//   from the core rows just written (read back through L2), the gate and the weight.
// The arithmetic is o_norm's, operation for operation, so the output bytes are its bytes; the
// model-arch example glm5next_kda_onorm_in_scan_parity compares them.
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim KDA64_D = 128; gate and out BF16 [T, out_stride] (the core's row stride), weight
//   BF16 [D]; part FP32 [slab chunk][H][C][D / 32]; count u32 [slab chunk][H], zero between
//   launches (the last block puts it back).
// - Rows t >= T are not written.

#ifndef KDA_CHUNK64_ONORM_CUH
#define KDA_CHUNK64_ONORM_CUH

#include <cuda_bf16.h>
#include <stdint.h>

#include "kda_chunk64.cuh"

// 2026-10-14: What the o_norm needs beyond the scan's arguments.
struct K64Onorm {
    const __nv_bfloat16* gate;
    const __nv_bfloat16* weight;
    __nv_bfloat16* out;
    float* part;
    unsigned int* count;
    float eps;
};

// 2026-10-14: Warp sums of this warp's 16 rows over its 32 columns (o: rows row0 + 8 hr, columns
// 8 nt + 2 tig + e at o[nt][2 hr + e]) by o_norm's tree, stored by the tig == 0 lanes at
// part[(cb * C + row) * (D / 32) + quarter].
template <int NTW>
__device__ __forceinline__ void k64_onorm_part(const float (&o)[NTW][4], float* __restrict__ part,
                                               size_t cb, unsigned int row0,
                                               unsigned int quarter, unsigned int tig) {
    static_assert(NTW == 4, "a warp's 32 columns: one o_norm warp");
    constexpr int C = KDA64_C, Q = KDA64_D / 32;
#pragma unroll
    for (int hr = 0; hr < 2; ++hr) {
        float s2[2];
#pragma unroll
        for (int e = 0; e < 2; ++e) {
            const int k = hr * 2 + e;
            // 2026-10-14: o_norm's `acc = 0.0f; acc += f * f` for each column.
            float q[4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) q[nt] = 0.0f + o[nt][k] * o[nt][k];
            const float s8 = (q[0] + q[2]) + (q[1] + q[3]);
            const float s4 = s8 + __shfl_down_sync(0xffffffffu, s8, 2);
            s2[e] = s4 + __shfl_down_sync(0xffffffffu, s4, 1);
        }
        if (tig == 0) {
            part[(cb * C + row0 + (unsigned int)hr * 8u) * Q + quarter] = s2[0] + s2[1];
        }
    }
}

// 2026-10-14: Called by every block of the scan after the chunk's core rows and warp sums are
// stored and fenced (and a __syncthreads): the last of the head's `nblk` blocks to arrive writes
// the chunk's normalised rows; the others return at once.
__device__ __forceinline__ void k64_onorm_tail(const K64Onorm& on, const float* __restrict__ core,
                                               size_t cb, unsigned int t0, unsigned int T,
                                               unsigned int h, unsigned int out_stride,
                                               unsigned int nblk, unsigned int tid,
                                               float* s_inv, unsigned int* s_last) {
    constexpr int C = KDA64_C, D = KDA64_D, Q = KDA64_D / 32;
    if (tid == 0) {
        const unsigned int prev = atomicAdd(on.count + cb, 1u);
        *s_last = prev == nblk - 1u;
    }
    __syncthreads();
    if (*s_last == 0u) return;
    // 2026-10-14: Every other block's rows and sums are visible after the fence; the counter goes
    // back to zero for the next launch.
    __threadfence();
    if (tid == 0) on.count[cb] = 0u;
    if (tid < (unsigned int)C) {
        const float* p = on.part + (cb * C + tid) * Q;
        // 2026-10-14: o_norm's warp-0 fold: lane w holds r_w + 0 (lanes 4.. are 0), then the
        // 2- and 1-steps.
        float r[Q];
#pragma unroll
        for (int w = 0; w < Q; ++w) r[w] = __ldcg(p + w) + 0.0f;
        const float sum = (r[0] + r[2]) + (r[1] + r[3]);
        s_inv[tid] = rsqrtf(sum / (float)D + on.eps);
    }
    __syncthreads();
    // 2026-10-14: 64 rows x 128 columns, four columns per thread, 8 rows per pass.
    constexpr unsigned int ROWS_PER_PASS = KDA64_SCAN_THREADS / (D / 4);
#pragma unroll
    for (unsigned int pass = 0; pass < (unsigned int)C / ROWS_PER_PASS; ++pass) {
        const unsigned int i = pass * ROWS_PER_PASS + tid / (D / 4), col = (tid % (D / 4)) * 4u;
        const unsigned int t = t0 + i;
        if (t >= T) continue;
        const size_t at = (size_t)t * out_stride + (size_t)h * D + col;
        const float4 x = __ldcg((const float4*)(core + at));
        const uint2 graw = *(const uint2*)(on.gate + at);
        const uint2 wraw = *(const uint2*)(on.weight + col);
        const __nv_bfloat162* g2 = (const __nv_bfloat162*)&graw;
        const __nv_bfloat162* w2 = (const __nv_bfloat162*)&wraw;
        const float inv = s_inv[i];
        const float xs[4] = {x.x, x.y, x.z, x.w};
        const float gs[4] = {__low2float(g2[0]), __high2float(g2[0]), __low2float(g2[1]),
                             __high2float(g2[1])};
        const float ws[4] = {__low2float(w2[0]), __high2float(w2[0]), __low2float(w2[1]),
                             __high2float(w2[1])};
        float y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float s = 1.0f / (1.0f + __expf(-gs[j]));
            y[j] = xs[j] * inv * ws[j] * s;
        }
        uint2 packed;
        *(__nv_bfloat162*)&packed.x = __floats2bfloat162_rn(y[0], y[1]);
        *(__nv_bfloat162*)&packed.y = __floats2bfloat162_rn(y[2], y[3]);
        *(uint2*)(on.out + at) = packed;
    }
}

#endif
