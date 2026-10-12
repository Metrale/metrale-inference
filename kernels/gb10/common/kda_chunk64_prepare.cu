// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: The per-chunk factors of the 64-token chunked KDA prefill (`METRALE_GLM_KDA_CHUNK64`;
// the recurrence and the WY form are in kda_chunk64.cu's header), one block of 256 threads per
// (slab-local chunk, head), FP32 on CUDA cores, 48.25 KiB of shared memory so two blocks share
// an SM; kda_chunk64.cuh has the scratch layout.
//
// Pairs: G = the in-chunk cumsum of the log gate (thread per channel), decay = exp(G_last), then
// A[i,j] (j < i) and P[i,j] (j <= i): one warp per row i, lanes over channels (4 each), 16
// columns j at a time with the 16 partial sums per lane reduced across the warp by a butterfly
// transpose (31 shuffles for 16 pairs, not 31 per pair; 16 columns, not 32, keep the kernel
// under 128 registers for two blocks per SM). q is read from global memory once per row. A rows
// go to the block's global scratch slot as they complete (a fully unrolled row loop holding them
// in registers spilled); once every warp is done with K, A is read back (L2) over K's shared
// memory. P goes out as BF16. G stays.
// Solve: the forward substitution U_i = beta_i (v_i - sum_{j<i} A_ij U_j) (threads 0..127, one
// channel each) and W_i = beta_i (exp(G_i) k_i - sum A_ij W_j) (threads 128..255), A rows read
// as float4, four partial sums per row, exp(G_i) from the staged G; Qg = scale exp(G_i) q_i by
// the U threads and Kd = exp(G_last - G_i) k_i by the W threads. The rows' q/k/v come from
// global memory KDA64_PRE rows ahead through a compile-time register ring.
//
// Every exponent taken is <= 0 (G_i - G_j for j <= i, G_i, G_last - G_i), so nothing overflows.
// Positions t >= T read as k = v = q = 0, gate = 0, beta = 0.
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim D = KDA64_D (128), chunk KDA64_C (64). q, k, v BF16 rows `qkv_stride` apart (q at
//   0, k at H * D, v at 2 H * D elements of a row), gate FP32 [T, gate_stride], beta FP32 [T, H].
//   Block (lc, h) covers tokens (c0 + lc) * C .. and writes slab slot cb = lc * H + h.
// - grid (slab chunks, H), block KDA64_PREP_THREADS, dynamic shared memory KDA64_PREP_SMEM.

#include <cuda_bf16.h>
#include <stdint.h>

#include "kda_chunk64.cuh"

#define KDA64_PRE 4

// 2026-10-12: 16-byte cp.async that zero-fills instead of reading when `live` is false (the
// source operand is then unused but must still be an address).
__device__ __forceinline__ void k64_cp16z(uint32_t saddr, const void* g, bool live) {
    const unsigned int n = live ? 16u : 0u;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(g), "r"(n));
}

// 2026-10-12: Butterfly transpose-reduce of 16 slots over 32 lanes: on return, lane l's v[0]
// holds the sum over the 32 lanes of their v[l & 15]. Lane halves are summed first; then step s
// (8, 4, 2, 1) pairs the live slots (r, r + s), r < s: a lane with bit s clear keeps slot r and
// sends slot r + s, a lane with bit s set keeps r + s and sends r; the kept value plus the
// partner's sent value lands in slot r, which then holds logical index r | (lane & ~(s - 1) & 15)
// summed over the lanes agreeing with this one on the bits >= s. Slot indices are compile-time,
// so v stays in registers.
__device__ __forceinline__ void k64_transpose_reduce(float (&v)[16], unsigned int lane) {
#pragma unroll
    for (int r = 0; r < 16; ++r) v[r] += __shfl_xor_sync(0xFFFFFFFFu, v[r], 16);
#pragma unroll
    for (int s = 8; s >= 1; s >>= 1) {
        const bool upper = (lane & (unsigned int)s) != 0u;
#pragma unroll
        for (int r = 0; r < s; ++r) {
            const float lo = v[r], hi = v[r + s];
            const float recv = __shfl_xor_sync(0xFFFFFFFFu, upper ? lo : hi, s);
            v[r] = (upper ? hi : lo) + recv;
        }
    }
}

extern "C" __global__ void __launch_bounds__(KDA64_PREP_THREADS, 2) kda_chunk64_prepare(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ u_out,
    __nv_bfloat16* __restrict__ w_out,
    __nv_bfloat16* __restrict__ qg_out,
    __nv_bfloat16* __restrict__ kd_out,
    __nv_bfloat16* __restrict__ p_out,
    float* __restrict__ ad_out,
    unsigned int H,
    unsigned int T,
    unsigned int c0,
    unsigned int qkv_stride,
    unsigned int gate_stride,
    float scale
) {
    constexpr int C = KDA64_C, D = KDA64_D, NW = KDA64_PREP_THREADS / 32, PRE = KDA64_PRE;
    extern __shared__ __align__(16) float k64p_sh[];
    float* sG = k64p_sh;                               // [C][D]
    __nv_bfloat16* sK = (__nv_bfloat16*)(sG + C * D);  // [C][D], then A FP32 [C][C]
    float* sA = sG + C * D;
    float* sB = sA + C * C;                            // [C]
    const unsigned int lc = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const unsigned int t0 = (c0 + lc) * C;
    const size_t cb = (size_t)lc * H + h;
    const size_t hd = (size_t)h * D;
    const uint32_t bG = (uint32_t)__cvta_generic_to_shared(sG);
    const uint32_t bK = (uint32_t)__cvta_generic_to_shared(sK);

    // 2026-10-12: gate rows (16 B = 4 floats) and k rows (16 B = 8 halves), zero past T.
    for (unsigned int e = tid; e < (unsigned int)(C * D / 4); e += blockDim.x) {
        const unsigned int i = e / (D / 4), q4 = e % (D / 4);
        const unsigned int t = t0 + i;
        const bool live = t < T;
        const float* src = gate + (size_t)(live ? t : 0u) * gate_stride + hd + q4 * 4u;
        k64_cp16z(bG + e * 16u, src, live);
    }
    for (unsigned int e = tid; e < (unsigned int)(C * D / 8); e += blockDim.x) {
        const unsigned int i = e / (D / 8), q8 = e % (D / 8);
        const unsigned int t = t0 + i;
        const bool live = t < T;
        const __nv_bfloat16* src = k + (size_t)(live ? t : 0u) * qkv_stride + hd + q8 * 8u;
        k64_cp16z(bK + e * 16u, src, live);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
    __syncthreads();
    if (tid < (unsigned int)D) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            acc += sG[i * D + tid];
            sG[i * D + tid] = acc;
        }
        ad_out[cb * KDA64_AD_STRIDE + C * C + tid] = __expf(acc);
    }
    __syncthreads();

    // 2026-10-12: Pairs. Warp w owns rows w, w + NW, ..; lanes 0..15 write each 16-column group.
    const unsigned int warp = tid >> 5, lane = tid & 31u, half = lane >> 4, lane16 = lane & 15u;
    __nv_bfloat16* P = p_out + cb * C * C;
    float* A = ad_out + cb * KDA64_AD_STRIDE;
    for (unsigned int i = warp; i < (unsigned int)C; i += NW) {
        const unsigned int t = t0 + i;
        float ki[D / 32], qi[D / 32], gi[D / 32];
#pragma unroll
        for (int e = 0; e < D / 32; ++e) {
            const unsigned int d = lane + 32u * (unsigned int)e;
            ki[e] = __bfloat162float(sK[i * D + d]);
            gi[e] = sG[i * D + d];
            qi[e] = t < T ? scale * __bfloat162float(q[(size_t)t * qkv_stride + hd + d]) : 0.0f;
        }
#pragma unroll
        for (int gidx = 0; gidx < C / 16; ++gidx) {
            const unsigned int j0 = 16u * (unsigned int)gidx, j = j0 + lane16;
            if (j0 > i) {
                if (half == 0u) {
                    A[i * C + j] = 0.0f;
                    P[i * C + j] = __float2bfloat16_rn(0.0f);
                }
                continue;
            }
            // 2026-10-12: Columns j0 .. min(j0 + 15, i); the bound is warp-uniform.
            const unsigned int n = min(16u, i - j0 + 1u);
            float a[16], p[16];
#pragma unroll
            for (int jj = 0; jj < 16; ++jj) {
                a[jj] = 0.0f;
                p[jj] = 0.0f;
                if ((unsigned int)jj < n) {
                    const unsigned int jr = j0 + (unsigned int)jj;
#pragma unroll
                    for (int e = 0; e < D / 32; ++e) {
                        const unsigned int d = lane + 32u * (unsigned int)e;
                        const float kj = __bfloat162float(sK[jr * D + d]) * __expf(gi[e] - sG[jr * D + d]);
                        a[jj] += ki[e] * kj;
                        p[jj] += qi[e] * kj;
                    }
                }
            }
            k64_transpose_reduce(a, lane);
            k64_transpose_reduce(p, lane);
            if (half == 0u) {
                A[i * C + j] = j < i ? a[0] : 0.0f;
                P[i * C + j] = __float2bfloat16_rn(j <= i ? p[0] : 0.0f);
            }
        }
    }
    // 2026-10-12: K is dead; A (this block's own writes, visible after the barrier) and beta
    // take its place. G stays for the solve.
    __syncthreads();
    for (unsigned int e = tid; e < (unsigned int)(C * C / 4); e += blockDim.x) {
        *(float4*)(sA + e * 4u) = *(const float4*)(A + e * 4u);
    }
    if (tid < (unsigned int)C) sB[tid] = (t0 + tid < T) ? beta[(size_t)(t0 + tid) * H + h] : 0.0f;
    __syncthreads();

    // 2026-10-12: Solve. Row i's q/k/v (this channel) are fetched PRE rows ahead into a ring.
    const unsigned int d = tid % D;
    const bool is_w = tid >= (unsigned int)D;
    const size_t base = cb * C * D;
    const float gl = sG[(C - 1) * D + d];
    auto fetch = [&](int i) -> float2 {
        const unsigned int t = t0 + (unsigned int)i;
        if (t >= T) return make_float2(0.0f, 0.0f);
        const size_t qo = (size_t)t * qkv_stride + hd + d;
        return is_w ? make_float2(__bfloat162float(k[qo]), 0.0f)
                    : make_float2(__bfloat162float(v[qo]), __bfloat162float(q[qo]));
    };
    float2 ring[PRE];
#pragma unroll
    for (int i = 0; i < PRE; ++i) ring[i] = fetch(i);
    float hist[C];
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const float2 in = ring[i % PRE];
        if (i + PRE < C) ring[i % PRE] = fetch(i + PRE);
        const float G = sG[i * D + d];
        const float eg = __expf(G);
        float rhs;
        if (is_w) {
            rhs = eg * in.x;
            kd_out[base + (size_t)i * D + d] = __float2bfloat16_rn(__expf(gl - G) * in.x);
        } else {
            rhs = in.x;
            qg_out[base + (size_t)i * D + d] = __float2bfloat16_rn(scale * eg * in.y);
        }
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < i; j += 4) {
            const float4 a4 = *(const float4*)(sA + i * C + j);
            acc[0] += a4.x * hist[j];
            if (j + 1 < i) acc[1] += a4.y * hist[j + 1];
            if (j + 2 < i) acc[2] += a4.z * hist[j + 2];
            if (j + 3 < i) acc[3] += a4.w * hist[j + 3];
        }
        hist[i] = sB[i] * (rhs - ((acc[0] + acc[1]) + (acc[2] + acc[3])));
        if (is_w) {
            w_out[base + (size_t)i * D + d] = __float2bfloat16_rn(hist[i]);
        } else {
            u_out[base + (size_t)i * D + d] = hist[i];
        }
    }
}
