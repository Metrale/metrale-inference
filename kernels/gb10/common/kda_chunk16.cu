// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: Chunked KDA prefill for long prompts, FP32 throughout, in 16-token chunks:
// kda_chunk16_prepare works on every (chunk, head) in parallel and kda_chunk16_scan carries the
// recurrent state across the chunks of one head, its 128 value columns split over four blocks
// (a state column depends only on its own column). It computes the recurrence of
// kda_recurrent.cu, per token t and head h:
//   S <- diag(exp(g_t)) S;  delta = (v_t - S^T k_t) * beta_t;  S <- S + k_t delta^T;
//   out_t = scale * S^T q_t
// over a chunk at once (the WY form). With G_i = sum_{p <= i} g_p inside the chunk (a log decay
// per channel, every g in (lower_bound, 0) so |G| <= 16 * |lower_bound|), S0 the state before
// the chunk:
//   A[i,j] = sum_d k_i[d] k_j[d] exp(G_i[d] - G_j[d])                (j < i)
//   delta_i = beta_i (v_i - S0^T (exp(G_i) k_i) - sum_{j<i} A[i,j] delta_j)
//     => delta = U - W S0 with U_i = beta_i (v_i - sum_{j<i} A[i,j] U_j),
//                              W_i = beta_i (exp(G_i) k_i - sum_{j<i} A[i,j] W_j)
//   out_i = (scale exp(G_i) q_i)^T S0 + sum_{j<=i} P[i,j] delta_j,
//     P[i,j] = scale sum_d q_i[d] k_j[d] exp(G_i[d] - G_j[d])        (j <= i)
//   S_end = diag(exp(G_last)) S0 + sum_i (exp(G_last - G_i) k_i) delta_i^T
// Every exponent is G_i - G_j <= 0, or G_i >= -16 * 5 = -80 at GLM-5.3's lower bound -5: no
// overflow, and exp(G_i) stays a normal FP32 number. The products and sums are in another order
// than the per-token walk's, so the outputs and the state differ from it at FP32 rounding level
// (model-arch example glm5next_kda_chunk16_bench measures the drift on checkpoint-like inputs).
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim D = KDA16_D (128); chunk KDA16_C (16). q, k, v BF16 rows `qkv_stride` apart (the
//   conv's [T, 3 * H * D] output: q at 0, k at H * D, v at 2 * H * D elements of a row), gate
//   FP32 [T, H * D], beta FP32 [T, H], out FP32 [T, H * D]; the state FP32 [H, D, D] indexed
//   [k][v], read before the first chunk and written after the last.
// - Scratch per (chunk, head), [nchunks, H, ...]: U, W, Qg, Kd as [C, D] FP32, P as [C, C] FP32,
//   decay as [D] FP32. Positions t >= T read as k = v = q = 0, gate = 0, beta = 0, so they add
//   nothing, and their out rows are not written.
// - prepare: grid (nchunks, H), block KDA16_D. scan: grid (H, D / KDA16_VB), block 256.

#include <cuda_bf16.h>
#include <math.h>

#define KDA16_D 128
#define KDA16_C 16
#define KDA16_VB 32
#define KDA16_SCAN_THREADS 256

extern "C" __global__ void __launch_bounds__(KDA16_D) kda_chunk16_prepare(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ u_out,
    float* __restrict__ w_out,
    float* __restrict__ qg_out,
    float* __restrict__ kd_out,
    float* __restrict__ p_out,
    float* __restrict__ decay_out,
    unsigned int H,
    unsigned int T,
    unsigned int qkv_stride,
    unsigned int gate_stride,
    float scale
) {
    constexpr int C = KDA16_C, D = KDA16_D;
    __shared__ float sG[C][D];
    __shared__ float sK[C][D];
    __shared__ float sQ[C][D];
    __shared__ float sE[C][D];
    __shared__ float sEi[C][D];
    __shared__ float sA[C][C];
    __shared__ float sB[C];
    const unsigned int c = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const unsigned int t0 = c * C;

    float vv[C];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const unsigned int t = t0 + i;
        const bool live = t < T;
        const size_t qo = (size_t)t * qkv_stride + (size_t)h * D + d;
        acc += live ? gate[(size_t)t * gate_stride + (size_t)h * D + d] : 0.0f;
        sG[i][d] = acc;
        sE[i][d] = expf(acc);
        sEi[i][d] = expf(-acc);
        sK[i][d] = live ? __bfloat162float(k[qo]) : 0.0f;
        sQ[i][d] = live ? __bfloat162float(q[qo]) : 0.0f;
        vv[i] = live ? __bfloat162float(v[qo]) : 0.0f;
    }
    if (d < C) sB[d] = (t0 + d < T) ? beta[(size_t)(t0 + d) * H + h] : 0.0f;
    __syncthreads();

    // 2026-10-10: A (j < i) and P (j <= i) share exp(G_i - G_j) = exp(G_i) exp(-G_j) (both
    // normal FP32 numbers for |G| <= 80); one warp per pair, lanes over d.
    const unsigned int warp = d >> 5, lane = d & 31u;
    float* P = p_out + ((size_t)c * H + h) * C * C;
    for (unsigned int pi = warp; pi < (unsigned int)(C * (C + 1) / 2); pi += D / 32) {
        unsigned int i = 0, rem = pi;
        while (rem > i) { rem -= i + 1; ++i; }
        const unsigned int j = rem;
        float a = 0.0f, p = 0.0f;
#pragma unroll
        for (int e = 0; e < D / 32; ++e) {
            const unsigned int dd = lane + 32u * e;
            const float x = sE[i][dd] * sEi[j][dd];
            a += sK[i][dd] * sK[j][dd] * x;
            p += sQ[i][dd] * sK[j][dd] * x;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
            p += __shfl_xor_sync(0xFFFFFFFFu, p, o);
        }
        if (lane == 0) {
            if (j < i) sA[i][j] = a;
            P[i * C + j] = p * scale;
        }
    }
    for (unsigned int e = d; e < (unsigned int)(C * C); e += D) {
        const unsigned int i = e / C, j = e % C;
        if (j > i) P[i * C + j] = 0.0f;
    }
    __syncthreads();

    // 2026-10-10: Forward substitution, one channel per thread.
    const size_t base = ((size_t)c * H + h) * C * D + d;
    const float gl = sG[C - 1][d];
    float uu[C], ww[C];
#pragma unroll
    for (int i = 0; i < C; ++i) {
        float su = vv[i], sw = sE[i][d] * sK[i][d];
#pragma unroll
        for (int j = 0; j < i; ++j) {
            su -= sA[i][j] * uu[j];
            sw -= sA[i][j] * ww[j];
        }
        uu[i] = sB[i] * su;
        ww[i] = sB[i] * sw;
        u_out[base + (size_t)i * D] = uu[i];
        w_out[base + (size_t)i * D] = ww[i];
        qg_out[base + (size_t)i * D] = scale * sE[i][d] * sQ[i][d];
        kd_out[base + (size_t)i * D] = expf(gl - sG[i][d]) * sK[i][d];
    }
    decay_out[((size_t)c * H + h) * D + d] = expf(gl);
}

extern "C" __global__ void __launch_bounds__(KDA16_SCAN_THREADS) kda_chunk16_scan(
    const float* __restrict__ u_in,
    const float* __restrict__ w_in,
    const float* __restrict__ qg_in,
    const float* __restrict__ kd_in,
    const float* __restrict__ p_in,
    const float* __restrict__ decay_in,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int T,
    unsigned int num_chunks,
    unsigned int out_stride
) {
    constexpr int C = KDA16_C, D = KDA16_D, VB = KDA16_VB, NT = KDA16_SCAN_THREADS;
    __shared__ float sS[D][VB];
    __shared__ float sW[C][D];
    __shared__ float sQ[C][D];
    __shared__ float sKd[C][D];
    __shared__ float sDel[C][VB];
    __shared__ float sP[C][C];
    __shared__ float sDec[D];
    const unsigned int h = blockIdx.x, v0 = blockIdx.y * VB, tid = threadIdx.x;
    float* S = state + (size_t)h * D * D;
    for (unsigned int e = tid; e < D * VB; e += NT) {
        const unsigned int kk = e / VB, vi = e % VB;
        sS[kk][vi] = S[(size_t)kk * D + v0 + vi];
    }
    // 2026-10-10: Thread roles: (i, two adjacent columns) for delta and out; (k, 16 columns) for
    // the state update.
    const unsigned int oi = (tid * 2) / VB, ov = (tid * 2) % VB;
    const unsigned int sv = tid % VB, sk0 = (tid / VB) * (D / (NT / VB));

    for (unsigned int c = 0; c < num_chunks; ++c) {
        const size_t cb = ((size_t)c * H + h);
        __syncthreads();
        for (unsigned int e = tid; e < C * D; e += NT) {
            const unsigned int i = e / D, kk = e % D;
            sW[i][kk] = w_in[cb * C * D + e];
            sQ[i][kk] = qg_in[cb * C * D + e];
            sKd[i][kk] = kd_in[cb * C * D + e];
        }
        sP[tid / C][tid % C] = p_in[cb * C * C + tid];
        if (tid < D) sDec[tid] = decay_in[cb * D + tid];
        __syncthreads();

        // 2026-10-10: delta = U - W S0 for this block's columns.
        // 2026-10-10: Four interleaved partial sums per output, for latency.
        float a0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, a1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int kk = 0; kk < D; ++kk) {
            const float wv = sW[oi][kk];
            a0[kk & 3] += wv * sS[kk][ov];
            a1[kk & 3] += wv * sS[kk][ov + 1];
        }
        sDel[oi][ov] = u_in[cb * C * D + (size_t)oi * D + v0 + ov] - ((a0[0] + a0[1]) + (a0[2] + a0[3]));
        sDel[oi][ov + 1] =
            u_in[cb * C * D + (size_t)oi * D + v0 + ov + 1] - ((a1[0] + a1[1]) + (a1[2] + a1[3]));
        __syncthreads();

        // 2026-10-10: out = Qg S0 + P delta (causal: P is zero above the diagonal).
        float b0[4] = {0.0f, 0.0f, 0.0f, 0.0f}, b1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int kk = 0; kk < D; ++kk) {
            const float qv = sQ[oi][kk];
            b0[kk & 3] += qv * sS[kk][ov];
            b1[kk & 3] += qv * sS[kk][ov + 1];
        }
        float o0 = (b0[0] + b0[1]) + (b0[2] + b0[3]), o1 = (b1[0] + b1[1]) + (b1[2] + b1[3]);
#pragma unroll
        for (int j = 0; j < C; ++j) {
            const float pv = sP[oi][j];
            o0 += pv * sDel[j][ov];
            o1 += pv * sDel[j][ov + 1];
        }
        const unsigned int t = c * C + oi;
        if (t < T) {
            float* o = out + (size_t)t * out_stride + (size_t)h * D + v0 + ov;
            o[0] = o0;
            o[1] = o1;
        }
        __syncthreads();

        // 2026-10-10: S_end = diag(decay) S0 + Kd^T delta; a warp updates 32 consecutive
        // columns of one row at a time (no bank conflicts).
#pragma unroll 4
        for (unsigned int kk = sk0; kk < sk0 + D / (NT / VB); ++kk) {
            float s = sDec[kk] * sS[kk][sv];
#pragma unroll
            for (int i = 0; i < C; ++i) s += sKd[i][kk] * sDel[i][sv];
            sS[kk][sv] = s;
        }
    }
    __syncthreads();
    for (unsigned int e = tid; e < D * VB; e += NT) {
        const unsigned int kk = e / VB, vi = e % VB;
        S[(size_t)kk * D + v0 + vi] = sS[kk][vi];
    }
}
