// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: Chunked KDA prefill in 64-token chunks with the chunk-level products on tensor
// cores (`METRALE_GLM_KDA_CHUNK64`). The recurrence is kda_recurrent.cu's (and kda_chunk16.cu's),
// per token t and head h:
//   S <- diag(exp(g_t)) S;  delta = (v_t - S^T k_t) * beta_t;  S <- S + k_t delta^T;
//   out_t = scale * S^T q_t
// taken a chunk of C = 64 tokens at a time in the WY form. With G_i = sum_{p <= i} g_p inside
// the chunk (a per-channel log decay, every g in (lower_bound, 0)) and S0 the state before it:
//   A[i,j] = sum_d k_i[d] k_j[d] exp(G_i[d] - G_j[d])                         (j < i)
//   U_i = beta_i (v_i - sum_{j<i} A[i,j] U_j)
//   W_i = beta_i (exp(G_i) k_i - sum_{j<i} A[i,j] W_j)
//   delta = U - W S0
//   out_i = (scale exp(G_i) q_i)^T S0 + sum_{j<=i} P[i,j] delta_j,
//   P[i,j] = scale sum_d q_i[d] k_j[d] exp(G_i[d] - G_j[d])                    (j <= i)
//   S_end = diag(exp(G_last)) S0 + sum_i (exp(G_last - G_i) k_i) delta_i^T
// Every exponent taken is G_i - G_j with j <= i, or G_i, or G_last - G_i: all <= 0, so nothing
// overflows (a large decay underflows toward 0, as the true weight does). Unlike kda_chunk16 this
// never forms exp(-G), which overflows FP32 past ~88 at 64 tokens of a -5 lower bound.
//
// kda_chunk64_prepare, one block per (chunk, head), FP32 on CUDA cores: G, the A and P pairs
// (one warp per pair, lanes over channels, __expf of the non-positive exponent), the forward
// substitution for U (threads 0..127, one channel each) and W (threads 128..255), and
// Qg = scale exp(G) q, Kd = exp(G_last - G) k, decay = exp(G_last).
// kda_chunk64_scan, one block of 4 warps per (head, 32-value-column block), serial over the
// chunks: delta = U - W S0 and Qg S0 (warp w: rows 16 w .. 16 w + 15), then P delta, then
// S_end = diag(decay) S0 + Kd^T delta (warp w: state rows 32 w .. 32 w + 31, held in registers
// in the MMA accumulator layout for the whole walk). Products on mma.sync.m16n8k16 BF16 with FP32
// accumulation (W, Qg, Kd, P, delta and the staged S0 rounded to BF16, as FLA's chunked KDA does);
// U, the state and the outputs stay FP32. So the outputs differ from the token walk at BF16
// rounding level: opt-in and accuracy-gated (model-arch example glm5next_kda_chunk64_bench).
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim D = KDA64_D (128), chunk KDA64_C (64). q, k, v BF16 rows `qkv_stride` apart (the
//   conv's [T, 3 * H * D] output: q at 0, k at H * D, v at 2 * H * D elements of a row), gate FP32
//   [T, gate_stride], beta FP32 [T, H], out FP32 [T, out_stride]; the state FP32 [H, D, D]
//   indexed [k][v], read before the first chunk and written after the last.
// - Scratch per (chunk, head), [nchunks, H, ...]: U FP32 [C, D]; W, Qg, Kd BF16 [C, D]; P FP32
//   [C, C] (zero above the diagonal); decay FP32 [D]. Positions t >= T read as k = v = q = 0,
//   gate = 0, beta = 0, so they add nothing, and their out rows are not written.
// - prepare: grid (nchunks, H), block 256, dynamic shared memory KDA64_PREP_SMEM.
//   scan: grid (H, D / KDA64_VB), block 128, dynamic shared memory KDA64_SCAN_SMEM.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#define KDA64_D 128
#define KDA64_C 64
#define KDA64_VB 32
#define KDA64_PREP_THREADS 256
#define KDA64_SCAN_THREADS 128
// 2026-10-12: Padded BF16 row strides (halves) so ldmatrix row addresses hit distinct banks.
#define KDA64_LD (KDA64_D + 8)
#define KDA64_LDC (KDA64_C + 8)
// 2026-10-12: prepare: G FP32 [C][D], A FP32 [C][C + 1], beta FP32 [C], K and Q BF16 [C][D]
// (the inputs' own precision).
#define KDA64_PREP_SMEM                                                                      \
    ((KDA64_C * KDA64_D + KDA64_C * (KDA64_C + 1) + KDA64_C) * 4 + 2 * KDA64_C * KDA64_D * 2)
// 2026-10-12: scan: W, Qg, Kd BF16 [C][LD]; P BF16 [C][LDC]; S^T BF16 [VB][LD]; delta^T BF16
// [VB][LDC]; decay FP32 [D].
#define KDA64_SCAN_SMEM                                                                      \
    ((3 * KDA64_C * KDA64_LD + KDA64_C * KDA64_LDC + KDA64_VB * KDA64_LD +                    \
      KDA64_VB * KDA64_LDC) * 2 + KDA64_D * 4)

extern "C" __global__ void __launch_bounds__(KDA64_PREP_THREADS) kda_chunk64_prepare(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ u_out,
    __nv_bfloat16* __restrict__ w_out,
    __nv_bfloat16* __restrict__ qg_out,
    __nv_bfloat16* __restrict__ kd_out,
    float* __restrict__ p_out,
    float* __restrict__ decay_out,
    unsigned int H,
    unsigned int T,
    unsigned int qkv_stride,
    unsigned int gate_stride,
    float scale
) {
    constexpr int C = KDA64_C, D = KDA64_D;
    extern __shared__ __align__(16) float k64_sh[];
    float* sG = k64_sh;                                     // [C][D]
    float* sA = sG + C * D;                                 // [C][C + 1]
    float* sB = sA + C * (C + 1);                           // [C]
    __nv_bfloat16* sK = (__nv_bfloat16*)(sB + C);           // [C][D]
    __nv_bfloat16* sQ = sK + C * D;                         // [C][D]
    const unsigned int c = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const unsigned int t0 = c * C;

    for (unsigned int e = tid; e < (unsigned int)(C * D); e += blockDim.x) {
        const unsigned int i = e / D, d = e % D;
        const unsigned int t = t0 + i;
        const bool live = t < T;
        const size_t qo = (size_t)t * qkv_stride + (size_t)h * D + d;
        sK[e] = live ? k[qo] : __float2bfloat16_rn(0.0f);
        sQ[e] = live ? q[qo] : __float2bfloat16_rn(0.0f);
        sG[e] = live ? gate[(size_t)t * gate_stride + (size_t)h * D + d] : 0.0f;
    }
    if (tid < (unsigned int)C) sB[tid] = (t0 + tid < T) ? beta[(size_t)(t0 + tid) * H + h] : 0.0f;
    __syncthreads();
    // 2026-10-12: G = in-chunk cumsum of the gate, one channel per thread.
    if (tid < (unsigned int)D) {
        float acc = 0.0f;
        for (int i = 0; i < C; ++i) {
            acc += sG[i * D + tid];
            sG[i * D + tid] = acc;
        }
    }
    __syncthreads();

    // 2026-10-12: A (j < i) and P (j <= i), one warp per pair, lanes over channels; the exponent
    // G_i - G_j is <= 0 for j <= i.
    const unsigned int warp = tid >> 5, lane = tid & 31u, nwarps = blockDim.x >> 5;
    float* P = p_out + ((size_t)c * H + h) * C * C;
    for (unsigned int pi = warp; pi < (unsigned int)(C * (C + 1) / 2); pi += nwarps) {
        // 2026-10-12: Pair pi -> (i, j <= i), row-major over the lower triangle.
        const unsigned int i = (unsigned int)((sqrtf(8.0f * (float)pi + 1.0f) - 1.0f) * 0.5f);
        unsigned int ii = i;
        while ((ii + 1) * (ii + 2) / 2 <= pi) ++ii;
        while (ii * (ii + 1) / 2 > pi) --ii;
        const unsigned int j = pi - ii * (ii + 1) / 2;
        float a = 0.0f, p = 0.0f;
#pragma unroll
        for (int e = 0; e < D / 32; ++e) {
            const unsigned int d = lane + 32u * (unsigned int)e;
            const float x = __expf(sG[ii * D + d] - sG[j * D + d]);
            const float kj = __bfloat162float(sK[j * D + d]) * x;
            a += __bfloat162float(sK[ii * D + d]) * kj;
            p += __bfloat162float(sQ[ii * D + d]) * kj;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(0xFFFFFFFFu, a, o);
            p += __shfl_xor_sync(0xFFFFFFFFu, p, o);
        }
        if (lane == 0) {
            if (j < ii) sA[ii * (C + 1) + j] = a;
            P[ii * C + j] = p * scale;
        }
    }
    for (unsigned int e = tid; e < (unsigned int)(C * C); e += blockDim.x) {
        const unsigned int i = e / C, j = e % C;
        if (j > i) P[i * C + j] = 0.0f;
    }
    __syncthreads();

    // 2026-10-12: Forward substitution: threads 0..127 solve U for channel d, 128..255 W.
    const size_t base = ((size_t)c * H + h) * C * D;
    const unsigned int d = tid % D;
    const bool is_w = tid >= (unsigned int)D;
    float hist[C];
#pragma unroll
    for (int i = 0; i < C; ++i) {
        const unsigned int t = t0 + (unsigned int)i;
        float s;
        if (is_w) {
            s = __expf(sG[i * D + d]) * __bfloat162float(sK[i * D + d]);
        } else {
            s = t < T ? __bfloat162float(v[(size_t)t * qkv_stride + (size_t)h * D + d]) : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < i; ++j) s -= sA[i * (C + 1) + j] * hist[j];
        hist[i] = sB[i] * s;
        if (is_w) {
            w_out[base + (size_t)i * D + d] = __float2bfloat16_rn(hist[i]);
        } else {
            u_out[base + (size_t)i * D + d] = hist[i];
        }
    }
    // 2026-10-12: Qg, Kd and the chunk's decay.
    const float gl = sG[(C - 1) * D + d];
    for (unsigned int i = is_w ? 1u : 0u; i < (unsigned int)C; i += 2u) {
        qg_out[base + (size_t)i * D + d] =
            __float2bfloat16_rn(scale * __expf(sG[i * D + d]) * __bfloat162float(sQ[i * D + d]));
        kd_out[base + (size_t)i * D + d] =
            __float2bfloat16_rn(__expf(gl - sG[i * D + d]) * __bfloat162float(sK[i * D + d]));
    }
    if (!is_w) decay_out[((size_t)c * H + h) * D + d] = __expf(gl);
}

__device__ __forceinline__ void k64_mma(float (&d)[4], const uint32_t (&a)[4], uint32_t b0,
                                        uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void k64_ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void k64_ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

__device__ __forceinline__ void k64_ldsm_x2(uint32_t addr, uint32_t& r0, uint32_t& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ uint32_t k64_pack(float x, float y) {
    __nv_bfloat162 b = __floats2bfloat162_rn(x, y);
    return *reinterpret_cast<uint32_t*>(&b);
}

// 2026-10-12: A fragment (m16 x k16, row-major) of the BF16 tile at smem `base` (row stride `ld`
// halves), rows r0 .., cols c0 ..: ldmatrix.x4 over (rows 0-7, cols 0-7), (rows 8-15, cols 0-7),
// (rows 0-7, cols 8-15), (rows 8-15, cols 8-15).
__device__ __forceinline__ void k64_frag_a(uint32_t base, unsigned int ld, unsigned int r0,
                                           unsigned int c0, unsigned int lane,
                                           uint32_t (&a)[4]) {
    const unsigned int m = lane >> 3, row = r0 + (lane & 7u) + (m & 1u) * 8u;
    const unsigned int col = c0 + (m >> 1) * 8u;
    k64_ldsm_x4(base + (row * ld + col) * 2u, a);
}

// 2026-10-12: A fragment of the TRANSPOSE of the BF16 tile X at smem `base` (X row stride `ld`):
// rows of X^T are cols of X. X^T rows r0 .. (X cols), X^T cols c0 .. (X rows).
__device__ __forceinline__ void k64_frag_a_t(uint32_t base, unsigned int ld, unsigned int r0,
                                             unsigned int c0, unsigned int lane,
                                             uint32_t (&a)[4]) {
    const unsigned int m = lane >> 3;
    // 2026-10-12: Matrix m: X^T rows r0 + (m & 1) * 8 .., cols c0 + (m >> 1) * 8 ..: X rows
    // c0 + (m >> 1) * 8 + (lane & 7), X cols r0 + (m & 1) * 8 .. + 7, transposed by ldmatrix.
    const unsigned int xrow = c0 + (m >> 1) * 8u + (lane & 7u);
    const unsigned int xcol = r0 + (m & 1u) * 8u;
    k64_ldsm_x4_t(base + (xrow * ld + xcol) * 2u, a);
}

// 2026-10-12: B fragment (k16 x n8, "col") from the BF16 tile Bt at smem `base` stored [n][k]
// (row stride `ld`): n rows n0 .., k cols k0 ...
__device__ __forceinline__ void k64_frag_b(uint32_t base, unsigned int ld, unsigned int n0,
                                           unsigned int k0, unsigned int lane, uint32_t& b0,
                                           uint32_t& b1) {
    const unsigned int row = n0 + (lane & 7u), col = k0 + ((lane >> 3) & 1u) * 8u;
    k64_ldsm_x2(base + (row * ld + col) * 2u, b0, b1);
}

extern "C" __global__ void __launch_bounds__(KDA64_SCAN_THREADS) kda_chunk64_scan(
    const float* __restrict__ u_in,
    const __nv_bfloat16* __restrict__ w_in,
    const __nv_bfloat16* __restrict__ qg_in,
    const __nv_bfloat16* __restrict__ kd_in,
    const float* __restrict__ p_in,
    const float* __restrict__ decay_in,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int T,
    unsigned int num_chunks,
    unsigned int out_stride
) {
    constexpr int C = KDA64_C, D = KDA64_D, VB = KDA64_VB, LD = KDA64_LD, LDC = KDA64_LDC;
    extern __shared__ __align__(16) unsigned char k64_scan_sh[];
    __nv_bfloat16* sW = (__nv_bfloat16*)k64_scan_sh;     // [C][LD]
    __nv_bfloat16* sQ = sW + C * LD;                      // [C][LD]
    __nv_bfloat16* sKd = sQ + C * LD;                     // [C][LD]
    __nv_bfloat16* sP = sKd + C * LD;                     // [C][LDC]
    __nv_bfloat16* sST = sP + C * LDC;                    // [VB][LD]   S0^T
    __nv_bfloat16* sDT = sST + VB * LD;                   // [VB][LDC]  delta^T
    float* sDec = (float*)(sDT + VB * LDC);               // [D]
    const uint32_t bW = (uint32_t)__cvta_generic_to_shared(sW);
    const uint32_t bQ = (uint32_t)__cvta_generic_to_shared(sQ);
    const uint32_t bKd = (uint32_t)__cvta_generic_to_shared(sKd);
    const uint32_t bP = (uint32_t)__cvta_generic_to_shared(sP);
    const uint32_t bST = (uint32_t)__cvta_generic_to_shared(sST);
    const uint32_t bDT = (uint32_t)__cvta_generic_to_shared(sDT);

    const unsigned int h = blockIdx.x, v0 = blockIdx.y * VB, tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31u, g = lane >> 2, tig = lane & 3u;
    float* S = state + (size_t)h * D * D;

    // 2026-10-12: The state rows 32 warp .. 32 warp + 31, all VB columns, in the m16n8 accumulator
    // layout: st[mt][nt][0..3] = rows 32 warp + 16 mt + (g, g, g + 8, g + 8), columns
    // v0 + 8 nt + 2 tig + (0, 1, 0, 1).
    float st[2][VB / 8][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < VB / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int row = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                const unsigned int col = v0 + (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                st[mt][nt][e] = S[(size_t)row * D + col];
            }

    for (unsigned int c = 0; c < num_chunks; ++c) {
        const size_t cb = (size_t)c * H + h;
        // 2026-10-12: The chunk's factors into shared memory (16-byte copies, padded rows).
        for (unsigned int e = tid; e < (unsigned int)(C * D / 8); e += blockDim.x) {
            const unsigned int i = e / (D / 8), q8 = e % (D / 8);
            const size_t src = cb * C * D + (size_t)i * D + q8 * 8u;
            *(uint4*)(sW + i * LD + q8 * 8u) = *(const uint4*)(w_in + src);
            *(uint4*)(sQ + i * LD + q8 * 8u) = *(const uint4*)(qg_in + src);
            *(uint4*)(sKd + i * LD + q8 * 8u) = *(const uint4*)(kd_in + src);
        }
        for (unsigned int e = tid; e < (unsigned int)(C * C / 2); e += blockDim.x) {
            const unsigned int i = e / (C / 2), j2 = (e % (C / 2)) * 2u;
            const float2 pv = *(const float2*)(p_in + cb * C * C + (size_t)i * C + j2);
            *(uint32_t*)(sP + i * LDC + j2) = k64_pack(pv.x, pv.y);
        }
        for (unsigned int e = tid; e < (unsigned int)D; e += blockDim.x) sDec[e] = decay_in[cb * D + e];
        // 2026-10-12: S0^T as BF16: this warp's state rows are k = columns of S^T.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < VB / 8; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const unsigned int kr = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                    const unsigned int vc = (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                    sST[vc * LD + kr] = __float2bfloat16_rn(st[mt][nt][e]);
                }
        __syncthreads();

        // 2026-10-12: delta = U - W S0 and o = Qg S0 for rows 16 warp .. 16 warp + 15.
        float dl[VB / 8][4], o[VB / 8][4];
#pragma unroll
        for (int nt = 0; nt < VB / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                dl[nt][e] = 0.0f;
                o[nt][e] = 0.0f;
            }
        const unsigned int r0 = warp * 16u;
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            uint32_t aw[4], aq[4];
            k64_frag_a(bW, LD, r0, (unsigned int)ks * 16u, lane, aw);
            k64_frag_a(bQ, LD, r0, (unsigned int)ks * 16u, lane, aq);
#pragma unroll
            for (int nt = 0; nt < VB / 8; ++nt) {
                uint32_t b0, b1;
                k64_frag_b(bST, LD, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                k64_mma(dl[nt], aw, b0, b1);
                k64_mma(o[nt], aq, b0, b1);
            }
        }
        // 2026-10-12: delta = U - (W S0); delta^T to shared memory as BF16.
#pragma unroll
        for (int nt = 0; nt < VB / 8; ++nt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const unsigned int i = r0 + g + (unsigned int)hr * 8u;
                const unsigned int vc = (unsigned int)nt * 8u + 2u * tig;
                const float2 uv = *(const float2*)(u_in + cb * C * D + (size_t)i * D + v0 + vc);
                const float d0 = uv.x - dl[nt][hr * 2], d1 = uv.y - dl[nt][hr * 2 + 1];
                sDT[vc * LDC + i] = __float2bfloat16_rn(d0);
                sDT[(vc + 1u) * LDC + i] = __float2bfloat16_rn(d1);
            }
        }
        __syncthreads();

        // 2026-10-12: o += P delta (P zero above the diagonal), K = the chunk's 64 rows.
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
            uint32_t ap[4];
            k64_frag_a(bP, LDC, r0, (unsigned int)ks * 16u, lane, ap);
#pragma unroll
            for (int nt = 0; nt < VB / 8; ++nt) {
                uint32_t b0, b1;
                k64_frag_b(bDT, LDC, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                k64_mma(o[nt], ap, b0, b1);
            }
        }
#pragma unroll
        for (int nt = 0; nt < VB / 8; ++nt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const unsigned int t = c * C + r0 + g + (unsigned int)hr * 8u;
                if (t < T) {
                    float2* op = (float2*)(out + (size_t)t * out_stride + (size_t)h * D + v0 +
                                           (unsigned int)nt * 8u + 2u * tig);
                    *op = make_float2(o[nt][hr * 2], o[nt][hr * 2 + 1]);
                }
            }
        }

        // 2026-10-12: S_end = diag(decay) S0 + Kd^T delta for this warp's 32 state rows.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < VB / 8; ++nt) {
                const unsigned int ka = warp * 32u + (unsigned int)mt * 16u + g;
                st[mt][nt][0] *= sDec[ka];
                st[mt][nt][1] *= sDec[ka];
                st[mt][nt][2] *= sDec[ka + 8u];
                st[mt][nt][3] *= sDec[ka + 8u];
            }
        }
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                uint32_t ak[4];
                k64_frag_a_t(bKd, LD, warp * 32u + (unsigned int)mt * 16u, (unsigned int)ks * 16u,
                             lane, ak);
#pragma unroll
                for (int nt = 0; nt < VB / 8; ++nt) {
                    uint32_t b0, b1;
                    k64_frag_b(bDT, LDC, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                    k64_mma(st[mt][nt], ak, b0, b1);
                }
            }
        }
        // 2026-10-12: The next chunk overwrites every shared buffer.
        __syncthreads();
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < VB / 8; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int row = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                const unsigned int col = v0 + (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                S[(size_t)row * D + col] = st[mt][nt][e];
            }
}
