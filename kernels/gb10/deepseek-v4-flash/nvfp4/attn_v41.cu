// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-09-25: DeepSeek-V4.1 Flash attention: the elementwise and reduction kernels between its projections,
// for every layer kind (compress ratio 0: sliding window only; ratio > 0: window plus compressed latent rows).
// The projections run elsewhere (model-arch attn_v41.rs: `dense_gemm_bf16` / `dense_gemv_bf16` or the Q2_K
// kernels). Numerics follow the CPU reference `deepseek_v41_ref::{attn, compress}`: f32 math, bf16 storage,
// and the reference quantisers' rounding.
//
// Owner: gb10 kernels (deepseek-v4-flash); launched by model-arch `attn_v41`.
// Invariants: none beyond the types.
//
// Kernels:
//   attn_v41_rmsnorm_bf16 / _f32   RMSNorm with an f32 weight, bf16 output
//   attn_v41_rope                  RoPE on the last rope_dim of a row: adjacent pairs, f32 rotation, bf16 out
//   attn_v41_act_quant_fp8         in-place fp8 round trip: blocks of 32, e4m3 values, power-of-two scale
//   attn_v41_fp4_quant             in-place fp4 round trip: e2m1 values, e8m0 (mode 0) or e4m3 (mode 1)
//                                  scale, any block size
//   attn_v41_gemm_f32              C[M,N] = A[M,K](bf16) * B[N,K](f32)^T in f32 (the compressor)
//   attn_v41_gemv_f32_staged       the same product, one output a block, summed in k order
//   attn_v41_scale_bf16            out = bf16(bf16(in) * s) (the indexer's head weights)
//   attn_v41_pool                  the compressor's per-dim softmax pooling over `ratio` rows
//   attn_v41_slice_cols / _scatter_cols   column slices for the grouped wo_a
//
// The cache rows (packed window, latent and index-key rows) and the kernels that read them are in
// attn_v41_kv.cu; the helpers both files use are in attn_v41_common.cuh.


#include "attn_v41_common.cuh"

// 2026-09-25: RMSNorm `w * (x * rsqrt(mean(x^2) + eps))` in f32, bf16 out; the _f32 variant reads an f32 x.
// Grid: (rows). Block: AV_BLOCK.
extern "C" __global__ void attn_v41_rmsnorm_bf16(
    const __nv_bfloat16* __restrict__ x, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ out, const unsigned int dim, const float eps) {
    __shared__ float red[AV_BLOCK];
    const size_t base = (size_t)blockIdx.x * dim;
    float ss = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = __bfloat162float(x[base + d]);
        ss += v * v;
    }
    ss = av_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)dim + eps);
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[base + d] = __float2bfloat16(w[d] * (__bfloat162float(x[base + d]) * r));
    }
}

extern "C" __global__ void attn_v41_rmsnorm_f32(
    const float* __restrict__ x, const float* __restrict__ w,
    __nv_bfloat16* __restrict__ out, const unsigned int dim, const float eps) {
    __shared__ float red[AV_BLOCK];
    const size_t base = (size_t)blockIdx.x * dim;
    float ss = 0.0f;
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = x[base + d];
        ss += v * v;
    }
    ss = av_block_sum(ss, red);
    const float r = rsqrtf(ss / (float)dim + eps);
    for (unsigned int d = threadIdx.x; d < dim; d += blockDim.x) {
        out[base + d] = __float2bfloat16(w[d] * (x[base + d] * r));
    }
}

// 2026-09-25: RoPE on the last `rope_dim` of each `row_len`-wide row.
// `fc` is [max_pos][rope_dim/2] of (cos, sin); `pos[r]` per row; `inverse`
// negates sin. Grid: (rows). Block: >= rope_dim/2.
extern "C" __global__ void attn_v41_rope(
    __nv_bfloat16* __restrict__ x, const int* __restrict__ pos,
    const float2* __restrict__ fc, const unsigned int row_len,
    const unsigned int rope_dim, const unsigned int inverse) {
    const unsigned int r = blockIdx.x;
    const unsigned int half = rope_dim / 2;
    const unsigned int k = threadIdx.x;
    if (k >= half) return;
    __nv_bfloat16* row = x + (size_t)r * row_len + (row_len - rope_dim);
    const float2 cs = fc[(size_t)pos[r] * half + k];
    const float c = cs.x;
    const float s = inverse ? -cs.y : cs.y;
    const float a = __bfloat162float(row[2 * k]);
    const float b = __bfloat162float(row[2 * k + 1]);
    row[2 * k] = __float2bfloat16(a * c - b * s);
    row[2 * k + 1] = __float2bfloat16(a * s + b * c);
}

// 2026-09-25: act_quant: blocks of 32, scale = pow2_ceil(amax / 448) with amax >= 1e-4.
// In place on bf16 rows; `n_blocks` blocks of 32. Grid: ceil(n_blocks / 8).
// Block: 256 = 8 blocks x 32 lanes, one warp per block.
extern "C" __global__ void attn_v41_act_quant_fp8(
    __nv_bfloat16* __restrict__ x, const unsigned int n_blocks) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    __nv_bfloat16* p = x + (size_t)gb * 32 + lane;
    const float v = __bfloat162float(*p);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float s = av_fp8_scale(amax);
    const float q = av_e4m3_round_trip(fminf(fmaxf(v / s, -448.0f), 448.0f));
    *p = __float2bfloat16(q * s);
}

// 2026-09-25: fp4_act_quant: e2m1 values; scale mode 0 = e8m0 (pow2 ceil of amax/6, amax
// >= 6*2^-126), mode 1 = e4m3 (amax/6 through e4m3, amax >= 6*2^-9).
// In place on bf16; `block` values per scale; one warp per block, lanes stride. Grid: ceil(n_blocks / 8). Block: 256.
extern "C" __global__ void attn_v41_fp4_quant(
    __nv_bfloat16* __restrict__ x, const unsigned int n_blocks,
    const unsigned int block, const unsigned int mode) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    __nv_bfloat16* p = x + (size_t)gb * block;
    float amax = 0.0f;
    for (unsigned int i = lane; i < block; i += 32) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float s = av_fp4_scale(amax, mode);
    for (unsigned int i = lane; i < block; i += 32) {
        const float v = __bfloat162float(p[i]);
        const float q = av_e2m1_rne(fminf(fmaxf(v / s, -6.0f), 6.0f));
        p[i] = __float2bfloat16(q * s);
    }
}

// 2026-09-25: f32 GEMM C[M,N] = A[M,K] * B[N,K]^T in 16x16 tiles: the compressor's f32 weights on bf16
// input, accumulated in f32 in k order. Grid: (ceil(N/16), ceil(M/16)). Block: (16, 16).
#define AV_TILE 16
extern "C" __global__ void attn_v41_gemm_f32(
    const __nv_bfloat16* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K) {
    const unsigned int row = blockIdx.y * AV_TILE + threadIdx.y;
    const unsigned int col = blockIdx.x * AV_TILE + threadIdx.x;
    __shared__ float sa[AV_TILE][AV_TILE];
    __shared__ float sb[AV_TILE][AV_TILE];
    float acc = 0.0f;
    for (unsigned int k0 = 0; k0 < K; k0 += AV_TILE) {
        const unsigned int ka = k0 + threadIdx.x;
        const unsigned int kb = k0 + threadIdx.y;
        sa[threadIdx.y][threadIdx.x] = (row < M && ka < K) ? __bfloat162float(A[(size_t)row * K + ka]) : 0.0f;
        sb[threadIdx.y][threadIdx.x] = (col < N && kb < K) ? B[(size_t)col * K + kb] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < AV_TILE; ++k) acc += sa[threadIdx.y][k] * sb[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

// 2026-09-25: The attn_v41_gemm_f32 product with one output a block (the launcher uses it at M = 1): the
// 256 threads store the K products bf16(A[k]) * B[n][k] in shared memory, then thread 0 sums them in k
// order from 0.0f, eight float4 at a time. attn_v41_gemm_f32 sums each output's products in the same order
// and this tree builds with --fmad=false (KERNEL.toml), so both give the same value; here no load sits on
// the add chain and no tile rows idle. Measured 2026-09-19 with nsys on GB10: 144 us a launch for a 10 MB
// weight in attn_v41_gemm_f32, 67 us in this form. K a multiple of 8, rows 16-byte aligned. Dynamic shared
// memory: K * 4 bytes. Grid: (N, M, 1)  Block: (256, 1, 1)



#define AV_CHAIN_UNROLL 8
extern "C" __global__ void __launch_bounds__(256) attn_v41_gemv_f32_staged(
    const __nv_bfloat16* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
    const unsigned int M, const unsigned int N, const unsigned int K) {
    extern __shared__ float av_gemv_products[];
    const unsigned int t = blockIdx.y;
    const unsigned int n = blockIdx.x;
    if (t >= M || n >= N) return;
    const uint4* a4 = (const uint4*)(A + (size_t)t * K);
    const float4* b4 = (const float4*)(B + (size_t)n * K);
    for (unsigned int i = threadIdx.x; i < K / 8; i += blockDim.x) {
        const uint4 av = a4[i];
        const float4 b0 = b4[2 * i], b1 = b4[2 * i + 1];
        const unsigned int ar[4] = {av.x, av.y, av.z, av.w};
        const float bb[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
        float pr[8];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(ar[q] & 0xFFFFu);
            *(unsigned short*)&a_hi = (unsigned short)(ar[q] >> 16);
            pr[2 * q] = __bfloat162float(a_lo) * bb[2 * q];
            pr[2 * q + 1] = __bfloat162float(a_hi) * bb[2 * q + 1];
        }
        float4* p4 = (float4*)(av_gemv_products + 8 * i);
        p4[0] = make_float4(pr[0], pr[1], pr[2], pr[3]);
        p4[1] = make_float4(pr[4], pr[5], pr[6], pr[7]);
    }
    __syncthreads();
    if (threadIdx.x != 0) return;
    const float4* p4 = (const float4*)av_gemv_products;
    const unsigned int k4n = K / 4;
    float acc = 0.0f;
    unsigned int k = 0;
    for (; k + AV_CHAIN_UNROLL <= k4n; k += AV_CHAIN_UNROLL) {
        float4 v[AV_CHAIN_UNROLL];
#pragma unroll
        for (int u = 0; u < AV_CHAIN_UNROLL; ++u) v[u] = p4[k + u];
#pragma unroll
        for (int u = 0; u < AV_CHAIN_UNROLL; ++u) {
            acc += v[u].x;
            acc += v[u].y;
            acc += v[u].z;
            acc += v[u].w;
        }
    }
    for (; k < k4n; ++k) {
        const float4 v = p4[k];
        acc += v.x;
        acc += v.y;
        acc += v.z;
        acc += v.w;
    }
    C[(size_t)t * N + n] = acc;
}

// 2026-09-25: Compressor pooling: per group g and dim d, softmax over the `ratio` member scores, weighted
// sum of kv; the result is rounded to bf16 and stored as f32 for the f32 RMSNorm that follows.
// kv, score: [groups * ratio, hd] f32. out: [groups, hd] f32 (bf16-valued).

extern "C" __global__ void attn_v41_pool(
    const float* __restrict__ kv, const float* __restrict__ score, float* __restrict__ out,
    const unsigned int ratio, const unsigned int hd) {
    const unsigned int g = blockIdx.x;
    for (unsigned int d = threadIdx.x; d < hd; d += blockDim.x) {
        float m = -CUDART_INF_F;
        for (unsigned int r = 0; r < ratio; ++r) m = fmaxf(m, score[((size_t)g * ratio + r) * hd + d]);
        float sum = 0.0f;
        for (unsigned int r = 0; r < ratio; ++r) sum += expf(score[((size_t)g * ratio + r) * hd + d] - m);
        float acc = 0.0f;
        for (unsigned int r = 0; r < ratio; ++r) {
            const float e = expf(score[((size_t)g * ratio + r) * hd + d] - m);
            acc += kv[((size_t)g * ratio + r) * hd + d] * (e / sum);
        }
        out[(size_t)g * hd + d] = av_bf16r(acc);
    }
}

// 2026-09-25: Column slices for the grouped wo_a: out[t, 0..w) = in[t, col0 + 0..w).
// in rows are `in_stride` wide, out rows `w` wide. Grid: (T). Block: 256.
extern "C" __global__ void attn_v41_slice_cols(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int in_stride, const unsigned int col0, const unsigned int w) {
    const unsigned int t = blockIdx.x;
    for (unsigned int c = threadIdx.x; c < w; c += blockDim.x)
        out[(size_t)t * w + c] = in[(size_t)t * in_stride + col0 + c];
}

// 2026-09-25: out[t, col0 + 0..w) = in[t, 0..w), out rows `out_stride` wide. Grid: (T). Block: 256.
extern "C" __global__ void attn_v41_scatter_cols(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int out_stride, const unsigned int col0, const unsigned int w) {
    const unsigned int t = blockIdx.x;
    for (unsigned int c = threadIdx.x; c < w; c += blockDim.x)
        out[(size_t)t * out_stride + col0 + c] = in[(size_t)t * w + c];
}

// 2026-09-25: out[i] = bf16(in[i] * s); `in` is bf16, so the product is what the reference's
// `to_bf16_rne(v * wscale)` computes. Grid: ceil(n / 256). Block: 256.
extern "C" __global__ void attn_v41_scale_bf16(
    const __nv_bfloat16* __restrict__ in, __nv_bfloat16* __restrict__ out,
    const unsigned int n, const float s) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(__bfloat162float(in[i]) * s);
}
