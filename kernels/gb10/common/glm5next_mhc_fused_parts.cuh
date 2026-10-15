// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The device parts of glm5next_mhc_fused.cu (its shared-memory block, the Sinkhorn on
// warp 0, and the collapse + RMS norm on the other warps), moved here unchanged so
// glm5next_mhc_post_fold.cu runs the same tail. The invariants are glm5next_mhc_fused.cu's.
//
// Owner: gb10 kernels.

#pragma once
#include <cuda_bf16.h>

#define GLM_HCF_BLOCK 256
#define GLM_HCF_MAX_MULT 4
#define GLM_HCF_MAX_MIX 24
#define GLM_HCF_MAX_H 8192
#define GLM_HCF_TAIL_THREADS (GLM_HCF_BLOCK - 32)

// 2026-10-09: rms_norm_vanilla's warp_reduce_sum: an xor butterfly, offsets 16 down to 1.
__device__ __forceinline__ float glm_hcf_warp_sum(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

// 2026-10-09: Named barrier over warps 1..7, the tail's y + norm threads; warp 0 runs the
// Sinkhorn meanwhile and never reaches it.
__device__ __forceinline__ void glm_hcf_tail_bar() {
    asm volatile("bar.sync 1, %0;" ::"n"(GLM_HCF_TAIL_THREADS));
}

// 2026-10-15: glm_hcf_tail_bar on named barrier BAR, for a block that runs one tail per group of
// GLM_HCF_BLOCK threads (glm5next_hc_pre_norm_tile8: group g's tail warps on barrier 4 + g).
template <int BAR>
__device__ __forceinline__ void glm_hcf_tail_bar_at() {
    asm volatile("bar.sync %0, %1;" ::"n"(BAR), "n"(GLM_HCF_TAIL_THREADS));
}

// 2026-10-09: post and the Sinkhorn of glm5next_hc_finish block 0, on warp 0. Lane l holds comb
// element (i, j) = (l / HC, l % HC); lanes at or past HC * HC hold a copy of lane l % (HC * HC)
// so every shuffle runs on the full warp. Each sum starts where the unfused one does and adds
// the same operands in the same order; every division stays a division.
template <int HC>
__device__ __forceinline__ void glm_hcf_sinkhorn(
    const float* s_mix,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    unsigned int t,
    unsigned int sinkhorn_iters,
    float hc_eps
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int l = lane % (HC * HC);
    const unsigned int i = l / HC;
    const unsigned int j = l % HC;

    if (lane < HC) {
        float po = s_mix[HC + lane] * hc_scale[1] + hc_base[HC + lane];
        post_out[(size_t)t * HC + lane] = 2.f * (1.f / (1.f + expf(-po)));
    }

    float v = s_mix[2 * HC + l] * hc_scale[2] + hc_base[2 * HC + l];

    float mx = -1e30f;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) mx = fmaxf(mx, __shfl_sync(0xFFFFFFFFu, v, i * HC + jj));
    const float e = expf(v - mx);
    float sum = 0.f;
#pragma unroll
    for (int jj = 0; jj < HC; ++jj) sum += __shfl_sync(0xFFFFFFFFu, e, i * HC + jj);
    v = e / sum + hc_eps;

    float c = hc_eps;
#pragma unroll
    for (int ii = 0; ii < HC; ++ii) c += __shfl_sync(0xFFFFFFFFu, v, ii * HC + j);
    v /= c;

    for (unsigned int it = 0; it + 1 < sinkhorn_iters; ++it) {
        float r = hc_eps;
#pragma unroll
        for (int jj = 0; jj < HC; ++jj) r += __shfl_sync(0xFFFFFFFFu, v, i * HC + jj);
        v /= r;
        float cc = hc_eps;
#pragma unroll
        for (int ii = 0; ii < HC; ++ii) cc += __shfl_sync(0xFFFFFFFFu, v, ii * HC + j);
        v /= cc;
    }
    if (lane < HC * HC) comb_out[(size_t)t * HC * HC + lane] = v;
}

// 2026-10-10: CG reads the streams through L2 (__ldcg), for a caller whose other blocks wrote
// them in this launch (glm5next_hc_post_pre_norm); the values read are the same.
// 2026-10-09: y and its RMS norm on warps 1..7 (u = threadIdx.x - 32). y[d] is
// glm5next_hc_finish's collapse; the norm is rms_norm_vanilla over the BF16 y with
// NT = min(H, 1024) virtual threads: virtual thread v sums pairs v, v + NT, .. of y as
// `s += a * a + b * b`, each virtual warp of 32 runs the same butterfly on one real warp, and
// the warp sums go through the same final butterfly, zeros past the last warp included.
// 2026-10-15: `u` is the tail thread's index (0 .. GLM_HCF_TAIL_THREADS - 1) and BAR the named
// barrier its GLM_HCF_TAIL_THREADS threads share; glm_hcf_collapse_norm below passes
// threadIdx.x - 32 and barrier 1, as before. No value depends on either.
template <int HC, bool CG, int BAR>
__device__ __forceinline__ void glm_hcf_collapse_norm_at(
    const unsigned int u,
    const float* __restrict__ x,
    const float* s_pre,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_bfloat16* s_y,
    float* s_ws,
    unsigned int H,
    float rms_eps
) {
    const unsigned int rw = u >> 5;
    const unsigned int lane = threadIdx.x & 31u;

    float pre[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) pre[i] = s_pre[i];
    // 2026-10-09: Four columns per thread and pass through 16-byte loads: one pass per
    // GLM_HCF_TAIL_THREADS quads keeps the single tail block from waiting on one L2 round trip
    // per column (measured 3.3 us scalar, 1.2 us here at H 4096). Each column's sum is still
    // acc = 0 + pre[0] x[0] + .. + pre[HC-1] x[HC-1].
    const unsigned int nq = H / 4u;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    for (unsigned int q = u; q < nq; q += GLM_HCF_TAIL_THREADS) {
        float4 v[HC];
#pragma unroll
        for (int i = 0; i < HC; ++i) v[i] = CG ? __ldcg(&x4[i * nq + q]) : x4[i * nq + q];
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
#pragma unroll
        for (int i = 0; i < HC; ++i) {
            a0 += pre[i] * v[i].x;
            a1 += pre[i] * v[i].y;
            a2 += pre[i] * v[i].z;
            a3 += pre[i] * v[i].w;
        }
        const __nv_bfloat162 lo = __halves2bfloat162(__float2bfloat16(a0), __float2bfloat16(a1));
        const __nv_bfloat162 hi = __halves2bfloat162(__float2bfloat16(a2), __float2bfloat16(a3));
        reinterpret_cast<__nv_bfloat162*>(y_out)[2 * q] = lo;
        reinterpret_cast<__nv_bfloat162*>(y_out)[2 * q + 1] = hi;
        reinterpret_cast<__nv_bfloat162*>(s_y)[2 * q] = lo;
        reinterpret_cast<__nv_bfloat162*>(s_y)[2 * q + 1] = hi;
    }
    glm_hcf_tail_bar_at<BAR>();

    const unsigned int nt = H < 1024u ? H : 1024u;
    const unsigned int nwv = nt / 32u;
    const unsigned int half = H / 2u;
    for (unsigned int vw = rw; vw < nwv; vw += GLM_HCF_TAIL_THREADS / 32) {
        float s = 0.f;
        for (unsigned int p = vw * 32u + lane; p < half; p += nt) {
            const float a = __bfloat162float(s_y[2 * p]);
            const float b = __bfloat162float(s_y[2 * p + 1]);
            s += a * a + b * b;
        }
        s = glm_hcf_warp_sum(s);
        if (lane == 0) s_ws[vw] = s;
    }
    glm_hcf_tail_bar_at<BAR>();
    if (rw == 0) {
        float v = (lane < nwv) ? s_ws[lane] : 0.0f;
        v = glm_hcf_warp_sum(v);
        if (lane == 0) s_ws[32] = v;
    }
    glm_hcf_tail_bar_at<BAR>();
    const float rms = rsqrtf(s_ws[32] / (float)H + rms_eps);

    // 2026-10-09: Four columns per thread and pass, the weights through one 8-byte load; each
    // output is rms_norm_vanilla's x * rms * w.
    const uint2* w4 = reinterpret_cast<const uint2*>(norm_w);
    for (unsigned int q = u; q < nq; q += GLM_HCF_TAIL_THREADS) {
        const uint2 wq = w4[q];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wq);
        unsigned short o[4];
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const float xv = __bfloat162float(s_y[4 * q + e]);
            const float wv = __bfloat162float(wb[e]);
            o[e] = __bfloat16_as_ushort(__float2bfloat16(xv * rms * wv));
        }
        uint2 pk;
        pk.x = (unsigned int)o[0] | ((unsigned int)o[1] << 16);
        pk.y = (unsigned int)o[2] | ((unsigned int)o[3] << 16);
        reinterpret_cast<uint2*>(normed_out)[q] = pk;
    }
}

template <int HC, bool CG = false>
__device__ __forceinline__ void glm_hcf_collapse_norm(
    const float* __restrict__ x,
    const float* s_pre,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    __nv_bfloat16* __restrict__ normed_out,
    __nv_bfloat16* s_y,
    float* s_ws,
    unsigned int H,
    float rms_eps
) {
    glm_hcf_collapse_norm_at<HC, CG, 1>(threadIdx.x - 32u, x, s_pre, norm_w, y_out, normed_out,
                                        s_y, s_ws, H, rms_eps);
}

// 2026-10-09: The block's shared memory, declared once in each entry point: a __shared__
// declared inside the per-HC instantiations would be reserved once per instantiation.
template <int ROWS>
struct GlmHcfSmem {
    float red_ss[GLM_HCF_BLOCK];
    float red[ROWS][GLM_HCF_BLOCK];
    float mix[GLM_HCF_MAX_MIX];
    float pre[GLM_HCF_MAX_MULT];
    float ws[33];
    __nv_bfloat16 y[GLM_HCF_MAX_H];
    bool last;
};

// 2026-10-12: The pre-norm entry points' argument list, shared by glm5next_mhc_fused.cu and
// glm5next_mhc_tok.cu (moved here unchanged from glm5next_mhc_fused.cu).
#define GLM_HCF_ARGS                                                                          \
    const float* __restrict__ streams, const __nv_bfloat16* __restrict__ hc_fn,               \
        const float* __restrict__ hc_scale, const float* __restrict__ hc_base,                \
        float* __restrict__ mix, unsigned int* __restrict__ done,                             \
        const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,          \
        float* __restrict__ post_out, float* __restrict__ comb_out,                           \
        __nv_bfloat16* __restrict__ normed_out, const unsigned int hidden_size,               \
        const unsigned int hc_mult, const unsigned int sinkhorn_iters, const float norm_eps,  \
        const float hc_eps, const float rms_eps
