// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: GLM-5.3 mHC pre-mix in one launch: glm5next_hc_mix_bf16 (or its 8-row variant),
// glm5next_hc_finish and the rms_norm_vanilla that follows it, fused with a last-block-done
// tail.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to the unfused sequence glm5next_hc_mix_bf16 (or
//   glm5next_hc_mix_bf16_rows) -> glm5next_hc_finish -> rms_norm_vanilla launched with
//   blockDim min(H, 1024): each value is computed with the same operations in the same order
//   (the build passes --fmad=false, so no product is contracted into an FMA on either side).
//   glm5next_hc_fused_gate (model-arch examples) checks y, post, comb and the normed row
//   byte for byte against those kernels.
// - blockDim == GLM_HCF_BLOCK (256); hc_mult is 1..=GLM_HCF_MAX_MULT (4); H is a multiple of
//   4, at most GLM_HCF_MAX_H, and min(H, 1024) is a multiple of 32. glm_hc_pre_norm (model-arch
//   glm5next_mhc_fused.rs) refuses any other shape.
// - `done` holds one u32 per token, zero before the launch; the last block of each token puts
//   it back to zero, so the next launch finds it zeroed.
//
// Mix phase: block (t, y) computes ROWS rows y*ROWS .. of token t's mixes. The RMS partials
// and the dot partials are taken in one pass over the streams; each partial is the same
// thread-strided sum in the same k order as the unfused kernel, and each 256-wide tree runs
// in the same order, so the RMS and every mix keep their bits.
//
// Tail: the block that finds itself last for its token (an atomic on done[t], after a fence
// on the mix stores) reads the mixes back. Warp 0 runs post and the Sinkhorn on 16 lanes, one
// comb element per lane in a register; row and column sums gather the operands by shuffle and
// add them in the unfused order. Warps 1..7 collapse y = sum_i pre[i] * x[i] and run the RMS
// norm of y, emulating rms_norm_vanilla's min(H, 1024) threads, their warp butterflies and its
// final warp butterfly over the warp sums.

#include "glm5next_mhc_fused_parts.cuh"

// 2026-10-09: The whole kernel for ROWS mix rows per block (1: glm5next_hc_mix_bf16's grid,
// 8: glm5next_hc_mix_bf16_rows's) and hc_mult HC.
// 2026-10-11: PF > 0 (ROWS 1 only, launched only where hc_mult * H == PF * GLM_HCF_BLOCK):
// each thread issues all PF of its stream loads and all PF of its hc_fn loads before the first
// add, then accumulates them in the loop's order, so every partial is the same chain of the
// same operations and the outputs keep their bits; only the loads stop waiting on one another
// (glm5next_hc_pre_norm_pf64, METRALE_GLM_MHC_PRE_PREFETCH).
template <int ROWS, int HC, int PF = 0>
__device__ __forceinline__ void glm_hcf_body(
    const float* __restrict__ streams,
    const __nv_bfloat16* __restrict__ hc_fn,
    const float* __restrict__ hc_scale,
    const float* __restrict__ hc_base,
    float* __restrict__ mix,
    unsigned int* __restrict__ done,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ y_out,
    float* __restrict__ post_out,
    float* __restrict__ comb_out,
    __nv_bfloat16* __restrict__ normed_out,
    unsigned int H,
    unsigned int sinkhorn_iters,
    float norm_eps,
    float hc_eps,
    float rms_eps,
    GlmHcfSmem<ROWS>& sm
) {
    constexpr unsigned int MIX = (2 + HC) * HC;
    const unsigned int t = blockIdx.x;
    const unsigned int m0 = blockIdx.y * ROWS;
    const unsigned int tid = threadIdx.x;
    const unsigned int hc_dim = HC * H;
    const unsigned int rows = min((unsigned int)ROWS, MIX - m0);
    const float* x = streams + (size_t)t * hc_dim;

    float* red_ss = sm.red_ss;
    float (*red)[GLM_HCF_BLOCK] = sm.red;
    float* s_mix = sm.mix;
    float* s_pre = sm.pre;

    float ss = 0.f;
    float acc[ROWS];
#pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.f;
    if constexpr (PF > 0) {
        static_assert(ROWS == 1, "the prefetching mix runs one mix row per block");
        float xv[PF];
        __nv_bfloat16 wv[PF];
        const __nv_bfloat16* __restrict__ wrow = hc_fn + (size_t)m0 * hc_dim;
#pragma unroll
        for (int j = 0; j < PF; ++j) xv[j] = x[tid + (unsigned int)j * GLM_HCF_BLOCK];
#pragma unroll
        for (int j = 0; j < PF; ++j) wv[j] = wrow[tid + (unsigned int)j * GLM_HCF_BLOCK];
#pragma unroll
        for (int j = 0; j < PF; ++j) {
            const float v = xv[j];
            ss += v * v;
            acc[0] += __bfloat162float(wv[j]) * v;
        }
    } else {
    // 2026-10-09: Eight iterations in flight per thread: at one token the 24 blocks are bound by
    // load latency (whole kernel at C1: 13.4 us at unroll 4, 11.1 us at 8, 16 no better).
#pragma unroll 8
    for (unsigned int k = tid; k < hc_dim; k += GLM_HCF_BLOCK) {
        const float v = x[k];
        ss += v * v;
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if ((unsigned int)r < rows) {
                acc[r] += __bfloat162float(hc_fn[(size_t)(m0 + r) * hc_dim + k]) * v;
            }
        }
    }
    }
    red_ss[tid] = ss;
#pragma unroll
    for (int r = 0; r < ROWS; ++r) red[r][tid] = acc[r];
    __syncthreads();
    for (unsigned int s = GLM_HCF_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) {
            red_ss[tid] += red_ss[tid + s];
#pragma unroll
            for (int r = 0; r < ROWS; ++r) red[r][tid] += red[r][tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float rsq = rsqrtf(red_ss[0] / (float)hc_dim + norm_eps);
#pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if ((unsigned int)r < rows) mix[(size_t)t * MIX + m0 + r] = red[r][0] * rsq;
        }
        __threadfence();
        const unsigned int prev = atomicAdd(&done[t], 1u);
        sm.last = (prev == gridDim.y - 1);
    }
    __syncthreads();
    if (!sm.last) return;

    // 2026-10-09: Last block of token t: every other block's mixes are visible after the
    // fence. The counter goes back to zero for the next launch.
    __threadfence();
    if (tid == 0) done[t] = 0;
    if (tid < MIX) s_mix[tid] = __ldcg(&mix[(size_t)t * MIX + tid]);
    __syncthreads();
    if (tid < HC) {
        float pr = s_mix[tid] * hc_scale[0] + hc_base[tid];
        s_pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __syncthreads();

    if (tid < 32) {
        glm_hcf_sinkhorn<HC>(s_mix, hc_scale, hc_base, post_out, comb_out, t, sinkhorn_iters,
                             hc_eps);
    } else {
        glm_hcf_collapse_norm<HC>(x, s_pre, norm_w, y_out + (size_t)t * H,
                                  normed_out + (size_t)t * H, sm.y, sm.ws, H, rms_eps);
    }
}

#define GLM_HCF_ARGS                                                                          \
    const float* __restrict__ streams, const __nv_bfloat16* __restrict__ hc_fn,               \
        const float* __restrict__ hc_scale, const float* __restrict__ hc_base,                \
        float* __restrict__ mix, unsigned int* __restrict__ done,                             \
        const __nv_bfloat16* __restrict__ norm_w, __nv_bfloat16* __restrict__ y_out,          \
        float* __restrict__ post_out, float* __restrict__ comb_out,                           \
        __nv_bfloat16* __restrict__ normed_out, const unsigned int hidden_size,               \
        const unsigned int hc_mult, const unsigned int sinkhorn_iters, const float norm_eps,  \
        const float hc_eps, const float rms_eps

#define GLM_HCF_DISPATCH(ROWS)                                                                \
    switch (hc_mult) {                                                                        \
    case 1: glm_hcf_body<ROWS, 1>(GLM_HCF_PASS); break;                                       \
    case 2: glm_hcf_body<ROWS, 2>(GLM_HCF_PASS); break;                                       \
    case 3: glm_hcf_body<ROWS, 3>(GLM_HCF_PASS); break;                                       \
    case 4: glm_hcf_body<ROWS, 4>(GLM_HCF_PASS); break;                                       \
    default: break;                                                                           \
    }

#define GLM_HCF_PASS                                                                          \
    streams, hc_fn, hc_scale, hc_base, mix, done, norm_w, y_out, post_out, comb_out,          \
        normed_out, hidden_size, sinkhorn_iters, norm_eps, hc_eps, rms_eps, sm

// 2026-10-09: One block per (token, mix row): grid (T, (2 + hc) * hc, 1).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<1> sm;
    GLM_HCF_DISPATCH(1)
}

// 2026-10-09: One block per (token, 8 mix rows): grid (T, ceil((2 + hc) * hc / 8), 1).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_rows(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<8> sm;
    GLM_HCF_DISPATCH(8)
}

// 2026-10-11: glm5next_hc_pre_norm for hc_mult 4 at hc_mult * H == 64 * GLM_HCF_BLOCK (GLM-5.3:
// H 4096), with every stream and hc_fn load of a thread in flight at once (glm_hcf_body PF 64).
// Same grid as glm5next_hc_pre_norm; any other shape returns without writing (the host never
// launches it there: glm5next_mhc_fused.rs `pre_norm_kernel`).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_pf64(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<1> sm;
    if (hc_mult != 4u || hc_mult * hidden_size != 64u * GLM_HCF_BLOCK) return;
    glm_hcf_body<1, 4, 64>(GLM_HCF_PASS);
}

// 2026-10-11: glm5next_hc_pre_norm for hc_mult 4 at hc_mult * H == 64 * GLM_HCF_BLOCK (GLM-5.3:
// H 4096) with the mix split along k across GLM_HCF_SK_SLICES blocks per token: grid (T, 64).
// Block (t, b) takes elements b*256 .. b*256+255 of token t's 16384 stream values, one per
// warp per three mix rows (16-byte hc_fn loads, the slice of x in shared memory), and forms
// the RMS partial and all 24 mix partials of that slice by warp butterflies, and stores 25 floats at mix + (t * 64 + b) * 25 (the
// site's `mix` scratch, which the host checks is large enough). The last block for the token
// adds the 64 slices in slice order, then runs glm_hcf_body's tail unchanged (pre, Sinkhorn,
// collapse + RMS norm). NOT bit-identical to glm5next_hc_pre_norm: the k order of every dot
// product changes (accuracy class RO, judged by the numerics gate). Uses 64 x T blocks instead
// of 24 x T, so the k loop is one load per array per thread instead of 64 chained.
#define GLM_HCF_SK_SLICES 64
#define GLM_HCF_SK_VALS 25
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK)
glm5next_hc_pre_norm_sk64(GLM_HCF_ARGS) {
    __shared__ GlmHcfSmem<1> sm;
    __shared__ float s_part[1][GLM_HCF_SK_VALS];
    constexpr unsigned int HC = 4, MIX = (2 + HC) * HC;
    if (hc_mult != HC || hc_mult * hidden_size != GLM_HCF_SK_SLICES * GLM_HCF_BLOCK) return;
    const unsigned int t = blockIdx.x, b = blockIdx.y, tid = threadIdx.x;
    const unsigned int lane = tid & 31u, warp = tid >> 5;
    const unsigned int hc_dim = HC * hidden_size;
    const float* x = streams + (size_t)t * hc_dim;
    // 2026-10-11: The slice's 256 stream values go to shared memory (64 float4 loads); warp w
    // takes mix rows w, w + 8, w + 16, lane l the 8 consecutive columns 8l .. 8l + 7 of each
    // row through one 16-byte hc_fn load, so every warp request is 512 contiguous bytes.
    __shared__ __align__(16) float s_x[GLM_HCF_BLOCK];
    if (tid < GLM_HCF_BLOCK / 4)
        reinterpret_cast<float4*>(s_x)[tid] =
            reinterpret_cast<const float4*>(x + b * GLM_HCF_BLOCK)[tid];
    uint4 wv[MIX / 8];
#pragma unroll
    for (unsigned int j = 0; j < MIX / 8; ++j) {
        const unsigned int r = warp + 8u * j;
        wv[j] = reinterpret_cast<const uint4*>(hc_fn + (size_t)r * hc_dim + b * GLM_HCF_BLOCK)[lane];
    }
    __syncthreads();
#pragma unroll
    for (unsigned int j = 0; j < MIX / 8; ++j) {
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv[j]);
        float a = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) a += __bfloat162float(wb[e]) * s_x[8u * lane + e];
        a = glm_hcf_warp_sum(a);
        if (lane == 0) s_part[0][1 + warp + 8u * j] = a;
    }
    if (warp == 0) {
        float q = 0.f;
#pragma unroll
        for (int e = 0; e < 8; ++e) q += s_x[8u * lane + e] * s_x[8u * lane + e];
        q = glm_hcf_warp_sum(q);
        if (lane == 0) s_part[0][0] = q;
    }
    __syncthreads();
    float* part = mix + ((size_t)t * GLM_HCF_SK_SLICES + b) * GLM_HCF_SK_VALS;
    if (tid < GLM_HCF_SK_VALS) {
        part[tid] = s_part[0][tid];
        __threadfence();
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        const unsigned int prev = atomicAdd(&done[t], 1u);
        sm.last = (prev == gridDim.y - 1);
    }
    __syncthreads();
    if (!sm.last) return;

    __threadfence();
    if (tid == 0) done[t] = 0;
    const float* all = mix + (size_t)t * GLM_HCF_SK_SLICES * GLM_HCF_SK_VALS;
    if (tid < GLM_HCF_SK_VALS) {
        float s = 0.f;
        float v[GLM_HCF_SK_SLICES];
#pragma unroll
        for (int q = 0; q < GLM_HCF_SK_SLICES; ++q) v[q] = __ldcg(&all[q * GLM_HCF_SK_VALS + tid]);
#pragma unroll
        for (int q = 0; q < GLM_HCF_SK_SLICES; ++q) s += v[q];
        sm.red[0][tid] = s;
    }
    __syncthreads();
    if (tid < MIX) {
        const float rsq = rsqrtf(sm.red[0][0] / (float)hc_dim + norm_eps);
        sm.mix[tid] = sm.red[0][tid + 1] * rsq;
    }
    __syncthreads();
    if (tid < HC) {
        float pr = sm.mix[tid] * hc_scale[0] + hc_base[tid];
        sm.pre[tid] = 1.f / (1.f + expf(-pr)) + hc_eps;
    }
    __syncthreads();
    if (tid < 32) {
        glm_hcf_sinkhorn<HC>(sm.mix, hc_scale, hc_base, post_out, comb_out, t, sinkhorn_iters,
                             hc_eps);
    } else {
        glm_hcf_collapse_norm<HC>(x, sm.pre, norm_w, y_out + (size_t)t * hidden_size,
                                  normed_out + (size_t)t * hidden_size, sm.y, sm.ws, hidden_size,
                                  rms_eps);
    }
}
