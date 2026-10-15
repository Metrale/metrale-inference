// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: glm5next_hc_post_rank_sum: the reduce step of the one-shot all-reduce
// (bf16_add_rank_sum) and the mHC post-mix after it (glm5next_hc_post) in one launch.
//
// Owner: gb10 kernels.
// Invariants:
// - Byte-identical to bf16_add_rank_sum (bf16_add.cu) followed by glm5next_hc_post
//   (glm5next_mhc.cu) on the same inputs: x is the FP32 sum of the ranks' partials in rank
//   order 0..n_ranks-1 from 0.0f, rounded to BF16 once; the post-mix then widens that BF16 value
//   and runs glm5next_hc_post's loops and order. Both outputs are written: `block_out` gets x (what
//   bf16_add_rank_sum leaves there) and `out` the post-mix. glm5next_hc_post_sum_gate (model-arch
//   examples) checks both byte for byte.
// - `block_out` holds this rank's partial, `peers` the others', one slot of `slot_elems` elements
//   per peer, peer r in slot (r < my_rank ? r : r - 1); element (t, d) is at t * H + d in both.
// - hc_mult <= 4; `out` may alias `residual` (same reasoning as glm5next_hc_post: column d is read
//   and written by one thread only).
// - 2026-10-14: Any grid covers the rows: block b handles rows blockIdx.x, + gridDim.x, ... < T
//   and, in each, columns blockIdx.y * blockDim.x + tid, + gridDim.y * blockDim.x, ... < H. An
//   element's arithmetic does not depend on which thread computes it, so every grid writes the
//   same bytes. The default grid is (T, NB) of 256 as glm5next_hc_post (one row per block row);
//   METRALE_GLM_MHC_POST_SUM_CTAS caps it at (CTAS, 1) of 1024 (glm5next_hc_post_sum_grid.rs).
// Callers: model-arch glm5next_mhc.rs glm_hc_post_rank_sum, opt-in METRALE_GLM_AR_FUSED_POST.

#include <cuda_bf16.h>

#define GLM_HCPS_MAX_MULT 4
// 2026-10-15: The most ranks glm5next_hc_post_rank_sum_vec8 holds in registers (host-checked).
#define GLM_HCPS_MAX_RANKS 8

extern "C" __global__ void __launch_bounds__(1024) glm5next_hc_post_rank_sum(
    __nv_bfloat16* __restrict__ block_out,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int num_tokens
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    __shared__ float s_p[GLM_HCPS_MAX_MULT];
    __shared__ float s_c[GLM_HCPS_MAX_MULT * GLM_HCPS_MAX_MULT];

    for (unsigned int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        __nv_bfloat16* x = block_out + (size_t)t * H;
        const float* res = residual + (size_t)t * hc * H;
        const float* p = post + (size_t)t * hc;
        const float* c = comb + (size_t)t * hc * hc;
        float* o = out + (size_t)t * hc * H;

        __syncthreads();  // the previous row's readers of s_p/s_c are done
        if (tid < hc) s_p[tid] = p[tid];
        if (tid < hc * hc) s_c[tid] = c[tid];
        __syncthreads();

        const unsigned int stride = gridDim.y * blockDim.x;
        for (unsigned int d = blockIdx.y * blockDim.x + tid; d < H; d += stride) {
            const long long e = (long long)t * H + d;
            // bf16_add_rank_sum's arithmetic for element e.
            float acc = 0.0f;
            for (int r = 0; r < n_ranks; ++r) {
                __nv_bfloat16 v = (r == my_rank) ? x[d]
                    : peers[(long long)(r < my_rank ? r : r - 1) * slot_elems + e];
                acc += __bfloat162float(v);
            }
            const __nv_bfloat16 xb = __float2bfloat16_rn(acc);
            x[d] = xb;
            // glm5next_hc_post's arithmetic for column d.
            float xd = (float)xb;
            float rv[GLM_HCPS_MAX_MULT];
#pragma unroll
            for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i)
                if (i < hc) rv[i] = res[i * H + d];
#pragma unroll
            for (unsigned int j = 0; j < GLM_HCPS_MAX_MULT; ++j) {
                if (j < hc) {
                    float a = s_p[j] * xd;
#pragma unroll
                    for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i)
                        if (i < hc) a += s_c[i * hc + j] * rv[i];
                    o[j * H + d] = a;
                }
            }
        }
    }
}

// 2026-10-15: glm5next_hc_post_rank_sum_vec8: glm5next_hc_post_rank_sum with each thread owning
// eight consecutive columns, so this rank's partial, every peer slot and the summed row move as
// 16-byte accesses and each residual and output stream as two 16-byte accesses.
// Invariants:
// - Every element is glm5next_hc_post_rank_sum's: the FP32 rank sum in rank order from 0.0f,
//   one __float2bfloat16_rn, then the post-mix of the widened BF16 value in the same loop order
//   (s_p[j] * xd, then + s_c[i * hc + j] * rv[i] for i = 0..hc-1; --fmad=false, KERNEL.toml).
//   glm5next_hc_post_sum_gate (model-arch examples) checks the bytes against the scalar kernel.
// - write_x != 0 stores the summed row to block_out as the scalar kernel does; write_x == 0
//   leaves block_out holding this rank's partial (the SP prefill post-mix, where nothing reads
//   the sum afterwards unless a dump tap does; the host passes 1 then).
// - The host launches it only when hidden_size % 8 == 0, slot_elems % 8 == 0 and block_out,
//   peers, residual and out are 16-byte aligned; grid (T', ceil(hidden_size / 8 / blockDim.x))
//   with T' <= T (rows grid-stride over blockIdx.x). hc_mult <= 4; `out` may alias `residual`
//   (a column group is read and written by one thread only).
extern "C" __global__ void __launch_bounds__(256) glm5next_hc_post_rank_sum_vec8(
    __nv_bfloat16* __restrict__ block_out,
    const __nv_bfloat16* __restrict__ peers,
    long long slot_elems,
    int n_ranks,
    int my_rank,
    const float* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    float* __restrict__ out,
    const unsigned int hidden_size,
    const unsigned int hc_mult,
    const unsigned int num_tokens,
    const int write_x
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const unsigned int hc = hc_mult;
    const unsigned int d = (blockIdx.y * blockDim.x + tid) * 8u;
    __shared__ float s_p[GLM_HCPS_MAX_MULT];
    __shared__ float s_c[GLM_HCPS_MAX_MULT * GLM_HCPS_MAX_MULT];

    for (unsigned int t = blockIdx.x; t < num_tokens; t += gridDim.x) {
        __syncthreads();  // the previous row's readers of s_p/s_c are done
        if (tid < hc) s_p[tid] = post[(size_t)t * hc + tid];
        if (tid < hc * hc) s_c[tid] = comb[(size_t)t * hc * hc + tid];
        __syncthreads();
        if (d >= H) continue;

        const long long e = (long long)t * H + d;
        __nv_bfloat16* x = block_out + e;
        uint4 v[GLM_HCPS_MAX_RANKS];
#pragma unroll
        for (int r = 0; r < GLM_HCPS_MAX_RANKS; ++r) {
            if (r < n_ranks) {
                const __nv_bfloat16* src = (r == my_rank)
                    ? x : peers + (long long)(r < my_rank ? r : r - 1) * slot_elems + e;
                v[r] = *reinterpret_cast<const uint4*>(src);
            }
        }
        const float* res = residual + (size_t)t * hc * H + d;
        float4 rv[GLM_HCPS_MAX_MULT][2];
#pragma unroll
        for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i) {
            if (i < hc) {
                rv[i][0] = *reinterpret_cast<const float4*>(res + (size_t)i * H);
                rv[i][1] = *reinterpret_cast<const float4*>(res + (size_t)i * H + 4);
            }
        }
        // bf16_add_rank_sum's arithmetic for the eight elements.
        float xd[8];
        uint4 xo;
        __nv_bfloat16* xob = reinterpret_cast<__nv_bfloat16*>(&xo);
#pragma unroll
        for (unsigned int k = 0; k < 8u; ++k) {
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < GLM_HCPS_MAX_RANKS; ++r)
                if (r < n_ranks) acc += __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(&v[r])[k]);
            xob[k] = __float2bfloat16_rn(acc);
            xd[k] = (float)xob[k];
        }
        if (write_x) *reinterpret_cast<uint4*>(x) = xo;
        // glm5next_hc_post's arithmetic for the eight columns.
        float* o = out + (size_t)t * hc * H + d;
#pragma unroll
        for (unsigned int j = 0; j < GLM_HCPS_MAX_MULT; ++j) {
            if (j < hc) {
                float a[8];
#pragma unroll
                for (unsigned int k = 0; k < 8u; ++k) a[k] = s_p[j] * xd[k];
#pragma unroll
                for (unsigned int i = 0; i < GLM_HCPS_MAX_MULT; ++i) {
                    if (i < hc) {
                        const float c = s_c[i * hc + j];
                        const float* r8 = reinterpret_cast<const float*>(&rv[i][0]);
#pragma unroll
                        for (unsigned int k = 0; k < 8u; ++k) a[k] += c * r8[k];
                    }
                }
                *reinterpret_cast<float4*>(o + (size_t)j * H) = make_float4(a[0], a[1], a[2], a[3]);
                *reinterpret_cast<float4*>(o + (size_t)j * H + 4) = make_float4(a[4], a[5], a[6], a[7]);
            }
        }
    }
}
