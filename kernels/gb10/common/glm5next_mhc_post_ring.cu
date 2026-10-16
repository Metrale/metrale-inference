// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-15: GLM-5.3 mHC post-mix + next pre-norm in one launch on tile4's ring
// (METRALE_GLM_MHC_POST_PRE_TILE, design (c); host side in model-arch glm5next_mhc_post_tile.rs):
// tile4's four-token group with its cp.async ring, trees and tails unchanged
// (glm5next_mhc_tile_ring.cuh), with vec8's post-mix fused in as the producer of the streams the
// ring reads. hidden 4096 and hc_mult 4 only.
//
// Owner: gb10 kernels (family glm_mhc).
// Invariants:
// - Every output is byte-identical to glm5next_hc_post_rank_sum_vec8 (glm5next_hc_post_sum.cu)
//   followed by glm5next_hc_pre_norm_tile4 (glm5next_mhc_tile.cu) on the same inputs. Thread t
//   forms column d = cc * 256 + t of the four streams of each of the group's tokens with vec8's
//   expression (glm_hcpr_store: the FP32 rank sum in rank order from 0.0f, one
//   __float2bfloat16_rn, then s_p[j] * xd + s_c[i * 4 + j] * rv[i] for i = 0..3, --fmad=false),
//   writes them to `streams_out`, and the ring reads them back as tile4 reads `streams`; a ring
//   step of stream i at column chunk cc is issued only after a barrier that follows the store of
//   chunk cc. The chains, trees and tails are tile4's. examples/glm5next_hc_post_pre_tile_gate.rs
//   checks every output byte.
// - One 256-thread group per block, blockDim 256; grid any size up to ceil(T / 4) (the host
//   launches min(ceil(T / 4), SM count)): block b runs token groups b, b + gridDim.x, ...;
//   dynamic shared memory GlmHcprLayout::BYTES (glm5next_mhc_post_tile.rs computes the same).
// - Self: a group forms its own chunks 0..3 before its ring starts and chunk c + 4 in chain step
//   c < 12. PIPE (ring1, ring1e): only a block's first group does; every later group's chunks
//   were formed in the previous group's steps 16..63 (chunk m loaded in step 16 + 3m, stored in
//   step 18 + 3m), its post and comb rows staged in step 12 after the last self store.
// - EVF (ring1e): the partial and residual loads carry an L2 evict_first policy; no value changes.
// - Aliasing (the SP prefill call): `streams_out` may be `residual` and `post_out`/`comb_out` may
//   be `post`/`comb`: a stream element is read and then written by one thread, and a token's
//   post/comb rows are read into shared memory before its group writes them.
// - write_x != 0 stores the summed row to block_out as vec8 does; 0 leaves this rank's partial.
// - n_ranks 1..GLM_HCPT_MAX_RANKS, hidden_size GLM_HCPT_H, blockDim 256 (traps otherwise);
//   `streams_out` and hc_fn 16-byte aligned (cp.async). The host refuses every other shape first.

#include "e4m3_mma_pipe.cuh"
#include "glm5next_mhc_fused_parts.cuh"
#include "glm5next_mhc_tile_parts.cuh"
#include "glm5next_mhc_tile_ring.cuh"
#include "glm5next_mhc_post_args.cuh"

#define GLM_HCPR_PC (GLM_HCT_HC + GLM_HCT_HC * GLM_HCT_HC)
#define GLM_HCPR_SELF (GLM_HCPT_COLS - GLM_HCT_STAGES)

// 2026-10-15: tile4's layout for one group, then the post and comb rows of four tokens.
struct GlmHcprLayout {
    static constexpr unsigned int PC_OFF = (GlmHctLayout<1>::BYTES + 15u) & ~15u;
    static constexpr unsigned int BYTES = PC_OFF + GLM_HCT_TOK * GLM_HCPR_PC * 4;
};

// 2026-10-15: One column chunk's operands for four tokens, in registers between load and store.
struct GlmHcprBuf {
    __nv_bfloat16 v[GLM_HCT_TOK][GLM_HCPT_MAX_RANKS];
    float rv[GLM_HCT_TOK][GLM_HCT_HC];
};

template <bool EVF>
__device__ __forceinline__ float glm_hcpr_ldf(const float* p, unsigned long long pol) {
    if constexpr (EVF) {
        float v;
        asm volatile("ld.global.L2::cache_hint.f32 %0, [%1], %2;" : "=f"(v) : "l"(p), "l"(pol));
        return v;
    } else {
        return *p;
    }
}

template <bool EVF>
__device__ __forceinline__ __nv_bfloat16 glm_hcpr_ldh(const __nv_bfloat16* p,
                                                      unsigned long long pol) {
    if constexpr (EVF) {
        unsigned short v;
        asm volatile("ld.global.L2::cache_hint.b16 %0, [%1], %2;" : "=h"(v) : "l"(p), "l"(pol));
        return __ushort_as_bfloat16(v);
    } else {
        return *p;
    }
}

// 2026-10-15: Tokens t0 .. t0 + nt - 1 (nt <= 4): stage their post and comb rows.
__device__ __forceinline__ void glm_hcpr_pc(float* s_pc, const unsigned int t0,
                                            const unsigned int nt, const unsigned int t,
                                            const float* post, const float* comb) {
    if (t < GLM_HCT_TOK * GLM_HCPR_PC) {
        const unsigned int q = t / GLM_HCPR_PC, k = t % GLM_HCPR_PC;
        if (q < nt)
            s_pc[t] = k < GLM_HCT_HC ? post[(size_t)(t0 + q) * GLM_HCT_HC + k]
                                     : comb[(size_t)(t0 + q) * GLM_HCT_HC * GLM_HCT_HC + k -
                                            GLM_HCT_HC];
    }
}

// 2026-10-15: Column cc * 256 + t of tokens t0 .. t0 + nt - 1: this rank's and the peers'
// partials and the four residual streams.
template <bool EVF>
__device__ __forceinline__ void glm_hcpr_load(GlmHcprBuf& b, const unsigned int t0,
                                              const unsigned int nt, const unsigned int cc,
                                              const unsigned int t, unsigned long long pol,
                                              GLM_HCPT_ARGS) {
    constexpr unsigned int H = GLM_HCPT_H;
    const unsigned int d = cc * GLM_HCF_BLOCK + t;
#pragma unroll
    for (int q = 0; q < GLM_HCT_TOK; ++q) {
        if ((unsigned int)q < nt) {
            const long long e = (long long)(t0 + q) * H + d;
#pragma unroll
            for (int r = 0; r < GLM_HCPT_MAX_RANKS; ++r)
                if (r < n_ranks)
                    b.v[q][r] = glm_hcpr_ldh<EVF>(
                        (r == my_rank)
                            ? block_out + e
                            : peers + (long long)(r < my_rank ? r : r - 1) * slot_elems + e,
                        pol);
            const float* res = residual + (size_t)(t0 + q) * GLM_HCT_HC * H + d;
#pragma unroll
            for (int i = 0; i < GLM_HCT_HC; ++i)
                b.rv[q][i] = glm_hcpr_ldf<EVF>(res + (size_t)i * H, pol);
        }
    }
}

// 2026-10-15: vec8's post for the loaded column: the rank sum, the summed row (write_x) and
// the four streams, stored to `streams_out`.
__device__ __forceinline__ void glm_hcpr_store(const GlmHcprBuf& b, const unsigned int t0,
                                               const unsigned int nt, const unsigned int cc,
                                               const unsigned int t, const float* s_pc,
                                               GLM_HCPT_ARGS) {
    constexpr unsigned int H = GLM_HCPT_H;
    const unsigned int d = cc * GLM_HCF_BLOCK + t;
#pragma unroll
    for (int q = 0; q < GLM_HCT_TOK; ++q) {
        if ((unsigned int)q < nt) {
            const float* s_p = s_pc + q * GLM_HCPR_PC;
            const float* s_c = s_p + GLM_HCT_HC;
            float acc = 0.0f;
#pragma unroll
            for (int r = 0; r < GLM_HCPT_MAX_RANKS; ++r)
                if (r < n_ranks) acc += __bfloat162float(b.v[q][r]);
            const __nv_bfloat16 xb = __float2bfloat16_rn(acc);
            if (write_x) block_out[(size_t)(t0 + q) * H + d] = xb;
            const float xd = (float)xb;
            float* o = streams_out + (size_t)(t0 + q) * GLM_HCT_HC * H + d;
#pragma unroll
            for (int j = 0; j < GLM_HCT_HC; ++j) {
                float a = s_p[j] * xd;
#pragma unroll
                for (int i = 0; i < GLM_HCT_HC; ++i) a += s_c[i * GLM_HCT_HC + j] * b.rv[q][i];
                o[(size_t)j * H] = a;
            }
        }
    }
}

template <bool PIPE, bool EVF>
__device__ __forceinline__ void glm_hcpr_body(GLM_HCPT_ARGS) {
    using L = GlmHctLayout<1>;
    constexpr unsigned int MIX = GLM_HCT_MIX;
    constexpr int ST = GLM_HCT_STAGES;
    constexpr unsigned int COLS = GLM_HCPT_COLS;
    constexpr unsigned int HC_DIM = GLM_HCT_HC * GLM_HCPT_H;
    constexpr unsigned int STEPS = HC_DIM / GLM_HCF_BLOCK;
    static_assert(ST <= (int)COLS && 3 * COLS <= STEPS - COLS, "post chunk schedule");
    extern __shared__ __align__(16) unsigned char glm_hcpr_smem[];
    const unsigned int t = threadIdx.x;
    // 2026-10-15: A launch outside the layout's shape traps (fail fast); the host refuses first.
    unsigned int dyn_smem;
    asm("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_smem));
    if (dyn_smem < GlmHcprLayout::BYTES || hidden_size != GLM_HCPT_H || n_ranks < 1 ||
        n_ranks > GLM_HCPT_MAX_RANKS || blockDim.x != GLM_HCF_BLOCK)
        __trap();
    const unsigned int groups = (num_tokens + GLM_HCT_TOK - 1) / GLM_HCT_TOK;
    float* s_pc = reinterpret_cast<float*>(glm_hcpr_smem + GlmHcprLayout::PC_OFF);
    unsigned long long pol = 0;
    if constexpr (EVF)
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    GlmHcprBuf buf;

    bool self = true;
    for (unsigned int grp = blockIdx.x; grp < groups; grp += gridDim.x) {
        const unsigned int t0 = grp * GLM_HCT_TOK;
        const unsigned int ntb = min((unsigned int)GLM_HCT_TOK, num_tokens - t0);
        const unsigned int nxt = grp + gridDim.x;
        const bool next = PIPE && nxt < groups;
        const unsigned int t1 = nxt * GLM_HCT_TOK;
        const unsigned int ntn = next ? min((unsigned int)GLM_HCT_TOK, num_tokens - t1) : 0u;
        const float* xb = streams_out + (size_t)t0 * HC_DIM;
        if (self) {
            glm_hcpr_pc(s_pc, t0, ntb, t, post, comb);
            __syncthreads();
#pragma unroll 1
            for (unsigned int cc = 0; cc < (unsigned int)ST; ++cc) {
                glm_hcpr_load<EVF>(buf, t0, ntb, cc, t, pol, GLM_HCPT_PASS);
                glm_hcpr_store(buf, t0, ntb, cc, t, s_pc, GLM_HCPT_PASS);
            }
            __syncthreads();
        }

        float ss[GLM_HCT_TOK];
        float acc[GLM_HCT_TOK][MIX];
#pragma unroll
        for (int q = 0; q < GLM_HCT_TOK; ++q) {
            ss[q] = 0.f;
#pragma unroll
            for (int r = 0; r < (int)MIX; ++r) acc[q][r] = 0.f;
        }
#pragma unroll
        for (int s = 0; s < ST - 1; ++s)
            glm_hct_issue<1>(glm_hcpr_smem + s * L::STAGE, xb, streams_out, hc_fn, HC_DIM,
                             s * GLM_HCF_BLOCK, ntb);
#pragma unroll 1
        for (unsigned int c = 0; c < STEPS; ++c) {
            e4m3g::cp_wait<ST - 2>();
            __syncthreads();
            // 2026-10-15: tile4's refill of the stage step c - 1 used. Its stream columns were
            // stored before this barrier (self: chunk c + ST - 1 in step c - 1 or before the ring).
            const unsigned int n = c + ST - 1;
            if (n < STEPS)
                glm_hct_issue<1>(glm_hcpr_smem + (n % ST) * L::STAGE, xb, streams_out, hc_fn,
                                 HC_DIM, n * GLM_HCF_BLOCK, ntb);
            else
                e4m3g::cp_commit();
            const bool sp = self && c < GLM_HCPR_SELF;
            const unsigned int m = (c - COLS) / 3, ph = (c - COLS) % 3;
            const bool np = next && c >= COLS;
            if (sp) glm_hcpr_load<EVF>(buf, t0, ntb, c + ST, t, pol, GLM_HCPT_PASS);
            if (next && c == GLM_HCPR_SELF) glm_hcpr_pc(s_pc, t1, ntn, t, post, comb);
            if (np && ph == 0) glm_hcpr_load<EVF>(buf, t1, ntn, m, t, pol, GLM_HCPT_PASS);
            const unsigned char* stage = glm_hcpr_smem + (c % ST) * L::STAGE;
            const __nv_bfloat16* sw = reinterpret_cast<const __nv_bfloat16*>(stage);
            const float* sv = reinterpret_cast<const float*>(stage + L::W_BYTES);
            float v[GLM_HCT_TOK];
#pragma unroll
            for (int q = 0; q < GLM_HCT_TOK; ++q) v[q] = sv[q * GLM_HCF_BLOCK + t];
#pragma unroll
            for (int q = 0; q < GLM_HCT_TOK; ++q) ss[q] += v[q] * v[q];
#pragma unroll
            for (int r = 0; r < (int)MIX; ++r) {
                const float w = __bfloat162float(sw[r * GLM_HCF_BLOCK + t]);
#pragma unroll
                for (int q = 0; q < GLM_HCT_TOK; ++q) acc[q][r] += w * v[q];
            }
            if (sp) glm_hcpr_store(buf, t0, ntb, c + ST, t, s_pc, GLM_HCPT_PASS);
            if (np && ph == 2) glm_hcpr_store(buf, t1, ntn, m, t, s_pc, GLM_HCPT_PASS);
        }
        e4m3g::cp_wait<0>();
        __syncthreads();
        glm_hct_group_finish<1, 0>(ss, acc, glm_hcpr_smem, t, t0, ntb, HC_DIM, xb, streams_out,
                                   hc_fn, hc_scale, hc_base, mix, nullptr, norm_w, y_out,
                                   post_out, comb_out, normed_out, hidden_size, GLM_HCT_HC,
                                   sinkhorn_iters, norm_eps, hc_eps, rms_eps);
        // 2026-10-15: The next group's ring reuses the shared memory the tails read.
        __syncthreads();
        self = !PIPE;
    }
}

// 2026-10-15: Each group forms its own streams (no lead).
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 1)
glm5next_hc_post_pre_ring0(GLM_HCPT_ARGS) {
    glm_hcpr_body<false, false>(GLM_HCPT_PASS);
}

// 2026-10-15: Each group forms the next group's streams during its own steps 16..63.
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 1)
glm5next_hc_post_pre_ring1(GLM_HCPT_ARGS) {
    glm_hcpr_body<true, false>(GLM_HCPT_PASS);
}

// 2026-10-15: ring1 with the partial and residual loads marked L2 evict_first.
extern "C" __global__ void __launch_bounds__(GLM_HCF_BLOCK, 1)
glm5next_hc_post_pre_ring1e(GLM_HCPT_ARGS) {
    glm_hcpr_body<true, true>(GLM_HCPT_PASS);
}
