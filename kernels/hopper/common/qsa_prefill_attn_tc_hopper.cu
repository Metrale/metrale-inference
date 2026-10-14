// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: `qsa_prefill_attn_tc`: the QSA selective prefill attention of `qsa_prefill_attn`
// (gb10/qwen3.8-flash-next/nvfp4/qsa_indexer.cu), same arguments and output layout, on tensor
// cores with one CTA per (row, KV head) instead of one per (row, query head).
//
// - The G = nq / nkv query heads of a KV head share the row's key list, so they are stacked into
//   the 16 rows of one m16n8k16 A tile (rows G..15 are zero) and every gathered K/V row is read
//   once per KV head instead of once per query head (12x less traffic at G = 12).
// - Keys are the row's `topk` selected blocks of `ratio` tokens in list order, then the incomplete
//   tail block, as the base kernel walks them; they are gathered from the paged cache in tiles of
//   QPA_KT tokens through a two-stage cp.async pipeline, zero-filled and masked past the end.
// - Per tile: S = Q K^T in FP32 (m16n8k16 BF16, ascending k16), scaled by `inv_sqrt_d`; an online
//   softmax per row in FP32 (`__expf`); P rounded to BF16 for P V; the accumulators rescaled by
//   exp(m_old - m_new) before each tile's P V. The output is acc / l rounded to BF16.
// - Every reduction has a fixed order that depends on the shape only, so a row's output is the
//   same bits on every run and in every slab.
//
// Numerics differ from the base kernel's (FP32 p, per-warp strided partials): Tier 2 compares
// them within a tolerance. Requirements (the host admits only these; the kernel traps on the
// rest): hd == QPA_HD, 1 <= nq / nkv <= 16, nq % nkv == 0, 16-byte aligned q / caches, and
// QPA_SMEM_BYTES of dynamic shared memory.
//
// Owner: hopper kernels.
// Invariants: attn_out rows [r * nq + kvh * G, +G) are written for every row r and KV head kvh;
// nothing else is written.

#include <cuda_bf16.h>

#define QPA_HD 256
#define QPA_KT 64
#define QPA_THREADS 256
#define QPA_WARPS (QPA_THREADS / 32)
// 2026-10-14: Row strides padded by 8 BF16 (16 bytes) so the eight rows of an ldmatrix land in
// distinct banks.
#define QPA_LD (QPA_HD + 8)
#define QPA_PLD (QPA_KT + 8)
#define QPA_SLD (QPA_KT + 4)
#define QPA_Q_BYTES (16 * QPA_LD * 2)
#define QPA_KV_BYTES (QPA_KT * QPA_LD * 2)
#define QPA_STAGE_BYTES (2 * QPA_KV_BYTES)
#define QPA_S_BYTES (16 * QPA_SLD * 4)
#define QPA_P_BYTES (16 * QPA_PLD * 2)
#define QPA_SMEM_BYTES (QPA_Q_BYTES + 2 * QPA_STAGE_BYTES + QPA_S_BYTES + QPA_P_BYTES + 2 * 16 * 4)

__device__ __forceinline__ unsigned int qpa_saddr(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// 2026-10-14: 16 bytes global -> shared; zero-filled (no global read) when !valid.
__device__ __forceinline__ void qpa_cp16(void* dst, const void* src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(qpa_saddr(dst)),
                 "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void qpa_ldm_x4(unsigned int (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(qpa_saddr(p)));
}

__device__ __forceinline__ void qpa_ldm_x4_t(unsigned int (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(qpa_saddr(p)));
}

__device__ __forceinline__ void qpa_mma(float (&c)[4], const unsigned int (&a)[4],
                                        unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, "
        "{%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// 2026-10-14: Issue the cp.async gathers of key tile `tile` (K and V rows of tokens
// [tile * QPA_KT, +QPA_KT)) into one stage. Thread i copies 16-byte chunk (i % 32) of tile rows
// i / 32, i / 32 + 8, ...: a warp reads one 512-byte row contiguously.
__device__ __forceinline__ void qpa_issue(
    __nv_bfloat16* ks, __nv_bfloat16* vs, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int* __restrict__ block_table,
    const int* __restrict__ list, unsigned int tile, unsigned int n_tok, unsigned int sel,
    unsigned int complete, unsigned int ratio, unsigned int block_size, unsigned int row_elems,
    unsigned long long page_stride, unsigned int kvh) {
    const unsigned int chunk = threadIdx.x & 31u;
    for (unsigned int kr = threadIdx.x >> 5; kr < QPA_KT; kr += QPA_WARPS) {
        const unsigned int j = tile * QPA_KT + kr;
        const bool valid = j < n_tok;
        unsigned long long off = 0;
        if (valid) {
            const unsigned int tok = (j < sel) ? (unsigned int)list[j / ratio] * ratio + (j % ratio)
                                               : complete * ratio + (j - sel);
            off = (unsigned long long)(unsigned int)block_table[tok / block_size] * page_stride +
                  (unsigned long long)(tok % block_size) * row_elems +
                  (unsigned long long)kvh * QPA_HD + chunk * 8u;
        }
        qpa_cp16(ks + kr * QPA_LD + chunk * 8u, k_cache + off, valid);
        qpa_cp16(vs + kr * QPA_LD + chunk * 8u, v_cache + off, valid);
    }
}

extern "C" __global__ void __launch_bounds__(QPA_THREADS, 1) qsa_prefill_attn_tc(
    const __nv_bfloat16* __restrict__ q,        // 2026-10-14: [rows, nq, hd], roped
    const __nv_bfloat16* __restrict__ k_cache,  // 2026-10-14: paged NHD
    const __nv_bfloat16* __restrict__ v_cache,
    const int* __restrict__ block_table,
    const int* __restrict__ lists,              // 2026-10-14: [rows, topk] block ids
    __nv_bfloat16* __restrict__ attn_out,       // 2026-10-14: [rows, nq, hd]
    const unsigned int first_pos, const unsigned int topk, const unsigned int ratio,
    const unsigned int block_size, const unsigned int nq, const unsigned int nkv,
    const unsigned int hd, const float inv_sqrt_d) {
    const unsigned int group = nkv ? nq / nkv : 0;
    if (hd != QPA_HD || group == 0 || group > 16 || group * nkv != nq) __trap();

    const unsigned int r = blockIdx.x;
    const unsigned int kvh = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31u;
    const unsigned int warp = tid >> 5;
    const unsigned int pos = first_pos + r;
    const unsigned int complete = (pos + 1) / ratio;
    const unsigned int sel = topk * ratio;
    const unsigned int n_tok = sel + (pos + 1) - complete * ratio;
    const unsigned int n_tiles = (n_tok + QPA_KT - 1) / QPA_KT;
    const unsigned int row_elems = nkv * QPA_HD;
    const unsigned long long page_stride = (unsigned long long)block_size * row_elems;
    const int* list = lists + (size_t)r * topk;

    extern __shared__ __align__(16) unsigned char qpa_smem[];
    __nv_bfloat16* qs = (__nv_bfloat16*)qpa_smem;
    unsigned char* stage0 = qpa_smem + QPA_Q_BYTES;
    float* s_tile = (float*)(stage0 + 2 * QPA_STAGE_BYTES);
    __nv_bfloat16* p_tile = (__nv_bfloat16*)((unsigned char*)s_tile + QPA_S_BYTES);
    float* row_scale = (float*)((unsigned char*)p_tile + QPA_P_BYTES);
    float* row_inv_l = row_scale + 16;

    // 2026-10-14: Q rows 0..G-1 are the group's heads; rows G..15 are zero.
    for (unsigned int i = tid; i < 16 * (QPA_HD / 8); i += QPA_THREADS) {
        const unsigned int qr = i / (QPA_HD / 8), c = i % (QPA_HD / 8);
        const bool valid = qr < group;
        const __nv_bfloat16* src =
            q + (((size_t)r * nq + kvh * group + (valid ? qr : 0)) * QPA_HD + c * 8);
        qpa_cp16(qs + qr * QPA_LD + c * 8, src, valid);
    }
    {
        __nv_bfloat16* ks = (__nv_bfloat16*)stage0;
        qpa_issue(ks, ks + QPA_KT * QPA_LD, k_cache, v_cache, block_table, list, 0, n_tok, sel,
                  complete, ratio, block_size, row_elems, page_stride, kvh);
    }
    asm volatile("cp.async.commit_group;\n" ::);

    // 2026-10-14: Softmax ownership: row `tid / 16`, columns 4 * (tid % 16) .. +3; the 16 threads
    // of a row hold its running max and sum redundantly.
    const unsigned int srow = tid >> 4;
    const unsigned int scol = (tid & 15u) * 4u;
    float m_run = -1e30f, l_run = 0.0f;

    // 2026-10-14: P V ownership: warp w accumulates output dims [32 w, 32 w + 32) as 4 n8 tiles.
    float acc[4][4];
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[n][e] = 0.0f;

    for (unsigned int t = 0; t < n_tiles; ++t) {
        if (t + 1 < n_tiles) {
            __nv_bfloat16* ks = (__nv_bfloat16*)(stage0 + ((t + 1) & 1u) * QPA_STAGE_BYTES);
            qpa_issue(ks, ks + QPA_KT * QPA_LD, k_cache, v_cache, block_table, list, t + 1, n_tok,
                      sel, complete, ratio, block_size, row_elems, page_stride, kvh);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 1;\n" ::);
        __syncthreads();

        const __nv_bfloat16* ks = (const __nv_bfloat16*)(stage0 + (t & 1u) * QPA_STAGE_BYTES);
        const __nv_bfloat16* vs = ks + QPA_KT * QPA_LD;

        // 2026-10-14: S[:, 8 w .. 8 w + 8) = Q K^T over hd in ascending k16 steps.
        {
            float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const unsigned int n0 = warp * 8u;
#pragma unroll
            for (unsigned int k0 = 0; k0 < QPA_HD; k0 += 32) {
                unsigned int a0[4], a1[4], b[4];
                qpa_ldm_x4(a0, qs + (lane & 15u) * QPA_LD + k0 + (lane >> 4) * 8u);
                qpa_ldm_x4(a1, qs + (lane & 15u) * QPA_LD + k0 + 16 + (lane >> 4) * 8u);
                // 2026-10-14: B (k x n, col): lanes 8i..8i+7 address key rows n0..n0+7 at
                // k0 + 8 i, giving {b0, b1} of step k0 and {b2, b3} of step k0 + 16.
                qpa_ldm_x4(b, ks + (n0 + (lane & 7u)) * QPA_LD + k0 + (lane >> 3) * 8u);
                qpa_mma(s, a0, b[0], b[1]);
                qpa_mma(s, a1, b[2], b[3]);
            }
            const unsigned int g = lane >> 2, c = n0 + (lane & 3u) * 2u;
            s_tile[g * QPA_SLD + c] = s[0] * inv_sqrt_d;
            s_tile[g * QPA_SLD + c + 1] = s[1] * inv_sqrt_d;
            s_tile[(g + 8) * QPA_SLD + c] = s[2] * inv_sqrt_d;
            s_tile[(g + 8) * QPA_SLD + c + 1] = s[3] * inv_sqrt_d;
        }
        __syncthreads();

        // 2026-10-14: Online softmax of row `srow`, the max and sum over its 16 threads by a
        // fixed xor butterfly.
        {
            float v[4];
            float mx = -INFINITY;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int j = t * QPA_KT + scol + e;
                v[e] = (j < n_tok) ? s_tile[srow * QPA_SLD + scol + e] : -INFINITY;
                mx = fmaxf(mx, v[e]);
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xFFFFFFFFu, mx, o));
            const float m_new = fmaxf(m_run, mx);
            const float scale = __expf(m_run - m_new);
            float sum = 0.0f;
            __nv_bfloat16 p[4];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float pe = __expf(v[e] - m_new);
                sum += pe;
                p[e] = __float2bfloat16(pe);
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
            l_run = l_run * scale + sum;
            m_run = m_new;
#pragma unroll
            for (int e = 0; e < 4; ++e) p_tile[srow * QPA_PLD + scol + e] = p[e];
            if ((tid & 15u) == 0) row_scale[srow] = scale;
        }
        __syncthreads();

        // 2026-10-14: acc = acc * scale + P V over the tile's keys in ascending k16 steps.
        {
            const unsigned int g = lane >> 2;
            const float sc_lo = row_scale[g], sc_hi = row_scale[g + 8];
#pragma unroll
            for (int n = 0; n < 4; ++n) {
                acc[n][0] *= sc_lo;
                acc[n][1] *= sc_lo;
                acc[n][2] *= sc_hi;
                acc[n][3] *= sc_hi;
            }
            const unsigned int d0 = warp * 32u;
#pragma unroll
            for (unsigned int k0 = 0; k0 < QPA_KT; k0 += 16) {
                unsigned int a[4];
                qpa_ldm_x4(a, p_tile + (lane & 15u) * QPA_PLD + k0 + (lane >> 4) * 8u);
#pragma unroll
                for (unsigned int np = 0; np < 2; ++np) {
                    // 2026-10-14: B (k = key, n = dim) from row-major V via .trans: lanes 0-15
                    // address key rows k0..k0+15 at dims n, lanes 16-31 the same rows at n + 8.
                    unsigned int b[4];
                    const unsigned int n = d0 + np * 16u;
                    qpa_ldm_x4_t(b, vs + (k0 + (lane & 15u)) * QPA_LD + n + (lane >> 4) * 8u);
                    qpa_mma(acc[2 * np], a, b[0], b[1]);
                    qpa_mma(acc[2 * np + 1], a, b[2], b[3]);
                }
            }
        }
        __syncthreads();
    }
    asm volatile("cp.async.wait_group 0;\n" ::);

    if ((tid & 15u) == 0) row_inv_l[srow] = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
    __syncthreads();

    const unsigned int g = lane >> 2;
    const float inv_lo = row_inv_l[g], inv_hi = row_inv_l[g + 8];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
        const unsigned int d = warp * 32u + n * 8u + (lane & 3u) * 2u;
        if (g < group) {
            __nv_bfloat162 o = __floats2bfloat162_rn(acc[n][0] * inv_lo, acc[n][1] * inv_lo);
            *(__nv_bfloat162*)(attn_out + ((size_t)r * nq + kvh * group + g) * QPA_HD + d) = o;
        }
        if (g + 8 < group) {
            __nv_bfloat162 o = __floats2bfloat162_rn(acc[n][2] * inv_hi, acc[n][3] * inv_hi);
            *(__nv_bfloat162*)(attn_out + ((size_t)r * nq + kvh * group + g + 8) * QPA_HD + d) =
                o;
        }
    }
}
