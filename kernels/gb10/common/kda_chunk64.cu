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
// The factors U, W, Qg = scale exp(G) q, Kd = exp(G_last - G) k, P and decay = exp(G_last) come
// from kda_chunk64_prepare.cu (kda_chunk64_prepare); kda_chunk64.cuh has the
// scratch layout.
//
// kda_chunk64_scan, one block of 4 warps per (head, 64-value-column block), serial over the
// slab's chunks (slab-local chunk lc covers tokens (c0 + lc) * C ..). Warp w holds the state rows 32 w .. 32 w + 31 x the block's 64 columns in registers in
// the MMA accumulator layout for the whole walk. Per chunk: (A) delta = U - W S0 and o = Qg S0
// (warp w: rows 16 w .. 16 w + 15), delta^T staged to shared memory; (B) S = diag(decay) S +
// Kd^T delta; (C) o += P delta, written out. Products on mma.sync.m16n8k16 BF16 with FP32
// accumulation (W, Qg, Kd, P, delta and the staged S0 rounded to BF16, as FLA's chunked KDA
// does); U, the state and the outputs stay FP32. So the outputs differ from the token walk at
// BF16 rounding level: opt-in and accuracy-gated (model-arch example glm5next_kda_chunk64_bench).
// The chunk's factors arrive by 16-byte cp.async: the next chunk's W and Qg are issued after
// phase A, Kd and decay after B, P after C, so they land while this chunk computes.
//
// Owner: gb10 kernels.
// Invariants:
// - head_dim D = KDA64_D (128), chunk KDA64_C (64), out FP32 [T, out_stride]; the state FP32
//   [H, D, D] indexed [k][v], read before the slab's first chunk and written after its last. Out
//   rows t >= T are not written.
// - grid (H, D / KDA64_VB), block KDA64_SCAN_THREADS, dynamic shared memory KDA64_SCAN_SMEM.

#include <cuda_bf16.h>
#include <stdint.h>

#include "kda_chunk64.cuh"

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

__device__ __forceinline__ void k64_cp16(uint32_t saddr, const void* g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(saddr), "l"(g));
}

__device__ __forceinline__ void k64_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }

__device__ __forceinline__ void k64_cp_wait_all() {
    asm volatile("cp.async.wait_group 0;\n" ::: "memory");
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

// 2026-10-12: One [C][D] BF16 factor of chunk `cb` into its padded smem tile, 16 B per copy.
__device__ __forceinline__ void k64_cp_tile(uint32_t sbase, const __nv_bfloat16* __restrict__ g,
                                            size_t cb, unsigned int tid) {
    constexpr int C = KDA64_C, D = KDA64_D, LD = KDA64_LD;
    for (unsigned int e = tid; e < (unsigned int)(C * D / 8); e += KDA64_SCAN_THREADS) {
        const unsigned int i = e / (D / 8), q8 = e % (D / 8);
        k64_cp16(sbase + (i * LD + q8 * 8u) * 2u, g + cb * C * D + (size_t)i * D + q8 * 8u);
    }
}

extern "C" __global__ void __launch_bounds__(KDA64_SCAN_THREADS) kda_chunk64_scan(
    const float* __restrict__ u_in,
    const __nv_bfloat16* __restrict__ w_in,
    const __nv_bfloat16* __restrict__ qg_in,
    const __nv_bfloat16* __restrict__ kd_in,
    const __nv_bfloat16* __restrict__ p_in,
    const float* __restrict__ ad_in,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int T,
    unsigned int c0,
    unsigned int num_chunks,
    unsigned int out_stride
) {
    constexpr int C = KDA64_C, D = KDA64_D, VB = KDA64_VB, LD = KDA64_LD, LDC = KDA64_LDC;
    constexpr int NT = VB / 8;
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
    const uint32_t bDec = (uint32_t)__cvta_generic_to_shared(sDec);

    const unsigned int h = blockIdx.x, v0 = blockIdx.y * VB, tid = threadIdx.x;
    const unsigned int warp = tid >> 5, lane = tid & 31u, g = lane >> 2, tig = lane & 3u;
    float* S = state + (size_t)h * D * D;

    // 2026-10-12: The chunk's factors, three cp.async groups: W + Qg; Kd + decay; P.
    auto load_wq = [&](size_t cb) {
        k64_cp_tile(bW, w_in, cb, tid);
        k64_cp_tile(bQ, qg_in, cb, tid);
        k64_cp_commit();
    };
    auto load_kd = [&](size_t cb) {
        k64_cp_tile(bKd, kd_in, cb, tid);
        if (tid < (unsigned int)(D / 4)) {
            k64_cp16(bDec + tid * 16u, ad_in + cb * KDA64_AD_STRIDE + C * C + tid * 4u);
        }
        k64_cp_commit();
    };
    auto load_p = [&](size_t cb) {
        for (unsigned int e = tid; e < (unsigned int)(C * C / 8); e += KDA64_SCAN_THREADS) {
            const unsigned int i = e / (C / 8), j8 = e % (C / 8);
            k64_cp16(bP + (i * LDC + j8 * 8u) * 2u, p_in + cb * C * C + (size_t)i * C + j8 * 8u);
        }
        k64_cp_commit();
    };

    // 2026-10-12: The state rows 32 warp .. 32 warp + 31, all VB columns, in the m16n8 accumulator
    // layout: st[mt][nt][0..3] = rows 32 warp + 16 mt + (g, g, g + 8, g + 8), columns
    // v0 + 8 nt + 2 tig + (0, 1, 0, 1).
    float st[2][NT][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int row = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                const unsigned int col = v0 + (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                st[mt][nt][e] = S[(size_t)row * D + col];
            }
    if (num_chunks > 0) {
        load_wq((size_t)h);
        load_kd((size_t)h);
        load_p((size_t)h);
    }

    const unsigned int r0 = warp * 16u;
    for (unsigned int c = 0; c < num_chunks; ++c) {
        const size_t cb = (size_t)c * H + h;
        const bool more = c + 1 < num_chunks;
        const size_t cbn = cb + H;
        // 2026-10-12: U rows for this warp's delta, requested now, consumed after phase A.
        float2 uv[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const unsigned int i = r0 + g + (unsigned int)hr * 8u;
                uv[nt][hr] = *(const float2*)(u_in + cb * C * D + (size_t)i * D + v0 +
                                              (unsigned int)nt * 8u + 2u * tig);
            }
        // 2026-10-12: S0^T as BF16: this warp's state rows are k = columns of S^T.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const unsigned int kr = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                    const unsigned int vc = (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                    sST[vc * LD + kr] = __float2bfloat16_rn(st[mt][nt][e]);
                }
        k64_cp_wait_all();
        __syncthreads();

        // 2026-10-12: (A) delta = U - W S0 and o = Qg S0 for rows 16 warp .. 16 warp + 15.
        float dl[NT][4], o[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                dl[nt][e] = 0.0f;
                o[nt][e] = 0.0f;
            }
#pragma unroll
        for (int ks = 0; ks < D / 16; ++ks) {
            uint32_t aw[4], aq[4];
            k64_frag_a(bW, LD, r0, (unsigned int)ks * 16u, lane, aw);
            k64_frag_a(bQ, LD, r0, (unsigned int)ks * 16u, lane, aq);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint32_t b0, b1;
                k64_frag_b(bST, LD, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                k64_mma(dl[nt], aw, b0, b1);
                k64_mma(o[nt], aq, b0, b1);
            }
        }
        // 2026-10-12: delta^T to shared memory as BF16.
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const unsigned int i = r0 + g + (unsigned int)hr * 8u;
                const unsigned int vc = (unsigned int)nt * 8u + 2u * tig;
                sDT[vc * LDC + i] = __float2bfloat16_rn(uv[nt][hr].x - dl[nt][hr * 2]);
                sDT[(vc + 1u) * LDC + i] = __float2bfloat16_rn(uv[nt][hr].y - dl[nt][hr * 2 + 1]);
            }
        }
        __syncthreads();
        if (more) load_wq(cbn);

        // 2026-10-12: (B) S = diag(decay) S0 + Kd^T delta for this warp's 32 state rows.
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const unsigned int ka = warp * 32u + (unsigned int)mt * 16u + g;
            const float d0 = sDec[ka], d1 = sDec[ka + 8u];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                st[mt][nt][0] *= d0;
                st[mt][nt][1] *= d0;
                st[mt][nt][2] *= d1;
                st[mt][nt][3] *= d1;
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
                for (int nt = 0; nt < NT; ++nt) {
                    uint32_t b0, b1;
                    k64_frag_b(bDT, LDC, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                    k64_mma(st[mt][nt], ak, b0, b1);
                }
            }
        }
        __syncthreads();
        if (more) load_kd(cbn);

        // 2026-10-12: (C) o += P delta (P zero above the diagonal), then out.
#pragma unroll
        for (int ks = 0; ks < C / 16; ++ks) {
            uint32_t ap[4];
            k64_frag_a(bP, LDC, r0, (unsigned int)ks * 16u, lane, ap);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                uint32_t b0, b1;
                k64_frag_b(bDT, LDC, (unsigned int)nt * 8u, (unsigned int)ks * 16u, lane, b0, b1);
                k64_mma(o[nt], ap, b0, b1);
            }
        }
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
            for (int hr = 0; hr < 2; ++hr) {
                const unsigned int t = (c0 + c) * C + r0 + g + (unsigned int)hr * 8u;
                if (t < T) {
                    float2* op = (float2*)(out + (size_t)t * out_stride + (size_t)h * D + v0 +
                                           (unsigned int)nt * 8u + 2u * tig);
                    *op = make_float2(o[nt][hr * 2], o[nt][hr * 2 + 1]);
                }
            }
        }
        // 2026-10-12: Every warp is past sP and sDT before the next chunk's P lands / S0^T staging.
        __syncthreads();
        if (more) load_p(cbn);
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const unsigned int row = warp * 32u + (unsigned int)mt * 16u + g + (e >= 2 ? 8u : 0u);
                const unsigned int col = v0 + (unsigned int)nt * 8u + 2u * tig + (unsigned int)(e & 1);
                S[(size_t)row * D + col] = st[mt][nt][e];
            }
}
