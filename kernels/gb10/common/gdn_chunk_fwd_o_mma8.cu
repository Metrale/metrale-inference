// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-28: `gated_delta_rule_chunk_fwd_o_mma8`: a bit-identical twin of `gated_delta_rule_chunk_fwd_o`
// (gated_delta_rule_fla.cu, kernel 3 of the chunked GDN prefill). For token i and value column v:
//   O_i[v] = (exp(gc_i) <q_i, S_c[:, v]> + sum_{l <= i} exp(gc_i - gc_l) <q_i, k_l> uc_l[v]) * rsqrt(k_dim)
// with the original's arithmetic, in its order: kq = Q K^T and o1 = bf16(Q S_c) from the same chain of BF16 m16n8k16
// MMAs per output (head-dim steps of 16 in ascending order from a zero accumulator), the same decay, and for each
// (i, v) the same sequential sum over l = 0..i. What differs: 16-byte cp.async staging into XOR-swizzled shared memory
// (S_c kept in its [k][v] layout and read as MMA B fragments with ldmatrix.trans instead of a strided transposing
// copy), both products on 8 warps instead of 4, and the (i, v) sums spread over all 512 threads instead of 128.
//
// Owner: gb10 kernels.
// Invariants:
// - K_DIM == V_DIM == 128 and CHUNK == 64, as in gated_delta_rule_fla.cu; same arguments, grid (num_chunks,
//   num_v_heads, batch_size), block 512 and dynamic shared memory (98,816 B) as the original.

#include <cuda_bf16.h>

#define K_DIM 128
#define V_DIM 128
#define CHUNK 64

// 2026-09-28: Per-sequence geometry; the definition of GDN_GEOM in gated_delta_rule_fla.cu.
struct GdnGeom { unsigned int seqlen, nchunks, choff; unsigned long long tokoff; };
#define GDN_GEOM(g)                                                            \
    GdnGeom g;                                                                 \
    (void)cu_chunks;                                                          \
    if (is_varlen) {                                                           \
        unsigned int _s0 = (unsigned int)cu_seqlens[b];                       \
        g.seqlen  = (unsigned int)cu_seqlens[b + 1] - _s0;                    \
        g.tokoff  = (unsigned long long)_s0;                                  \
        unsigned int _co = 0;                                                  \
        for (unsigned int _i = 0; _i < b; _i++)                               \
            _co += ((unsigned int)(cu_seqlens[_i + 1] - cu_seqlens[_i])       \
                    + CHUNK - 1) / CHUNK;                                      \
        g.choff   = _co;                                                       \
        g.nchunks = (g.seqlen + CHUNK - 1) / CHUNK;                           \
    } else {                                                                  \
        g.seqlen  = seq_len;                                                   \
        g.tokoff  = (unsigned long long)b * seq_len;                          \
        g.choff   = b * num_chunks;                                            \
        g.nchunks = num_chunks;                                                \
    }

namespace gdnx {
__device__ __forceinline__ unsigned sa(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp16p(void* dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa(dst)), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void ldsm4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void mma16816(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
// 2026-09-28: Row-major [rows][128] BF16 tile with 256-byte rows; 16-byte chunk ch of row r, XOR-swizzled by the row's low 3 bits.
__device__ __forceinline__ unsigned sw(unsigned r, unsigned ch) { return r * 256u + ((ch ^ (r & 7u)) << 4); }

// 2026-09-28: mma_gram's product C[m][n] = sum_k A[m][k] B[n][k] (m < 64, k < 128, n < NT*8) with A and B [rows][128] swizzled; the
// MMA chain per output is mma_gram's (ks ascending from a zero accumulator). 8 warps: warp w takes rows (w & 3) * 16 and
// n-tiles [(w >> 2) * NT/2, +NT/2).
template <int NT, bool B_TRANS>
__device__ __forceinline__ void gram8(const unsigned char* sA, const unsigned char* sB, float (&acc)[NT / 2][4]) {
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u, lmat = lane >> 3, lrow = lane & 7u;
    const unsigned m0 = (warp & 3u) * 16, nt0 = (warp >> 2) * (NT / 2);
    #pragma unroll
    for (int i = 0; i < NT / 2; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    #pragma unroll
    for (unsigned ks = 0; ks < 8; ks++) {
        unsigned a[4];
        ldsm4(sa(sA) + sw(m0 + (lane & 15u), 2 * ks + (lane >> 4)), a[0], a[1], a[2], a[3]);
        #pragma unroll
        for (int j = 0; j < NT / 4; j++) {
            const unsigned n = (nt0 + 2 * j) * 8;
            unsigned b00, b01, b10, b11;
            if (!B_TRANS) {
                // 2026-09-28: B[n][k] row-major: rows n, k contiguous.
                ldsm4(sa(sB) + sw(n + lrow + ((lmat >> 1) << 3), 2 * ks + (lmat & 1u)), b00, b01, b10, b11);
            } else {
                // 2026-09-28: B given as Bt[k][n] row-major (k rows, n contiguous): transposed 8x8 loads.
                ldsm4t(sa(sB) + sw(ks * 16 + lrow + ((lmat & 1u) << 3), (n >> 3) + (lmat >> 1)), b00, b01, b10, b11);
            }
            mma16816(acc[2 * j], a, b00, b01);
            mma16816(acc[2 * j + 1], a, b10, b11);
        }
    }
}
}
// 2026-09-28: Shared memory: sq, sk (each 64 x 256 B), kq (64 x 64 FP32), uc (64 x 256 B),
// S (128 x 256 B, natural [k][v] layout), gc, egc: 16K + 16K + 16K + 16K + 32K + 512 = 98,816 B (same as the original).
extern "C" __global__ void __launch_bounds__(512, 1) gated_delta_rule_chunk_fwd_o_mma8(
    const __nv_bfloat16* __restrict__ query, const __nv_bfloat16* __restrict__ key, const float* __restrict__ gate,
    const float* __restrict__ gc_in, const __nv_bfloat16* __restrict__ S_in, const __nv_bfloat16* __restrict__ uc_in,
    __nv_bfloat16* __restrict__ output, unsigned int batch_size, unsigned int seq_len, unsigned int num_chunks,
    unsigned int num_k_heads, unsigned int num_v_heads, unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride,
    unsigned int gb_stride, const int* __restrict__ cu_seqlens, const int* __restrict__ cu_chunks, unsigned int is_varlen) {
    using namespace gdnx;
    const unsigned int c = blockIdx.x, vh = blockIdx.y, b = blockIdx.z;
    if (vh >= num_v_heads || b >= batch_size) return;
    GDN_GEOM(g);
    if (c >= g.nchunks) return;
    const unsigned tid = threadIdx.x;
    const unsigned kh = vh / (num_v_heads / num_k_heads);
    const float inv_sqrt_d = rsqrtf((float)k_dim);
    const unsigned cs = c * CHUNK;
    const unsigned ce = (g.seqlen - cs) < CHUNK ? (g.seqlen - cs) : CHUNK;
    const unsigned long long base = ((unsigned long long)(g.choff + c) * num_v_heads + vh);
    const unsigned long long out_base = (g.tokoff * num_v_heads + vh) * v_dim;
    query += g.tokoff * qk_stride;
    key += g.tokoff * qk_stride;
    extern __shared__ __align__(128) unsigned char sm[];
    unsigned char* sq = sm;
    unsigned char* sk = sq + CHUNK * 256;
    float* kq = (float*)(sk + CHUNK * 256);
    unsigned char* suc = (unsigned char*)(kq + CHUNK * CHUNK);
    unsigned char* sS = suc + CHUNK * 256;
    float* gc = (float*)(sS + K_DIM * 256);
    float* egc = gc + CHUNK;
    for (unsigned e = tid; e < CHUNK * 16; e += 512) {
        const unsigned i = e >> 4, ch = e & 15u;
        const bool ok = i < ce;
        const unsigned long long off = ok ? (unsigned long long)(cs + i) * qk_stride + kh * k_dim + ch * 8 : 0ull;
        cp16p(sq + sw(i, ch), query + off, ok);
        cp16p(sk + sw(i, ch), key + off, ok);
        cp16p(suc + sw(i, ch), uc_in + base * CHUNK * V_DIM + (ok ? i * v_dim + ch * 8 : 0), ok);
    }
    for (unsigned e = tid; e < K_DIM * 16; e += 512) {
        const unsigned r = e >> 4, ch = e & 15u;
        cp16p(sS + sw(r, ch), S_in + base * K_DIM * V_DIM + r * V_DIM + ch * 8, true);
    }
    asm volatile("cp.async.commit_group;\n" ::);
    for (unsigned i = tid; i < ce; i += 512) {
        const float gv = gc_in[base * CHUNK + i];
        gc[i] = gv;
        egc[i] = expf(gv);
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();

    const unsigned warp = tid >> 5, lane = tid & 31u, grp = lane >> 2, q4 = lane & 3u;
    const unsigned m0 = (warp & 3u) * 16;
    if (warp < 8) {
        float acc[4][4];
        gram8<8, false>(sq, sk, acc);
        const unsigned nt0 = (warp >> 2) * 4;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const unsigned n0 = (nt0 + j) * 8 + q4 * 2;
            kq[(m0 + grp) * CHUNK + n0] = acc[j][0];
            kq[(m0 + grp) * CHUNK + n0 + 1] = acc[j][1];
            kq[(m0 + grp + 8) * CHUNK + n0] = acc[j][2];
            kq[(m0 + grp + 8) * CHUNK + n0 + 1] = acc[j][3];
        }
    }
    __syncthreads();
    for (unsigned p = tid; p < CHUNK * CHUNK; p += 512) {
        const unsigned i = p / CHUNK, l = p % CHUNK;
        if (i < ce && l <= i) kq[p] = expf(gc[i] - gc[l]) * kq[p];
    }
    // 2026-09-28: o1 = bf16(Q S_c): rows 64, n = v 0..127 (16 n-tiles), S given as [k][v]. Written into sk (free after the first gram).
    __syncthreads();
    if (warp < 8) {
        float acc[8][4];
        gram8<16, true>(sq, sS, acc);
        const unsigned nt0 = (warp >> 2) * 8;
        __nv_bfloat16* o1 = (__nv_bfloat16*)sk;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            const unsigned n0 = (nt0 + j) * 8 + q4 * 2;
            o1[(m0 + grp) * V_DIM + n0] = __float2bfloat16(acc[j][0]);
            o1[(m0 + grp) * V_DIM + n0 + 1] = __float2bfloat16(acc[j][1]);
            o1[(m0 + grp + 8) * V_DIM + n0] = __float2bfloat16(acc[j][2]);
            o1[(m0 + grp + 8) * V_DIM + n0 + 1] = __float2bfloat16(acc[j][3]);
        }
    }
    __syncthreads();
    // 2026-09-28: Each (i, v): t2 over l = 0..i in order, exactly the original's per-thread loop body.
    const __nv_bfloat16* o1 = (const __nv_bfloat16*)sk;
    const unsigned v = tid & 127u;
    for (unsigned i = tid >> 7; i < ce; i += 4) {
        const float t1 = egc[i] * (float)o1[i * V_DIM + v];
        float t2 = 0.0f;
        for (unsigned l = 0; l <= i; l++) {
            const __nv_bfloat16 u = *(const __nv_bfloat16*)(suc + sw(l, v >> 3) + (v & 7u) * 2);
            t2 += kq[i * CHUNK + l] * (float)u;
        }
        output[out_base + (unsigned long long)(cs + i) * num_v_heads * v_dim + v] = __float2bfloat16((t1 + t2) * inv_sqrt_d);
    }
}
