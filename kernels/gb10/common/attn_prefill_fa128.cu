// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-27: `attn_prefill_fa128_paged` and `attn_prefill_fa128`: causal prefill flash attention for HDIM 256 over a
// BF16 K/V (paged cache, or contiguous buffers), bit-identical twins of `attn_prefill_paged_64`
// (prefill_paged_compute.cuh through attn_prefill_paged.cu) and `attn_prefill_64` (attn_prefill.cu).
//
// Per query row the arithmetic is the original's, in the same order: the same 32-key blocks in ascending order; S from
// sixteen chained bf16 m16n8k16 MMAs over the head dim in ascending order; S * inv_sqrt_d; the same -1e30 masks; the
// row max over the block by fmaxf and the quad butterfly; the same rescale of l and O (conditional on a new max in the
// paged twin, unconditional in the contiguous one, as in each original); p = __expf(s - m); the per-thread sum over the
// four n-tiles then the quad butterfly; P rounded to FP16 (paged: with V converted BF16 -> FP16 by the same
// round-to-nearest) or kept BF16 (contiguous); O += P V as two k16 MMAs per n-tile in ascending key order; O / l in BF16.
// A key block that is fully masked for a row adds exactly nothing to it (p = 0, l += +0, O += +0), so a row may see
// more masked blocks than in the original, or skip them, without changing a bit.
//
// What differs is the organization: one CTA of 8 warps takes 128 query rows of one head (16 rows per warp), every warp
// computes both QK^T and P V for its rows over all 256 head dims, P stays in registers (the S accumulator layout is the
// P V A-fragment layout), K/V tiles are read once per 128 query rows, fragments come from XOR-swizzled shared memory
// through ldmatrix, and a warp whose rows are all before a key block skips it. Q blocks run in reverse so the longest
// (causal) tiles start first.
//
// Owner: gb10 kernels.
// Invariants:
// - HDIM 256 and BC 32 only; causal masking on and no sliding window (the host falls back otherwise).
// - Block 256 threads; grid (num_q_heads, ceil(q_len / 128), batch); dynamic shared memory FA128_SMEM (96 KiB).

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace fa128 {

constexpr int HD = 256;
constexpr int BC = 32;
constexpr int BR = 128;
constexpr int THREADS = 256;
constexpr int ROW_BYTES = HD * 2;
constexpr int ROW_CHUNKS = ROW_BYTES / 16;
constexpr int NT = HD / 8;

__device__ __forceinline__ unsigned smem_addr(const void* p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(unsigned dst, const void* src, bool pred) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::); }

// 2026-09-27: Byte offset of 16-byte chunk `ch` of row `row` in a [rows][256] BF16 tile; the XOR with the row's low 3
// bits makes the 8 rows of each ldmatrix phase hit 8 distinct bank groups.
__device__ __forceinline__ unsigned swz(unsigned row, unsigned ch) {
    return row * ROW_BYTES + ((ch ^ (row & 7u)) << 4);
}

__device__ __forceinline__ void ldsm_x4(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(unsigned a, unsigned& d0, unsigned& d1, unsigned& d2, unsigned& d3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}

__device__ __forceinline__ void mma_bf16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
__device__ __forceinline__ void mma_f16(float* c, const unsigned* a, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                 : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

__device__ __forceinline__ unsigned pack_f16(float lo, float hi) {
    __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    const unsigned l = __bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned h = __bfloat16_as_ushort(__float2bfloat16(hi));
    return l | (h << 16);
}

// 2026-09-27: Where K/V row `pos` of kv head `kvh` lives: the paged cache through its block table, or a contiguous
// [seq, num_kv_heads, HD] buffer.
struct PagedKv {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const int* block_table;
    unsigned block_size, num_kv_heads;
    __device__ __forceinline__ unsigned long long row(unsigned pos, unsigned kvh) const {
        const unsigned long long page = (unsigned long long)(unsigned)block_table[pos / block_size];
        return (page * block_size + pos % block_size) * num_kv_heads * HD + (unsigned long long)kvh * HD;
    }
};
struct ContigKv {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    unsigned num_kv_heads;
    __device__ __forceinline__ unsigned long long row(unsigned pos, unsigned kvh) const {
        return ((unsigned long long)pos * num_kv_heads + kvh) * HD;
    }
};

// 2026-09-27: Thread t copies 16-byte chunk (t % 32) of tile rows t / 32 + 8 i, i = 0..3, so each thread needs the
// element offsets of 4 rows per key block. ROW_NONE marks a row at or past kv_len (zero-filled).
constexpr unsigned long long ROW_NONE = ~0ull;

template <typename Kv>
__device__ __forceinline__ void row_offsets(const Kv& kv, unsigned start, unsigned kv_len, unsigned kvh,
                                            unsigned long long (&off)[4]) {
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned pos = start + (threadIdx.x >> 5) + 8 * i;
        off[i] = pos < kv_len ? kv.row(pos, kvh) : ROW_NONE;
    }
}

__device__ __forceinline__ void load_kv(unsigned char* dst, const __nv_bfloat16* src,
                                        const unsigned long long (&off)[4]) {
    const unsigned ch = threadIdx.x & 31u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const unsigned r = (threadIdx.x >> 5) + 8 * i;
        const bool ok = off[i] != ROW_NONE;
        cp16(smem_addr(dst + swz(r, ch)), src + (ok ? off[i] : 0ull) + ch * 8, ok);
    }
}

// 2026-09-27: The shared body. FP16_PV selects the paged original's arithmetic (FP16 P, V converted, rescale only on a
// new max); otherwise the contiguous original's (BF16 P and V, unconditional rescale).
template <bool FP16_PV, typename Kv>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ O, const Kv& kv,
                                     unsigned q_len, unsigned kv_len, unsigned q_offset, unsigned num_q_heads,
                                     float inv_sqrt_d) {
    extern __shared__ __align__(128) unsigned char smem[];
    unsigned char* sQ = smem;
    unsigned char* sK = sQ + BR * ROW_BYTES;
    unsigned char* sV = sK + BC * ROW_BYTES;

    const unsigned q_head = blockIdx.x;
    const unsigned q_start = (gridDim.y - 1 - blockIdx.y) * BR;
    if (q_head >= num_q_heads || q_start >= q_len) return;
    const unsigned q_tile_end = min(q_start + BR, q_len);
    const unsigned q_seq_stride = num_q_heads * HD;
    const unsigned kvh = q_head / (num_q_heads / kv.num_kv_heads);
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const unsigned gid = lane >> 2, tig = lane & 3u, lmat = lane >> 3, lrow = lane & 7u;
    const unsigned wr0 = warp * 16;
    const unsigned row0 = wr0 + gid, row1 = row0 + 8;
    // 2026-09-27: The warp's last query row by absolute position; key blocks past it are skipped by this warp.
    const unsigned warp_last_pos = q_offset + q_start + wr0 + 15;
    const bool warp_live = q_start + wr0 < q_len;

    unsigned num_kv_blocks = (kv_len + BC - 1) / BC;
    num_kv_blocks = min(num_kv_blocks, (q_offset + q_tile_end - 1) / BC + 1);

    #pragma unroll
    for (unsigned c = threadIdx.x; c < (unsigned)(BR * ROW_CHUNKS); c += THREADS) {
        const unsigned r = c / ROW_CHUNKS, ch = c % ROW_CHUNKS;
        const bool ok = q_start + r < q_len;
        const __nv_bfloat16* g = Q + (ok ? (unsigned long long)(q_start + r) * q_seq_stride + q_head * HD : 0ull) + ch * 8;
        cp16(smem_addr(sQ + swz(r, ch)), g, ok);
    }
    unsigned long long off_cur[4], off_nxt[4];
    row_offsets(kv, 0, kv_len, kvh, off_cur);
    if (num_kv_blocks > 0) load_kv(sK, kv.k, off_cur);
    cp_commit();
    cp_wait_all();
    __syncthreads();

    float acc_o[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) { acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f; acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f; }
    float m_r0 = -1e30f, m_r1 = -1e30f, l_r0 = 0.0f, l_r1 = 0.0f;
    const unsigned q_base = smem_addr(sQ), k_base = smem_addr(sK), v_base = smem_addr(sV);

    for (unsigned kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const unsigned kv_start = kv_block * BC;
        const unsigned kv_tile_len = min(kv_start + BC, kv_len) - kv_start;
        load_kv(sV, kv.v, off_cur);
        cp_commit();
        // 2026-09-27: The next block's row offsets, computed now so their block-table reads overlap QK^T.
        if (kv_block + 1 < num_kv_blocks) row_offsets(kv, kv_start + BC, kv_len, kvh, off_nxt);

        const bool active = warp_live && kv_start <= warp_last_pos;
        unsigned pa[2][4];
        if (active) {
            float acc_s[4][4];
            #pragma unroll
            for (int i = 0; i < 4; i++) { acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f; acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f; }
            #pragma unroll
            for (unsigned ks = 0; ks < HD / 16; ks++) {
                unsigned a[4];
                ldsm_x4(q_base + swz(wr0 + (lane & 15u), 2 * ks + (lane >> 4)), a[0], a[1], a[2], a[3]);
                #pragma unroll
                for (int j = 0; j < 2; j++) {
                    unsigned b00, b01, b10, b11;
                    const unsigned key = j * 16 + lrow + ((lmat >> 1) << 3);
                    ldsm_x4(k_base + swz(key, 2 * ks + (lmat & 1u)), b00, b01, b10, b11);
                    mma_bf16(acc_s[2 * j], a, b00, b01);
                    mma_bf16(acc_s[2 * j + 1], a, b10, b11);
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                const unsigned c0 = nt * 8 + tig * 2, c1 = c0 + 1;
                const unsigned qr0 = q_offset + q_start + row0, qr1 = q_offset + q_start + row1;
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
                if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (q_start + row0 >= q_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (q_start + row1 >= q_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }
            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));
            const float mn0 = fmaxf(m_r0, rmax0), mn1 = fmaxf(m_r1, rmax1);
            if (FP16_PV) {
                if (mn0 != m_r0) {
                    const float eo0 = __expf(m_r0 - mn0);
                    l_r0 *= eo0;
                    #pragma unroll
                    for (int i = 0; i < NT; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                    m_r0 = mn0;
                }
                if (mn1 != m_r1) {
                    const float eo1 = __expf(m_r1 - mn1);
                    l_r1 *= eo1;
                    #pragma unroll
                    for (int i = 0; i < NT; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                    m_r1 = mn1;
                }
            } else {
                const float eo0 = __expf(m_r0 - mn0), eo1 = __expf(m_r1 - mn1);
                l_r0 *= eo0; l_r1 *= eo1;
                #pragma unroll
                for (int i = 0; i < NT; i++) {
                    acc_o[i][0] *= eo0; acc_o[i][1] *= eo0;
                    acc_o[i][2] *= eo1; acc_o[i][3] *= eo1;
                }
                m_r0 = mn0; m_r1 = mn1;
            }
            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                const float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                const float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                // 2026-09-27: A fragment of the k16 step nt / 2: a0/a1 hold key n-tile 2ks (rows gid, gid + 8),
                // a2/a3 key n-tile 2ks + 1.
                const int ks = nt >> 1, hi = (nt & 1) * 2;
                pa[ks][hi + 0] = FP16_PV ? pack_f16(p00, p01) : pack_bf16(p00, p01);
                pa[ks][hi + 1] = FP16_PV ? pack_f16(p10, p11) : pack_bf16(p10, p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1); sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1); sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0; l_r1 += sum1;
        }

        cp_wait_all();
        __syncthreads();  // 2026-09-27: V landed; every warp is done with sK
        if (kv_block + 1 < num_kv_blocks) {
            load_kv(sK, kv.k, off_nxt);
            cp_commit();
        }
        if (FP16_PV) {
            // 2026-09-27: V to FP16 in place, each value as bf16 -> float -> half (round to nearest), as the paged
            // original converts it per MMA.
            #pragma unroll
            for (unsigned c = threadIdx.x; c < (unsigned)(BC * ROW_CHUNKS); c += THREADS) {
                uint4* p = (uint4*)(sV + c * 16);
                uint4 v = *p;
                unsigned* w = (unsigned*)&v;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    const __nv_bfloat162 b = *reinterpret_cast<const __nv_bfloat162*>(&w[i]);
                    w[i] = pack_f16(__bfloat162float(b.x), __bfloat162float(b.y));
                }
                *p = v;
            }
            __syncthreads();
        }
        if (active) {
            #pragma unroll
            for (int ks = 0; ks < 2; ks++) {
                const unsigned key = ks * 16 + lrow + ((lmat & 1u) << 3);
                #pragma unroll
                for (int nt = 0; nt < NT; nt += 2) {
                    unsigned b00, b01, b10, b11;
                    ldsm_x4_t(v_base + swz(key, nt + (lmat >> 1)), b00, b01, b10, b11);
                    if (FP16_PV) {
                        mma_f16(acc_o[nt], pa[ks], b00, b01);
                        mma_f16(acc_o[nt + 1], pa[ks], b10, b11);
                    } else {
                        mma_bf16(acc_o[nt], pa[ks], b00, b01);
                        mma_bf16(acc_o[nt + 1], pa[ks], b10, b11);
                    }
                }
            }
        }
        cp_wait_all();
        __syncthreads();  // 2026-09-27: next K landed; every warp is done with sV
        #pragma unroll
        for (int i = 0; i < 4; i++) off_cur[i] = off_nxt[i];
    }

    const float il0 = (l_r0 > 0) ? (1.f / l_r0) : 0, il1 = (l_r1 > 0) ? (1.f / l_r1) : 0;
    __nv_bfloat16* ob = O + q_head * HD;
    #pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        const unsigned c0 = nt * 8 + tig * 2;
        const unsigned gr0 = q_start + row0, gr1 = q_start + row1;
        if (gr0 < q_len) {
            const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * il0));
            const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * il0));
            *(unsigned*)&ob[(unsigned long long)gr0 * q_seq_stride + c0] = lo | (hi << 16);
        }
        if (gr1 < q_len) {
            const unsigned lo = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * il1));
            const unsigned hi = __bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * il1));
            *(unsigned*)&ob[(unsigned long long)gr1 * q_seq_stride + c0] = lo | (hi << 16);
        }
    }
}

}  // namespace fa128

// 2026-09-27: Twin of attn_prefill_paged_64 at HDIM 256 with causal masking and no sliding window.
extern "C" __global__ void __launch_bounds__(256, 1) attn_prefill_fa128_paged(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K_cache,
    const __nv_bfloat16* __restrict__ V_cache,
    __nv_bfloat16* __restrict__ O,
    const int* __restrict__ block_table,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int q_offset,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int cache_block_size,
    const float inv_sqrt_d
) {
    const fa128::PagedKv kv{K_cache, V_cache, block_table, cache_block_size, num_kv_heads};
    fa128::body<true>(Q, O, kv, q_len, kv_len, q_offset, num_q_heads, inv_sqrt_d);
}

// 2026-09-27: Twin of attn_prefill_64 at HDIM 256 with causal masking and no sliding window; grid z is the batch.
extern "C" __global__ void __launch_bounds__(256, 1) attn_prefill_fa128(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const float inv_sqrt_d
) {
    const unsigned long long b = blockIdx.z;
    const unsigned long long q_off = b * seq_len * num_q_heads * fa128::HD;
    const unsigned long long kv_off = b * seq_len * num_kv_heads * fa128::HD;
    const fa128::ContigKv kv{K + kv_off, V + kv_off, num_kv_heads};
    fa128::body<false>(Q + q_off, O + q_off, kv, seq_len, seq_len, 0, num_q_heads, inv_sqrt_d);
}
