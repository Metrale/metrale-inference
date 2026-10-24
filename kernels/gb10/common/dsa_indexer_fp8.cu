// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-16: The DSA indexer over an FP8 cache (`--indexer-cache-dtype fp8`): each complete
// pool of KP tokens is stored once, as its compressed key rotated by the unnormalized 128-point
// Walsh-Hadamard transform H and quantized to E4M3 with one power-of-two scale; the tokens of a
// pool still being filled sit in a small BF16 ring. The store fuses the compression and the
// quantization; the compression and the fused scores read a pool's key back as
// H(dequant(q) * scale) / 128 (H * H = 128 I).
//
// Owner: gb10 kernels.
// Invariants:
// - Block layout (one DSA layer's V block of bs tokens, E = bs / KP pools, S ring sets), from the
//   block's start: [E][D] E4M3 rotated keys, [E] f32 scales, [S][KP-1][D] BF16 ring keys,
//   [S][KP-1][D] BF16 ring gates. Pool p = x / KP of token x is entry (x % bs) / KP of block
//   bt[x / bs]; token x with x % KP < KP-1 has ring slot (set (x / KP) % S, x % KP) of that
//   block. glm5next_dsa/fp8_rows.rs carries the same offsets (Fp8IndexerLayout).
// - D is 128 and every launch has blockDim.x == 128 for the store and the compression (one
//   thread per channel). first_key is 0 and the cache is paged (valid NULL).
// - A pool's key is computed by dsa_pool_key_rows, the expressions dsa_kpool_compress uses, so
//   the value quantized is the value the BF16 cache's compression stores for that pool.
// - The scale is 2^k, k = ceil(log2(amax / 448)) clamped to [-126, 127], from the f32 bits of
//   amax / 448; the quantization is a saturating round-to-nearest-even E4M3 conversion of
//   y / scale. Every step but __expf has a bit-exact host twin (fp8_twin.rs): the kernels build
//   with --fmad=false, so f32 expression order fixes the result.
// - The store's rows are segments of `seg_rows` consecutive rows of one sequence at consecutive
//   positions (pos[r] = pos[seg0] + r - seg0); row r's block table is bt + r * bt_stride. With
//   seg_rows 1 every row is its own sequence and one launch (phase 3) is race-free; with
//   seg_rows > 1 the host launches phase 1 (pool entries) then phase 2 (ring rows), so a pool
//   entry reads ring rows written before this call and staged rows of this call, never a ring
//   row this call writes.

#include "dsa_indexer_pools.cuh"
#include <cuda_fp8.h>

#define DSA_FP8_D 128u

// 2026-10-16: In-place unnormalized Walsh-Hadamard transform of v[0, 128) in shared memory.
// The caller has synchronised after writing v; it returns synchronised.
__device__ __forceinline__ void dsa_fwht128(float* v, unsigned int tid) {
    for (unsigned int h = 1u; h < DSA_FP8_D; h <<= 1) {
        for (unsigned int i = tid; i < DSA_FP8_D / 2u; i += blockDim.x) {
            const unsigned int a = (i / h) * 2u * h + i % h;
            const float x = v[a];
            const float y = v[a + h];
            v[a] = x + y;
            v[a + h] = x - y;
        }
        __syncthreads();
    }
}

// 2026-10-16: The power-of-two scale that maps amax into E4M3's finite range: 2^k with
// k = ceil(log2(amax / 448)), read from the f32 bits of amax / 448 (exponent, plus one when any
// mantissa bit is set), clamped to [-126, 127]. amax 0 gives 2^-126.
__device__ __forceinline__ float dsa_ue8m0_scale(float amax) {
    const float r = amax / 448.0f;
    const unsigned int bits = __float_as_uint(r);
    int k = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
    k = k < -126 ? -126 : (k > 127 ? 127 : k);
    return __uint_as_float((unsigned int)(k + 127) << 23);
}

// 2026-10-16: E4M3 byte to f32 (no NaN: the store saturates). Exponent 0 is subnormal, m * 2^-9.
__device__ __forceinline__ float dsa_e4m3_to_f32(unsigned char b) {
    const unsigned int e = (b >> 3) & 0xfu;
    const unsigned int m = b & 0x7u;
    const float mag = e == 0u ? (float)m * 0.001953125f
                              : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -mag : mag;
}

// 2026-10-16: Byte offsets inside one FP8 block (see the layout invariant).
struct DsaFp8Block {
    unsigned int D, KP, S, bs;
    __device__ __forceinline__ unsigned int entries() const { return bs / KP; }
    __device__ __forceinline__ unsigned int entry(unsigned int x) const {
        return ((x % bs) / KP) * D;
    }
    __device__ __forceinline__ unsigned int scale(unsigned int x) const {
        return entries() * D + 4u * ((x % bs) / KP);
    }
    __device__ __forceinline__ unsigned int ring_key(unsigned int x) const {
        return entries() * (D + 4u) + (((x / KP) % S) * (KP - 1u) + x % KP) * D * 2u;
    }
    __device__ __forceinline__ unsigned int ring_gate(unsigned int x) const {
        return ring_key(x) + S * (KP - 1u) * D * 2u;
    }
};

// 2026-10-16: Slot rows of the pool a store row completes: rows at or after the segment's first
// position are this launch's staged rows, earlier ones the block's ring.
struct DsaFp8WriteRows {
    const __nv_bfloat16* __restrict__ stage_k;
    const __nv_bfloat16* __restrict__ stage_gate;
    const unsigned char* __restrict__ blk;
    DsaFp8Block g;
    long long x0;
    unsigned int seg0;
    __device__ __forceinline__ const __nv_bfloat16* row(long long raw, bool gate) const {
        if (raw >= x0) {
            const size_t r = (size_t)seg0 + (size_t)(raw - x0);
            return (gate ? stage_gate : stage_k) + r * g.D;
        }
        const unsigned int off = gate ? g.ring_gate((unsigned int)raw) : g.ring_key((unsigned int)raw);
        return (const __nv_bfloat16*)(blk + off);
    }
    __device__ __forceinline__ __nv_bfloat16 gate_at(long long raw, unsigned int d) const {
        return row(raw, true)[d];
    }
    __device__ __forceinline__ __nv_bfloat16 key_at(long long raw, unsigned int d) const {
        return row(raw, false)[d];
    }
};

// 2026-10-16: grid (1, rows), block 128. Row r places staged row r (token pos[r]): phase bit 1
// quantizes the pool a row with pos % KP == KP-1 completes into its entry and scale; phase bit 2
// writes a row with pos % KP < KP-1 into its ring slot, unless row r + S * KP of the same
// segment writes the same slot of the same block (the last writer stays, without a race).
// `smem` is static: 128 floats for the rotation, 4 for the amax reduction.
extern "C" __global__ void dsa_indexer_store_fp8(
    const __nv_bfloat16* __restrict__ stage_k,
    const __nv_bfloat16* __restrict__ stage_gate,
    const int* __restrict__ pos,
    unsigned char* __restrict__ v_pool,
    const float* __restrict__ ape,
    unsigned int KP,
    unsigned int S,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int bs,
    unsigned int blk_bytes,
    unsigned int seg_rows,
    unsigned int rows,
    unsigned int phase
) {
    __shared__ float xs[DSA_FP8_D];
    __shared__ float red[DSA_FP8_D / 32u];
    const unsigned int r = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int seg0 = (r / seg_rows) * seg_rows;
    const unsigned int seg_last = min(seg0 + seg_rows, rows) - 1u;
    const unsigned int x = (unsigned int)pos[r];
    const unsigned int j = x % KP;
    const DsaFp8Block g{DSA_FP8_D, KP, S, bs};
    unsigned char* blk = v_pool + (size_t)bt[(size_t)r * bt_stride + x / bs] * blk_bytes;

    if (j == KP - 1u) {
        if (!(phase & 1u)) return;
        const DsaFp8WriteRows src{stage_k, stage_gate, blk, g, (long long)pos[seg0], seg0};
        xs[tid] = dsa_pool_key_rows(src, nullptr, ape, x + 1u, DSA_FP8_D, KP, 0, x / KP, tid);
        __syncthreads();
        dsa_fwht128(xs, tid);
        const float y = xs[tid];
        float m = fabsf(y);
        for (int off = 16; off > 0; off >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, off));
        if ((tid & 31u) == 0u) red[tid >> 5] = m;
        __syncthreads();
        float amax = red[0];
        for (unsigned int w = 1u; w < DSA_FP8_D / 32u; ++w) amax = fmaxf(amax, red[w]);
        const float sc = dsa_ue8m0_scale(amax);
        blk[g.entry(x) + tid] =
            (unsigned char)__nv_cvt_float_to_fp8(y / sc, __NV_SATFINITE, __NV_E4M3);
        if (tid == 0u) *(float*)(blk + g.scale(x)) = sc;
        return;
    }
    if (!(phase & 2u)) return;
    const unsigned int later = x + S * KP;
    if (r + S * KP <= seg_last && later / bs == x / bs) return;
    const size_t src_row = (size_t)r * DSA_FP8_D;
    ((__nv_bfloat16*)(blk + g.ring_key(x)))[tid] = stage_k[src_row + tid];
    ((__nv_bfloat16*)(blk + g.ring_gate(x)))[tid] = stage_gate[src_row + tid];
}

// 2026-10-16: Pool keys read back from an FP8 cache: entry (x % bs) / KP of block bt[x / bs],
// x = p * KP, dequantized, scaled, rotated back and divided by 128 in `sm` (128 shared floats;
// `out` may be `sm`). A pool that is not complete below S has no entry yet: its key is 0 (the
// bodies never score or select it: it is not a valid pool).
struct DsaFp8Pools {
    const unsigned char* __restrict__ v_pool;
    DsaFp8Block g;
    unsigned int blk_bytes;
    float* sm;
    __device__ __forceinline__ void key_block(unsigned int p, unsigned int S, const int* bt,
                                              float* out, unsigned int base,
                                              unsigned int tid) const {
        const unsigned int x = p * g.KP;
        if (x + g.KP > S) {
            for (unsigned int d = tid; d < g.D; d += blockDim.x) out[base + d] = 0.0f;
            return;
        }
        const unsigned char* blk = v_pool + (size_t)bt[x / g.bs] * blk_bytes;
        const float sc = *(const float*)(blk + g.scale(x));
        for (unsigned int d = tid; d < g.D; d += blockDim.x)
            sm[d] = dsa_e4m3_to_f32(blk[g.entry(x) + d]) * sc;
        __syncthreads();
        dsa_fwht128(sm, tid);
        for (unsigned int d = tid; d < g.D; d += blockDim.x)
            out[base + d] = sm[d] * (1.0f / 128.0f);
        // 2026-10-16: The next pool rewrites `sm`.
        __syncthreads();
    }
};

// 2026-10-16: dsa_kpool_compress over an FP8 cache: grid as dsa_kpool_compress, block 128.
// The token ids and validity are dsa_kpool_compress's (valid NULL); the keys are DsaFp8Pools'.
extern "C" __global__ void dsa_kpool_compress_fp8(
    const unsigned char* __restrict__ v_pool,
    float* __restrict__ pool_keys,
    int* __restrict__ pool_indices,
    unsigned char* __restrict__ pool_valid,
    unsigned int S,
    unsigned int KP,
    unsigned int S_sets,
    const int* __restrict__ geom,
    const int* __restrict__ bt,
    unsigned int bs,
    unsigned int blk_bytes
) {
    __shared__ float sm[DSA_FP8_D];
    const DsaFp8Pools src{v_pool, DsaFp8Block{DSA_FP8_D, KP, S_sets, bs}, blk_bytes, sm};
    dsa_kpool_compress_body(src, nullptr, pool_keys, pool_indices, pool_valid, S, DSA_FP8_D, KP,
                            0, geom, bt);
}

// 2026-10-16: dsa_pool_scores_rows over an FP8 cache: grid, block and dynamic shared memory
// ((D + H) floats) as dsa_pool_scores_rows. The pool key is rotated back in the `key` buffer.
extern "C" __global__ void dsa_pool_scores_rows_fp8(
    const unsigned char* __restrict__ v_pool,
    const float* __restrict__ q,
    const float* __restrict__ weights,
    const int* __restrict__ q_pos,
    float* __restrict__ out,
    unsigned char* __restrict__ valid_cand,
    unsigned int H,
    unsigned int KP,
    unsigned int S_sets,
    float scale,
    const int* __restrict__ geom,
    const int* __restrict__ bt,
    unsigned int bt_stride,
    unsigned int bs,
    unsigned int blk_bytes,
    unsigned int sc_stride
) {
    extern __shared__ float sh_fp8[];
    const DsaFp8Pools src{v_pool, DsaFp8Block{DSA_FP8_D, KP, S_sets, bs}, blk_bytes, sh_fp8};
    dsa_pool_scores_rows_body(src, q, weights, q_pos, out, valid_cand, H, DSA_FP8_D, KP, 0, scale,
                              geom, bt, bt_stride, sc_stride, sh_fp8, sh_fp8 + DSA_FP8_D);
}
