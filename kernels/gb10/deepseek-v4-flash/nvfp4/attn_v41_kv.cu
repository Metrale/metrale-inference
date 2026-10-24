// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

// 2026-10-19: DeepSeek-V4.1 Flash attention caches in the layout the checkpoint's inference code declares
// (kv-codec registry rows mxfp8, nvfp4 and mxfp4; model-arch attn_v41/kv_codec.rs), and the kernels that write and read them.
//
//   window row  (every layer)     hd e4m3 codes, then hd/32 e8m0 scales          (act_quant)
//   latent row  (kv sources)      hd/2 bytes of e2m1 pairs, then hd/16 e4m3 scales (fp4_act_quant, 16, e4m3)
//   index row   (kv sources)      ihd/2 bytes of e2m1 pairs, then ihd/32 e8m0 scales (fp4_act_quant, 32)
//
// Two e2m1 codes share a byte, the lower-indexed value in the low nibble. The reference keeps these rows as
// bf16 after an in-place quantise-dequantise, so each stored value is `code * scale`, a product that is exact
// in f32; the readers decode that product where they used to load the bf16, in the same order, so their
// results are the bf16 storage's bit for bit.
//
// Owner: gb10 kernels (deepseek-v4-flash); launched by model-arch `attn_v41`.
// Invariants: the group sizes here (32 and 16) and the row layouts are the ones kv_codec.rs declares; the
// launcher refuses any other plan.
//
// Kernels:
//   attn_v41_kv_pack_fp8        act_quant of bf16 rows into packed window rows
//   attn_v41_kv_pack_fp4        fp4_act_quant of bf16 rows into packed latent (mode 1) or index (mode 0) rows
//   attn_v41_index_score_packed the indexer's rectified, weighted head scores over packed index rows
//   attn_v41_sparse_attn_packed softmax over gathered packed window and latent rows plus the sink
//   attn_v41_ring_put_row       one packed row into the window ring at pos % win, pos read from device memory

#include "attn_v41_common.cuh"

#define AV_WINDOW_GROUP 32
#define AV_LATENT_GROUP 16
#define AV_INDEX_GROUP 32

// 2026-10-19: act_quant (blocks of 32, scale pow2_ceil(amax / 448), amax >= 1e-4) of bf16 `x`, written as
// window rows of `row_values` values into `out`. Grid: ceil(n_blocks / 8). Block: 256, one warp per block.
extern "C" __global__ void attn_v41_kv_pack_fp8(
    const __nv_bfloat16* __restrict__ x, const unsigned int n_blocks,
    unsigned char* __restrict__ out, const unsigned int row_values) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    const float v = __bfloat162float(x[(size_t)gb * AV_WINDOW_GROUP + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float s = av_fp8_scale(amax);
    const unsigned int gpr = row_values / AV_WINDOW_GROUP;
    unsigned char* row = out + (size_t)(gb / gpr) * (row_values + gpr);
    const unsigned int g = gb % gpr;
    row[g * AV_WINDOW_GROUP + lane] = (unsigned char)av_e4m3_code(fminf(fmaxf(v / s, -448.0f), 448.0f));
    if (lane == 0) row[row_values + g] = (unsigned char)av_e8m0_code(s);
}

// 2026-10-19: fp4_act_quant (`block` values a scale; mode 0 = e8m0 scale, mode 1 = e4m3 scale) of bf16 `x`,
// written as rows of `row_values` values into `out`. Even lanes write a byte of two codes, the odd lane's
// code from a shuffle. Grid: ceil(n_blocks / 8). Block: 256, one warp per block.
extern "C" __global__ void attn_v41_kv_pack_fp4(
    const __nv_bfloat16* __restrict__ x, const unsigned int n_blocks, const unsigned int block,
    const unsigned int mode, unsigned char* __restrict__ out, const unsigned int row_values) {
    const unsigned int gb = blockIdx.x * 8 + threadIdx.x / 32;
    const unsigned int lane = threadIdx.x % 32;
    if (gb >= n_blocks) return;
    const __nv_bfloat16* p = x + (size_t)gb * block;
    float amax = 0.0f;
    for (unsigned int i = lane; i < block; i += 32) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float s = av_fp4_scale(amax, mode);
    const unsigned int gpr = row_values / block;
    unsigned char* row = out + (size_t)(gb / gpr) * (row_values / 2 + gpr);
    const unsigned int g = gb % gpr;
    for (unsigned int base = 0; base < block; base += 32) {
        const unsigned int i = base + lane;
        unsigned int code = 0;
        if (i < block) code = av_e2m1_code(fminf(fmaxf(__bfloat162float(p[i]) / s, -6.0f), 6.0f));
        const unsigned int odd = __shfl_down_sync(0xffffffffu, code, 1);
        if (i < block && (i & 1u) == 0) row[(g * block + i) / 2] = (unsigned char)(code | (odd << 4));
    }
    if (lane == 0) row[row_values / 2 + g] = (unsigned char)(mode == 0 ? av_e8m0_code(s) : av_e4m3_code(s));
}

// 2026-10-19: Value d of a packed index row of `ihd` values.
__device__ __forceinline__ float av_index_key(const unsigned char* __restrict__ kr, unsigned int d, unsigned int ihd) {
    return av_e2m1_value(kr[d / 2] >> (4 * (d & 1u))) * av_e8m0_value(kr[ihd / 2 + d / AV_INDEX_GROUP]);
}

__device__ __forceinline__ float av_bf16_at(const uint4& v, const int j) {
    const unsigned int word = j < 2 ? v.x : j < 4 ? v.y : j < 6 ? v.z : v.w;
    const unsigned short bits = (j & 1) ? (unsigned short)(word >> 16) : (unsigned short)(word & 0xFFFFu);
    __nv_bfloat16 b;
    *(unsigned short*)&b = bits;
    return __bfloat162float(b);
}

__device__ __forceinline__ unsigned int av_bf16_bits(float v) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v));
}

// 2026-10-19: One (t, p) score with the key row decoded once into registers as bf16 (exact: e2m1 times a
// power of two), then the products, order and roundings of the generic loop.
template <int IHD>
__device__ __forceinline__ float av_index_score_row(
    const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ kr,
    const __nv_bfloat16* __restrict__ w, const unsigned int nh) {
    const unsigned int* kw = (const unsigned int*)kr;
    const unsigned int sw = kw[IHD / 8];
    uint4 kk[IHD / 8];
#pragma unroll
    for (int i = 0; i < IHD / 8; ++i) {
        const unsigned int word = kw[i];
        const float s = av_e8m0_value((sw >> (8 * ((8 * i) / AV_INDEX_GROUP))) & 0xFFu);
        unsigned int b[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) b[j] = av_bf16_bits(av_e2m1_value(word >> (4 * j)) * s);
        kk[i] = make_uint4(b[0] | (b[1] << 16), b[2] | (b[3] << 16), b[4] | (b[5] << 16), b[6] | (b[7] << 16));
    }
    float acc = 0.0f;
    for (unsigned int h = 0; h < nh; ++h) {
        const uint4* q4 = (const uint4*)(q + (size_t)h * IHD);
        float dot = 0.0f;
#pragma unroll
        for (int i = 0; i < IHD / 8; ++i) {
            const uint4 qq = q4[i];
#pragma unroll
            for (int j = 0; j < 8; ++j) dot += av_bf16_at(qq, j) * av_bf16_at(kk[i], j);
        }
        dot = av_bf16r(dot);
        acc += av_bf16r(fmaxf(dot, 0.0f) * __bfloat162float(w[h]));
    }
    return acc;
}

// 2026-10-19: Indexer scores: score[t, p] = bf16(sum_h bf16(relu(bf16(q[t,h] . k[p])) * w[t,h])).
// q: [T, nh, ihd] bf16, k: [width] packed index rows, w: [T, nh] bf16 (already scaled), out: [T, width] f32.
// Grid: (ceil(width / 256), T). Block: 256, one (t, p) per thread. With `ihd` 128 a thread keeps its key row
// in registers; measured 2026-09-19 with nsys on GB10, the generic loop over bf16 keys took 122 us a launch.
extern "C" __global__ void attn_v41_index_score_packed(
    const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ k,
    const __nv_bfloat16* __restrict__ w, float* __restrict__ out,
    const unsigned int width, const unsigned int nh, const unsigned int ihd) {
    const unsigned int t = blockIdx.y;
    const unsigned int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= width) return;
    const unsigned char* kr = k + (size_t)p * (ihd / 2 + ihd / AV_INDEX_GROUP);
    const __nv_bfloat16* qt = q + (size_t)t * nh * ihd;
    const __nv_bfloat16* wt = w + (size_t)t * nh;
    float acc;
    if (ihd == 128) {
        acc = av_index_score_row<128>(qt, kr, wt, nh);
    } else {
        acc = 0.0f;
        for (unsigned int h = 0; h < nh; ++h) {
            const __nv_bfloat16* qv = qt + (size_t)h * ihd;
            float dot = 0.0f;
            for (unsigned int d = 0; d < ihd; ++d) dot += __bfloat162float(qv[d]) * av_index_key(kr, d, ihd);
            dot = av_bf16r(dot);
            acc += av_bf16r(fmaxf(dot, 0.0f) * __bfloat162float(wt[h]));
        }
    }
    out[(size_t)t * width + p] = av_bf16r(acc);
}

// 2026-10-19: Value d of a packed window row (`latent` false) or latent row (`latent` true) of `hd` values.
__device__ __forceinline__ float av_cache_value(
    const unsigned char* __restrict__ kr, const bool latent, const unsigned int d, const unsigned int hd) {
    if (latent) {
        return av_e2m1_value(kr[d / 2] >> (4 * (d & 1u))) * av_e4m3_value(kr[hd / 2 + d / AV_LATENT_GROUP]);
    }
    return av_e4m3_value(kr[d]) * av_e8m0_value(kr[hd + d / AV_WINDOW_GROUP]);
}

// 2026-10-19: Sparse attention with the sink over packed rows.
// q: [T, nh, hd] bf16. Index i < split reads window row rows_a[i], else latent row rows_b[i - split].
// idx: [T, topk] i32, -1 = absent. sink: [nh] f32. o: [T, nh, hd] bf16.
// Grid: (T, nh, S). Block: 256. Scores in shared memory (topk <= AV_MAX_TOPK).
// S (METRALE_DS41_SPARSE_SPLIT, 1 to 8, default 2) splits the output dimension hd across S blocks per
// (token, head): every block recomputes the scores and the softmax (the same values in the same thread
// layout) and writes its own hd / S outputs, each by the same j-ordered sum as one block would, so the bytes
// are the same for any S; only the number of blocks in flight changes.
#define AV_MAX_TOPK 2048
extern "C" __global__ void attn_v41_sparse_attn_packed(
    const __nv_bfloat16* __restrict__ q, const unsigned char* __restrict__ rows_a,
    const unsigned char* __restrict__ rows_b, const unsigned int split,
    const int* __restrict__ idx, const float* __restrict__ sink,
    __nv_bfloat16* __restrict__ o, const unsigned int nh, const unsigned int hd,
    const unsigned int topk, const float scale) {
    const unsigned int t = blockIdx.x;
    const unsigned int h = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid % 32;
    const unsigned int warp = tid / 32;
    const unsigned int nwarps = blockDim.x / 32;
    const size_t ra = hd + hd / AV_WINDOW_GROUP;
    const size_t rb = hd / 2 + hd / AV_LATENT_GROUP;
    __shared__ float sc[AV_MAX_TOPK];
    __shared__ float red[AV_BLOCK];
    const __nv_bfloat16* qv = q + ((size_t)t * nh + h) * hd;
    const int* ids = idx + (size_t)t * topk;

    // 2026-10-19: One warp per candidate row: the dot product over hd.
    for (unsigned int j = warp; j < topk; j += nwarps) {
        const int i = ids[j];
        float s = -CUDART_INF_F;
        if (i >= 0) {
            const bool latent = (unsigned int)i >= split;
            const unsigned char* kr = latent ? rows_b + ((unsigned int)i - split) * rb : rows_a + (unsigned int)i * ra;
            float acc = 0.0f;
            for (unsigned int d = lane; d < hd; d += 32) acc += __bfloat162float(qv[d]) * av_cache_value(kr, latent, d, hd);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
            s = acc * scale;
        }
        if (lane == 0) sc[j] = s;
    }
    __syncthreads();
    // 2026-10-19: Softmax over the scores plus the sink, which enters the denominator only.
    float m = sink[h];
    for (unsigned int j = tid; j < topk; j += blockDim.x) m = fmaxf(m, sc[j]);
    m = av_block_max(m, red);
    float den = 0.0f;
    for (unsigned int j = tid; j < topk; j += blockDim.x) {
        const float e = sc[j] == -CUDART_INF_F ? 0.0f : expf(sc[j] - m);
        sc[j] = e;
        den += e;
    }
    den = av_block_sum(den, red) + expf(sink[h] - m);
    __syncthreads();
    // 2026-10-19: o[d] = sum_j p_j * row_j[d] over this block's slice of d.
    __nv_bfloat16* ov = o + ((size_t)t * nh + h) * hd;
    const unsigned int d0 = blockIdx.z * hd / gridDim.z;
    const unsigned int d1 = (blockIdx.z + 1) * hd / gridDim.z;
    for (unsigned int d = d0 + tid; d < d1; d += blockDim.x) {
        float acc = 0.0f;
        for (unsigned int j = 0; j < topk; ++j) {
            const int i = ids[j];
            if (i < 0) continue;
            const bool latent = (unsigned int)i >= split;
            const unsigned char* kr = latent ? rows_b + ((unsigned int)i - split) * rb : rows_a + (unsigned int)i * ra;
            acc += (sc[j] / den) * av_cache_value(kr, latent, d, hd);
        }
        ov[d] = __float2bfloat16(acc);
    }
}

// 2026-10-19: The window ring write of the graph-captured decode step: ring row (pos[0] % win) = src, `row_bytes`
// bytes. The eager step does this as a device copy at a host-computed slot; a captured
// graph bakes its pointers, so the slot has to come from device memory. Bit-identical to the copy.
// Grid: (1). Block: 256.
extern "C" __global__ void attn_v41_ring_put_row(
    const unsigned char* __restrict__ src, unsigned char* __restrict__ ring,
    const int* __restrict__ pos, const unsigned int win, const unsigned int row_bytes) {
    const unsigned int slot = ((unsigned int)pos[0]) % win;
    unsigned char* dst = ring + (size_t)slot * row_bytes;
    for (unsigned int c = threadIdx.x; c < row_bytes; c += blockDim.x) dst[c] = src[c];
}
