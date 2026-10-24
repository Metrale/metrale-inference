// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-18: Paged decode and prefill attention over the 128-group rotated KV codecs
// (turbo3_tcq, turbo2_tcq, turbo1_tcq, turbo4_n128). The kernel rotates the query once per
// 128-group and scores it against the cached keys in the rotated domain, which is exact because
// the rotation is orthogonal (q . k = Rq . Rk; arXiv:2504.19874, TurboQuant). It accumulates
// rotated values and un-rotates only the output row. Elements decode as codebook[index] *
// (fp16(norm) * decode_scale), with trellis states read as two-byte windows (arXiv:2406.11235,
// QTIP). Format after spiritbuun/buun-llama-cpp (MIT, credit spiritbuun);
// docs/kv-cache-codecs.md.
//
// Owner: gb10 kernels.
// Invariants:
// - Decode launch: grid (num_q_heads, num_seqs, 1); prefill launch: grid (num_q_heads,
//   num_tokens, 1). Both 256 threads (NUM_WARPS warps). Each block computes one query row: each
//   warp takes a contiguous chunk of the attended positions and the warps merge through shared
//   memory.
// - Prefill is causal over one sequence: token t attends positions [0, seq_len_start + t], all
//   of them already written to the cache (the chunk's own rows included).
// - head_dim is 128, 256 or 512; any other value returns without writing.
// - num_q_heads is a multiple of num_kv_heads (GQA: kv_head = q_head / (num_q_heads /
//   num_kv_heads)).
// - K and V caches have their own block strides; the row of (position, kv_head) sits at
//   block_table[pos / block_size] * stride + ((pos % block_size) * num_kv_heads + kv_head) *
//   head_dim / 128 * BLOCK.
// - k_book / v_book are the sides' codebooks, k_scale / v_scale their decode factors
//   (crates/kv-tq128 Tq128Codec::decode_scale).
// - Q rows are q_stride elements apart (per sequence for decode, per token for prefill); O is
//   [rows, num_q_heads, head_dim] bf16. A decode sequence with seq_len == 0 returns before any
//   write. There is no sliding window and no sparse-V skip.
// - Softmax uses __expf and plain float sums: the result matches attention over the reference
//   dequant within a tolerance (crates/gpu-examples tq128_microtest), not bit for bit.

#include "tq128_common.cuh"

namespace {

constexpr int NUM_WARPS = 8;
constexpr int MAX_GROUPS = 4;
constexpr int MAX_BOOK = 512;

struct Args {
    const __nv_bfloat16* q;
    const unsigned char* __restrict__ k_cache;
    const unsigned char* __restrict__ v_cache;
    __nv_bfloat16* o;
    const int* block_tables;
    const int* seq_lens;
    const float* k_book;
    const float* v_book;
    unsigned max_blocks_per_seq;
    unsigned seq_len_start;
    unsigned num_q_heads;
    unsigned num_kv_heads;
    unsigned head_dim;
    unsigned block_size;
    float inv_sqrt_d;
    float k_scale;
    float v_scale;
    unsigned q_stride;
    unsigned long long k_block_stride_bytes;
    unsigned long long v_block_stride_bytes;
};

// 2026-10-18: One query row: the head, its attended length, its block table and its row.
struct Row {
    unsigned q_head;
    unsigned seq_len;
    const int* table;
    unsigned long long index;
};

struct Smem {
    float k_book[MAX_BOOK];
    float v_book[MAX_BOOK];
    float m[NUM_WARPS];
    float l[NUM_WARPS];
    float o[NUM_WARPS][MAX_GROUPS * tq128::GROUP];
};

template <class C>
__device__ __forceinline__ const unsigned char* row_ptr(const unsigned char* cache,
                                                        unsigned long long stride, int phys,
                                                        unsigned off, unsigned kv_heads,
                                                        unsigned kv_head, unsigned groups) {
    return cache + (unsigned long long)phys * stride +
           ((unsigned long long)off * kv_heads + kv_head) * groups * C::BLOCK;
}

template <class KC, class VC, int G>
__device__ void attend(const Args& a, Smem& sm, const Row r) {
    const unsigned tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;

    for (unsigned i = tid; i < KC::BOOK; i += blockDim.x) sm.k_book[i] = a.k_book[i];
    for (unsigned i = tid; i < VC::BOOK; i += blockDim.x) sm.v_book[i] = a.v_book[i];
    __syncthreads();

    const unsigned kv_head = r.q_head / (a.num_q_heads / a.num_kv_heads);
    const __nv_bfloat16* qp = a.q + r.index * a.q_stride +
                              (unsigned long long)r.q_head * a.head_dim + lane * 4;
    float q[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g) {
#pragma unroll
        for (int i = 0; i < 4; ++i) q[g][i] = __bfloat162float(qp[g * tq128::GROUP + i]);
        tq128::rotate(q[g], lane);
    }

    const unsigned chunk = (r.seq_len + NUM_WARPS - 1) / NUM_WARPS;
    const unsigned start = min(warp * chunk, r.seq_len);
    const unsigned end = min(start + chunk, r.seq_len);

    float m = -1e30f, l = 0.0f;
    float o[G][4];
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) o[g][i] = 0.0f;

    for (unsigned pos = start; pos < end; ++pos) {
        const int phys = r.table[pos / a.block_size];
        const unsigned off = pos % a.block_size;
        const unsigned char* kr =
            row_ptr<KC>(a.k_cache, a.k_block_stride_bytes, phys, off, a.num_kv_heads, kv_head, G);
        float dot = 0.0f;
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const unsigned char* blk = kr + g * KC::BLOCK;
            unsigned idx[4];
            tq128::lane_indices<KC>(blk, lane, idx);
            float part = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) part += q[g][i] * sm.k_book[idx[i]];
            dot += part * (tq128::load_norm(blk) * a.k_scale);
        }
#pragma unroll
        for (int off_x = 16; off_x > 0; off_x >>= 1) dot += __shfl_xor_sync(tq128::FULL, dot, off_x);
        const float score = dot * a.inv_sqrt_d;
        const float m_new = fmaxf(m, score);
        const float e_old = __expf(m - m_new);
        const float e_new = __expf(score - m_new);
        l = l * e_old + e_new;
        m = m_new;

        const unsigned char* vr =
            row_ptr<VC>(a.v_cache, a.v_block_stride_bytes, phys, off, a.num_kv_heads, kv_head, G);
#pragma unroll
        for (int g = 0; g < G; ++g) {
            const unsigned char* blk = vr + g * VC::BLOCK;
            unsigned idx[4];
            tq128::lane_indices<VC>(blk, lane, idx);
            const float w = e_new * (tq128::load_norm(blk) * a.v_scale);
#pragma unroll
            for (int i = 0; i < 4; ++i) o[g][i] = o[g][i] * e_old + w * sm.v_book[idx[i]];
        }
    }

    if (lane == 0) {
        sm.m[warp] = m;
        sm.l[warp] = l;
    }
#pragma unroll
    for (int g = 0; g < G; ++g)
#pragma unroll
        for (int i = 0; i < 4; ++i) sm.o[warp][g * tq128::GROUP + lane * 4 + i] = o[g][i];
    __syncthreads();
    if (warp != 0) return;

    float m_all = sm.m[0];
#pragma unroll
    for (int w = 1; w < NUM_WARPS; ++w) m_all = fmaxf(m_all, sm.m[w]);
    float l_all = 0.0f, f[NUM_WARPS];
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        f[w] = __expf(sm.m[w] - m_all);
        l_all += sm.l[w] * f[w];
    }
    const float inv_l = l_all > 0.0f ? 1.0f / l_all : 0.0f;
    __nv_bfloat16* op =
        a.o + (r.index * a.num_q_heads + r.q_head) * a.head_dim + lane * 4;
#pragma unroll
    for (int g = 0; g < G; ++g) {
        float res[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            float acc = 0.0f;
#pragma unroll
            for (int w = 0; w < NUM_WARPS; ++w) acc += sm.o[w][g * tq128::GROUP + lane * 4 + i] * f[w];
            res[i] = acc * inv_l;
        }
        tq128::unrotate(res, lane);
#pragma unroll
        for (int i = 0; i < 4; ++i) op[g * tq128::GROUP + i] = __float2bfloat16_rn(res[i]);
    }
}

template <class KC, class VC>
__device__ __forceinline__ void dispatch(const Args& a, const Row r) {
    static_assert(KC::BOOK <= MAX_BOOK && VC::BOOK <= MAX_BOOK, "codebook exceeds shared table");
    __shared__ Smem sm;
    switch (a.head_dim) {
        case 128: attend<KC, VC, 1>(a, sm, r); break;
        case 256: attend<KC, VC, 2>(a, sm, r); break;
        case 512: attend<KC, VC, 4>(a, sm, r); break;
        default: break;
    }
}

// 2026-10-18: Decode: block (q_head, seq) is sequence seq's one new query row.
template <class KC, class VC>
__device__ __forceinline__ void decode(const Args& a) {
    const unsigned q_head = blockIdx.x, seq = blockIdx.y;
    if (q_head >= a.num_q_heads) return;
    const unsigned seq_len = (unsigned)a.seq_lens[seq];
    if (seq_len == 0) return;
    dispatch<KC, VC>(
        a, Row{q_head, seq_len, a.block_tables + (unsigned long long)seq * a.max_blocks_per_seq,
               seq});
}

// 2026-10-18: Prefill: block (q_head, t) is chunk token t, at position seq_len_start + t.
template <class KC, class VC>
__device__ __forceinline__ void prefill(const Args& a) {
    const unsigned q_head = blockIdx.x, t = blockIdx.y;
    if (q_head >= a.num_q_heads) return;
    dispatch<KC, VC>(a, Row{q_head, a.seq_len_start + t + 1, a.block_tables, t});
}

}  // namespace

#define TQ128_DECODE_ENTRY(NAME, KC, VC)                                                         \
    extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32) NAME(                           \
        const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ K_cache,          \
        const unsigned char* __restrict__ V_cache, __nv_bfloat16* __restrict__ O,                \
        const int* __restrict__ block_tables, const int* __restrict__ seq_lens,                  \
        const float* __restrict__ k_book, const float* __restrict__ v_book,                      \
        const unsigned max_blocks_per_seq, const unsigned num_q_heads,                           \
        const unsigned num_kv_heads, const unsigned head_dim, const unsigned block_size,         \
        const float inv_sqrt_d, const float k_scale, const float v_scale,                        \
        const unsigned q_stride, const unsigned long long k_block_stride_bytes,                  \
        const unsigned long long v_block_stride_bytes) {                                         \
        const Args a{Q,           K_cache,      V_cache,    O,          block_tables,            \
                     seq_lens,    k_book,       v_book,     max_blocks_per_seq,                  \
                     0u,          num_q_heads,  num_kv_heads, head_dim, block_size,              \
                     inv_sqrt_d,  k_scale,      v_scale,    q_stride,   k_block_stride_bytes,    \
                     v_block_stride_bytes};                                                      \
        decode<KC, VC>(a);                                                                       \
    }

#define TQ128_PREFILL_ENTRY(NAME, KC, VC)                                                        \
    extern "C" __global__ void __launch_bounds__(NUM_WARPS * 32) NAME(                           \
        const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ K_cache,          \
        const unsigned char* __restrict__ V_cache, __nv_bfloat16* __restrict__ O,                \
        const int* __restrict__ block_table, const float* __restrict__ k_book,                   \
        const float* __restrict__ v_book, const unsigned seq_len_start,                          \
        const unsigned num_q_heads, const unsigned num_kv_heads, const unsigned head_dim,        \
        const unsigned block_size, const float inv_sqrt_d, const float k_scale,                  \
        const float v_scale, const unsigned q_stride,                                            \
        const unsigned long long k_block_stride_bytes,                                           \
        const unsigned long long v_block_stride_bytes) {                                         \
        const Args a{Q,           K_cache,      V_cache,    O,          block_table,             \
                     nullptr,     k_book,       v_book,     0u,                                  \
                     seq_len_start, num_q_heads, num_kv_heads, head_dim, block_size,             \
                     inv_sqrt_d,  k_scale,      v_scale,    q_stride,   k_block_stride_bytes,    \
                     v_block_stride_bytes};                                                      \
        prefill<KC, VC>(a);                                                                      \
    }

#define TQ128_ENTRIES(S, C)                                    \
    TQ128_DECODE_ENTRY(paged_decode_attn_##S, C, C)            \
    TQ128_PREFILL_ENTRY(paged_prefill_attn_##S, C, C)

TQ128_ENTRIES(tcq3, tq128::Tcq3)
TQ128_ENTRIES(tcq2, tq128::Tcq2)
TQ128_ENTRIES(tcq1, tq128::Tcq1)
TQ128_ENTRIES(turbo4_n128, tq128::Turbo4N128)
