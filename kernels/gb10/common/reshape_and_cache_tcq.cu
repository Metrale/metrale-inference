// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-18: Cache write for the trellis-coded KV codecs turbo3_tcq, turbo2_tcq and turbo1_tcq:
// each 128-group is normalized, rotated, and coded along the minimum-squared-error path of a
// bitshift trellis found by a warp-cooperative Viterbi search (arXiv:2406.11235, QTIP; format,
// codebooks and tie rules after spiritbuun/buun-llama-cpp, MIT, credit spiritbuun; "Closing the
// Gap: Trellis-Coded Quantization for KV Cache at 2-3 Bits", spiritbuun 2026). The output is
// byte-identical to crates/kv-tq128 encode_head (docs/kv-cache-codecs.md).
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: 128 threads (WARPS warps), grid ceil(num_tokens * num_kv_heads * head_dim / 128 /
//   WARPS); one warp codes one (token, head, group). One launch writes one side (K or V) with
//   that side's codebook (2^L floats).
// - head_dim is a positive multiple of 128; a token whose slot is negative writes nothing.
// - The cache row of (slot, head) is head_dim / 128 consecutive blocks at
//   (slot / block_size) * block_stride_bytes + ((slot % block_size) * num_kv_heads + head) *
//   head_dim / 128 * BLOCK.
// - Tie rules (the reference's): a group's best predecessor is the first minimum over
//   p = 0, 1, ..; the final state is the lowest index among the minimum costs.

#include "tq128_common.cuh"

namespace {

constexpr int WARPS = 4;

template <class C>
struct Viterbi {
    static constexpr int STATES = 1 << C::L;
    static constexpr int GROUPS = 1 << (C::L - C::K);
    static constexpr int FAN = 1 << C::K;
    static constexpr int LANE_STATES = STATES / 32;   // lane owns states lane + 32 i
    static constexpr int LANE_GROUPS = GROUPS / 32;   // and groups lane + 32 j
    static constexpr int BACK_WORDS = LANE_GROUPS * C::K;  // ballot words per step
    static constexpr int COST_LEN = STATES + STATES / 32;
    static_assert(GROUPS >= 32, "a lane needs at least one predecessor group");
};

// 2026-10-18: Cost slot of state s, padded by one word per 32 so the predecessor reads
// ((g << k) | p) and the state writes (lane + 32 i) both hit 32 distinct banks.
__device__ __forceinline__ int padded(int s) { return s + (s >> 5); }

template <class C>
struct WarpSmem {
    float x[tq128::GROUP];
    float cost[Viterbi<C>::COST_LEN];
    unsigned back[tq128::GROUP * Viterbi<C>::BACK_WORDS];
    unsigned short states[tq128::GROUP];
};

// 2026-10-18: Viterbi over sm.x (128 rotated values). Back-pointers are packed by ballot: word
// (t, j, b) holds bit b of the chosen predecessor p of group lane + 32 j at step t, one bit per
// lane. Lane 0 backtracks into sm.states.
template <class C>
__device__ void viterbi_warp(WarpSmem<C>& sm, const float* __restrict__ codebook, unsigned lane) {
    using V = Viterbi<C>;
    float cb[V::LANE_STATES];
#pragma unroll
    for (int i = 0; i < V::LANE_STATES; ++i) {
        cb[i] = codebook[lane + 32 * i];
        sm.cost[padded(lane + 32 * i)] = 0.0f;
    }
    __syncwarp();
    for (int t = 0; t < tq128::GROUP; ++t) {
        const float xt = sm.x[t];
        float best[V::LANE_GROUPS];
#pragma unroll
        for (int j = 0; j < V::LANE_GROUPS; ++j) {
            const int base = (int)(lane + 32 * j) << C::K;
            float m = sm.cost[padded(base)];
            unsigned arg = 0;
#pragma unroll
            for (int p = 1; p < V::FAN; ++p) {
                const float c = sm.cost[padded(base | p)];
                if (c < m) {
                    m = c;
                    arg = (unsigned)p;
                }
            }
            best[j] = m;
#pragma unroll
            for (int b = 0; b < C::K; ++b) {
                const unsigned w = __ballot_sync(tq128::FULL, (arg >> b) & 1u);
                if (lane == 0) sm.back[t * V::BACK_WORDS + j * C::K + b] = w;
            }
        }
        __syncwarp();
        // 2026-10-18: (lane + 32 i) & (GROUPS - 1) == lane + 32 (i % LANE_GROUPS): a lane's
        // states draw only on the groups it just reduced.
#pragma unroll
        for (int i = 0; i < V::LANE_STATES; ++i) {
            const float d = __fsub_rn(xt, cb[i]);
            sm.cost[padded(lane + 32 * i)] = __fadd_rn(best[i % V::LANE_GROUPS], __fmul_rn(d, d));
        }
        __syncwarp();
    }
    float bc = sm.cost[padded(lane)];
    int bs = (int)lane;
#pragma unroll
    for (int i = 1; i < V::LANE_STATES; ++i) {
        const float c = sm.cost[padded(lane + 32 * i)];
        if (c < bc) {
            bc = c;
            bs = (int)lane + 32 * i;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float oc = __shfl_xor_sync(tq128::FULL, bc, off);
        const int os = __shfl_xor_sync(tq128::FULL, bs, off);
        if (oc < bc || (oc == bc && os < bs)) {
            bc = oc;
            bs = os;
        }
    }
    if (lane == 0) {
        int s = bs;
        for (int t = tq128::GROUP - 1; t >= 0; --t) {
            sm.states[t] = (unsigned short)s;
            const int g = s & (V::GROUPS - 1);
            const unsigned* w = sm.back + t * V::BACK_WORDS + (g >> 5) * C::K;
            int p = 0;
#pragma unroll
            for (int b = 0; b < C::K; ++b) p |= (int)((w[b] >> (g & 31)) & 1u) << b;
            s = (g << C::K) | p;
        }
    }
    __syncwarp();
}

template <class C>
__device__ void encode_tcq(const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ cache,
                           const long long* __restrict__ slot_mapping,
                           const float* __restrict__ codebook, unsigned num_tokens,
                           unsigned num_kv_heads, unsigned head_dim, unsigned block_size,
                           unsigned src_stride, unsigned long long block_stride_bytes) {
    __shared__ WarpSmem<C> smem[WARPS];
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned groups = head_dim / tq128::GROUP;
    const unsigned long long item = (unsigned long long)blockIdx.x * WARPS + warp;
    if (item >= (unsigned long long)num_tokens * num_kv_heads * groups) return;
    const unsigned g = (unsigned)(item % groups);
    const unsigned head = (unsigned)((item / groups) % num_kv_heads);
    const unsigned token = (unsigned)(item / ((unsigned long long)groups * num_kv_heads));
    const long long slot = slot_mapping[token];
    if (slot < 0) return;
    WarpSmem<C>& sm = smem[warp];

    const __nv_bfloat16* x = src + (unsigned long long)token * src_stride +
                             (unsigned long long)head * head_dim + g * tq128::GROUP + lane * 4;
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(x[i]);
    const float norm = __fsqrt_rn(tq128::group_sum_squares(v));
    const float inv = norm > tq128::TINY ? __fdiv_rn(1.0f, norm) : 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], inv);
    tq128::rotate(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) sm.x[lane * 4 + i] = v[i];
    __syncwarp();

    viterbi_warp<C>(sm, codebook, lane);

    float r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) r[i] = codebook[sm.states[lane * 4 + i]];
    const float recon = __fsqrt_rn(tq128::group_sum_squares(r));
    const float corrected = recon > tq128::TINY ? __fdiv_rn(norm, recon) : norm;

    unsigned char* dst = cache + (unsigned long long)(slot / block_size) * block_stride_bytes +
                         (((unsigned long long)(slot % block_size) * num_kv_heads + head) * groups + g) *
                             C::BLOCK;
    if (lane == 0) tq128::store_norm(dst, corrected);
    // 2026-10-18: Stream bit q is bit (q - k t) of state t for t = min(q / k, 127); past the
    // stream's end the shift reaches L and the bit reads 0, which zero-fills the padding.
    for (int byte = (int)lane; byte < C::BLOCK - 2; byte += 32) {
        unsigned out = 0;
#pragma unroll
        for (int q = 0; q < 8; ++q) {
            const int bit = byte * 8 + q;
            const int t = min(bit / C::K, tq128::GROUP - 1);
            out |= ((unsigned)(sm.states[t] >> (bit - C::K * t)) & 1u) << q;
        }
        dst[2 + byte] = (unsigned char)out;
    }
}

}  // namespace

#define TQ128_RESHAPE_ENTRY(NAME, CODEC)                                                        \
    extern "C" __global__ void __launch_bounds__(WARPS * 32)                                    \
        NAME(const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ cache,          \
             const long long* __restrict__ slot_mapping, const float* __restrict__ codebook,    \
             const unsigned num_tokens, const unsigned num_kv_heads, const unsigned head_dim,   \
             const unsigned block_size, const unsigned src_stride,                              \
             const unsigned long long block_stride_bytes) {                                     \
        encode_tcq<CODEC>(src, cache, slot_mapping, codebook, num_tokens, num_kv_heads,         \
                          head_dim, block_size, src_stride, block_stride_bytes);                \
    }

TQ128_RESHAPE_ENTRY(reshape_and_cache_tcq3, tq128::Tcq3)
TQ128_RESHAPE_ENTRY(reshape_and_cache_tcq2, tq128::Tcq2)
TQ128_RESHAPE_ENTRY(reshape_and_cache_tcq1, tq128::Tcq1)
