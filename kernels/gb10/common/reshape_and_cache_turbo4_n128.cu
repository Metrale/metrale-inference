// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-18: Cache write for turbo4_n128 (4.125 bits per value): each 128-group is normalized,
// rotated (arXiv:2504.19874, TurboQuant), and each coordinate takes the nearest of 16 levels;
// one fp16 corrected norm per group. Codebook and format after spiritbuun/buun-llama-cpp's
// turbo4 (MIT, credit spiritbuun). Byte-identical to crates/kv-tq128 encode_head
// (docs/kv-cache-codecs.md). This is not the engine's 16-group turbo4.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: 128 threads (WARPS warps), grid ceil(num_tokens * num_kv_heads * head_dim / 128 /
//   WARPS); one warp codes one (token, head, group).
// - levels holds 16 floats and mids the 15 decision midpoints between them; a value on a
//   midpoint takes the upper level.
// - head_dim is a positive multiple of 128; a token whose slot is negative writes nothing.
// - Row layout as reshape_and_cache_tcq.cu, with 66-byte blocks.

#include "tq128_common.cuh"

namespace {
constexpr int WARPS = 4;
}

extern "C" __global__ void __launch_bounds__(WARPS * 32) reshape_and_cache_turbo4_n128(
    const __nv_bfloat16* __restrict__ src, unsigned char* __restrict__ cache,
    const long long* __restrict__ slot_mapping, const float* __restrict__ levels,
    const float* __restrict__ mids, const unsigned num_tokens, const unsigned num_kv_heads,
    const unsigned head_dim, const unsigned block_size, const unsigned src_stride,
    const unsigned long long block_stride_bytes) {
    using C = tq128::Turbo4N128;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned groups = head_dim / tq128::GROUP;
    const unsigned long long item = (unsigned long long)blockIdx.x * WARPS + warp;
    if (item >= (unsigned long long)num_tokens * num_kv_heads * groups) return;
    const unsigned g = (unsigned)(item % groups);
    const unsigned head = (unsigned)((item / groups) % num_kv_heads);
    const unsigned token = (unsigned)(item / ((unsigned long long)groups * num_kv_heads));
    const long long slot = slot_mapping[token];
    if (slot < 0) return;

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

    unsigned idx[4];
    float r[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned n = 0;
        for (int m = 0; m < C::BOOK - 1; ++m) n += v[i] >= mids[m] ? 1u : 0u;
        idx[i] = n;
        r[i] = levels[n];
    }
    const float recon = __fsqrt_rn(tq128::group_sum_squares(r));
    const float corrected = recon > tq128::TINY ? __fdiv_rn(norm, recon) : norm;

    unsigned char* dst = cache + (unsigned long long)(slot / block_size) * block_stride_bytes +
                         (((unsigned long long)(slot % block_size) * num_kv_heads + head) * groups + g) *
                             C::BLOCK;
    if (lane == 0) tq128::store_norm(dst, corrected);
    dst[2 + lane * 2] = (unsigned char)(idx[0] | (idx[1] << 4));
    dst[2 + lane * 2 + 1] = (unsigned char)(idx[2] | (idx[3] << 4));
}
