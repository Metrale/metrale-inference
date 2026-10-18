// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-18: Dequantize cache rows of the 128-group rotated KV codecs (turbo3_tcq, turbo2_tcq,
// turbo1_tcq, turbo4_n128) to bf16 in the input domain: y = codebook[index] * (fp16(norm) *
// decode_scale), then the inverse rotation, then round to nearest-even bf16. For prefill over a
// cached prefix. Format after spiritbuun/buun-llama-cpp (MIT, credit spiritbuun); rotation per
// arXiv:2504.19874. Byte-identical to crates/kv-tq128 dequant_head_bf16.
//
// Owner: gb10 kernels.
// Invariants:
// - Launch: 128 threads (WARPS warps), grid ceil(num_rows * num_kv_heads * head_dim / 128 /
//   WARPS); one warp decodes one (row, head, group).
// - Row r reads cache slot slots[r] and writes out + r * out_stride + head * head_dim; a negative
//   slot writes nothing.
// - codebook is the side's book (2^L floats, 16 for turbo4_n128) and decode_scale the side's
//   factor (crates/kv-tq128 Tq128Codec::decode_scale).
// - Cache layout as reshape_and_cache_tcq.cu.

#include "tq128_common.cuh"

namespace {

constexpr int WARPS = 4;

template <class C>
__device__ void dequant(const unsigned char* __restrict__ cache, __nv_bfloat16* __restrict__ out,
                        const long long* __restrict__ slots, const float* __restrict__ codebook,
                        unsigned num_rows, unsigned num_kv_heads, unsigned head_dim,
                        unsigned block_size, unsigned out_stride,
                        unsigned long long block_stride_bytes, float decode_scale) {
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const unsigned groups = head_dim / tq128::GROUP;
    const unsigned long long item = (unsigned long long)blockIdx.x * WARPS + warp;
    if (item >= (unsigned long long)num_rows * num_kv_heads * groups) return;
    const unsigned g = (unsigned)(item % groups);
    const unsigned head = (unsigned)((item / groups) % num_kv_heads);
    const unsigned row = (unsigned)(item / ((unsigned long long)groups * num_kv_heads));
    const long long slot = slots[row];
    if (slot < 0) return;

    const unsigned char* block =
        cache + (unsigned long long)(slot / block_size) * block_stride_bytes +
        (((unsigned long long)(slot % block_size) * num_kv_heads + head) * groups + g) * C::BLOCK;
    const float scale = __fmul_rn(tq128::load_norm(block), decode_scale);
    unsigned idx[4];
    tq128::lane_indices<C>(block, lane, idx);
    float v[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(codebook[idx[i]], scale);
    tq128::unrotate(v, lane);

    __nv_bfloat16* dst = out + (unsigned long long)row * out_stride +
                         (unsigned long long)head * head_dim + g * tq128::GROUP + lane * 4;
#pragma unroll
    for (int i = 0; i < 4; ++i) dst[i] = __float2bfloat16_rn(v[i]);
}

}  // namespace

#define TQ128_DEQUANT_ENTRY(NAME, CODEC)                                                        \
    extern "C" __global__ void __launch_bounds__(WARPS * 32)                                    \
        NAME(const unsigned char* __restrict__ cache, __nv_bfloat16* __restrict__ out,          \
             const long long* __restrict__ slots, const float* __restrict__ codebook,           \
             const unsigned num_rows, const unsigned num_kv_heads, const unsigned head_dim,     \
             const unsigned block_size, const unsigned out_stride,                              \
             const unsigned long long block_stride_bytes, const float decode_scale) {           \
        dequant<CODEC>(cache, out, slots, codebook, num_rows, num_kv_heads, head_dim,           \
                       block_size, out_stride, block_stride_bytes, decode_scale);               \
    }

TQ128_DEQUANT_ENTRY(tq128_dequant_bf16_tcq3, tq128::Tcq3)
TQ128_DEQUANT_ENTRY(tq128_dequant_bf16_tcq2, tq128::Tcq2)
TQ128_DEQUANT_ENTRY(tq128_dequant_bf16_tcq1, tq128::Tcq1)
TQ128_DEQUANT_ENTRY(tq128_dequant_bf16_turbo4_n128, tq128::Turbo4N128)
