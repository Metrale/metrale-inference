// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-18: Device helpers shared by the 128-group rotated KV codecs (turbo3_tcq, turbo2_tcq,
// turbo1_tcq, turbo4_n128): block geometry, the warp-level randomized Hadamard rotation, the
// group sum of squares, and index extraction. Format after spiritbuun/buun-llama-cpp (MIT;
// credit spiritbuun, https://github.com/spiritbuun/buun-llama-cpp); rotation and norm per
// arXiv:2504.19874 (TurboQuant); trellis indexing per arXiv:2406.11235 (QTIP). The bit-exact
// reference is crates/kv-tq128; docs/kv-cache-codecs.md describes the format.
//
// Owner: gb10 kernels.
// Invariants:
// - One warp holds one 128-group: lane l owns elements 4l..4l+3.
// - Every operation the encoder and dequant results depend on uses an explicit round-to-nearest
//   intrinsic (__fadd_rn, __fmul_rn, __fsqrt_rn, __fdiv_rn), so their bytes do not depend on
//   contraction flags. The order of each sum is the reference's (crates/kv-tq128 numeric.rs).
// - Callers pass codebooks as arguments; this header holds no codebook values.

#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tq_plus_signs.cuh"

namespace tq128 {

constexpr unsigned FULL = 0xffffffffu;
constexpr int GROUP = 128;
constexpr float TINY = 1e-10f;

// 2026-10-18: Block geometry per codec. A block is a little-endian fp16 norm, then the payload,
// zero-padded to an even length. TRELLIS codecs store a k-bits-per-step bitshift stream over
// 2^L states; the scalar codec stores 4-bit levels, even element in the low nibble.
template <int K_, int L_>
struct Trellis {
    static constexpr bool TRELLIS = true;
    static constexpr int K = K_;
    static constexpr int L = L_;
    static constexpr int BOOK = 1 << L_;
    static constexpr int PAYLOAD = ((L_ - K_) + K_ * GROUP + 7) / 8;
    static constexpr int BLOCK = (2 + PAYLOAD + 1) & ~1;
};
using Tcq3 = Trellis<3, 9>;
using Tcq2 = Trellis<2, 8>;
using Tcq1 = Trellis<1, 8>;

struct Turbo4N128 {
    static constexpr bool TRELLIS = false;
    static constexpr int BOOK = 16;
    static constexpr int PAYLOAD = GROUP / 2;
    static constexpr int BLOCK = 2 + PAYLOAD;
};

static_assert(Tcq3::BLOCK == 52 && Tcq2::BLOCK == 36 && Tcq1::BLOCK == 20, "tcq block sizes");
static_assert(Turbo4N128::BLOCK == 66, "turbo4_n128 block size");

__device__ __forceinline__ float inv_sqrt_128() { return __int_as_float(0x3DB504F3); }

// 2026-10-18: v += shfl_xor(v, off) for off = 16, 8, 4, 2, 1; every lane ends with the same sum.
__device__ __forceinline__ float warp_xor_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = __fadd_rn(v, __shfl_xor_sync(FULL, v, off));
    return v;
}

// 2026-10-18: Sum of squares of the warp's group: ((a + b) + c) + d in lane, then the butterfly.
__device__ __forceinline__ float group_sum_squares(const float v[4]) {
    const float s = __fadd_rn(
        __fadd_rn(__fadd_rn(__fmul_rn(v[0], v[0]), __fmul_rn(v[1], v[1])), __fmul_rn(v[2], v[2])),
        __fmul_rn(v[3], v[3]));
    return warp_xor_sum(s);
}

// 2026-10-18: The 128-point Walsh-Hadamard transform, butterflies h = 1, 2, .., 64 producing
// (a + b, a - b). h = 1 and 2 pair elements inside a lane; h >= 4 pairs lane l with l ^ (h / 4).
__device__ __forceinline__ void fwht128(float v[4], unsigned lane) {
    float a = v[0], b = v[1];
    v[0] = __fadd_rn(a, b);
    v[1] = __fsub_rn(a, b);
    a = v[2];
    b = v[3];
    v[2] = __fadd_rn(a, b);
    v[3] = __fsub_rn(a, b);
    a = v[0];
    b = v[2];
    v[0] = __fadd_rn(a, b);
    v[2] = __fsub_rn(a, b);
    a = v[1];
    b = v[3];
    v[1] = __fadd_rn(a, b);
    v[3] = __fsub_rn(a, b);
#pragma unroll
    for (unsigned m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float o = __shfl_xor_sync(FULL, v[i], m);
            v[i] = upper ? __fsub_rn(o, v[i]) : __fadd_rn(v[i], o);
        }
    }
}

// 2026-10-18: x * first, WHT, x * (1/sqrt(128) * last). rotate = (SIGNS1, SIGNS2); unrotate
// swaps the tables. The seed-42 sign tables are the engine's TQP_SIGNS{1,2}_128.
__device__ __forceinline__ void apply_rotation(float v[4], unsigned lane, const float* first,
                                               const float* last) {
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], first[lane * 4 + i]);
    fwht128(v, lane);
#pragma unroll
    for (int i = 0; i < 4; ++i) v[i] = __fmul_rn(v[i], __fmul_rn(inv_sqrt_128(), last[lane * 4 + i]));
}

__device__ __forceinline__ void rotate(float v[4], unsigned lane) {
    apply_rotation(v, lane, tq_plus::TQP_SIGNS1_128, tq_plus::TQP_SIGNS2_128);
}

__device__ __forceinline__ void unrotate(float v[4], unsigned lane) {
    apply_rotation(v, lane, tq_plus::TQP_SIGNS2_128, tq_plus::TQP_SIGNS1_128);
}

__device__ __forceinline__ float load_norm(const unsigned char* block) {
    const unsigned short h = (unsigned short)(block[0] | (block[1] << 8));
    return __half2float(__ushort_as_half(h));
}

// 2026-10-18: The fp16 bits are widened to 32 bits inside the asm. With CUDA 13.0 ptxas,
// `block[0] = (unsigned char)(__half_as_ushort(__float2half_rn(x)) & 0xFF)` compiles to
// F2I.U8.F16.TRUNC: it stores the truncated VALUE (11.68 -> 0x0b), not the low byte of the bits.
// A byte store whose source is a 32-bit integer register is not folded that way.
__device__ __forceinline__ void store_norm(unsigned char* block, float norm) {
    unsigned bits;
    asm("{ .reg .b16 h; cvt.rn.f16.f32 h, %1; cvt.u32.u16 %0, h; }" : "=r"(bits) : "f"(norm));
    block[0] = (unsigned char)(bits & 0xFFu);
    block[1] = (unsigned char)(bits >> 8);
}

// 2026-10-18: Trellis state t: the L-bit window at stream bit k * t (a two-byte read).
template <class C>
__device__ __forceinline__ unsigned state_at(const unsigned char* payload, unsigned t) {
    const unsigned bit = C::K * t;
    const unsigned w = payload[bit >> 3] | ((unsigned)payload[(bit >> 3) + 1] << 8);
    return (w >> (bit & 7)) & ((1u << C::L) - 1);
}

// 2026-10-18: The codebook indices of this lane's four elements of the block's group.
template <class C>
__device__ __forceinline__ void lane_indices(const unsigned char* block, unsigned lane,
                                             unsigned idx[4]) {
    const unsigned char* payload = block + 2;
    if constexpr (C::TRELLIS) {
#pragma unroll
        for (int i = 0; i < 4; ++i) idx[i] = state_at<C>(payload, lane * 4 + i);
    } else {
        const unsigned char b0 = payload[lane * 2], b1 = payload[lane * 2 + 1];
        idx[0] = b0 & 0xF;
        idx[1] = b0 >> 4;
        idx[2] = b1 & 0xF;
        idx[3] = b1 >> 4;
    }
}

}  // namespace tq128
