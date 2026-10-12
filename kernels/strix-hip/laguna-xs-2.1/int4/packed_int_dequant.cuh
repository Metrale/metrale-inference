// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Packed INT4 / INT8 (compressed-tensors pack-quantized, symmetric, group 128)
// decode and the per-lane dot product of the strix-hip W4A16 / W8A16 GEMVs.
//
// Owner: strix-hip kernels (laguna-xs-2.1/int4).
// Invariants:
// - Layout (crates/config precision_plan/packed_int.rs is the SSOT): weight row n is
//   K * BITS / 32 little-endian 32-bit words; word j holds K indices j * (32 / BITS) + i
//   in bits [BITS * i, BITS * (i + 1)), least significant first; a field u is the code
//   q = u - 2^(BITS - 1) (offset binary). One BF16 scale per 128 K values, row-major
//   [N, K / 128]. No zero points.
// - Every function here is plain C++ over integers and floats, compiled both by hipcc and
//   by the host check scripts/laguna/packed_int_gemv_host_check.cpp, so the decode and
//   the lane arithmetic the GPU runs are the ones the host check exercises.
// - Lane arithmetic is fp32 with no fused multiply-add written out: per 32-bit word,
//   inner = sum_i float(q_i) * x_i in order i = 0, 1, ...; then acc += inner * scale.
//   KERNEL.toml builds with --fmad=false (-ffp-contract=off on HIP), so the compiler
//   does not contract these either.
//
// TODO(gfx1151 int8-dot): replace the fp32 inner product with V_DOT4_I32_IU8 over
// codes and a per-128 dynamically quantized INT8 activation (acc += idot * w_scale *
// a_scale). The word-at-a-time split below is the unit that upgrade replaces.

#pragma once

#if defined(__HIPCC__) || defined(__CUDACC__)
#define PI_HD __host__ __device__ __forceinline__
#else
#define PI_HD inline
#endif

// 2026-10-07: K values per scale.
#define PI_GROUP 128

// 2026-10-07: BF16 bits to fp32 (exact).
PI_HD float pi_bf16_to_f32(unsigned short b) {
    union {
        unsigned int u;
        float f;
    } v;
    v.u = ((unsigned int)b) << 16;
    return v.f;
}

// 2026-10-07: fp32 to BF16 bits, round to nearest even; NaN stays a quiet NaN.
PI_HD unsigned short pi_f32_to_bf16_rn(float f) {
    union {
        unsigned int u;
        float f;
    } v;
    v.f = f;
    if ((v.u & 0x7f800000u) == 0x7f800000u && (v.u & 0x007fffffu) != 0u) {
        return (unsigned short)((v.u >> 16) | 0x0040u);
    }
    unsigned int rounding = 0x7fffu + ((v.u >> 16) & 1u);
    return (unsigned short)((v.u + rounding) >> 16);
}

// 2026-10-07: Code i of a packed word: unsigned field i, least significant first, minus
// the offset 2^(BITS - 1).
template <int BITS>
PI_HD int pi_code(unsigned int word, int i) {
    return (int)((word >> (BITS * i)) & ((1u << BITS) - 1u)) - (1 << (BITS - 1));
}

// 2026-10-07: sum_i float(code_i) * x[i] over the 32 / BITS codes of one word, in order.
template <int BITS>
PI_HD float pi_word_dot(unsigned int word, const float* x) {
    float inner = 0.0f;
#pragma unroll
    for (int i = 0; i < 32 / BITS; ++i) {
        float prod = (float)pi_code<BITS>(word, i) * x[i];
        inner = inner + prod;
    }
    return inner;
}

// 2026-10-07: One lane's partial of output row n: the lane reads words lane, lane + lanes,
// lane + 2 * lanes, ... of the row (coalesced across the wave), dots each with its
// CODES activations (BF16, widened exactly) and folds in that word's group scale.
// `row_words` is the row's K * BITS / 32 words, `row_scales` its K / 128 BF16 scales,
// `x` the activation row (K BF16 values). K must be a multiple of 128.
template <int BITS>
PI_HD float pi_row_lane_partial(const unsigned int* row_words, const unsigned short* row_scales,
                                const unsigned short* x, int k, int lane, int lanes) {
    const int codes = 32 / BITS;
    const int words = k / codes;
    float acc = 0.0f;
    for (int w = lane; w < words; w += lanes) {
        float xs[32 / BITS];
#pragma unroll
        for (int i = 0; i < 32 / BITS; ++i) {
            xs[i] = pi_bf16_to_f32(x[w * codes + i]);
        }
        float inner = pi_word_dot<BITS>(row_words[w], xs);
        float scale = pi_bf16_to_f32(row_scales[(w * codes) / PI_GROUP]);
        float term = inner * scale;
        acc = acc + term;
    }
    return acc;
}
