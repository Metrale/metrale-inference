// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The GLM-5.3 KDA decode gate chain of ONE token in one launch. Unfused, a KDA layer runs, after the
// projections from the hidden state:
//   w8a8_act_quant_row(f_low) -> w8a8_gemv_rowscale_mb1_ku8(f_b) -> g_raw -> kda_gate_bf16 -> gate
//   kda_sigmoid_bf16_f32(beta_bf16) -> beta
//   w8a8_act_quant_row(g_low) -> w8a8_gemv_rowscale_mb1_ku8(g_b) -> out_gate
// six launches, each a few microseconds of latency for 0.7 MB of weights. Here f_b and g_b are two segments of one
// grid, one 16-row tile per warp (wxay::k128_tile), the activation quantization is each warp's prologue, the gate is
// f_b's epilogue and the beta sigmoid rides on block 0.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to the unfused chain (the build passes --fmad=false on both sides):
//   * quantization: quant_row's amax (a max, so exact in any order), scale fmaxf(amax / 448, 1e-12) and quant8
//     encoder (w8a8_quant.cuh), on the bytes the MMA operand needs;
//   * GEMV: the W8A8 rowscale policy's loads and MMA chunk at K = 128 with mma_gemv's 8-warp partial sum
//     (wxay::k128_tile), then its store expression bf16_rn(r * (w_scale[n] * a_scale));
//   * epilogues: kda_gate_scalar / kda_beta_scalar (kda_gate_scalar.cuh), the functions the unfused kernels call.
//   glm5next_c1_fuse_microtest checks g_raw, gate, out_gate and beta byte for byte against the unfused kernels.
// - One token. K == 128 (the low rank), N % 16 == 0, N == heads * D; f_low, g_low 16-byte aligned. Per-row weight
//   scales (Fp8PerRow). Grid (ceil(2 * N / 16 / 8), 1, 1), block (256, 1, 1), no dynamic shared memory. The
//   activation scratch and scales of the unfused path are not written.

#include "kda_gate_scalar.cuh"
#include "w8a8_policy.cuh"
#include "w8a8_quant.cuh"

namespace kdag {

using F = w8a8::W8A8<false, false>;
using u8 = unsigned char;

// 2026-10-10: The single token's operand for the MMA (lanes with g == 0 hold it, as F::tok / F::load_a give token 0
// to those lanes and zeros to the rest) and its scale. Each lane reads 4 of the 128 values for the amax; lanes g == 0
// read and encode the 32 values of their two 16-byte operand runs, k in [16t, 16t + 16) and [64 + 16t, 64 + 16t + 16).
__device__ __forceinline__ F::AReg quant_a(const __nv_bfloat16* __restrict__ x, const wxay::Lane& L, float& s) {
    const uint2 raw = *reinterpret_cast<const uint2*>(x + L.lane * 4u);
    float m = 0.f;
    {
        const uint32_t w[2] = {raw.x, raw.y};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m = fmaxf(m, fabsf(__uint_as_float(w[i] << 16)));
            m = fmaxf(m, fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    s = fmaxf(m / w8a8q::E4M3_MAX, 1e-12f);

    F::AReg b;
    b.b0 = make_uint4(0u, 0u, 0u, 0u);
    b.b1 = make_uint4(0u, 0u, 0u, 0u);
    b.s0 = 0.0f;
    b.s1 = 0.0f;
    if (L.g == 0u) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x);
        const uint2 q0 = w8a8q::quant8(x4[2u * L.t], s);
        const uint2 q1 = w8a8q::quant8(x4[2u * L.t + 1u], s);
        const uint2 q2 = w8a8q::quant8(x4[8u + 2u * L.t], s);
        const uint2 q3 = w8a8q::quant8(x4[8u + 2u * L.t + 1u], s);
        b.b0 = make_uint4(q0.x, q0.y, q1.x, q1.y);
        b.b1 = make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
    return b;
}

}  // namespace kdag

extern "C" __global__ __launch_bounds__(256) void kda_gates_w8a8_k128(
    const __nv_bfloat16* __restrict__ f_low, const __nv_bfloat16* __restrict__ g_low,
    const unsigned char* __restrict__ Wf, const float* __restrict__ Sf, const unsigned char* __restrict__ Wg,
    const float* __restrict__ Sg, __nv_bfloat16* __restrict__ g_raw, __nv_bfloat16* __restrict__ out_gate,
    const float* __restrict__ dt_bias, const float* __restrict__ A_log, float* __restrict__ gate,
    const __nv_bfloat16* __restrict__ beta_bf16, float* __restrict__ beta, unsigned N, unsigned D, unsigned heads,
    float lower_bound) {
    using kdag::F;
    if (blockIdx.x == 0u) {
        for (unsigned i = threadIdx.x; i < heads; i += blockDim.x) beta[i] = kda_beta_scalar(__bfloat162float(beta_bf16[i]));
    }
    const wxay::Lane L = wxay::lane_ids();
    const unsigned tiles = N / 16u;
    const unsigned wt = blockIdx.x * wxay::WARPS + L.warp;
    if (wt >= 2u * tiles) return;
    const bool g_seg = wt >= tiles;
    const unsigned n0 = (g_seg ? wt - tiles : wt) * 16u;

    float s;
    const F::AReg b = kdag::quant_a(g_seg ? g_low : f_low, L, s);
    const float* S = g_seg ? Sg : Sf;
    const w8a8::Args a = {nullptr, nullptr, {g_seg ? Wg : Wf, nullptr, nullptr}, {S, nullptr, nullptr}, N, N, 128u, 0u};
    const F::Tile tile = F::tile(a, n0, L, N, 128u);
    float r[4];
    wxay::k128_tile<F>(tile, b, L, r);

    // 2026-10-10: F::store at token 0: lanes with t == 0 hold the token's two outputs, rows g and g + 8.
    if (L.t != 0u) return;
#pragma unroll
    for (int c = 0; c < 4; c += 2) {
        const unsigned n = n0 + L.g + (c == 0 ? 0u : 8u);
        if (n >= N) continue;
        const __nv_bfloat16 o = __float2bfloat16_rn(r[c] * (__ldg(S + n) * s));
        if (g_seg) {
            out_gate[n] = o;
        } else {
            g_raw[n] = o;
            gate[n] = kda_gate_scalar(__bfloat162float(o), dt_bias[n], expf(A_log[n / D]), lower_bound);
        }
    }
}
