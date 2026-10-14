// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-10: The GLM-5.3 KDA decode gate chain of ONE token in one launch. Unfused, a KDA layer runs, after the
// projections from the hidden state:
//   w8a8_act_quant_row(f_low) -> w8a8_gemv_rowscale_mb1_ku8(f_b) -> g_raw -> kda_gate_bf16 -> gate
//   kda_sigmoid_bf16_f32(beta_bf16) -> beta
//   w8a8_act_quant_row(g_low) -> w8a8_gemv_rowscale_mb1_ku8(g_b) -> out_gate
// six launches, each a few microseconds of latency for 0.7 MB of weights. Here f_b and g_b are two segments of one
// grid, one 16-row tile per warp (wxay::k128_tile), the activation quantization is each warp's prologue, the gate is
// f_b's epilogue and the beta sigmoid rides on block 0.
// 2026-10-14: M tokens (an MTP verify group: 2 rows at C1, 4 at C2, ...). Token tile blockIdx.y covers tokens
// [8y, 8y + 8), the MMA's n8 side; at K = 128 the unfused W8A8 GEMV (every MB entry) runs each output on warp 0 of
// a CTA with seven idle warps, which held it at 33 GB/s for 2816 x 128 bytes per call.
//
// Owner: gb10 kernels.
// Invariants:
// - Every output is byte-identical to the unfused chain at the same M (the build passes --fmad=false on both sides):
//   * quantization: quant_row's amax (a max, so exact in any order), scale fmaxf(amax / 448, 1e-12) and quant8
//     encoder (w8a8_quant.cuh), per token, on the bytes the MMA operand needs;
//   * GEMV: the W8A8 rowscale policy's loads and MMA chunk at K = 128 with mma_gemv's 8-warp partial sum
//     (wxay::k128_tile), then its store expression bf16_rn(r * (w_scale[n] * a_scale[m])); an MMA column is one
//     token, so a token's bits do not depend on the others in its tile (mma_gemv's row invariance);
//   * epilogues: kda_gate_scalar / kda_beta_scalar (kda_gate_scalar.cuh), the functions the unfused kernels call.
//   glm5next_c1_fuse_microtest checks g_raw, gate, out_gate and beta byte for byte against the unfused kernels at
//   M = 1..=16, 24 and 32.
// - K == 128 (the low rank), N % 16 == 0, N == heads * D; f_low, g_low [M, 128] contiguous BF16, 16-byte aligned;
//   g_raw, out_gate, gate [M, N]; beta_bf16, beta [M, heads]. Per-row weight scales (Fp8PerRow). Grid
//   (ceil(2 * N / 16 / 8), ceil(M / 8), 1), block (256, 1, 1), no dynamic shared memory. The activation scratch and
//   scales of the unfused path are not written.

#include "kda_gate_scalar.cuh"
#include "w8a8_policy.cuh"
#include "w8a8_quant.cuh"

namespace kdag {

using F = w8a8::W8A8<false, false>;
using u8 = unsigned char;

// 2026-10-14: The operand of token tile `j0` for the MMA and the scale of this lane's token j0 + g. Lane (g, t)
// reads its token's two 16-value operand runs, k in [16t, 16t + 16) and [64 + 16t, 64 + 16t + 16); the four lanes
// of a quad together hold all 128 values, so the token's amax is a two-step xor over t. A token at or past M gives
// a zero operand (as F::load_a does) and an unused scale.
__device__ __forceinline__ F::AReg quant_a(const __nv_bfloat16* __restrict__ x, unsigned j0, unsigned M,
                                           const wxay::Lane& L, float& s) {
    const unsigned tk = j0 + L.g;
    const bool live = tk < M;
    uint4 v[4] = {make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u), make_uint4(0u, 0u, 0u, 0u),
                  make_uint4(0u, 0u, 0u, 0u)};
    if (live) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (unsigned long long)tk * 128u);
        v[0] = x4[2u * L.t];
        v[1] = x4[2u * L.t + 1u];
        v[2] = x4[8u + 2u * L.t];
        v[3] = x4[8u + 2u * L.t + 1u];
    }
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float f[8];
        w8a8q::unpack8(v[i], f);
#pragma unroll
        for (int e = 0; e < 8; ++e) m = fmaxf(m, fabsf(f[e]));
    }
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    s = fmaxf(m / w8a8q::E4M3_MAX, 1e-12f);

    F::AReg b;
    b.b0 = make_uint4(0u, 0u, 0u, 0u);
    b.b1 = make_uint4(0u, 0u, 0u, 0u);
    b.s0 = 0.0f;
    b.s1 = 0.0f;
    if (live) {
        const uint2 q0 = w8a8q::quant8(v[0], s);
        const uint2 q1 = w8a8q::quant8(v[1], s);
        const uint2 q2 = w8a8q::quant8(v[2], s);
        const uint2 q3 = w8a8q::quant8(v[3], s);
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
    const __nv_bfloat16* __restrict__ beta_bf16, float* __restrict__ beta, unsigned M, unsigned N, unsigned D,
    unsigned heads, float lower_bound) {
    using kdag::F;
    if (blockIdx.x == 0u && blockIdx.y == 0u) {
        for (unsigned i = threadIdx.x; i < M * heads; i += blockDim.x)
            beta[i] = kda_beta_scalar(__bfloat162float(beta_bf16[i]));
    }
    const wxay::Lane L = wxay::lane_ids();
    const unsigned tiles = N / 16u;
    const unsigned wt = blockIdx.x * wxay::WARPS + L.warp;
    if (wt >= 2u * tiles) return;
    const bool g_seg = wt >= tiles;
    const unsigned n0 = (g_seg ? wt - tiles : wt) * 16u;
    const unsigned j0 = blockIdx.y * 8u;

    float s;
    const F::AReg b = kdag::quant_a(g_seg ? g_low : f_low, j0, M, L, s);
    const float* S = g_seg ? Sg : Sf;
    const w8a8::Args a = {nullptr, nullptr, {g_seg ? Wg : Wf, nullptr, nullptr}, {S, nullptr, nullptr}, N, N, 128u, 0u};
    const F::Tile tile = F::tile(a, n0, L, N, 128u);
    float r[4];
    wxay::k128_tile<F>(tile, b, L, r);

    // 2026-10-14: F::store's layout: r[c] is weight row g (c < 2) or g + 8, token j0 + 2t + (c & 1); its scale is
    // held by the lanes of that token's quad (g == 2t + (c & 1)).
    const float s_even = __shfl_sync(0xffffffffu, s, 8u * L.t);
    const float s_odd = __shfl_sync(0xffffffffu, s, 8u * L.t + 4u);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const unsigned n = n0 + L.g + (c < 2 ? 0u : 8u);
        const unsigned tk = j0 + 2u * L.t + (unsigned)(c & 1);
        if (n >= N || tk >= M) continue;
        const __nv_bfloat16 o = __float2bfloat16_rn(r[c] * (__ldg(S + n) * ((c & 1) ? s_odd : s_even)));
        const unsigned long long at = (unsigned long long)tk * N + n;
        if (g_seg) {
            out_gate[at] = o;
        } else {
            g_raw[at] = o;
            gate[at] = kda_gate_scalar(__bfloat162float(o), dt_bias[n], expf(A_log[n / D]), lower_bound);
        }
    }
}
