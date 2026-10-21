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
// - kda_gates_w8a8_k128: one token. kda_gates_w8a8_k128_rows (2026-10-21): M rows, 1 <= M <= 8, in the MMA's n8
//   token columns (each row its own activation scale, so each row's bytes do not depend on the others; M == 1
//   gives the one-token kernel's bytes); M == 0 or M > 8 returns without writing.
//   glm5next_kda_gates_rows_gate checks it against the unfused chain and the one-token kernel.
// - K == 128 (the low rank), N % 16 == 0, N == heads * D; f_low, g_low 16-byte aligned. Per-row weight
//   scales (Fp8PerRow). Grid (ceil(2 * N / 16 / 8), 1, 1), block (256, 1, 1), no dynamic shared memory. The
//   activation scratch and scales of the unfused path are not written.

#include "kda_gate_scalar.cuh"
#include "w8a8_policy.cuh"
#include "w8a8_quant.cuh"

namespace kdag {

using F = w8a8::W8A8<false, false>;
using u8 = unsigned char;

// 2026-10-21: The rows' values one lane reads: values 4l .. 4l + 3 of every row r < M (for the amaxes), and, for
// lanes with g < M, the 32 values of row g in its two 16-byte operand runs, k in [16t, 16t + 16) and
// [64 + 16t, 64 + 16t + 16). All loads are issued before any is used.
constexpr unsigned MAX_ROWS = 8;
struct RowLoads {
    uint2 amax_part[MAX_ROWS];
    uint4 run[4];
};

__device__ __forceinline__ RowLoads load_rows(const __nv_bfloat16* __restrict__ x, unsigned M, const wxay::Lane& L) {
    RowLoads v;
#pragma unroll
    for (unsigned r = 0; r < MAX_ROWS; ++r)
        v.amax_part[r] = r < M ? *reinterpret_cast<const uint2*>(x + r * 128u + L.lane * 4u) : make_uint2(0u, 0u);
    const uint4* x4 = reinterpret_cast<const uint4*>(x + L.g * 128u);
    const bool live = L.g < M;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    v.run[0] = live ? x4[2u * L.t] : z;
    v.run[1] = live ? x4[2u * L.t + 1u] : z;
    v.run[2] = live ? x4[8u + 2u * L.t] : z;
    v.run[3] = live ? x4[8u + 2u * L.t + 1u] : z;
    return v;
}

// 2026-10-21: The quantization scale of each row r < M, in every lane: the warp's max of the |values| (exact in any
// order; fmaxf drops NaN), then quant_row's scale. Rows at or past M get 0.
__device__ __forceinline__ void row_scales(const RowLoads& v, unsigned M, float (&s)[MAX_ROWS]) {
#pragma unroll
    for (unsigned r = 0; r < MAX_ROWS; ++r) {
        float m = 0.f;
        const uint32_t w[2] = {v.amax_part[r].x, v.amax_part[r].y};
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            m = fmaxf(m, fabsf(__uint_as_float(w[i] << 16)));
            m = fmaxf(m, fabsf(__uint_as_float(w[i] & 0xFFFF0000u)));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        s[r] = r < M ? fmaxf(m / w8a8q::E4M3_MAX, 1e-12f) : 0.f;
    }
}

__device__ __forceinline__ float pick(const float (&s)[MAX_ROWS], unsigned r) {
    float v = 0.f;
#pragma unroll
    for (unsigned i = 0; i < MAX_ROWS; ++i) v = (i == r) ? s[i] : v;
    return v;
}

// 2026-10-10: The MMA operand of the tokens (lanes with g < M hold token g's, as F::tok / F::load_a give token g to
// lane group g and zeros past M): lane (g, t) encodes its two runs of row g with that row's scale.
__device__ __forceinline__ F::AReg quant_a(const RowLoads& v, unsigned M, const float (&s)[MAX_ROWS],
                                           const wxay::Lane& L) {
    F::AReg b;
    b.b0 = make_uint4(0u, 0u, 0u, 0u);
    b.b1 = make_uint4(0u, 0u, 0u, 0u);
    b.s0 = 0.0f;
    b.s1 = 0.0f;
    if (L.g < M) {
        const float sg = pick(s, L.g);
        const uint2 q0 = w8a8q::quant8(v.run[0], sg);
        const uint2 q1 = w8a8q::quant8(v.run[1], sg);
        const uint2 q2 = w8a8q::quant8(v.run[2], sg);
        const uint2 q3 = w8a8q::quant8(v.run[3], sg);
        b.b0 = make_uint4(q0.x, q0.y, q1.x, q1.y);
        b.b1 = make_uint4(q2.x, q2.y, q3.x, q3.y);
    }
    return b;
}

// 2026-10-21: The chain for M tokens (rows of f_low / g_low at stride 128; outputs at stride N, beta at stride
// heads). Each token's arithmetic is the one-token chain's: the tokens sit in separate MMA columns, and quant_row's
// scale and encoder are per row.
__device__ __forceinline__ void gates_body(
    const __nv_bfloat16* __restrict__ f_low, const __nv_bfloat16* __restrict__ g_low, const unsigned char* __restrict__ Wf,
    const float* __restrict__ Sf, const unsigned char* __restrict__ Wg, const float* __restrict__ Sg,
    __nv_bfloat16* __restrict__ g_raw, __nv_bfloat16* __restrict__ out_gate, const float* __restrict__ dt_bias,
    const float* __restrict__ A_log, float* __restrict__ gate, const __nv_bfloat16* __restrict__ beta_bf16,
    float* __restrict__ beta, unsigned M, unsigned N, unsigned D, unsigned heads, float lower_bound) {
    if (blockIdx.x == 0u) {
        for (unsigned i = threadIdx.x; i < M * heads; i += blockDim.x)
            beta[i] = kda_beta_scalar(__bfloat162float(beta_bf16[i]));
    }
    const wxay::Lane L = wxay::lane_ids();
    const unsigned tiles = N / 16u;
    const unsigned wt = blockIdx.x * wxay::WARPS + L.warp;
    if (wt >= 2u * tiles) return;
    const bool g_seg = wt >= tiles;
    const unsigned n0 = (g_seg ? wt - tiles : wt) * 16u;
    const __nv_bfloat16* x = g_seg ? g_low : f_low;

    const float* S = g_seg ? Sg : Sf;
    const w8a8::Args a = {nullptr, nullptr, {g_seg ? Wg : Wf, nullptr, nullptr}, {S, nullptr, nullptr}, N, N, 128u, 0u};
    const F::Tile tile = F::tile(a, n0, L, N, 128u);
    const F::WReg w = F::load_w(tile, 0u, true);
    const RowLoads v = load_rows(x, M, L);
    float ws[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const unsigned n = n0 + L.g + (c < 2 ? 0u : 8u);
        ws[c] = n < N ? __ldg(S + n) : 0.f;
    }
    float s[MAX_ROWS];
    row_scales(v, M, s);
    const F::AReg b = quant_a(v, M, s, L);
    float r[4];
    wxay::k128_tile_w<F>(w, b, L, r);

    // 2026-10-10: F::store's layout: lane (g, t) holds rows n0 + g (c 0, 1) and n0 + g + 8 (c 2, 3) of tokens 2t
    // (c even) and 2t + 1 (c odd).
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const unsigned tk = L.t * 2u + (unsigned)(c & 1);
        const unsigned n = n0 + L.g + (c < 2 ? 0u : 8u);
        if (tk >= M || n >= N) continue;
        const __nv_bfloat16 o = __float2bfloat16_rn(r[c] * (ws[c] * pick(s, tk)));
        const size_t i = (size_t)tk * N + n;
        if (g_seg) {
            out_gate[i] = o;
        } else {
            g_raw[i] = o;
            gate[i] = kda_gate_scalar(__bfloat162float(o), dt_bias[n], expf(A_log[n / D]), lower_bound);
        }
    }
}

}  // namespace kdag

#define KDAG_ARGS                                                                                                   \
    const __nv_bfloat16 *__restrict__ f_low, const __nv_bfloat16 *__restrict__ g_low,                                   \
        const unsigned char *__restrict__ Wf, const float *__restrict__ Sf, const unsigned char *__restrict__ Wg,       \
        const float *__restrict__ Sg, __nv_bfloat16 *__restrict__ g_raw, __nv_bfloat16 *__restrict__ out_gate,          \
        const float *__restrict__ dt_bias, const float *__restrict__ A_log, float *__restrict__ gate,                    \
        const __nv_bfloat16 *__restrict__ beta_bf16, float *__restrict__ beta

extern "C" __global__ __launch_bounds__(256) void kda_gates_w8a8_k128(KDAG_ARGS, unsigned N, unsigned D, unsigned heads,
                                                                      float lower_bound) {
    kdag::gates_body(f_low, g_low, Wf, Sf, Wg, Sg, g_raw, out_gate, dt_bias, A_log, gate, beta_bf16, beta, 1u, N, D,
                     heads, lower_bound);
}

// 2026-10-21: The chain for M = 1..8 tokens in one launch (the verify step's rows), same grid.
extern "C" __global__ __launch_bounds__(256) void kda_gates_w8a8_k128_rows(KDAG_ARGS, unsigned M, unsigned N,
                                                                           unsigned D, unsigned heads,
                                                                           float lower_bound) {
    if (M == 0u || M > kdag::MAX_ROWS) return;
    kdag::gates_body(f_low, g_low, Wf, Sf, Wg, Sg, g_raw, out_gate, dt_bias, A_log, gate, beta_bf16, beta, M, N, D,
                     heads, lower_bound);
}
