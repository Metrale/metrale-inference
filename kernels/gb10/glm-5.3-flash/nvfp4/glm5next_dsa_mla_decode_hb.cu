// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-09: GLM-5.3-Flash selected-index MLA decode, head-batched on tensor cores
// (module `glm5next_dsa_mla_decode_hb`). Same contract and inputs as
// glm5next_dsa_mla_decode_fp8 in glm5next_dsa_mla_decode.cu (family glm_mla_decode), which
// runs one CTA per (head, row) and so reads every selected latent token once per head.
// Owner: gb10 kernels (glm-5.3-flash).
//
// Absorbed NoPE MLA: every query head of a row attends the SAME selected latent tokens (one
// 512-byte FP8 latent per token, K == V). Here one CTA owns (split, row, group of 24 heads)
// and loads each 32-token tile of its slice of the selection row once, widened to bf16 in
// shared memory, for all its heads:
//   S^T[tok, head] = K[tok, :] . Q[head, :]    mma m16n8k16 bf16 (K is FP8, exact in bf16)
//   online softmax per head over the slice     fp32, __expf, as the per-head kernel
//   O^T[dim, head] += V^T[dim, tok] P^T[tok, head]   mma bf16, P split hi + lo bf16
// then writes (m, l, o) per head for glm5next_dsa_mla_merge, or, with one split, the
// normalised BF16 output directly.
//
// QK: warp w < 6 owns token m-tile (w & 1) and head n-tile (w >> 1) and keeps that n-tile's Q
// B-fragments for all 32 k-steps in registers; PV: warp w owns latent dims [64w, 64w + 64).
// One CTA per SM (about 250 registers, 40 KB of shared memory). 2026-10-09: measured against
// a two-CTA-per-SM variant that keeps the tile FP8 and widens it per fragment (123 registers,
// 49 KB): that one was 1.6x slower at 16 rows (139 against 85 us, 8 splits, 2051 entries),
// its per-fragment widening repeated by each of the three head n-tiles.
//
// Numerics differ from the per-head kernel in summation order only, plus the P rounding: P
// is carried as bf16 hi + bf16 lo (relative error <= 2^-16 per weight) into a fp32 MMA
// accumulator. The split of a row depends only on that row (its own selection and
// num_splits), never on the other rows of the launch, so a row's output is the same
// whatever it is batched with.
//
// Launch contract:
//   * glm5next_dsa_mla_decode_hb_fp8: grid (num_splits, rows, ceil(num_q_heads / 24)), block
//     256, no dynamic shared memory.
//   * kv_lora_dim == 512; K and V are one buffer (the caller refuses distinct ones).
//   * A selection entry of -1, or any index outside [0, seq_len), is skipped; nothing
//     deduplicates. A row with no valid index writes zeros; a row whose seq_len is 0 writes
//     nothing to O (and its partials are never read).
//   * ws_o [rows, num_splits, num_q_heads, 512] f32 and ws_ml [rows, num_splits,
//     num_q_heads, 2] f32 when num_splits > 1; unused (may be null) when it is 1.
//   * glm5next_dsa_mla_merge: grid (num_q_heads, rows), block 128.
//   * 2026-10-12: glm5next_dsa_mla_decode_hb_fp8_phi: the same launch contract, P as bf16 hi
//     only (see its definition).

#include "glm5next_dsa_mla_hb_tile.cuh"

// 2026-10-12: The body. P_LO: carry P as bf16 hi + lo (the decode contract above); false: hi
// only (`glm5next_dsa_mla_decode_hb_fp8_phi`), which drops the second PV MMA of every tile.
// 2026-10-14: the per-tile steps live in glm5next_dsa_mla_hb_tile.cuh (shared with the
// pipelined prefill entry); this function only schedules them.
template <bool P_LO>
__device__ __forceinline__ void hb_body(HB_PARAMS) {
    __shared__ __align__(128) unsigned char kb[HB_TT * HB_CHUNKS * 16];   // 32 KB bf16 tile
    __shared__ __align__(16) float s_sc[HB_HEADS * HB_S_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 s_ph[HB_HEADS * HB_P_STRIDE];
    __shared__ __align__(16) __nv_bfloat16 s_pl[HB_HEADS * HB_P_STRIDE];
    __shared__ float s_alpha[HB_HEADS];
    __shared__ float s_l[HB_HEADS];
    __shared__ int s_valid[HB_TT];
    __shared__ unsigned int s_red[HB_WARPS];

    const unsigned int split = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int head_base = blockIdx.z * HB_HEADS;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    const unsigned int g = lane >> 2;
    const unsigned int c = lane & 3u;
    if (head_base >= num_q_heads) return;
    const unsigned int nh = min((unsigned int)HB_HEADS, num_q_heads - head_base);

    const unsigned int seq_len = (unsigned int)seq_lens[row];
    if (seq_len == 0) return;

    const int* sel = sel_indices + (size_t)row * sel_width;
    const int* bt = block_tables + (size_t)row * max_blocks_per_seq;
    const unsigned int kb_base = (unsigned int)__cvta_generic_to_shared(kb);

    hb_stage_q(kb, Q + ((size_t)row * num_q_heads + head_base) * HB_KVL, nh, tid);
    const unsigned int n_end = hb_row_extent(sel, sel_width, seq_len, s_red, tid, warp, lane);

    const unsigned int tiles = (n_end + HB_TT - 1) / HB_TT;
    const unsigned int per = (tiles + num_splits - 1) / num_splits;
    const unsigned int t_begin = min(split * per, tiles);
    const unsigned int t_end = min(t_begin + per, tiles);

    // 2026-10-09: QK warps: warp w < 2 * HB_NT owns token m-tile (w & 1) and head n-tile (w >> 1),
    // and keeps that n-tile's Q B-fragments for all 32 k-steps in registers.
    const bool qk_warp = warp < 2u * HB_NT;
    const unsigned int qk_mt = warp & 1u;
    const unsigned int qk_nt = warp >> 1;
    unsigned int qf[32][2];
    if (qk_warp) {
        #pragma unroll
        for (int s = 0; s < 32; s += 2) {
            const unsigned int addr = kb_base + hb_swz(qk_nt * 8u + (lane & 7u), 2u * s + (lane >> 3));
            hb_ldsm_x4(addr, qf[s][0], qf[s][1], qf[s + 1][0], qf[s + 1][1]);
        }
    }

    const unsigned int ld_tok = tid >> 3;
    const unsigned int ld_ch = tid & 7u;
    uint4 pre[4];
    bool pre_ok = false;
    auto load_tile = [&](unsigned int tile) {
        pre_ok = hb_gather(pre, tile * HB_TT + ld_tok, n_end, sel, seq_len, bt, KV, block_size,
                           cache_stride_bytes, ld_ch);
    };

    // 2026-10-09: Softmax state, per head, held identically by the 8 threads of the head's
    // group (thread -> head tid >> 3, tokens 4 * (tid & 7) .. +3).
    const unsigned int sm_h = tid >> 3;
    const unsigned int sm_sub = tid & 7u;
    float m_run = HB_NEG;
    float l_run = 0.0f;

    float acc[4][HB_NT][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < HB_NT; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

    if (t_begin < t_end) load_tile(t_begin);
    const float sc = k_scale;

    for (unsigned int tile = t_begin; tile < t_end; tile++) {
        __syncthreads();   // 2026-10-09: the previous tile's PV (and the Q fragment loads) are done
        hb_widen(kb, pre, ld_tok, ld_ch);
        if (ld_ch == 0) s_valid[ld_tok] = pre_ok ? 1 : 0;
        if (tile + 1 < t_end) load_tile(tile + 1);
        __syncthreads();

        if (qk_warp) {
            // 2026-10-09: Four accumulators over interleaved k-steps, summed at the end: four
            // MMA dependency chains of 8 instead of one of 32.
            float dk[4][4];
            #pragma unroll
            for (int a = 0; a < 4; a++)
                #pragma unroll
                for (int e = 0; e < 4; e++) dk[a][e] = 0.0f;
            const unsigned int a_row = qk_mt * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
            #pragma unroll
            for (int s = 0; s < 32; s++) {
                unsigned int a0, a1, a2, a3;
                hb_ldsm_x4(kb_base + hb_swz(a_row, 2u * s + (lane >> 4)), a0, a1, a2, a3);
                hb_mma(dk[s & 3], a0, a1, a2, a3, qf[s][0], qf[s][1]);
            }
            float d[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) d[e] = (dk[0][e] + dk[1][e]) + (dk[2][e] + dk[3][e]);
            const unsigned int h0 = qk_nt * 8u + 2u * c;
            const unsigned int t0 = qk_mt * 16u + g;
            s_sc[h0 * HB_S_STRIDE + t0] = (d[0] * sc) * inv_sqrt_d;
            s_sc[(h0 + 1u) * HB_S_STRIDE + t0] = (d[1] * sc) * inv_sqrt_d;
            s_sc[h0 * HB_S_STRIDE + t0 + 8u] = (d[2] * sc) * inv_sqrt_d;
            s_sc[(h0 + 1u) * HB_S_STRIDE + t0 + 8u] = (d[3] * sc) * inv_sqrt_d;
        }
        __syncthreads();

        if (sm_h < (unsigned int)HB_HEADS) {
            const float4 sv = *(const float4*)(s_sc + sm_h * HB_S_STRIDE + 4u * sm_sub);
            const float s4[4] = {sv.x, sv.y, sv.z, sv.w};
            hb_softmax<P_LO>(s4, s_valid, sm_h, sm_sub, nh, m_run, l_run, s_ph, s_pl, s_alpha);
        }
        __syncthreads();

        hb_pv<P_LO>(acc, kb_base, s_ph, s_pl, s_alpha, warp, lane);
    }

    hb_finish(acc, m_run, l_run, s_l, sm_h, sm_sub, nh, row, split, head_base, t_begin >= t_end,
              warp, lane, O, ws_o, ws_ml, num_q_heads, v_scale, num_splits);
}

extern "C" __global__ void __launch_bounds__(HB_THREADS, 1) glm5next_dsa_mla_decode_hb_fp8(HB_PARAMS) {
    hb_body<true>(HB_ARGS);
}

// 2026-10-12: The prefill twin (METRALE_GLM_DSA_HB_PREFILL_P_HI=1, glm5next_dsa/attend.rs): the
// same contract with P carried as one bf16 (relative error <= 2^-9 per weight instead of 2^-16),
// so each tile's PV runs one MMA per (dim tile, head tile) instead of two: a third of the
// kernel's MMAs. Not bit-identical to the entry above (accuracy-gated; the gate is
// examples/glm5next_dsa_hb_bench.rs).
extern "C" __global__ void __launch_bounds__(HB_THREADS, 1) glm5next_dsa_mla_decode_hb_fp8_phi(HB_PARAMS) {
    hb_body<false>(HB_ARGS);
}

// 2026-10-09: Merge the num_splits partials of one (head, row) in split order, skipping
// l == 0, into BF16 O. grid (num_q_heads, rows), block 128 (four latent dims per thread).
extern "C" __global__ void __launch_bounds__(128) glm5next_dsa_mla_merge(
    const float* __restrict__ ws_o, const float* __restrict__ ws_ml,
    const int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ O,
    const unsigned int num_q_heads, const unsigned int num_splits, const float v_scale) {
    const unsigned int h = blockIdx.x;
    const unsigned int row = blockIdx.y;
    if (seq_lens[row] == 0) return;
    const size_t base = (size_t)row * num_splits * num_q_heads + h;
    float m = HB_NEG;
    for (unsigned int s = 0; s < num_splits; s++) {
        const float* ml = ws_ml + (base + (size_t)s * num_q_heads) * 2u;
        if (ml[1] > 0.0f) m = fmaxf(m, ml[0]);
    }
    float l = 0.0f;
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int d0 = threadIdx.x * 4u;
    for (unsigned int s = 0; s < num_splits; s++) {
        const size_t at = base + (size_t)s * num_q_heads;
        const float ls = ws_ml[at * 2u + 1u];
        if (!(ls > 0.0f)) continue;
        const float w = __expf(ws_ml[at * 2u] - m);
        l += ls * w;
        const float4 v = *(const float4*)(ws_o + at * HB_KVL + d0);
        o[0] += v.x * w;
        o[1] += v.y * w;
        o[2] += v.z * w;
        o[3] += v.w * w;
    }
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    __nv_bfloat16* out = O + ((size_t)row * num_q_heads + h) * HB_KVL + d0;
    #pragma unroll
    for (int e = 0; e < 4; e++) out[e] = __float2bfloat16((o[e] * v_scale) * inv_l);
}
