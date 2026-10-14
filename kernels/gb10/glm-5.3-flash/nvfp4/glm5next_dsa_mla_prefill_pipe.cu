// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: GLM-5.3-Flash selected-index MLA attention for prefill, pipelined (module
// `glm5next_dsa_mla_prefill_pipe`). The same contract, inputs and output BYTES as
// glm5next_dsa_mla_decode_hb_fp8_phi (glm5next_dsa_mla_decode_hb.cu): a reschedule of that
// kernel, not a new numerics point. Owner: gb10 kernels (glm-5.3-flash).
//
// Why: the phi kernel runs each 32-token tile as four phases separated by barriers (gather +
// widen, QK on 6 of 8 warps, softmax, PV), so the tensor pipe idles through the widen and the
// softmax, and two warps idle through QK. Here:
//   * the tile buffer is double-buffered (2 x 32 KB of dynamic shared memory), so tile t + 1
//     is widened while tile t's softmax runs, and QK of tile t + 1 is issued in the same phase
//     as PV of tile t: two barriers per tile instead of four;
//   * QK runs on all 8 warps: warp w owns token m-tile (w & 1) and accumulator chain
//     a = w >> 1 for all three head n-tiles, i.e. k-steps a, a + 4, ..., a + 28: exactly the
//     MMAs, in exactly the order, that the phi kernel's chain dk[a] performs. The four chain
//     partials go to four shared planes and the softmax threads combine them as
//     ((p0 + p1) + (p2 + p3)) * k_scale * inv_sqrt_d, the phi kernel's expression, in
//     round-to-nearest intrinsics so no contraction can differ.
//   * softmax, P V and the store are the shared functions of glm5next_dsa_mla_hb_tile.cuh.
// Every (row, head) therefore goes through the same IEEE operations in the same order as in
// the phi kernel, so its output bytes are identical; and since nothing a CTA computes depends
// on another row, a row's bytes do not depend on what it is launched with (sub-chunk, group
// size, prefix-cache hit or miss). The gate is examples/glm5next_dsa_prefill_pipe_gate.rs.
//
// Launch contract: as glm5next_dsa_mla_decode_hb_fp8 (grid (num_splits, rows,
// ceil(num_q_heads / 24)), block 256; ws_o / ws_ml and glm5next_dsa_mla_merge when
// num_splits > 1), plus PP_SMEM_BYTES of dynamic shared memory; launched with less, the
// kernel traps rather than corrupt memory. One CTA per SM.

#include "glm5next_dsa_mla_hb_tile.cuh"

#define PP_TILE_BYTES (HB_TT * HB_CHUNKS * 16)            // 2026-10-14: one bf16 tile, 32 KB
#define PP_PLANE_FLOATS (HB_HEADS * HB_S_STRIDE)          // 2026-10-14: one QK chain plane
#define PP_CHAINS 4
#define PP_SMEM_BYTES (2 * PP_TILE_BYTES + PP_CHAINS * PP_PLANE_FLOATS * 4)   // 79360

__device__ __forceinline__ void pp_ldsm_x2(unsigned int addr, unsigned int& r0, unsigned int& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r0), "=r"(r1) : "r"(addr));
}

// 2026-10-14: Chain `chain` of the QK of one tile for token m-tile `mt`, all three head
// n-tiles: k-steps chain, chain + 4, ..., each accumulator started at zero, stored raw.
__device__ __forceinline__ void pp_qk_chain(unsigned int tile_base, const unsigned int (&qf)[8][HB_NT][2],
                                            float* plane, unsigned int mt, unsigned int chain,
                                            unsigned int lane) {
    float dk[HB_NT][4];
    #pragma unroll
    for (int j = 0; j < HB_NT; j++)
        #pragma unroll
        for (int e = 0; e < 4; e++) dk[j][e] = 0.0f;
    const unsigned int a_row = mt * 16u + (lane & 7u) + ((lane >> 3) & 1u) * 8u;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int s = chain + 4u * i;
        unsigned int a0, a1, a2, a3;
        hb_ldsm_x4(tile_base + hb_swz(a_row, 2u * s + (lane >> 4)), a0, a1, a2, a3);
        #pragma unroll
        for (int j = 0; j < HB_NT; j++) hb_mma(dk[j], a0, a1, a2, a3, qf[i][j][0], qf[i][j][1]);
    }
    const unsigned int g = lane >> 2;
    const unsigned int c = lane & 3u;
    const unsigned int t0 = mt * 16u + g;
    #pragma unroll
    for (int j = 0; j < HB_NT; j++) {
        const unsigned int h0 = j * 8u + 2u * c;
        plane[h0 * HB_S_STRIDE + t0] = dk[j][0];
        plane[(h0 + 1u) * HB_S_STRIDE + t0] = dk[j][1];
        plane[h0 * HB_S_STRIDE + t0 + 8u] = dk[j][2];
        plane[(h0 + 1u) * HB_S_STRIDE + t0 + 8u] = dk[j][3];
    }
}

extern "C" __global__ void __launch_bounds__(HB_THREADS, 1) glm5next_dsa_mla_prefill_pipe_phi(HB_PARAMS) {
    extern __shared__ __align__(128) unsigned char pp_smem[];
    __shared__ __align__(16) __nv_bfloat16 s_ph[HB_HEADS * HB_P_STRIDE];
    __shared__ float s_alpha[HB_HEADS];
    __shared__ float s_l[HB_HEADS];
    __shared__ int s_valid[2][HB_TT];
    __shared__ unsigned int s_red[HB_WARPS];

    unsigned int dyn_bytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_bytes));
    if (dyn_bytes < (unsigned int)PP_SMEM_BYTES) __trap();

    const unsigned int split = blockIdx.x;
    const unsigned int row = blockIdx.y;
    const unsigned int head_base = blockIdx.z * HB_HEADS;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid >> 5;
    const unsigned int lane = tid & 31u;
    if (head_base >= num_q_heads) return;
    const unsigned int nh = min((unsigned int)HB_HEADS, num_q_heads - head_base);

    const unsigned int seq_len = (unsigned int)seq_lens[row];
    if (seq_len == 0) return;

    const int* sel = sel_indices + (size_t)row * sel_width;
    const int* bt = block_tables + (size_t)row * max_blocks_per_seq;
    unsigned char* kb[2] = {pp_smem, pp_smem + PP_TILE_BYTES};
    const unsigned int kb_base[2] = {(unsigned int)__cvta_generic_to_shared(kb[0]),
                                     (unsigned int)__cvta_generic_to_shared(kb[1])};
    float* planes = (float*)(pp_smem + 2 * PP_TILE_BYTES);

    // 2026-10-14: Q goes to buffer 1, which the first tile does not use.
    hb_stage_q(kb[1], Q + ((size_t)row * num_q_heads + head_base) * HB_KVL, nh, tid);
    const unsigned int n_end = hb_row_extent(sel, sel_width, seq_len, s_red, tid, warp, lane);

    const unsigned int tiles = (n_end + HB_TT - 1) / HB_TT;
    const unsigned int per = (tiles + num_splits - 1) / num_splits;
    const unsigned int t_begin = min(split * per, tiles);
    const unsigned int t_end = min(t_begin + per, tiles);

    const unsigned int qk_mt = warp & 1u;
    const unsigned int qk_chain = warp >> 1;
    float* qk_plane = planes + qk_chain * PP_PLANE_FLOATS;
    unsigned int qf[8][HB_NT][2];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int s = qk_chain + 4u * i;
        #pragma unroll
        for (int j = 0; j < HB_NT; j++)
            pp_ldsm_x2(kb_base[1] + hb_swz(j * 8u + (lane & 7u), 2u * s + ((lane >> 3) & 1u)),
                       qf[i][j][0], qf[i][j][1]);
    }

    const unsigned int ld_tok = tid >> 3;
    const unsigned int ld_ch = tid & 7u;
    uint4 pre[4];
    bool pre_ok = false;
    auto load_tile = [&](unsigned int tile) {
        pre_ok = hb_gather(pre, tile * HB_TT + ld_tok, n_end, sel, seq_len, bt, KV, block_size,
                           cache_stride_bytes, ld_ch);
    };

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

    const float sc = k_scale;
    if (t_begin < t_end) {
        load_tile(t_begin);
        hb_widen(kb[0], pre, ld_tok, ld_ch);
        if (ld_ch == 0) s_valid[0][ld_tok] = pre_ok ? 1 : 0;
        if (t_begin + 1 < t_end) load_tile(t_begin + 1);
        __syncthreads();
        pp_qk_chain(kb_base[0], qf, qk_plane, qk_mt, qk_chain, lane);
    }

    for (unsigned int tile = t_begin; tile < t_end; tile++) {
        const unsigned int b = (tile - t_begin) & 1u;
        // 2026-10-14: QK(tile) is in the planes; P V(tile - 1) and the Q fragment loads are
        // done, so buffer b ^ 1, s_valid[b ^ 1], P and alpha are free.
        __syncthreads();
        if (tile + 1 < t_end) {
            hb_widen(kb[b ^ 1u], pre, ld_tok, ld_ch);
            if (ld_ch == 0) s_valid[b ^ 1u][ld_tok] = pre_ok ? 1 : 0;
            if (tile + 2 < t_end) load_tile(tile + 2);
        }
        if (sm_h < (unsigned int)HB_HEADS) {
            const unsigned int at = sm_h * HB_S_STRIDE + 4u * sm_sub;
            float s4[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float d = __fadd_rn(__fadd_rn(planes[at + e], planes[PP_PLANE_FLOATS + at + e]),
                                          __fadd_rn(planes[2 * PP_PLANE_FLOATS + at + e],
                                                    planes[3 * PP_PLANE_FLOATS + at + e]));
                s4[e] = __fmul_rn(__fmul_rn(d, sc), inv_sqrt_d);
            }
            hb_softmax<false>(s4, s_valid[b], sm_h, sm_sub, nh, m_run, l_run, s_ph, nullptr, s_alpha);
        }
        // 2026-10-14: P, alpha and tile + 1 are published; the planes are consumed.
        __syncthreads();
        hb_pv<false>(acc, kb_base[b], s_ph, nullptr, s_alpha, warp, lane);
        if (tile + 1 < t_end) pp_qk_chain(kb_base[b ^ 1u], qf, qk_plane, qk_mt, qk_chain, lane);
    }

    hb_finish(acc, m_run, l_run, s_l, sm_h, sm_sub, nh, row, split, head_base, t_begin >= t_end,
              warp, lane, O, ws_o, ws_ml, num_q_heads, v_scale, num_splits);
}
