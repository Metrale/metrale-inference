// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: GLM-5.3-Flash selected-index MLA attention for prefill, warp-specialized (module
// `glm5next_dsa_mla_prefill_tc`). The same inputs and output BYTES as
// glm5next_dsa_mla_decode_hb_fp8_phi (glm5next_dsa_mla_decode_hb.cu) at one split: a reschedule
// of that kernel, not a new numerics point. Owner: gb10 kernels (glm-5.3-flash).
//
// Why: the phi kernel spends most of a 32k prefill tile outside the tensor pipe (gather, widen,
// softmax, four barriers per tile; see the ablations in the ledger entry for
// METRALE_GLM_DSA_HB_PREFILL_TC). Here, one CTA per (row, head group):
//   * a prologue stages the row's Q in shared memory and builds a per-row table of 16-byte KV
//     offsets (the selection and block table read once) and the row's extent;
//   * 4 producer warps gather one 32-slot tile per step through that table, widen it exactly to
//     bf16 and publish it into one of two tile buffers (named barriers FULL 1/2, EMPTY 3/4),
//     two tiles ahead of the consumers;
//   * 8 consumer warps run QK, softmax and P V of a tile (barrier 5 between the phases): QK is
//     the phi kernel's four accumulator chains (warp w: token m-tile w & 1, k-steps (w >> 1) +
//     4i), combined as ((p0 + p1) + (p2 + p3)) * k_scale * inv_sqrt_d in round-to-nearest
//     intrinsics; softmax, P V and the final store are the shared functions of
//     glm5next_dsa_mla_hb_tile.cuh.
// The widen: a 32-bit word of four finite E4M3 bytes goes E4M3 -> f16 -> f32 -> bf16 (round to
// nearest), exact because every finite E4M3 value is a bf16; a word holding an E4M3 NaN byte
// (0x7f / 0xff) takes hb_e4m3x2_to_bf16x2, the phi kernel's own path (which maps NaN to a
// finite bf16), so every byte widens to the phi kernel's bf16.
// Every (row, head) goes through the same IEEE operations in the same order as in the phi
// kernel, so its bytes are identical; nothing a CTA computes reads another row's data, so a
// row's bytes do not depend on the rows launched beside it. The gate is
// examples/glm5next_dsa_prefill_tc_gate.rs.
//
// Launch contract: grid (1, rows, ceil(num_q_heads / 24)), block 384 (TC_THREADS),
// TC_SMEM_BYTES of dynamic shared memory, num_splits == 1, sel_width <= TC_MAX_WIDTH,
// cache_stride_bytes a multiple of 16 and every KV offset below 2^36 bytes. Launched otherwise,
// the kernel traps rather than corrupt memory.

#include "glm5next_dsa_mla_hb_tile.cuh"

#define TC_THREADS 384                                  // 2026-10-14: 8 consumer + 4 producer warps
#define TC_CONS 256
#define TC_PROD 128
#define TC_TILE_BYTES (HB_TT * HB_CHUNKS * 16)          // 2026-10-14: one bf16 tile, 32 KB
#define TC_PLANE_FLOATS (HB_HEADS * HB_S_STRIDE)        // 2026-10-14: one QK chain plane
#define TC_MAX_WIDTH 2064                               // 2026-10-14: offset-table entries
#define TC_OFF_NONE 0xffffffffu
#define TC_SMEM_BYTES (2 * TC_TILE_BYTES + 4 * TC_PLANE_FLOATS * 4 + TC_MAX_WIDTH * 4)   // 87616
#define TC_PER (32 / (TC_PROD / HB_TT))                 // 2026-10-14: 16-byte chunks per producer

__device__ __forceinline__ void tc_bar_sync(unsigned int id, unsigned int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tc_bar_arrive(unsigned int id, unsigned int n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void tc_ldsm_x2(unsigned int addr, unsigned int& r0, unsigned int& r1) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r0), "=r"(r1) : "r"(addr));
}

// 2026-10-14: Two E4M3 bytes to a bf16x2 word through the hardware converts (finite bytes only).
__device__ __forceinline__ unsigned int tc_cvt2(unsigned short v) {
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    const float2 f = __half22float2(*(const __half2*)&h);
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x, f.y);
    return *(const unsigned int*)&b;
}
__device__ __forceinline__ void tc_cv_word(unsigned int w, unsigned int& lo, unsigned int& hi) {
    if (__vcmpeq4(w & 0x7f7f7f7fu, 0x7f7f7f7fu)) {
        lo = hb_e4m3x2_to_bf16x2((unsigned short)(w & 0xffffu));
        hi = hb_e4m3x2_to_bf16x2((unsigned short)(w >> 16));
    } else {
        lo = tc_cvt2((unsigned short)(w & 0xffffu));
        hi = tc_cvt2((unsigned short)(w >> 16));
    }
}

// 2026-10-14: Producer p owns token p / 4 of a tile and FP8 chunks (p % 4) + 4k, k < 8.
__device__ __forceinline__ bool tc_gather(uint4 (&r)[TC_PER], unsigned int tile, unsigned int tok,
                                          unsigned int c4, unsigned int n_end, const unsigned int* s_off,
                                          const unsigned char* __restrict__ KV) {
    const unsigned int slot = tile * HB_TT + tok;
    const unsigned int o = slot < n_end ? s_off[slot] : TC_OFF_NONE;
    const bool ok = o != TC_OFF_NONE;
    const unsigned char* src = KV + (unsigned long long)o * 16ull + c4 * 16u;
    #pragma unroll
    for (int k = 0; k < TC_PER; k++)
        r[k] = ok ? __ldg((const uint4*)(src + 64u * k)) : make_uint4(0u, 0u, 0u, 0u);
    return ok;
}
__device__ __forceinline__ void tc_widen(unsigned char* kb, const uint4 (&r)[TC_PER], unsigned int tok,
                                         unsigned int c4) {
    #pragma unroll
    for (int k = 0; k < TC_PER; k++) {
        uint4 lo, hi;
        tc_cv_word(r[k].x, lo.x, lo.y);
        tc_cv_word(r[k].y, lo.z, lo.w);
        tc_cv_word(r[k].z, hi.x, hi.y);
        tc_cv_word(r[k].w, hi.z, hi.w);
        const unsigned int ch = 2u * (c4 + 4u * k);
        *(uint4*)(kb + hb_swz(tok, ch)) = lo;
        *(uint4*)(kb + hb_swz(tok, ch + 1u)) = hi;
    }
}

// 2026-10-14: Chain `chain` of one tile's QK for token m-tile `mt`, all three head n-tiles, from
// zero accumulators, stored to the chain's plane.
__device__ __forceinline__ void tc_qk_chain(unsigned int tile_base, const unsigned int (&qf)[8][HB_NT][2],
                                            float* plane, unsigned int mt, unsigned int chain, unsigned int lane) {
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
    const unsigned int g = lane >> 2, c = lane & 3u, t0 = mt * 16u + g;
    #pragma unroll
    for (int j = 0; j < HB_NT; j++) {
        const unsigned int h0 = j * 8u + 2u * c;
        plane[h0 * HB_S_STRIDE + t0] = dk[j][0];
        plane[(h0 + 1u) * HB_S_STRIDE + t0] = dk[j][1];
        plane[h0 * HB_S_STRIDE + t0 + 8u] = dk[j][2];
        plane[(h0 + 1u) * HB_S_STRIDE + t0 + 8u] = dk[j][3];
    }
}

// 2026-10-14: The producer warps: tiles 0, 1 gathered before Q leaves buffer 1, then each tile is
// widened into its buffer once the consumers release it and the tile two ahead is gathered.
__device__ __forceinline__ void tc_produce(unsigned int p, unsigned int tiles, unsigned int n_end,
                                           const unsigned int* s_off, unsigned char* const (&kb)[2],
                                           int (*s_valid)[HB_TT], const unsigned char* __restrict__ KV) {
    const unsigned int tok = p >> 2, c4 = p & 3u;
    uint4 ra[TC_PER], rb[TC_PER];
    bool va = false, vb = false;
    if (tiles > 0) va = tc_gather(ra, 0, tok, c4, n_end, s_off, KV);
    if (tiles > 1) vb = tc_gather(rb, 1, tok, c4, n_end, s_off, KV);
    __syncthreads();   // 2026-10-14: the consumers hold their Q fragments; buffer 1 is free
    for (unsigned int n = 0; n < tiles; n += 2) {
        if (n >= 2) tc_bar_sync(3, TC_THREADS);
        tc_widen(kb[0], ra, tok, c4);
        if (c4 == 0) s_valid[0][tok] = va ? 1 : 0;
        tc_bar_arrive(1, TC_THREADS);
        if (n + 2 < tiles) va = tc_gather(ra, n + 2, tok, c4, n_end, s_off, KV);
        if (n + 1 < tiles) {
            if (n + 1 >= 2) tc_bar_sync(4, TC_THREADS);
            tc_widen(kb[1], rb, tok, c4);
            if (c4 == 0) s_valid[1][tok] = vb ? 1 : 0;
            tc_bar_arrive(2, TC_THREADS);
            if (n + 3 < tiles) vb = tc_gather(rb, n + 3, tok, c4, n_end, s_off, KV);
        }
    }
    __syncthreads();   // 2026-10-14: the barrier inside hb_finish
}

extern "C" __global__ void __launch_bounds__(TC_THREADS, 1) glm5next_dsa_mla_prefill_tc_phi(HB_PARAMS) {
    extern __shared__ __align__(128) unsigned char tc_smem[];
    __shared__ __align__(16) __nv_bfloat16 s_ph[HB_HEADS * HB_P_STRIDE];
    __shared__ float s_alpha[HB_HEADS];
    __shared__ float s_l[HB_HEADS];
    __shared__ int s_valid[2][HB_TT];
    __shared__ unsigned int s_red[TC_THREADS / 32];

    unsigned int dyn_bytes;
    asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(dyn_bytes));
    if (dyn_bytes < (unsigned int)TC_SMEM_BYTES || blockDim.x != TC_THREADS || sel_width > TC_MAX_WIDTH ||
        num_splits != 1 || gridDim.x != 1 || (cache_stride_bytes & 15ull) != 0ull)
        __trap();

    const unsigned int row = blockIdx.y;
    const unsigned int head_base = blockIdx.z * HB_HEADS;
    const unsigned int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31u;
    if (head_base >= num_q_heads) return;
    const unsigned int nh = min((unsigned int)HB_HEADS, num_q_heads - head_base);
    const unsigned int seq_len = (unsigned int)seq_lens[row];
    if (seq_len == 0) return;

    const int* sel = sel_indices + (size_t)row * sel_width;
    const int* bt = block_tables + (size_t)row * max_blocks_per_seq;
    unsigned char* const kb[2] = {tc_smem, tc_smem + TC_TILE_BYTES};
    const unsigned int kb_base[2] = {(unsigned int)__cvta_generic_to_shared(kb[0]),
                                     (unsigned int)__cvta_generic_to_shared(kb[1])};
    float* planes = (float*)(tc_smem + 2 * TC_TILE_BYTES);
    unsigned int* s_off = (unsigned int*)(tc_smem + 2 * TC_TILE_BYTES + 4 * TC_PLANE_FLOATS * 4);

    // 2026-10-14: Prologue: Q into buffer 1 (heads past nh zero); the offset table and extent.
    const __nv_bfloat16* q_row = Q + ((size_t)row * num_q_heads + head_base) * HB_KVL;
    for (unsigned int i = tid; i < (unsigned int)HB_HEADS * HB_CHUNKS; i += TC_THREADS) {
        const unsigned int h = i / HB_CHUNKS, ch = i % HB_CHUNKS;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (h < nh) v = *(const uint4*)(q_row + (size_t)h * HB_KVL + ch * 8u);
        *(uint4*)(kb[1] + hb_swz(h, ch)) = v;
    }
    unsigned int last = 0;
    for (unsigned int i = tid; i < sel_width; i += TC_THREADS) {
        const int t = sel[i];
        unsigned int o = TC_OFF_NONE;
        if (hb_valid(t, seq_len)) {
            const unsigned long long a =
                (unsigned long long)(unsigned int)bt[(unsigned int)t / block_size] * cache_stride_bytes +
                (unsigned long long)((unsigned int)t % block_size) * HB_KVL;
            if ((a >> 4) >= 0xffffffffull) __trap();
            o = (unsigned int)(a >> 4);
            last = i + 1;
        }
        s_off[i] = o;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) last = max(last, __shfl_xor_sync(0xffffffffu, last, o));
    if (lane == 0) s_red[warp] = last;
    __syncthreads();
    unsigned int n_end = 0;
    #pragma unroll
    for (int w = 0; w < TC_THREADS / 32; w++) n_end = max(n_end, s_red[w]);
    const unsigned int tiles = (n_end + HB_TT - 1) / HB_TT;

    if (tid >= TC_CONS) {
        tc_produce(tid - TC_CONS, tiles, n_end, s_off, kb, s_valid, KV);
        return;
    }

    // 2026-10-14: The consumer warps: Q fragments out of buffer 1, then the tiles in order.
    const unsigned int qk_mt = warp & 1u, qk_chain = warp >> 1;
    float* qk_plane = planes + qk_chain * TC_PLANE_FLOATS;
    unsigned int qf[8][HB_NT][2];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const unsigned int s = qk_chain + 4u * i;
        #pragma unroll
        for (int j = 0; j < HB_NT; j++)
            tc_ldsm_x2(kb_base[1] + hb_swz(j * 8u + (lane & 7u), 2u * s + ((lane >> 3) & 1u)), qf[i][j][0],
                       qf[i][j][1]);
    }
    __syncthreads();

    const unsigned int sm_h = tid >> 3, sm_sub = tid & 7u;
    float m_run = HB_NEG, l_run = 0.0f;
    float acc[4][HB_NT][4];
    #pragma unroll
    for (int i = 0; i < 4; i++)
        #pragma unroll
        for (int j = 0; j < HB_NT; j++)
            #pragma unroll
            for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
    const float sc = k_scale;
    if (tiles > 0) {
        tc_bar_sync(1, TC_THREADS);
        tc_qk_chain(kb_base[0], qf, qk_plane, qk_mt, qk_chain, lane);
    }
    for (unsigned int n = 0; n < tiles; n++) {
        const unsigned int b = n & 1u;
        tc_bar_sync(5, TC_CONS);   // 2026-10-14: the four QK planes of tile n are complete
        if (sm_h < (unsigned int)HB_HEADS) {
            const unsigned int at = sm_h * HB_S_STRIDE + 4u * sm_sub;
            float s4[4];
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const float d = __fadd_rn(__fadd_rn(planes[at + e], planes[TC_PLANE_FLOATS + at + e]),
                                          __fadd_rn(planes[2 * TC_PLANE_FLOATS + at + e],
                                                    planes[3 * TC_PLANE_FLOATS + at + e]));
                s4[e] = __fmul_rn(__fmul_rn(d, sc), inv_sqrt_d);
            }
            hb_softmax<false>(s4, s_valid[b], sm_h, sm_sub, nh, m_run, l_run, s_ph, nullptr, s_alpha);
        }
        tc_bar_sync(5, TC_CONS);   // 2026-10-14: P and alpha of tile n are published
        hb_pv<false>(acc, kb_base[b], s_ph, nullptr, s_alpha, warp, lane);
        if (n + 2 < tiles) tc_bar_arrive(3u + b, TC_THREADS);
        if (n + 1 < tiles) {
            tc_bar_sync(1u + (b ^ 1u), TC_THREADS);
            tc_qk_chain(kb_base[b ^ 1u], qf, qk_plane, qk_mt, qk_chain, lane);
        }
    }
    hb_finish(acc, m_run, l_run, s_l, sm_h, sm_sub, nh, row, 0u, head_base, tiles == 0, warp, lane, O, ws_o, ws_ml,
              num_q_heads, v_scale, 1u);
}
