// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The per-tile pieces of the head-batched selected-index MLA attention, shared by
// glm5next_dsa_mla_decode_hb.cu (the decode entries and their prefill twin) and
// glm5next_dsa_mla_prefill_pipe.cu (the pipelined prefill entry). Owner: gb10 kernels
// (glm-5.3-flash).
//
// One source for every operation that decides an output bit: the tile gather and its exact
// E4M3 -> bf16 widen, the online softmax of one 32-token tile, the P V step and the final
// store. The two files differ only in how they schedule these steps across warps and shared
// memory buffers, so a row computed by either goes through the same arithmetic in the same
// order. Every function is force-inlined; the decode module's SASS is unchanged by the move
// (sm_121f cubins of the pre- and post-move source disassemble identically; the PTX differs
// only in virtual register numbering and the placement of one compare per entry).

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define HB_THREADS 256
#define HB_WARPS 8
#define HB_TT 32          // 2026-10-09: tokens per tile (two m16 tiles)
#define HB_KVL 512
#define HB_CHUNKS 64      // 2026-10-09: 16-byte bf16 chunks per 512-wide row
#define HB_S_STRIDE 36    // 2026-10-09: f32 per S row (bank-conflict-free fragment stores)
#define HB_P_STRIDE 40    // 2026-10-09: bf16 per P row (bank-conflict-free B-fragment loads)
#define HB_HEADS 24       // 2026-10-09: heads per CTA: three n8 tiles
#define HB_NT 3
#define HB_NEG (-1e30f)

// 2026-10-09: Byte offset of 16-byte chunk `chunk` of row `row` in a [rows][64 chunks] bf16
// tile, XOR-swizzled so eight consecutive rows at one chunk hit eight bank groups.
__device__ __forceinline__ unsigned int hb_swz(unsigned int row, unsigned int chunk) {
    return row * (HB_CHUNKS * 16u) + ((chunk ^ (row & 7u)) * 16u);
}

// 2026-10-09: Two E4M3 bytes (low byte = lower index) to a bf16x2 word, exactly: through
// f16 (every finite E4M3 value is a normal f16), then f16 -> bf16 by moving the exponent
// bias (112) and dropping three mantissa bits that are zero.
__device__ __forceinline__ unsigned int hb_e4m3x2_to_bf16x2(unsigned short v) {
    __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)v, __NV_E4M3);
    const unsigned int x = (unsigned int)h.x | ((unsigned int)h.y << 16);
    const unsigned int mag = x & 0x7fff7fffu;
    const unsigned int r = ((mag >> 3) & 0x0fff0fffu) + (0x38003800u & __vcmpne2(mag, 0u));
    return r | (x & 0x80008000u);
}

__device__ __forceinline__ void hb_ldsm_x4(unsigned int addr, unsigned int& r0, unsigned int& r1,
                                           unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void hb_ldsm_x4_t(unsigned int addr, unsigned int& r0, unsigned int& r1,
                                             unsigned int& r2, unsigned int& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void hb_mma(float* d, unsigned int a0, unsigned int a1, unsigned int a2,
                                       unsigned int a3, unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// 2026-10-09: Whether selection entry `t` names a token the row may attend.
__device__ __forceinline__ bool hb_valid(int t, unsigned int seq_len) {
    return t >= 0 && (unsigned int)t < seq_len;
}

// 2026-10-12: The kernel's parameter list and the names, shared by every entry.
#define HB_PARAMS                                                                                  \
    const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ KV,                     \
    __nv_bfloat16* __restrict__ O, const int* __restrict__ block_tables,                           \
    const int* __restrict__ seq_lens, const int* __restrict__ sel_indices,                         \
    float* __restrict__ ws_o, float* __restrict__ ws_ml, const unsigned int sel_width,             \
    const unsigned int max_blocks_per_seq, const unsigned int num_q_heads,                         \
    const unsigned int block_size, const float inv_sqrt_d, const float k_scale,                    \
    const float v_scale, const unsigned long long cache_stride_bytes, const unsigned int num_splits
#define HB_ARGS                                                                                    \
    Q, KV, O, block_tables, seq_lens, sel_indices, ws_o, ws_ml, sel_width, max_blocks_per_seq,     \
    num_q_heads, block_size, inv_sqrt_d, k_scale, v_scale, cache_stride_bytes, num_splits

// 2026-10-14: Gather this thread's share of selection slot `slot` (loader geometry: thread ->
// token tid >> 3 of the tile, FP8 16-byte chunks (tid & 7) + 8k, k = 0..3) into `pre`; zeros
// for an invalid or out-of-extent slot. Returns whether the slot is valid.
__device__ __forceinline__ bool hb_gather(uint4 (&pre)[4], unsigned int slot, unsigned int n_end,
                                          const int* __restrict__ sel, unsigned int seq_len,
                                          const int* __restrict__ bt,
                                          const unsigned char* __restrict__ KV,
                                          unsigned int block_size,
                                          unsigned long long cache_stride_bytes,
                                          unsigned int ld_ch) {
    const int t = slot < n_end ? sel[slot] : -1;
    const bool ok = hb_valid(t, seq_len);
    if (ok) {
        const unsigned int lb = (unsigned int)t / block_size;
        const unsigned int p = (unsigned int)t % block_size;
        const unsigned char* src = KV + (unsigned long long)(unsigned int)bt[lb] * cache_stride_bytes
                                   + (unsigned long long)p * HB_KVL;
        #pragma unroll
        for (int k = 0; k < 4; k++) pre[k] = *(const uint4*)(src + (ld_ch + 8u * k) * 16u);
    } else {
        #pragma unroll
        for (int k = 0; k < 4; k++) pre[k] = make_uint4(0u, 0u, 0u, 0u);
    }
    return ok;
}

// 2026-10-14: Widen this thread's gathered FP8 chunks to bf16 into tile buffer `kb`.
__device__ __forceinline__ void hb_widen(unsigned char* kb, const uint4 (&pre)[4],
                                         unsigned int ld_tok, unsigned int ld_ch) {
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned int words[4] = {pre[k].x, pre[k].y, pre[k].z, pre[k].w};
        uint4 lo, hi;
        lo.x = hb_e4m3x2_to_bf16x2((unsigned short)(words[0] & 0xffffu));
        lo.y = hb_e4m3x2_to_bf16x2((unsigned short)(words[0] >> 16));
        lo.z = hb_e4m3x2_to_bf16x2((unsigned short)(words[1] & 0xffffu));
        lo.w = hb_e4m3x2_to_bf16x2((unsigned short)(words[1] >> 16));
        hi.x = hb_e4m3x2_to_bf16x2((unsigned short)(words[2] & 0xffffu));
        hi.y = hb_e4m3x2_to_bf16x2((unsigned short)(words[2] >> 16));
        hi.z = hb_e4m3x2_to_bf16x2((unsigned short)(words[3] & 0xffffu));
        hi.w = hb_e4m3x2_to_bf16x2((unsigned short)(words[3] >> 16));
        const unsigned int ch = 2u * (ld_ch + 8u * k);
        *(uint4*)(kb + hb_swz(ld_tok, ch)) = lo;
        *(uint4*)(kb + hb_swz(ld_tok, ch + 1u)) = hi;
    }
}

// 2026-10-14: One tile's online softmax for head `sm_h`, tokens 4 * sm_sub .. +3 (the 8
// threads of a head hold its state identically): `s4` the scaled scores, `valid` the tile's
// slot flags. Updates (m_run, l_run), writes P (hi, and lo when P_LO) and the head's rescale.
template <bool P_LO>
__device__ __forceinline__ void hb_softmax(const float (&s4)[4], const int* valid,
                                           unsigned int sm_h, unsigned int sm_sub, unsigned int nh,
                                           float& m_run, float& l_run, __nv_bfloat16* s_ph,
                                           __nv_bfloat16* s_pl, float* s_alpha) {
    bool ok[4];
    float tmax = HB_NEG;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        ok[e] = valid[4u * sm_sub + e] != 0 && sm_h < nh;
        if (ok[e]) tmax = fmaxf(tmax, s4[e]);
    }
    #pragma unroll
    for (int o = 1; o < 8; o <<= 1) tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, o));
    const float m_new = fmaxf(m_run, tmax);
    const float alpha = __expf(m_run - m_new);
    float psum = 0.0f;
    unsigned short ph[4], pl[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        const float p = ok[e] ? __expf(s4[e] - m_new) : 0.0f;
        psum += p;
        const __nv_bfloat16 hb = __float2bfloat16(p);
        ph[e] = __bfloat16_as_ushort(hb);
        pl[e] = P_LO ? __bfloat16_as_ushort(__float2bfloat16(p - __bfloat162float(hb)))
                     : (unsigned short)0;
    }
    #pragma unroll
    for (int o = 1; o < 8; o <<= 1) psum += __shfl_xor_sync(0xffffffffu, psum, o);
    l_run = l_run * alpha + psum;
    m_run = m_new;
    uint2 hv, lv;
    hv.x = (unsigned int)ph[0] | ((unsigned int)ph[1] << 16);
    hv.y = (unsigned int)ph[2] | ((unsigned int)ph[3] << 16);
    lv.x = (unsigned int)pl[0] | ((unsigned int)pl[1] << 16);
    lv.y = (unsigned int)pl[2] | ((unsigned int)pl[3] << 16);
    *(uint2*)(s_ph + sm_h * HB_P_STRIDE + 4u * sm_sub) = hv;
    if (P_LO) *(uint2*)(s_pl + sm_h * HB_P_STRIDE + 4u * sm_sub) = lv;
    if (sm_sub == 0) s_alpha[sm_h] = alpha;
}

// 2026-10-14: One tile's P V for warp `warp`'s latent dims [64 warp, 64 warp + 64) as four m16
// tiles: rescale by the heads' alpha, then tokens 0-15, then 16-31, V^T from tile `kb_base`.
template <bool P_LO>
__device__ __forceinline__ void hb_pv(float (&acc)[4][HB_NT][4], unsigned int kb_base,
                                      const __nv_bfloat16* s_ph, const __nv_bfloat16* s_pl,
                                      const float* s_alpha, unsigned int warp, unsigned int lane) {
    const unsigned int g = lane >> 2;
    const unsigned int c = lane & 3u;
    #pragma unroll
    for (int j = 0; j < HB_NT; j++) {
        const float a0 = s_alpha[j * 8 + 2 * c];
        const float a1 = s_alpha[j * 8 + 2 * c + 1];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            acc[i][j][0] *= a0;
            acc[i][j][1] *= a1;
            acc[i][j][2] *= a0;
            acc[i][j][3] *= a1;
        }
    }
    #pragma unroll
    for (int q = 0; q < 2; q++) {
        unsigned int bh[HB_NT][2], bl[HB_NT][2];
        #pragma unroll
        for (int j = 0; j < HB_NT; j++) {
            const unsigned int pr = (j * 8u + g) * HB_P_STRIDE + 16u * q + 2u * c;
            bh[j][0] = *(const unsigned int*)(s_ph + pr);
            bh[j][1] = *(const unsigned int*)(s_ph + pr + 8u);
            bl[j][0] = P_LO ? *(const unsigned int*)(s_pl + pr) : 0u;
            bl[j][1] = P_LO ? *(const unsigned int*)(s_pl + pr + 8u) : 0u;
        }
        const unsigned int v_tok = 16u * q + (lane & 7u) + ((lane >> 4) & 1u) * 8u;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            unsigned int a0, a1, a2, a3;
            hb_ldsm_x4_t(kb_base + hb_swz(v_tok, 8u * warp + 2u * i + ((lane >> 3) & 1u)), a0, a1, a2, a3);
            #pragma unroll
            for (int j = 0; j < HB_NT; j++) {
                hb_mma(acc[i][j], a0, a1, a2, a3, bh[j][0], bh[j][1]);
                if (P_LO) hb_mma(acc[i][j], a0, a1, a2, a3, bl[j][0], bl[j][1]);
            }
        }
    }
}

// 2026-10-14: The end of a split: publish (m, l) per head (a split with no tile publishes
// l = 0), then this warp's dims of O: f32 partials when num_splits > 1, else the normalised
// BF16 output. Every thread of the block must call it.
__device__ __forceinline__ void hb_finish(const float (&acc)[4][HB_NT][4], float m_run,
                                          float l_run, float* s_l, unsigned int sm_h,
                                          unsigned int sm_sub, unsigned int nh, unsigned int row,
                                          unsigned int split, unsigned int head_base,
                                          bool no_tile, unsigned int warp, unsigned int lane,
                                          __nv_bfloat16* __restrict__ O, float* __restrict__ ws_o,
                                          float* __restrict__ ws_ml, unsigned int num_q_heads,
                                          float v_scale, unsigned int num_splits) {
    const unsigned int g = lane >> 2;
    const unsigned int c = lane & 3u;
    if (sm_h < (unsigned int)HB_HEADS && sm_sub == 0) {
        s_l[sm_h] = l_run;
        if (num_splits > 1 && sm_h < nh) {
            float* ml = ws_ml + (((size_t)row * num_splits + split) * num_q_heads + head_base + sm_h) * 2u;
            ml[0] = m_run;
            ml[1] = l_run;
        }
    }
    __syncthreads();
    if (num_splits > 1 && no_tile) return;   // 2026-10-09: nothing to write; merge skips l == 0

    #pragma unroll
    for (int j = 0; j < HB_NT; j++) {
        #pragma unroll
        for (int e = 0; e < 2; e++) {
            const unsigned int h = j * 8u + 2u * c + e;
            if (h >= nh) continue;
            const float l = s_l[h];
            const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                #pragma unroll
                for (int r = 0; r < 2; r++) {
                    const unsigned int dim = 64u * warp + 16u * i + g + 8u * r;
                    const float v = acc[i][j][2 * r + e];
                    if (num_splits > 1) {
                        ws_o[(((size_t)row * num_splits + split) * num_q_heads + head_base + h) * HB_KVL + dim] = v;
                    } else {
                        O[((size_t)row * num_q_heads + head_base + h) * HB_KVL + dim] =
                            __float2bfloat16((v * v_scale) * inv_l);
                    }
                }
            }
        }
    }
}

// 2026-10-14: A row's valid extent, one past its last valid selection entry (the selection is
// a prefix of pools plus a tail with -1 holes). Uses `s_red` (HB_WARPS words); every thread of
// the block must call it, and it ends on a barrier.
__device__ __forceinline__ unsigned int hb_row_extent(const int* __restrict__ sel,
                                                      unsigned int sel_width, unsigned int seq_len,
                                                      unsigned int* s_red, unsigned int tid,
                                                      unsigned int warp, unsigned int lane) {
    unsigned int last = 0;
    for (unsigned int i = tid; i < sel_width; i += HB_THREADS)
        if (hb_valid(sel[i], seq_len)) last = i + 1;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) last = max(last, __shfl_xor_sync(0xffffffffu, last, o));
    if (lane == 0) s_red[warp] = last;
    __syncthreads();
    unsigned int n_end = 0;
    #pragma unroll
    for (int w = 0; w < HB_WARPS; w++) n_end = max(n_end, s_red[w]);
    return n_end;
}

// 2026-10-14: Stage the head group's queries (rows past nh zero) as a swizzled bf16 tile.
__device__ __forceinline__ void hb_stage_q(unsigned char* kb, const __nv_bfloat16* __restrict__ q_row,
                                           unsigned int nh, unsigned int tid) {
    for (unsigned int i = tid; i < (unsigned int)HB_HEADS * HB_CHUNKS; i += HB_THREADS) {
        const unsigned int h = i / HB_CHUNKS, ch = i % HB_CHUNKS;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (h < nh) v = *(const uint4*)(q_row + (size_t)h * HB_KVL + ch * 8u);
        *(uint4*)(kb + hb_swz(h, ch)) = v;
    }
}
