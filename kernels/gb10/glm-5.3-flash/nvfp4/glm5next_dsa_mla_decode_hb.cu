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

extern "C" __global__ void __launch_bounds__(HB_THREADS, 1) glm5next_dsa_mla_decode_hb_fp8(
    const __nv_bfloat16* __restrict__ Q, const unsigned char* __restrict__ KV,
    __nv_bfloat16* __restrict__ O, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, const int* __restrict__ sel_indices,
    float* __restrict__ ws_o, float* __restrict__ ws_ml, const unsigned int sel_width,
    const unsigned int max_blocks_per_seq, const unsigned int num_q_heads,
    const unsigned int block_size, const float inv_sqrt_d, const float k_scale,
    const float v_scale, const unsigned long long cache_stride_bytes,
    const unsigned int num_splits) {
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

    // 2026-10-09: Stage this head group's queries in the tile buffer, rows past nh zero.
    const __nv_bfloat16* q_row = Q + ((size_t)row * num_q_heads + head_base) * HB_KVL;
    for (unsigned int i = tid; i < (unsigned int)HB_HEADS * HB_CHUNKS; i += HB_THREADS) {
        const unsigned int h = i / HB_CHUNKS, ch = i % HB_CHUNKS;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (h < nh) v = *(const uint4*)(q_row + (size_t)h * HB_KVL + ch * 8u);
        *(uint4*)(kb + hb_swz(h, ch)) = v;
    }

    // 2026-10-09: The row's valid extent: one past its last valid entry. The selection is
    // a prefix of pools plus a tail with -1 holes, so the split covers [0, n_end).
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

    // 2026-10-09: Loader geometry: thread -> token (tid >> 3) of the tile, FP8 16-byte chunks
    // (tid & 7) + 8k, k = 0..3.
    const unsigned int ld_tok = tid >> 3;
    const unsigned int ld_ch = tid & 7u;
    uint4 pre[4];
    bool pre_ok = false;
    auto load_tile = [&](unsigned int tile) {
        const unsigned int slot = tile * HB_TT + ld_tok;
        const int t = slot < n_end ? sel[slot] : -1;
        pre_ok = hb_valid(t, seq_len);
        if (pre_ok) {
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
            bool ok[4];
            float tmax = HB_NEG;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                ok[e] = s_valid[4u * sm_sub + e] != 0 && sm_h < nh;
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
                const __nv_bfloat16 lb = __float2bfloat16(p - __bfloat162float(hb));
                ph[e] = __bfloat16_as_ushort(hb);
                pl[e] = __bfloat16_as_ushort(lb);
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
            *(uint2*)(s_pl + sm_h * HB_P_STRIDE + 4u * sm_sub) = lv;
            if (sm_sub == 0) s_alpha[sm_h] = alpha;
        }
        __syncthreads();

        // 2026-10-09: PV: warp w owns latent dims [64w, 64w + 64) as four m16 tiles.
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
                bl[j][0] = *(const unsigned int*)(s_pl + pr);
                bl[j][1] = *(const unsigned int*)(s_pl + pr + 8u);
            }
            const unsigned int v_tok = 16u * q + (lane & 7u) + ((lane >> 4) & 1u) * 8u;
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                unsigned int a0, a1, a2, a3;
                hb_ldsm_x4_t(kb_base + hb_swz(v_tok, 8u * warp + 2u * i + ((lane >> 3) & 1u)), a0, a1, a2, a3);
                #pragma unroll
                for (int j = 0; j < HB_NT; j++) {
                    hb_mma(acc[i][j], a0, a1, a2, a3, bh[j][0], bh[j][1]);
                    hb_mma(acc[i][j], a0, a1, a2, a3, bl[j][0], bl[j][1]);
                }
            }
        }
    }

    // 2026-10-09: Publish (m, l) per head; a split with no tile publishes l = 0.
    if (sm_h < (unsigned int)HB_HEADS && sm_sub == 0) {
        s_l[sm_h] = l_run;
        if (num_splits > 1 && sm_h < nh) {
            float* ml = ws_ml + (((size_t)row * num_splits + split) * num_q_heads + head_base + sm_h) * 2u;
            ml[0] = m_run;
            ml[1] = l_run;
        }
    }
    __syncthreads();
    if (num_splits > 1 && t_begin >= t_end) return;   // 2026-10-09: nothing to write; merge skips l == 0

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
