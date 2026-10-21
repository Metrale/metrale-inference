// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: The screened LM head: the greedy argmax of the BF16 head (dense_gemv_bf16_batchm /
// dense_gemv_bf16 per row, then argmax_bf16 over the whole row) without reading the whole BF16
// head. An INT8 copy of this rank's head rows with one FP32 scale per 128 weights is read
// instead; a rigorous per-row bound on |BF16-path logit - screen logit| keeps every row that can
// still be the argmax, and those rows are recomputed with the BF16 kernel's own arithmetic, so the
// token is the same one, ties included. The bound's derivation is the host specification
// crates/model-layers/src/layers/screen_head_bound.rs (`screen_kappa`, `screen_pad`).
//
// Owner: gb10 kernels.
// Invariants:
// - K % 512 == 0 and M * K <= SCREEN_MAX_MK (the host wrappers refuse anything else).
// - The state block (SCREEN_STATE_BYTES) is zeroed before screen_head_scan; the scan, rescore and
//   resolve of one call read and write only it, `upper` and their outputs.
// - screen_exact_dot is dense_gemv_bf16_batchm_body's per-row arithmetic for one row with one
//   warp: lane t keeps the accumulators of the reference's lanes t and t + 32, in the same kv order
//   and add order, and both are folded by the same shuffle tree before the cross-warp add.
//   Its result is therefore the reference's FP32 sum, bit for bit (the family builds with
//   --fmad=false, and every operation here is an explicit _rn intrinsic besides).
// - The rescore keys a row with argmax_key.cuh's plain_key of its BF16-rounded value, so the max
//   key over all ranks names argmax_bf16's token for the full row.

#include <cuda_bf16.h>
#include <math_constants.h>

#include "argmax_key.cuh"

#define SCREEN_BLK 128u
#define SCREEN_MAX_M 4u
#define SCREEN_MAX_MK 16384u
#define SCREEN_WARPS 8u

// 2026-10-21: State block offsets (bytes): key pairs u64[4][2] (p, q) @0, L cells u32[4] @64,
// masked L cells u32[4] @80, full flags u32[4] @96, candidate counts u32[4] @112. Mirrored by
// SCREEN_STATE_* in the host wrappers. A row's pair is argmax_key.cu's [p, q] layout, so a one-row
// state feeds argmax_key_resolve directly.
struct ScreenState {
    unsigned long long keys[SCREEN_MAX_M][2];
    unsigned int lcell[SCREEN_MAX_M];
    unsigned int lcell_q[SCREEN_MAX_M];
    unsigned int full[SCREEN_MAX_M];
    unsigned int count[SCREEN_MAX_M];
};

// 2026-10-21: The mask ids (argmax_key.cu's masks[2]); none for a NULL `masks`.
__device__ __forceinline__ void screen_masks(
    const unsigned int* __restrict__ masks, unsigned int* m0, unsigned int* m1) {
    *m0 = masks != nullptr ? masks[0] : 0xFFFFFFFFu;
    *m1 = masks != nullptr ? masks[1] : 0xFFFFFFFFu;
}

__device__ __forceinline__ float screen_bf16_bits(unsigned int b16) {
    return __uint_as_float(b16 << 16);
}

// 2026-10-21: The decode of key_order_bits (argmax_key.cuh): the float a lower-bound cell holds.
__device__ __forceinline__ float screen_unorder(unsigned int k) {
    const unsigned int b = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return __uint_as_float(b);
}

// 2026-10-21: One weight vector (8 BF16) times the matching activations, added into `a` in the
// reference's order: lo then hi of each 32-bit word, each product rounded then added.
__device__ __forceinline__ float screen_ref_step(float a, uint4 w, uint4 x) {
    const unsigned int wr[4] = {w.x, w.y, w.z, w.w};
    const unsigned int xr[4] = {x.x, x.y, x.z, x.w};
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const float wl = screen_bf16_bits(wr[i] & 0xFFFFu);
        const float wh = screen_bf16_bits(wr[i] >> 16);
        const float xl = screen_bf16_bits(xr[i] & 0xFFFFu);
        const float xh = screen_bf16_bits(xr[i] >> 16);
        a = __fadd_rn(a, __fmul_rn(xl, wl));
        a = __fadd_rn(a, __fmul_rn(xh, wh));
    }
    return a;
}

// 2026-10-21: The reference's FP32 dot product of one head row with one activation row, by a
// whole warp; every lane returns it.
__device__ __forceinline__ float screen_exact_dot(
    const __nv_bfloat16* __restrict__ w_row,
    const __nv_bfloat16* __restrict__ x_row,
    unsigned int K,
    unsigned int lane
) {
    const unsigned int kvec = K / 8u;
    const uint4* w = (const uint4*)w_row;
    const uint4* x = (const uint4*)x_row;
    float a0 = 0.0f;
    float a1 = 0.0f;
    for (unsigned int base = 0; base < kvec; base += 64u) {
        const unsigned int k0 = base + lane;
        const unsigned int k1 = base + 32u + lane;
        if (k0 < kvec) {
            a0 = screen_ref_step(a0, w[k0], x[k0]);
        }
        if (k1 < kvec) {
            a1 = screen_ref_step(a1, w[k1], x[k1]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a0 = __fadd_rn(a0, __shfl_down_sync(0xFFFFFFFFu, a0, off));
        a1 = __fadd_rn(a1, __shfl_down_sync(0xFFFFFFFFu, a1, off));
    }
    const float r = __fadd_rn(a0, a1);
    return __shfl_sync(0xFFFFFFFFu, r, 0);
}

// 2026-10-21: Load-time build of the screen copy: one warp per (row, 128-weight block).
// s = max|w| / 127 and q = rint(w / s), the quotient taken in FP64 so q is the nearest point.
// `bad` gets bit 0 for a non-finite weight and bit 1 for a block whose exact residual
// max|w - s q| exceeds s/2 (1 + 2^-20), the bound the scan assumes.
extern "C" __global__ void screen_head_quant_int8(
    const __nv_bfloat16* __restrict__ W,
    signed char* __restrict__ Q,
    float* __restrict__ S,
    unsigned int* __restrict__ bad,
    unsigned int rows,
    unsigned int K
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int nb = K / SCREEN_BLK;
    const unsigned long long blk =
        (unsigned long long)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (blk >= (unsigned long long)rows * nb) {
        return;
    }
    const unsigned long long off = blk * SCREEN_BLK + lane * 4u;
    const uint2 raw = *(const uint2*)(W + off);
    float w[4] = {screen_bf16_bits(raw.x & 0xFFFFu), screen_bf16_bits(raw.x >> 16),
                  screen_bf16_bits(raw.y & 0xFFFFu), screen_bf16_bits(raw.y >> 16)};
    float amax = 0.0f;
    bool finite = true;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        finite = finite && (fabsf(w[i]) <= 3.0e38f);
        amax = fmaxf(amax, fabsf(w[i]));
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    }
    const float s = __fdiv_rn(amax, 127.0f);
    char4 q;
    signed char* qp = (signed char*)&q;
    bool tight = true;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        double r = s > 0.0f ? rint((double)w[i] / (double)s) : 0.0;
        r = fmin(fmax(r, -127.0), 127.0);
        qp[i] = (signed char)(int)r;
        const double res = fabs((double)w[i] - (double)s * (double)r);
        tight = tight && (res <= 0.5 * (double)s * (1.0 + 0x1p-20));
    }
    *(char4*)(Q + off) = q;
    if (lane == 0) {
        S[blk] = s;
    }
    if (!finite) {
        atomicOr(bad, 1u);
    }
    if (!tight) {
        atomicOr(bad, 2u);
    }
}

// 2026-10-21: The screen: per row and activation row, S = sum_b s_b sum_k x_k q_k and
// eps = kappa * sum_b s_b |x_b|_1 + 2^-100, `upper` = S + eps rounded up, and the max over the
// rows of S - eps rounded down into the L cell (key_order_bits), and the same max over the rows
// other than the mask ids (global index `base + row`) into the masked L cell. A non-finite S or
// eps, or one past 2^100, sets the row's full flag: every row is then a candidate. One warp per
// row; blockDim (32 * SCREEN_WARPS).
extern "C" __global__ void __launch_bounds__(256) screen_head_scan(
    const signed char* __restrict__ Q,
    const float* __restrict__ S,
    const __nv_bfloat16* __restrict__ X,
    unsigned int M,
    unsigned int rows,
    unsigned int K,
    float kappa,
    float* __restrict__ upper,
    ScreenState* __restrict__ st,
    const unsigned int* __restrict__ masks,
    unsigned int base
) {
    __shared__ uint4 xs[SCREEN_MAX_MK / 8u];
    __shared__ float xl1[SCREEN_MAX_MK / 16u];
    __shared__ float lo_max[SCREEN_WARPS][SCREEN_MAX_M];
    __shared__ float lo_qmax[SCREEN_WARPS][SCREEN_MAX_M];
    unsigned int m0;
    unsigned int m1;
    screen_masks(masks, &m0, &m1);
    const unsigned int nvec = M * K / 8u;
    for (unsigned int i = threadIdx.x; i < nvec; i += blockDim.x) {
        xs[i] = ((const uint4*)X)[i];
    }
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < nvec / 2u; c += blockDim.x) {
        float a = 0.0f;
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            const uint4 v = xs[2u * c + h];
            const unsigned int r[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                a = __fadd_ru(a, fabsf(screen_bf16_bits(r[i] & 0xFFFFu)));
                a = __fadd_ru(a, fabsf(screen_bf16_bits(r[i] >> 16)));
            }
        }
        xl1[c] = a;
    }
    __syncthreads();

    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int nchunk = K / 16u;
    const unsigned int nb = K / SCREEN_BLK;
    float my_lo[SCREEN_MAX_M];
    float my_lo_q[SCREEN_MAX_M];
    #pragma unroll
    for (int m = 0; m < (int)SCREEN_MAX_M; m++) {
        my_lo[m] = -CUDART_INF_F;
        my_lo_q[m] = -CUDART_INF_F;
    }

    for (unsigned int row = blockIdx.x * SCREEN_WARPS + warp; row < rows;
         row += gridDim.x * SCREEN_WARPS) {
        const uint4* qrow = (const uint4*)(Q + (unsigned long long)row * K);
        const float* srow = S + (unsigned long long)row * nb;
        float acc[SCREEN_MAX_M];
        float err[SCREEN_MAX_M];
        #pragma unroll
        for (int m = 0; m < (int)SCREEN_MAX_M; m++) {
            acc[m] = 0.0f;
            err[m] = 0.0f;
        }
        for (unsigned int c0 = lane; c0 < nchunk; c0 += 256u) {
            uint4 qv[8];
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned int c = c0 + 32u * j;
                qv[j] = c < nchunk ? __ldg(qrow + c) : make_uint4(0u, 0u, 0u, 0u);
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                const unsigned int c = c0 + 32u * j;
                if (c >= nchunk) {
                    continue;
                }
                const float s = __ldg(srow + c / 8u);
                // 2026-10-21: q as 2^23 + (q + 128) by a byte permute, less 2^23 + 128: exact,
                // and a full-rate add instead of an integer conversion.
                const unsigned int qw[4] = {qv[j].x ^ 0x80808080u, qv[j].y ^ 0x80808080u,
                                            qv[j].z ^ 0x80808080u, qv[j].w ^ 0x80808080u};
                float qf[16];
                #pragma unroll
                for (int i = 0; i < 16; i++) {
                    const unsigned int sel = 0x7540u | (unsigned int)(i & 3);
                    qf[i] = __fsub_rn(__uint_as_float(__byte_perm(qw[i >> 2], 0x4B000000u, sel)),
                                      8388736.0f);
                }
                #pragma unroll
                for (int m = 0; m < (int)SCREEN_MAX_M; m++) {
                    if (m >= (int)M) {
                        break;
                    }
                    const uint4 x0 = xs[(m * K) / 8u + 2u * c];
                    const uint4 x1 = xs[(m * K) / 8u + 2u * c + 1u];
                    const unsigned int xr[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                    float p = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 8; i++) {
                        p = __fmaf_rn(screen_bf16_bits(xr[i] & 0xFFFFu), qf[2 * i], p);
                        p = __fmaf_rn(screen_bf16_bits(xr[i] >> 16), qf[2 * i + 1], p);
                    }
                    acc[m] = __fmaf_rn(s, p, acc[m]);
                    err[m] = __fmaf_ru(s, xl1[m * nchunk + c], err[m]);
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < (int)SCREEN_MAX_M; m++) {
            if (m >= (int)M) {
                break;
            }
            float a = acc[m];
            float e = err[m];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                a = __fadd_rn(a, __shfl_xor_sync(0xFFFFFFFFu, a, o));
                e = __fadd_ru(e, __shfl_xor_sync(0xFFFFFFFFu, e, o));
            }
            const float eps = __fadd_ru(__fmul_ru(kappa, e), 0x1p-100f);
            const bool ok = fabsf(a) <= 0x1p100f && eps <= 0x1p100f;
            const float up = ok ? __fadd_ru(a, eps) : CUDART_INF_F;
            const float lo = ok ? __fsub_rd(a, eps) : -CUDART_INF_F;
            if (lane == 0) {
                upper[(unsigned long long)m * rows + row] = up;
                if (!ok) {
                    atomicOr(&st->full[m], 1u);
                }
            }
            my_lo[m] = fmaxf(my_lo[m], lo);
            const unsigned int gi = base + row;
            if (gi != m0 && gi != m1) {
                my_lo_q[m] = fmaxf(my_lo_q[m], lo);
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < (int)SCREEN_MAX_M; m++) {
            lo_max[warp][m] = my_lo[m];
            lo_qmax[warp][m] = my_lo_q[m];
        }
    }
    __syncthreads();
    if (threadIdx.x < M) {
        float l = -CUDART_INF_F;
        float lq = -CUDART_INF_F;
        for (unsigned int w = 0; w < SCREEN_WARPS; w++) {
            l = fmaxf(l, lo_max[w][threadIdx.x]);
            lq = fmaxf(lq, lo_qmax[w][threadIdx.x]);
        }
        atomicMax(&st->lcell[threadIdx.x], key_order_bits(l));
        atomicMax(&st->lcell_q[threadIdx.x], key_order_bits(lq));
    }
}

// 2026-10-21: The rescore: every row whose `upper` reaches L - pad (pad = 2^-7 (1 + 2^-7) |L| +
// 2^-120, the BF16 tie margin) for the masked L cell (at most the plain one, and L - pad is
// monotone in L, so the plain threshold's rows are among them), or every row under a full flag,
// is recomputed with screen_exact_dot, rounded to BF16 and keyed at its global index `base + row`
// with argmax_key.cuh's plain_key and masked_key; the max keys go to keys[m]. One warp per 32-row
// group; blockDim (32 * SCREEN_WARPS).
extern "C" __global__ void __launch_bounds__(256) screen_head_rescore(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ X,
    unsigned int M,
    unsigned int rows,
    unsigned int K,
    unsigned int base,
    const float* __restrict__ upper,
    ScreenState* __restrict__ st,
    const unsigned int* __restrict__ masks
) {
    const unsigned int lane = threadIdx.x & 31u;
    const unsigned int groups = (rows + 31u) / 32u;
    unsigned int m0;
    unsigned int m1;
    screen_masks(masks, &m0, &m1);
    for (unsigned int m = 0; m < M; m++) {
        const bool full = st->full[m] != 0u;
        const float L = screen_unorder(st->lcell_q[m]);
        const float pad = __fmaf_ru(fabsf(L), 0x1.02p-7f, 0x1p-120f);
        const float thr = __fsub_rd(L, pad);
        const __nv_bfloat16* x = X + (unsigned long long)m * K;
        for (unsigned int g = blockIdx.x * SCREEN_WARPS + (threadIdx.x >> 5); g < groups;
             g += gridDim.x * SCREEN_WARPS) {
            const unsigned int row = g * 32u + lane;
            const bool cand =
                row < rows && (full || upper[(unsigned long long)m * rows + row] >= thr);
            unsigned int bits = __ballot_sync(0xFFFFFFFFu, cand);
            if (bits == 0u) {
                continue;
            }
            if (lane == 0) {
                atomicAdd(&st->count[m], (unsigned int)__popc(bits));
            }
            unsigned long long best = 0ull;
            unsigned long long best_q = 0ull;
            while (bits != 0u) {
                const unsigned int j = __ffs(bits) - 1u;
                bits &= bits - 1u;
                const unsigned int r = g * 32u + j;
                const float v = screen_exact_dot(W + (unsigned long long)r * K, x, K, lane);
                const float b = __bfloat162float(__float2bfloat16(v));
                const unsigned long long k = plain_key(b, base + r);
                const unsigned long long kq = masked_key(b, base + r, m0, m1);
                best = k > best ? k : best;
                best_q = kq > best_q ? kq : best_q;
            }
            if (lane == 0 && best != 0ull) {
                atomicMax(&st->keys[m][0], best);
            }
            if (lane == 0 && best_q != 0ull) {
                atomicMax(&st->keys[m][1], best_q);
            }
        }
    }
}

// 2026-10-21: out[m] = the token named by the max of this rank's p key keys[m][0] and the
// `n_peers` peer key blocks a one-shot exchange left at `peers + s * slot_bytes` (their keys at
// the same offsets). One thread per activation row. (A masked one-row pick resolves through
// argmax_key_resolve over the same pair.)
extern "C" __global__ void screen_head_resolve(
    const ScreenState* __restrict__ st,
    const unsigned char* __restrict__ peers,
    unsigned long long slot_bytes,
    unsigned int n_peers,
    unsigned int M,
    unsigned int* __restrict__ out
) {
    const unsigned int m = threadIdx.x;
    if (blockIdx.x != 0 || m >= M) {
        return;
    }
    unsigned long long kp = st->keys[m][0];
    for (unsigned int s = 0; s < n_peers; ++s) {
        const unsigned long long* pk =
            (const unsigned long long*)(peers + (unsigned long long)s * slot_bytes);
        kp = pk[2u * m] > kp ? pk[2u * m] : kp;
    }
    out[m] = plain_key_index(kp);
}

// 2026-10-21: Bytes of the stamp a screened verify leaves at the head of the logits buffer
// (SCREEN_STAMP_BYTES on the host): 0xFF bytes, BF16 NaNs no head writes.
#define SCREEN_STAMP_WORDS 4u

// 2026-10-21: out[0] = the BF16 logit the full head writes at row `row`, id `id`, for the lazy
// logits of a screened verify: recomputed from head row `id` of the whole head `W` and activation
// row `row` of `X` while `stamp` still holds the verify's stamp (or `stamp` is NULL), else copied
// from `logits` (a later head call has written the row). One warp.
extern "C" __global__ void screen_head_exact_id(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ X,
    unsigned int K,
    unsigned int row,
    unsigned int id,
    unsigned int vocab,
    const __nv_bfloat16* __restrict__ logits,
    const unsigned int* __restrict__ stamp,
    __nv_bfloat16* __restrict__ out
) {
    const unsigned int lane = threadIdx.x & 31u;
    bool lazy = true;
    if (stamp != nullptr) {
        const unsigned int v = lane < SCREEN_STAMP_WORDS ? stamp[lane] : 0xFFFFFFFFu;
        lazy = __all_sync(0xFFFFFFFFu, v == 0xFFFFFFFFu);
    }
    if (!lazy) {
        if (lane == 0) {
            out[0] = logits[(unsigned long long)row * vocab + id];
        }
        return;
    }
    const float v = screen_exact_dot(
        W + (unsigned long long)id * K, X + (unsigned long long)row * K, K, lane);
    if (lane == 0) {
        out[0] = __float2bfloat16(v);
    }
}
