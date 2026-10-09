// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-09-25: KDA (Kimi Delta Attention) recurrent decode: one token, per head h:
//   S[k][v] <- S[k][v] * exp(gate[k])
//   delta[v] = (v[v] - sum_k S[k][v] k[k]) * beta
//   S[k][v] <- S[k][v] + k[k] delta[v]
//   out[v]   = sum_k S[k][v] q[k] * scale
//
// Owner: gb10 kernels.
// Invariants:
// - q, k, v and gate are [H, D] and beta is [H]; the state is FP32 [H, D, D], indexed
//   [k][v] and updated in place; out is FP32 [H, D]. Gate, beta, state and out are FP32 in
//   every entry point.
// - blockIdx.x is the head. The two-pass kernels stride threads over v; the _smem kernel
//   owns VPB columns per block, starting at blockIdx.y * VPB.
// - The caller supplies q and k already L2-normalised (the decode conv,
//   causal_conv1d_update_l2norm, applies it), beta already sigmoided, and
//   scale = 1/sqrt(D).
//
// gate is kda_gate's output, a log-decay, and this kernel applies exp() to it.
// compute_gdn_gates (ssm_preprocess.cu) stores its gate already exponentiated and one per
// head, so a GDN gate passed here would be exponentiated twice, with no error raised.
















































#include <cuda_bf16.h>
#include <math.h>

// 2026-09-25: Dynamic shared memory: 3 * D floats, holding exp(gate), k and q * scale.
#define KDA_REC_BODY(LOAD_QKV)                                                        \
    extern __shared__ float sh[];                                                     \
    const unsigned int h = blockIdx.x;                                                \
    if (h >= H) return;                                                               \
    float* sh_decay = sh;                                                             \
    float* sh_k = sh + D;                                                             \
    float* sh_q = sh + 2u * D;                                                        \
    const size_t hd = (size_t)h * D;                                                  \
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {                       \
        sh_decay[i] = expf(gate[hd + i]);                                             \
        sh_k[i] = LOAD_QKV(k[hd + i]);                                                \
        sh_q[i] = LOAD_QKV(q[hd + i]) * scale;                                        \
    }                                                                                 \
    __syncthreads();                                                                  \
    const float b = beta[h];                                                          \
    float* S = state + hd * D;                                                        \
    for (unsigned int vi = threadIdx.x; vi < D; vi += blockDim.x) {                    \
        float kv = 0.0f;                                                              \
        /* 2026-09-25: Pass 1 decays column vi of S and accumulates kv = sum_k S[k][vi] k[k];\
           pass 2 adds k[k] * delta and accumulates o = sum_k S[k][vi] q[k] * scale.  \
           One thread per column: a warp reads consecutive vi for a fixed kk, so the  \
           state accesses are coalesced. unroll 8 keeps eight independent state loads \
           in flight per thread; it does not reorder the kv and o sums, which run     \
           kk = 0..D-1 in both passes.                                                \
                                                                                      \
                                                                                      \
           */                                                                         \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] * sh_decay[kk];                                    \
            S[idx] = s;                                                               \
            kv += s * sh_k[kk];                                                       \
        }                                                                             \
        const float delta = (LOAD_QKV(v[hd + vi]) - kv) * b;                          \
        float o = 0.0f;                                                               \
        _Pragma("unroll 8")                                                           \
        for (unsigned int kk = 0; kk < D; ++kk) {                                      \
            const size_t idx = (size_t)kk * D + vi;                                   \
            const float s = S[idx] + sh_k[kk] * delta;                                \
            S[idx] = s;                                                               \
            o += s * sh_q[kk];                                                        \
        }                                                                             \
        out[hd + vi] = o;                                                             \
    }

#define KDA_REC_IDENT(x) (x)
#define KDA_REC_BF16(x) __bfloat162float(x)

// 2026-09-25: FP32-input twin, used by the kda_recurrent and kda_chunk microtest examples.
extern "C" __global__ void kda_recurrent_decode_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_IDENT)
}

// 2026-09-25: BF16 q, k and v. glm5next_kda launches this when it does not launch the
// _smem kernel below, with block = min(128, D) and 3 * D floats of shared memory.


extern "C" __global__ void kda_recurrent_decode_bf16(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale
) {
    KDA_REC_BODY(KDA_REC_BF16)
}

// 2026-09-25: Single-pass-over-global variant: the same expressions in the same order as
// kda_recurrent_decode_bf16 (decay, kv over kk = 0..D-1, delta, update, o over kk = 0..D-1),
// but each thread keeps its decayed column in shared memory between the two passes, so the
// state is read once and written once from global memory.
//
// The v axis has no cross-thread dependency (kv, delta and o are per (h, vi)), so a block
// owns VPB columns and grid.y covers D / VPB. Shared memory: 3 * D + VPB * (D + 1) floats.
// The launcher must make VPB divide D and launch blockDim.x == VPB; otherwise columns are
// dropped with no error. glm5next_kda checks both, uses this kernel only when the target
// has it and the request fits KDA_SMEM_BUDGET, and skips it when
// METRALE_GLM_KDA_NO_SMEM=1.















// 2026-10-09: The body of kda_recurrent_decode_bf16_smem, for one sequence's row; the
// _rows entry below runs it for several sequences, each with its own state.
__device__ __forceinline__ void kda_recurrent_decode_bf16_smem_body(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB
) {
    extern __shared__ float sh_rec[];
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    const unsigned int v0 = blockIdx.y * VPB;
    if (v0 >= D) return;

    float* sh_decay = sh_rec;
    float* sh_k = sh_rec + D;
    float* sh_q = sh_rec + 2u * D;
// 2026-09-25: [VPB, D + 1]: column threadIdx.x of this block's slice, k-major. The +1 pad
// avoids bank conflicts: at a stride of D = 128 floats every thread of a warp would hit
// the same bank (128 % 32 == 0); a stride of D + 1 moves each thread to the next bank.



    float* sh_s = sh_rec + 3u * D;
    const unsigned int col_stride = D + 1u;

    const size_t hd = (size_t)h * D;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sh_decay[i] = expf(gate[hd + i]);
        sh_k[i] = __bfloat162float(k[hd + i]);
        sh_q[i] = __bfloat162float(q[hd + i]) * scale;
    }
    __syncthreads();

    const float b = beta[h];
    float* S = state + hd * D;
    const unsigned int vi = v0 + threadIdx.x;
    if (threadIdx.x >= VPB || vi >= D) return;
    float* col = sh_s + (size_t)threadIdx.x * col_stride;

    float kv = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll 8
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}

extern "C" __global__ void kda_recurrent_decode_bf16_smem(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB
) {
    kda_recurrent_decode_bf16_smem_body(q, k, v, gate, beta, state, out, H, D, scale, VPB);
}

// 2026-10-09: kda_recurrent_decode_bf16_smem for up to KDA_ROWS_MAX rows of different
// sequences in one launch: grid (H, D / VPB, rows), and block (h, v-slice, r) is the block
// (h, v-slice) of the single-row kernel on row r, so each row's state and output are the
// single-row launch's. Grid z index r takes workspace row w<r> (2026-10-09: a batched verify
// steps row t of every sequence in one launch, and those rows are not adjacent): it reads
// q/k/v at w<r> * qkv_row_stride elements past the given bases, gate at w<r> *
// gate_row_stride, beta at w<r> * beta_row_stride, writes out at w<r> * out_row_stride, and
// updates the state at s<r>. The per-row state pointers are kernel
// arguments, so a captured graph keeps the ones it was captured with; a null pointer skips
// the row. The launcher must pass exactly KDA_ROWS_MAX state arguments.
#define KDA_ROWS_MAX 16
#include "rows_pick.cuh"
extern "C" __global__ void kda_recurrent_decode_bf16_smem_rows(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ out,
    unsigned int H,
    unsigned int D,
    float scale,
    unsigned int VPB,
    unsigned int qkv_row_stride,
    unsigned int gate_row_stride,
    unsigned int beta_row_stride,
    unsigned int out_row_stride,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long s4, unsigned long long s5, unsigned long long s6, unsigned long long s7,
    unsigned long long s8, unsigned long long s9, unsigned long long s10, unsigned long long s11,
    unsigned long long s12, unsigned long long s13, unsigned long long s14, unsigned long long s15,
    unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3,
    unsigned int w4, unsigned int w5, unsigned int w6, unsigned int w7,
    unsigned int w8, unsigned int w9, unsigned int w10, unsigned int w11,
    unsigned int w12, unsigned int w13, unsigned int w14, unsigned int w15
) {
    const unsigned int r = blockIdx.z;
    if (r >= KDA_ROWS_MAX) return;
    float* state = (float*)rows_pick16(r, s0, s1, s2, s3, s4, s5, s6, s7,
                                       s8, s9, s10, s11, s12, s13, s14, s15);
    if (state == nullptr) return;
    const size_t w = rows_pick16(r, w0, w1, w2, w3, w4, w5, w6, w7,
                                 w8, w9, w10, w11, w12, w13, w14, w15);
    kda_recurrent_decode_bf16_smem_body(
        q + w * qkv_row_stride,
        k + w * qkv_row_stride,
        v + w * qkv_row_stride,
        gate + w * gate_row_stride,
        beta + w * beta_row_stride,
        state,
        out + w * out_row_stride,
        H, D, scale, VPB);
}

// 2026-10-09: The rows kernel with the decayed column in registers instead of shared memory.
// The smem kernel keeps (D + 1) floats of column per thread in shared memory, which limits a
// GB10 SM to a few resident warps (18,048 B per 32-thread block at D = 128); here head_dim is
// the compile-time KDA_REG_D, both kk loops are unrolled, the column lives in KDA_REG_D
// registers, and shared memory holds only the 3 * D staged vectors. Per (h, vi, row) the
// expressions and their order are kda_recurrent_decode_bf16_smem_body's: the same staging, the
// same pass-one products and kv sum, the same delta, the same pass-two update and o sum. Launch:
// grid (H, 1, rows), block KDA_REG_D (one thread per v column of the head), 3 * D floats of
// dynamic shared memory; D must equal KDA_REG_D and the state and row arguments are as for
// _rows.
#define KDA_REG_D 128

extern "C" __global__ void kda_recurrent_decode_bf16_rows_reg(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ out,
    unsigned int H,
    float scale,
    unsigned int qkv_row_stride,
    unsigned int gate_row_stride,
    unsigned int beta_row_stride,
    unsigned int out_row_stride,
    unsigned long long s0, unsigned long long s1, unsigned long long s2, unsigned long long s3,
    unsigned long long s4, unsigned long long s5, unsigned long long s6, unsigned long long s7,
    unsigned long long s8, unsigned long long s9, unsigned long long s10, unsigned long long s11,
    unsigned long long s12, unsigned long long s13, unsigned long long s14, unsigned long long s15,
    unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3,
    unsigned int w4, unsigned int w5, unsigned int w6, unsigned int w7,
    unsigned int w8, unsigned int w9, unsigned int w10, unsigned int w11,
    unsigned int w12, unsigned int w13, unsigned int w14, unsigned int w15
) {
    const unsigned int r = blockIdx.z;
    const unsigned int h = blockIdx.x;
    if (r >= KDA_ROWS_MAX || h >= H) return;
    float* state = (float*)rows_pick16(r, s0, s1, s2, s3, s4, s5, s6, s7,
                                       s8, s9, s10, s11, s12, s13, s14, s15);
    if (state == nullptr) return;
    const size_t w = rows_pick16(r, w0, w1, w2, w3, w4, w5, w6, w7,
                                 w8, w9, w10, w11, w12, w13, w14, w15);
    q += w * qkv_row_stride;
    k += w * qkv_row_stride;
    v += w * qkv_row_stride;
    gate += w * gate_row_stride;
    beta += w * beta_row_stride;
    out += w * out_row_stride;

    constexpr unsigned int D = KDA_REG_D;
    extern __shared__ float sh_reg[];
    float* sh_decay = sh_reg;
    float* sh_k = sh_reg + D;
    float* sh_q = sh_reg + 2u * D;
    const size_t hd = (size_t)h * D;
    for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
        sh_decay[i] = expf(gate[hd + i]);
        sh_k[i] = __bfloat162float(k[hd + i]);
        sh_q[i] = __bfloat162float(q[hd + i]) * scale;
    }
    __syncthreads();

    const float b = beta[h];
    float* S = state + hd * D;
    const unsigned int vi = threadIdx.x;
    if (vi >= D) return;

    float col[D];
    float kv = 0.0f;
    #pragma unroll
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = S[(size_t)kk * D + vi] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (__bfloat162float(v[hd + vi]) - kv) * b;
    float o = 0.0f;
    #pragma unroll
    for (unsigned int kk = 0; kk < D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        S[(size_t)kk * D + vi] = s;
        o += s * sh_q[kk];
    }
    out[hd + vi] = o;
}

// 2026-10-09: One token of the register-resident step on one v column: `col` holds the column
// S[:, vi] on entry and the updated column on exit; the staged decay, k and scaled q are the
// token's. The expressions and their order are kda_recurrent_decode_bf16_smem_body's pass one and
// pass two, with the column read from and written to registers instead of S in memory.
// _rows_reg keeps its own copy with the state loads and stores inside the two passes: built
// on this helper (load, step, store) it needed 255 registers and spilled.
__device__ __forceinline__ float kda_reg_token(
    float (&col)[KDA_REG_D],
    const float* __restrict__ sh_decay,
    const float* __restrict__ sh_k,
    const float* __restrict__ sh_q,
    float v,
    float b
) {
    float kv = 0.0f;
    #pragma unroll
    for (unsigned int kk = 0; kk < KDA_REG_D; ++kk) {
        const float s = col[kk] * sh_decay[kk];
        col[kk] = s;
        kv += s * sh_k[kk];
    }
    const float delta = (v - kv) * b;
    float o = 0.0f;
    #pragma unroll
    for (unsigned int kk = 0; kk < KDA_REG_D; ++kk) {
        const float s = col[kk] + sh_k[kk] * delta;
        col[kk] = s;
        o += s * sh_q[kk];
    }
    return o;
}

// 2026-10-09: T consecutive tokens of ONE sequence in one launch, the state column kept in
// registers between tokens: grid (H, KDA_REG_D / VB), block VB (VB divides KDA_REG_D; one
// thread per v column), 3 * KDA_REG_D floats of dynamic shared memory. Token t reads q/k/v at
// t * qkv_row_stride elements past the bases, gate at t * gate_row_stride, beta at
// t * beta_row_stride, and writes out at t * out_row_stride. Per (h, vi, t) the arithmetic is
// the per-token kernels' (kda_reg_token; the staging below is _rows_reg's), on the column the
// previous token left, so every output and the final state equal T per-token launches. The
// state is read once before token 0 and written once after token T - 1.
extern "C" __global__ void kda_recurrent_decode_bf16_seq_reg(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state,
    float* __restrict__ out,
    unsigned int H,
    float scale,
    unsigned int T,
    unsigned int qkv_row_stride,
    unsigned int gate_row_stride,
    unsigned int beta_row_stride,
    unsigned int out_row_stride
) {
    constexpr unsigned int D = KDA_REG_D;
    const unsigned int h = blockIdx.x;
    if (h >= H) return;
    extern __shared__ float sh_seq[];
    float* sh_decay = sh_seq;
    float* sh_k = sh_seq + D;
    float* sh_q = sh_seq + 2u * D;
    const size_t hd = (size_t)h * D;
    const unsigned int vi = blockIdx.y * blockDim.x + threadIdx.x;
    const bool live = vi < D;
    float* S = state + hd * D;

    float col[D];
    if (live) {
        #pragma unroll
        for (unsigned int kk = 0; kk < D; ++kk) col[kk] = S[(size_t)kk * D + vi];
    }
    for (unsigned int t = 0; t < T; ++t) {
        const size_t qo = (size_t)t * qkv_row_stride + hd;
        const size_t go = (size_t)t * gate_row_stride + hd;
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < D; i += blockDim.x) {
            sh_decay[i] = expf(gate[go + i]);
            sh_k[i] = __bfloat162float(k[qo + i]);
            sh_q[i] = __bfloat162float(q[qo + i]) * scale;
        }
        __syncthreads();
        if (live) {
            const float b = beta[(size_t)t * beta_row_stride + h];
            out[(size_t)t * out_row_stride + hd + vi] =
                kda_reg_token(col, sh_decay, sh_k, sh_q, __bfloat162float(v[qo + vi]), b);
        }
    }
    if (live) {
        #pragma unroll
        for (unsigned int kk = 0; kk < D; ++kk) S[(size_t)kk * D + vi] = col[kk];
    }
}
