// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: The KDA stateful step of a batched speculative verify: up to KDA_SEQS_MAX
// sequences, each with its own run of consecutive workspace rows, in three launches (conv, conv
// window advance, recurrence) instead of a conv and a recurrent launch per row index. Grid z
// picks the sequence; each sequence is the per-sequence token kernels' launch
// (causal_conv1d_update_l2norm_tokens, causal_conv1d_window_advance,
// kda_recurrent_decode_bf16_seq_reg_snap) on its own rows and state, so its bits are theirs, and
// those are the per-row walk's (glm5next_kda/seq_tokens.rs).
//
// Owner: gb10 kernels.
// Invariants:
// - Sequence r: state pointers c<r> (conv window) and s<r> (recurrent), first workspace row
//   w<r>, rows n<r>. A null state pointer skips the sequence. No two sequences share a state.
// - Snapshot t of sequence r, p<t>_<r> / sp<t>_<r> (nullable, t < KDA_SEQS_SNAP_MAX), is the state
//   after the sequence's row t: the recurrent state the per-row walk holds after row t, and the
//   conv window it holds then (2026-10-16; before, row 0 only). Stored wherever set, t < n<r>.
// - The state reads and writes of a sequence are its own launch's; nothing crosses sequences.

#include "causal_conv1d_l2norm_body.cuh"
#include "kda_reg_token.cuh"
#include "rows_pick.cuh"

#define KDA_SEQS_MAX 16
// 2026-10-16: KDA_SEQS_SNAP_MAX in glm5next_kda/seqs_tokens.rs: the snapshots one sequence keeps.
#define KDA_SEQS_SNAP_MAX 7

#define KDA_SEQS_PTR_ARGS(p)                                                                     \
    unsigned long long p##0, unsigned long long p##1, unsigned long long p##2,                    \
        unsigned long long p##3, unsigned long long p##4, unsigned long long p##5,                \
        unsigned long long p##6, unsigned long long p##7, unsigned long long p##8,                \
        unsigned long long p##9, unsigned long long p##10, unsigned long long p##11,              \
        unsigned long long p##12, unsigned long long p##13, unsigned long long p##14,             \
        unsigned long long p##15
#define KDA_SEQS_U32_ARGS(p)                                                                     \
    unsigned int p##0, unsigned int p##1, unsigned int p##2, unsigned int p##3,                   \
        unsigned int p##4, unsigned int p##5, unsigned int p##6, unsigned int p##7,               \
        unsigned int p##8, unsigned int p##9, unsigned int p##10, unsigned int p##11,             \
        unsigned int p##12, unsigned int p##13, unsigned int p##14, unsigned int p##15
#define KDA_SEQS_PICK(r, p)                                                                      \
    rows_pick16(r, p##0, p##1, p##2, p##3, p##4, p##5, p##6, p##7, p##8, p##9, p##10, p##11,      \
                p##12, p##13, p##14, p##15)
// 2026-10-16: The snapshot slots: KDA_SEQS_SNAP_MAX blocks of KDA_SEQS_MAX pointers, block t
// (p<t>_0 .. p<t>_15) holding each sequence's snapshot t; KDA_SEQS_SNAP_PICK(t, r, p) is sequence
// r's snapshot t, 0 for t >= KDA_SEQS_SNAP_MAX. Evaluated only on a row that may snapshot.
#define KDA_SEQS_SNAP_ARGS(p)                                                                    \
    KDA_SEQS_PTR_ARGS(p##0_), KDA_SEQS_PTR_ARGS(p##1_), KDA_SEQS_PTR_ARGS(p##2_),                 \
        KDA_SEQS_PTR_ARGS(p##3_), KDA_SEQS_PTR_ARGS(p##4_), KDA_SEQS_PTR_ARGS(p##5_),             \
        KDA_SEQS_PTR_ARGS(p##6_)
#define KDA_SEQS_SNAP_PICK(t, r, p)                                                              \
    ((t) == 0u   ? KDA_SEQS_PICK(r, p##0_)                                                      \
     : (t) == 1u ? KDA_SEQS_PICK(r, p##1_)                                                      \
     : (t) == 2u ? KDA_SEQS_PICK(r, p##2_)                                                      \
     : (t) == 3u ? KDA_SEQS_PICK(r, p##3_)                                                      \
     : (t) == 4u ? KDA_SEQS_PICK(r, p##4_)                                                      \
     : (t) == 5u ? KDA_SEQS_PICK(r, p##5_)                                                      \
     : (t) == 6u ? KDA_SEQS_PICK(r, p##6_)                                                      \
                 : 0ull)

// 2026-10-16: SNAPZ (`METRALE_GLM_KDA_SNAPZ=1`): sequence r's state source. f<r> (nullable) is its
// deferred-commit word: when set, the state the previous verdict kept is still in the snapshot
// slot z<r> (the commit's copy into the live state was skipped), so the launch reads the state
// from z<r>. Writes go where they always do. f<r> null (the lever off) reads the live state.
__device__ __forceinline__ unsigned long long kda_seqs_src(unsigned long long live,
                                                           unsigned long long f,
                                                           unsigned long long z) {
    return (f != 0ull && *(const unsigned int*)f != 0u) ? z : live;
}

// 2026-10-14: causal_conv1d_update_l2norm_tokens for every sequence: grid (ceil(dim / 256),
// max rows, sequences), block 256. Block (x, t, r) is the tokens kernel's block (x, t) on
// sequence r's rows (new_input and output offset by w<r> rows) and window c<r>; t >= n<r> is
// idle. The window is only read; causal_conv1d_window_advance_seqs moves it afterwards.
extern "C" __global__ void causal_conv1d_update_l2norm_tokens_seqs(
    const __nv_bfloat16* __restrict__ new_input,
    const __nv_bfloat16* __restrict__ weight,
    const float* __restrict__ bias,
    __nv_bfloat16* __restrict__ output,
    unsigned int dim,
    unsigned int d_conv,
    unsigned int qk_channels,
    unsigned int head_dim,
    float l2_eps,
    KDA_SEQS_PTR_ARGS(c),
    KDA_SEQS_U32_ARGS(w),
    KDA_SEQS_U32_ARGS(n),
    KDA_SEQS_PTR_ARGS(f),
    KDA_SEQS_PTR_ARGS(z)
) {
    const unsigned int r = blockIdx.z;
    if (r >= KDA_SEQS_MAX) return;
    const float* conv_state = (const float*)KDA_SEQS_PICK(r, c);
    const unsigned int t = blockIdx.y;
    if (conv_state == nullptr || t >= KDA_SEQS_PICK(r, n)) return;
    conv_state = (const float*)kda_seqs_src((unsigned long long)conv_state, KDA_SEQS_PICK(r, f),
                                            KDA_SEQS_PICK(r, z));
    const size_t row0 = KDA_SEQS_PICK(r, w);
    const __nv_bfloat16* in = new_input + row0 * dim;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    float win[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    if (ch < dim) {
        for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
            const int j = (int)t - (int)d_conv + (int)s;
            win[s] = j >= 0 ? (float)in[(size_t)j * dim + ch]
                            : conv_state[(size_t)ch * d_conv + (unsigned int)(j + (int)d_conv)];
        }
    }
    causal_conv1d_update_l2norm_body(nullptr, in, weight, bias, output + row0 * dim, 0u, t, true,
                                     dim, d_conv, qk_channels, head_dim, l2_eps, win);
}

// 2026-10-14: causal_conv1d_window_advance for every sequence: grid (ceil(dim / 256),
// sequences), block 256. Each thread reads its channel's window whole, writes the window after
// the sequence's row t to p<t>_<r> for each t < n<r> where that is set (the snapshot a copy of the
// window followed by a (t + 1)-row advance leaves), then the window after all n<r> rows to c<r>.
extern "C" __global__ void causal_conv1d_window_advance_seqs(
    const __nv_bfloat16* __restrict__ new_input,
    unsigned int dim,
    unsigned int d_conv,
    KDA_SEQS_PTR_ARGS(c),
    KDA_SEQS_SNAP_ARGS(p),
    KDA_SEQS_U32_ARGS(w),
    KDA_SEQS_U32_ARGS(n),
    KDA_SEQS_PTR_ARGS(f),
    KDA_SEQS_PTR_ARGS(z)
) {
    const unsigned int r = blockIdx.y;
    const unsigned int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= KDA_SEQS_MAX || ch >= dim) return;
    float* conv_state = (float*)KDA_SEQS_PICK(r, c);
    if (conv_state == nullptr) return;
    const unsigned int T = KDA_SEQS_PICK(r, n);
    const __nv_bfloat16* in = new_input + (size_t)KDA_SEQS_PICK(r, w) * dim;
    float* st = conv_state + (size_t)ch * d_conv;
    const float* src = (const float*)kda_seqs_src((unsigned long long)conv_state,
                                                  KDA_SEQS_PICK(r, f), KDA_SEQS_PICK(r, z)) +
                       (size_t)ch * d_conv;
    float old[4];
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) old[s] = src[s];
    for (unsigned int t = 0; t < T && t < KDA_SEQS_SNAP_MAX; ++t) {
        float* snap = (float*)KDA_SEQS_SNAP_PICK(t, r, p);
        if (snap == nullptr) continue;
        float* sp = snap + (size_t)ch * d_conv;
        for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
            const int j = (int)t + 1 - (int)d_conv + (int)s;
            sp[s] = j >= 0 ? (float)in[(size_t)j * dim + ch] : old[j + (int)d_conv];
        }
    }
    for (unsigned int s = 0; s < d_conv && s < 4u; ++s) {
        const int j = (int)T - (int)d_conv + (int)s;
        st[s] = j >= 0 ? (float)in[(size_t)j * dim + ch] : old[j + (int)d_conv];
    }
}

// 2026-10-14: kda_recurrent_decode_bf16_seq_reg_snap for every sequence: grid (H, KDA_REG_D /
// VB, sequences), block VB, 3 * KDA_REG_D floats of dynamic shared memory. Block (h, y, r) is
// that kernel's block (h, y) on sequence r: rows w<r> .. w<r> + n<r> - 1 (the bases offset by
// w<r> rows), state s<r>, and snapshot slots sp<t>_<r> (row t's state stored where set; before
// 2026-10-16 one slot, row 0's). The scaffolding repeats seq_reg_snap's rather than sharing it,
// so the C1 kernels' code is untouched; the per-token arithmetic is the shared kda_reg_token.
extern "C" __global__ void kda_recurrent_decode_bf16_seqs_reg(
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
    KDA_SEQS_PTR_ARGS(s),
    KDA_SEQS_SNAP_ARGS(sp),
    KDA_SEQS_U32_ARGS(w),
    KDA_SEQS_U32_ARGS(n),
    KDA_SEQS_PTR_ARGS(f),
    KDA_SEQS_PTR_ARGS(z)
) {
    constexpr unsigned int D = KDA_REG_D;
    const unsigned int r = blockIdx.z;
    const unsigned int h = blockIdx.x;
    if (r >= KDA_SEQS_MAX || h >= H) return;
    float* state = (float*)KDA_SEQS_PICK(r, s);
    if (state == nullptr) return;
    const unsigned int T = KDA_SEQS_PICK(r, n);
    const size_t row0 = KDA_SEQS_PICK(r, w);
    q += row0 * qkv_row_stride;
    k += row0 * qkv_row_stride;
    v += row0 * qkv_row_stride;
    gate += row0 * gate_row_stride;
    beta += row0 * beta_row_stride;
    out += row0 * out_row_stride;

    extern __shared__ float sh_seqs[];
    float* sh_decay = sh_seqs;
    float* sh_k = sh_seqs + D;
    float* sh_q = sh_seqs + 2u * D;
    const size_t hd = (size_t)h * D;
    const unsigned int vi = blockIdx.y * blockDim.x + threadIdx.x;
    const bool live = vi < D;
    float* S = state + hd * D;
    const float* S_src = (const float*)kda_seqs_src((unsigned long long)state,
                                                    KDA_SEQS_PICK(r, f), KDA_SEQS_PICK(r, z)) +
                         hd * D;

    float col[D];
    if (live) {
        #pragma unroll
        for (unsigned int kk = 0; kk < D; ++kk) col[kk] = S_src[(size_t)kk * D + vi];
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
            float* P = t < KDA_SEQS_SNAP_MAX ? (float*)KDA_SEQS_SNAP_PICK(t, r, sp) : nullptr;
            if (P != nullptr) {
                P += hd * D;
                #pragma unroll
                for (unsigned int kk = 0; kk < D; ++kk) P[(size_t)kk * D + vi] = col[kk];
            }
        }
    }
    if (live) {
        #pragma unroll
        for (unsigned int kk = 0; kk < D; ++kk) S[(size_t)kk * D + vi] = col[kk];
    }
}

// 2026-10-16: SNAPZ: the deferred commit of one state blob, for a verify that does not take the
// seqs kernels (`seqs_tokens.rs`, `snapz_resolve`): when *flag is nonzero, dst <- src, n16
// 16-byte words; else nothing. Grid-stride, block 256. The flag is read on the device, so a
// captured graph replays the decision of the step it runs in.
extern "C" __global__ void kda_snapz_resolve(const unsigned int* __restrict__ flag,
                                             const uint4* __restrict__ src,
                                             uint4* __restrict__ dst,
                                             unsigned long long n16) {
    if (*flag == 0u) return;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}
