// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-14: kda_reg_token, moved unchanged from kda_recurrent.cu (which includes it where it
// stood) so kda_seqs_tokens.cu runs the same per-token step.
//
// Owner: gb10 kernels.
// Invariants: see the function's comment.

#pragma once

#include <cuda_bf16.h>

#ifndef KDA_REG_D
#define KDA_REG_D 128
#endif

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
