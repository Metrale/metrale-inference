// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-21: Compact block scales (METRALE_GLM_MOE_SCALE_DICT; layout in w4a4_mx_core.cuh, CS)
// in the grouped W4A4 GEMM's weight-scale staging (w4g_body, w4n_body, `_cs` entries). A plain
// tensor stages row n's chunk c as two 4-byte copies into the stage's ws[r][0..8] (groups 0..3,
// 4..7); a compact one as two 4-byte copies into the same bytes: its lo-plane word, and the
// aligned hi-plane word that holds the (n, c) hi byte (the hi plane is zero-padded to whole
// words). w4g_cs_words rebuilds the MMA's two scale words from them: every byte is base +
// offset, the stored E4M3 code, so the words are the plain staging's byte permutes and every
// output byte is the plain entry's.
//
// Owner: gb10 kernels.
// Invariants: as w4a4_grouped_mx.cu. A compact tensor's base pointer is 4-byte aligned.
#pragma once

#include "w4a4_mx_core.cuh"

// 2026-10-21: Copy h's row base of row n: the lo plane (h = 0) or the hi plane (h = 1). The
// copy of chunk c reads w4g_cs_at(base, c, h).
__device__ __forceinline__ const unsigned char* w4g_cs_row(const unsigned char* bs, unsigned int n,
                                                           unsigned int h, unsigned int num_c,
                                                           unsigned int N)
{
    return h == 0u ? bs + (unsigned long long)n * num_c * 4u
                   : bs + (unsigned long long)N * num_c * 4u + (unsigned long long)n * num_c;
}

__device__ __forceinline__ const unsigned char* w4g_cs_at(const unsigned char* row, unsigned int c,
                                                          unsigned int h)
{
    return h == 0u ? row + c * 4u
                   : (const unsigned char*)((unsigned long long)(row + c) & ~3ull);
}

// 2026-10-21: The MMA's two scale words of row n's chunk c from its staged (lo word, hi word):
// zero for a dead row (n >= N, as the plain staging's zero fill), groups 4..7 zero in a `_k64`
// tail chunk (as the plain staging's skipped second copy). The hi byte sits at byte
// (n * num_c + c) % 4 of its word: the hi plane starts N * num_c * 4 bytes in.
__device__ __forceinline__ void w4g_cs_words(const uint2 sw, unsigned int n, unsigned int c,
                                             unsigned int num_c, bool live, bool tail,
                                             unsigned int base4, uint32_t& s0, uint32_t& s1)
{
    const uint32_t hb = (sw.y >> (((n * num_c + c) & 3u) * 8u)) & 0xFFu;
    const uint32_t keep = !live ? 0u : tail ? 0x00FF00FFu : 0xFFFFFFFFu;
    s0 = w4a4_cs_word(sw.x, hb & 0xFu, base4) & keep;
    s1 = w4a4_cs_word(sw.x >> 4, hb >> 4, base4) & keep;
}
