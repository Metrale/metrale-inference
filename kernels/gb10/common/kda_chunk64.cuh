// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-12: Shapes, launch constants and the scratch layout shared by the 64-token chunked KDA
// prefill kernels: kda_chunk64_prepare.cu (kda_chunk64_prepare) and kda_chunk64.cu
// (kda_chunk64_scan). The Rust side (model-arch glm5next_kda/chunk64.rs) mirrors every value.
//
// The prefill runs in slabs of up to KDA64_SLAB chunks: one prepare launch writes the slab's
// factors, one scan launch consumes them, and the next slab reuses the same scratch, so the
// factors stay in L2 (4 chunks x 22 heads x 88.5 KiB = 7.8 MiB of GB10's 24 MiB; measured best
// of 4/8/16/32 for the scan) instead of
// round-tripping through DRAM. Scratch per (slab-local chunk lc, head h), cb = lc * H + h:
// - U     FP32 [C][D]  at u     + cb * C * D
// - W     BF16 [C][D]  at w     + cb * C * D      (W_i = beta_i (exp(G_i) k_i - sum A_ij W_j))
// - Qg    BF16 [C][D]  at qg    + cb * C * D      (scale exp(G_i) q_i)
// - Kd    BF16 [C][D]  at kd    + cb * C * D      (exp(G_last - G_i) k_i)
// - P     BF16 [C][C]  at p     + cb * C * C      (zero above the diagonal)
// - A, decay FP32      at ad    + cb * KDA64_AD_STRIDE: A [C][C] (zero at and above the
//                      diagonal; the prepare's own scratch, read back for the solve) then
//                      decay [D] = exp(G_last)
//
// Owner: gb10 kernels.

#ifndef KDA_CHUNK64_CUH
#define KDA_CHUNK64_CUH

#define KDA64_D 128
#define KDA64_C 64
#define KDA64_SLAB 4
// 2026-10-12: Value columns per scan block: 2 blocks per head, 44 blocks at 22 heads, one wave
// on 48 SMs.
#define KDA64_VB 64
#define KDA64_PREP_THREADS 256
#define KDA64_SCAN_THREADS 128
// 2026-10-12: Padded BF16 row strides (halves) so ldmatrix row addresses hit distinct banks.
#define KDA64_LD (KDA64_D + 8)
#define KDA64_LDC (KDA64_C + 8)
#define KDA64_AD_STRIDE (KDA64_C * KDA64_C + KDA64_D)
// 2026-10-12: prepare: G FP32 [C][D], K BF16 [C][D] (A FP32 [C][C], read back from global, overlays
// K once the pair products are done), beta FP32 [C]: 48.25 KiB, two blocks per SM
// (2 x (48.25 + 1) KiB <= 100).
#define KDA64_PREP_SMEM (KDA64_C * KDA64_D * 4 + KDA64_C * KDA64_D * 2 + KDA64_C * 4)
// 2026-10-12: scan: W, Qg, Kd BF16 [C][LD]; P BF16 [C][LDC]; S^T BF16 [VB][LD]; delta^T BF16
// [VB][LDC]; decay FP32 [D]. 88,576 B: one block per SM.
#define KDA64_SCAN_SMEM                                                                      \
    ((3 * KDA64_C * KDA64_LD + KDA64_C * KDA64_LDC + KDA64_VB * KDA64_LD +                    \
      KDA64_VB * KDA64_LDC) * 2 + KDA64_D * 4)

#endif
