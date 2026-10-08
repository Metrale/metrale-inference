// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (gpt-oss-20b/mxfp4).
// Invariants:
// - strix-hip/common/w4a16_gemm.cu is an NVFP4 GEMM that no strix-hip target
//   compiles: every model target replaces it ([shadow] in KERNEL.toml). This
//   file keeps that true for GPT-OSS, whose weights are MXFP4 and whose
//   dense projections run through dense_gemv_bf16 and dense_gemv_bf16_batchm.
// - A w4a16 lookup returns handle 0 and the boot kernel gate refuses it.
