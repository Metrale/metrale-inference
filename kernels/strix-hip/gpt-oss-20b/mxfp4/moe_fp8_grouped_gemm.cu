// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (gpt-oss-20b/mxfp4).
// Invariants:
// - strix-hip/common/moe_fp8_grouped_gemm.cu defines moe_fp8_grouped_gemm_v2
//   as a no-op. This file shadows it ([shadow] in KERNEL.toml) so the GPT-OSS
//   target never resolves that stub: a lookup returns handle 0 and the boot
//   kernel gate refuses it.
// - GPT-OSS has no FP8 expert path; its experts are MXFP4.
