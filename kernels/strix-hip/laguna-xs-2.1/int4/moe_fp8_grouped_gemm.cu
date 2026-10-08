// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (laguna-xs-2.1/int4).
// Invariants:
// - strix-hip/common/moe_fp8_grouped_gemm.cu defines moe_fp8_grouped_gemm_v2 as a no-op.
//   This file shadows it ([shadow] in KERNEL.toml) so the Laguna INT4 target never
//   resolves that stub: a lookup returns handle 0 and the boot kernel gate refuses it.
// - Laguna INT4 has no FP8 expert weights.
