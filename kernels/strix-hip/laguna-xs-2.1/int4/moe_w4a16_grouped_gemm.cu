// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (laguna-xs-2.1/int4).
// Invariants:
// - strix-hip/common/moe_w4a16_grouped_gemm.cu holds no-op compile stubs. This file
//   shadows it ([shadow] in KERNEL.toml) so the Laguna INT4 target never resolves them:
//   a moe_w4a16 lookup returns handle 0 and the boot kernel gate refuses it, rather than
//   launching a kernel that writes nothing.
// - Laguna INT4 / INT8 experts run through packed_int_gemv, not this module.
