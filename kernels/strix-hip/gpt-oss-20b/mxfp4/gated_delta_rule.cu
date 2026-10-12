// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (gpt-oss-20b/mxfp4).
// Invariants:
// - GPT-OSS has no linear-attention (gated delta rule) layers. The strix-hip
//   common copy, gb10/common/gated_delta_rule.cu, is replaced by every other
//   strix-hip target, so no target has compiled it with hipcc; this file keeps
//   the GPT-OSS target from being the first.
// - A gated_delta_rule lookup returns handle 0 and the boot kernel gate refuses it.
