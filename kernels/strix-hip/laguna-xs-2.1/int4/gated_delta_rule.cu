// SPDX-License-Identifier: MIT OR Apache-2.0

// 2026-10-07: Intentionally defines no kernel entry points.
//
// Owner: strix-hip kernels (laguna-xs-2.1/int4).
// Invariants:
// - Laguna has no linear-attention (gated delta rule) layers. The strix-hip common copy,
//   gb10/common/gated_delta_rule.cu, is replaced by every strix-hip target, so no target
//   has compiled it with hipcc; this file keeps Laguna from being the first.
// - A gated_delta_rule lookup returns handle 0 and the boot kernel gate refuses it.
