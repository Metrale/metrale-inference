// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Routing capacity checks run before allocation or kernel lookup.
pub(super) fn validate_routing(top_k: usize, num_experts: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        top_k > 0 && top_k <= num_experts && num_experts > 0,
        "MoE config invalid: num_experts_per_tok={} must be in 1..={}",
        top_k,
        num_experts,
    );
    // 2026-09-25: The sigmoid routing kernels hold at most MAX_TOP_K
    // selections and MAX_EXPERTS experts in shared memory and ignore the
    // rest, so such configs are refused here.
    anyhow::ensure!(
        top_k <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
            && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        "MoE config exceeds the routing kernels' fixed shared-memory bounds: \
             num_experts_per_tok={} (max {}), num_experts={} (max {}). Raise \
             MAX_TOP_K / MAX_EXPERTS in kernels/gb10/common/moe_topk_sigmoid.cu \
             and their mirrors in layers::ops together.",
        top_k,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
        num_experts,
        crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::validate_routing;
    use crate::layers::ops::{MOE_TOPK_SIGMOID_MAX_EXPERTS, MOE_TOPK_SIGMOID_MAX_TOP_K};

    #[test]
    fn supported_boundary_shapes_pass() {
        assert!(validate_routing(1, 1).is_ok());
        assert!(validate_routing(8, 256).is_ok());
        assert!(validate_routing(MOE_TOPK_SIGMOID_MAX_TOP_K, MOE_TOPK_SIGMOID_MAX_EXPERTS).is_ok());
    }

    #[test]
    fn invalid_counts_and_shared_capacity_overflow_fail() {
        for (top_k, experts) in [
            (1, 0),
            (0, 1),
            (9, 8),
            (MOE_TOPK_SIGMOID_MAX_TOP_K + 1, MOE_TOPK_SIGMOID_MAX_EXPERTS),
            (1, MOE_TOPK_SIGMOID_MAX_EXPERTS + 1),
        ] {
            assert!(validate_routing(top_k, experts).is_err());
        }
    }
}
