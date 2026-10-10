// SPDX-License-Identifier: MIT OR Apache-2.0

//! Pure startup validation for DFlash tensor-parallel combinations.

/// Reject TP configurations that the current sharded weights or propose kernels cannot support.
/// The inputs are resolved before GPU allocations so these checks can be unit-tested without a
/// model or GPU backend.
pub(super) fn validate_tp_startup(
    tensor_parallel: bool,
    fp8_requested: bool,
    topk16_split: bool,
    dflash2_selector_active: bool,
    topk16_part_present: bool,
    topk16_merge_present: bool,
    local_vocab_size: usize,
) -> anyhow::Result<()> {
    if !tensor_parallel {
        return Ok(());
    }

    anyhow::ensure!(
        !fp8_requested,
        "DFlash TP does not support drafter FP8 yet: row scales must be preserved over the full unsharded rows; set METRALE_DFLASH_DRAFTER_FP8=0"
    );
    anyhow::ensure!(
        topk16_split,
        "DFlash TP requires the split top-16 path; METRALE_DFLASH2_TOPK_SPLIT=0 is unsupported"
    );
    if dflash2_selector_active {
        anyhow::ensure!(
            topk16_part_present && topk16_merge_present,
            "DFlash TP with an active DFlash2 selector requires both dflash2_topk16_part and dflash2_topk16_merge kernels"
        );
        validate_tp_topk16_merge_capacity(local_vocab_size)?;
    }

    Ok(())
}

/// Reject a TP rank's local vocab slice when its per-part candidates exceed the merge block.
/// This runs during head construction, before any serving propose can launch the merge.
fn validate_tp_topk16_merge_capacity(local_vocab_size: usize) -> anyhow::Result<()> {
    anyhow::ensure!(
        super::super::dflash2::topk16_merge_fits_vocab(local_vocab_size),
        "DFlash TP local vocab slice ({local_vocab_size} rows) exceeds the DFlash2 top-16 merge capacity"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{validate_tp_startup, validate_tp_topk16_merge_capacity};

    #[test]
    fn non_tp_is_unaffected_by_tp_only_settings() {
        assert!(validate_tp_startup(false, true, false, true, false, false, 51_648).is_ok());
    }

    #[test]
    fn tp_refuses_requested_drafter_fp8() {
        let err = validate_tp_startup(true, true, true, false, true, true, 51_648).unwrap_err();
        assert!(err.to_string().contains("does not support drafter FP8"));
        assert!(err.to_string().contains("METRALE_DFLASH_DRAFTER_FP8=0"));
    }

    #[test]
    fn tp_refuses_disabled_split_topk() {
        let err = validate_tp_startup(true, false, false, false, true, true, 51_648).unwrap_err();
        assert!(err.to_string().contains("METRALE_DFLASH2_TOPK_SPLIT=0"));
    }

    #[test]
    fn active_tp_selector_requires_both_distributed_topk_kernels() {
        for (part_present, merge_present) in [(false, false), (false, true), (true, false)] {
            let err = validate_tp_startup(
                true,
                false,
                true,
                true,
                part_present,
                merge_present,
                51_648,
            )
            .unwrap_err();
            assert!(err.to_string().contains("requires both dflash2_topk16_part"));
        }
    }

    #[test]
    fn active_tp_selector_succeeds_with_both_kernels() {
        assert!(validate_tp_startup(true, false, true, true, true, true, 51_648).is_ok());
    }

    #[test]
    fn inactive_tp_selector_does_not_require_distributed_topk_kernels() {
        assert!(validate_tp_startup(true, false, true, false, false, false, 51_648).is_ok());
    }

    #[test]
    fn glm_53_three_rank_local_vocab_slices_fit_the_merge_capacity() {
        for local_vocab_size in [51_648, 51_648, 51_584] {
            assert!(
                validate_tp_topk16_merge_capacity(local_vocab_size).is_ok(),
                "local vocab slice {local_vocab_size} should fit"
            );
        }
    }

    #[test]
    fn tp_local_vocab_slice_over_merge_capacity_is_rejected() {
        let err = validate_tp_topk16_merge_capacity(usize::MAX).unwrap_err();
        assert!(err.to_string().contains("local vocab slice"));
        assert!(err.to_string().contains("merge capacity"));
    }
}
