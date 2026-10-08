// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: The available strided kernel multiplies normalized inputs by
//! `1 + weight`. A checkpoint with plain norm weights must keep its vanilla
//! per-sequence kernel until a matching strided realization exists.

pub(super) fn additive_strided_norm_eligible(
    vanilla_weights: bool,
    rows: usize,
    kernel_present: bool,
    enabled: bool,
    whole_elements: bool,
) -> bool {
    !vanilla_weights && rows > 1 && kernel_present && enabled && whole_elements
}

#[cfg(test)]
mod tests {
    use super::additive_strided_norm_eligible as eligible;
    use crate::model_type_ships_vanilla_norm_weights;

    #[test]
    fn laguna_plain_weights_never_take_the_additive_kernel() {
        let vanilla = model_type_ships_vanilla_norm_weights("laguna");
        assert!(vanilla);
        for rows in [1, 2, 3, 4, 8, 16, 128] {
            assert!(!eligible(vanilla, rows, true, true, true));
        }
        // 2026-10-06: Known-bad arithmetic control at Laguna's 128-wide heads:
        // an all-ones input and zero weights must yield zero under plain weights,
        // while the rejected additive convention yields approximately one.
        let input = [1.0f32; 128];
        let inv_rms = (input.iter().map(|x| x * x).sum::<f32>() / 128.0 + 1e-6)
            .sqrt()
            .recip();
        let plain = input.map(|x| x * inv_rms * 0.0);
        let wrong_additive = input.map(|x| x * inv_rms * (1.0 + 0.0));
        assert!(plain.iter().all(|x| *x == 0.0));
        assert!(wrong_additive.iter().all(|x| *x > 0.99));
    }

    #[test]
    fn additive_models_keep_the_existing_strided_route_and_guards() {
        let vanilla = model_type_ships_vanilla_norm_weights("qwen3");
        assert!(!vanilla);
        for rows in [2, 3, 4, 8, 16, 128] {
            assert!(eligible(vanilla, rows, true, true, true));
        }
        assert!(!eligible(vanilla, 1, true, true, true));
        assert!(!eligible(vanilla, 2, false, true, true));
        assert!(!eligible(vanilla, 2, true, false, true));
        assert!(!eligible(vanilla, 2, true, true, false));
    }
}
