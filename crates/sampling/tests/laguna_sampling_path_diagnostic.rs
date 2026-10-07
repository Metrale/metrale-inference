// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: CPU diagnostic of existing sampler contracts; not a GPU capture
//! and not an explanation of Laguna's observed response variability.
use metrale_sampling::{
    SamplingParams, argmax_first_wins_f32, feed_argmax::plain_kernel_argmax, sample_with_params,
};
fn host(row: &[f32]) -> u32 {
    let bytes: Vec<u8> = row.iter().flat_map(|x| x.to_le_bytes()).collect();
    sample_with_params(
        &bytes,
        &SamplingParams {
            temperature: 0.0,
            ..SamplingParams::greedy(1)
        },
    )
}
#[test]
fn logit_readback_route_is_not_transparent_at_exact_bf16_ties() {
    // 2026-10-07: Exactly BF16-representable inputs. The plain kernel reference simulates
    // the checked-in1024-thread scan/tree, while production host sampling runs.
    let row = [2.0, 2.0, -1.0];
    assert_eq!(plain_kernel_argmax(&row), 0);
    assert_eq!(host(&row), 1);
    // 2026-10-07: Three existing policies can all disagree, not merely first vs last.
    let mut wide = vec![-1.0; 1026];
    wide[1] = 3.0;
    wide[2] = 3.0;
    wide[1025] = 3.0;
    assert_eq!(argmax_first_wins_f32(&wide), 1);
    assert_eq!(plain_kernel_argmax(&wide), 2);
    assert_eq!(host(&wide), 1025);
}
#[test]
fn unique_maximum_control_removes_tie_policy_difference() {
    for winner in [0, 1, 2, 511, 512, 1023, 1024, 1025] {
        let mut row = vec![-1.0; 1026];
        row[winner] = 3.0;
        assert_eq!(plain_kernel_argmax(&row), winner as u32);
        assert_eq!(argmax_first_wins_f32(&row), winner as u32);
        assert_eq!(host(&row), winner as u32);
    }
}
