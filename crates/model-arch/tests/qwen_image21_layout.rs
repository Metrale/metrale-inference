// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Independent expanded-slot placement and padded-key controls.
use metrale_model_arch::qwen_image21::layout::JointLayout;

#[test]
fn image_slots_expand_and_samples_keep_separate_projection_rows() {
    let layout = JointLayout::new(
        2,
        3,
        &[false, true, false, true],
        &[[1, 2, 2], [1, 2, 2]],
        &[true, true, false, false, true, true],
    )
    .unwrap();
    assert_eq!(layout.geometry(), (2, 3, 8, 10, 4));
    assert_eq!(layout.image_ids(), &[-1, 0, 0, 0, 0, -1, 1, 1, 1, 1]);
    assert_eq!(
        layout.projection_gather(),
        &[
            0, 6, 7, 8, 9, 2, 10, 11, 12, 13, 3, 14, 15, 16, 17, 5, 18, 19, 20, 21
        ]
    );
    assert_eq!(layout.target_gather(), &[6, 7, 8, 9, 16, 17, 18, 19]);
    assert_eq!(
        layout
            .key_valid()
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (!v).then_some(i))
            .collect::<Vec<_>>(),
        vec![5, 10]
    );
    assert_eq!(
        layout.target_mask(),
        &[
            false, false, false, false, false, false, true, true, true, true
        ]
    );
}

#[test]
fn adjacent_images_retain_distinct_visibility_blocks() {
    let layout = JointLayout::new(1, 1, &[true, true], &[[1, 2, 2], [1, 2, 2]], &[true]).unwrap();
    assert_eq!(layout.image_ids(), &[0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(layout.projection_gather(), &[1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(layout.target_gather(), &[4, 5, 6, 7]);
}

#[test]
fn malformed_slot_geometry_is_refused() {
    for (slots, shapes, valid) in [
        (vec![false, false], vec![[1, 2, 2]], vec![true]),
        (vec![false, true], vec![[1, 1, 3]], vec![true]),
        (vec![true, true], vec![[1, 2, 2]], vec![true]),
        (vec![false, true], vec![[1, 2, 2]], vec![]),
        (vec![false, true], vec![[0, 2, 2]], vec![true]),
    ] {
        assert!(JointLayout::new(1, 1, &slots, &shapes, &valid).is_err());
    }
}
