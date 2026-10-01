// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The load summary states what the arms built, including the Qwen3.6-27B-FP8
//! load that the previous policy-derived line misdescribed.
//!
//! Owner: model-arch weight loader tests.
//! Invariants: none beyond the types.

use super::*;

const WIDE: Served = Served::Nvfp4 {
    from_checkpoint: false,
    act: Nvfp4Act::Wide,
};

/// 2026-09-30: Qwen3.6-27B-FP8 under `declared` as it loads on GB10 today: 16 attention and 64
/// FFN groups requantized to NVFP4, 48 GDN groups native FP8, everything declared FP8. Only
/// attention and FFN are below declared, and the GDN count says FP8, not NVFP4.
#[test]
fn the_fp8_checkpoint_load_names_gdn_fp8_and_only_the_requantized_groups_below() {
    let mut f = ServedFormats::default();
    for layer in 0..64 {
        let mixer = if layer % 4 == 3 {
            Group::Attention
        } else {
            Group::Gdn
        };
        let served = if mixer == Group::Gdn {
            Served::Fp8
        } else {
            WIDE
        };
        f.record(mixer, layer, served, 8);
        f.record(Group::Ffn, layer, WIDE, 8);
    }
    assert_eq!(
        f.summary("declared"),
        "--weight-quantization declared: decode weights by group (layers): \
         attention: 16 NVFP4 requantized at load, W4A16; GDN: 48 FP8 W8A16; \
         dense FFN: 64 NVFP4 requantized at load, W4A16; \
         below the checkpoint's declared weight precision: attention 16, dense FFN 64."
    );
}

/// 2026-09-30: An NVFP4 checkpoint served from its own NVFP4 weights, W4A4 where declared, is
/// not below anything; a 16-bit group served NVFP4 is.
#[test]
fn below_is_judged_on_weight_bits_against_the_declaration() {
    let mut f = ServedFormats::default();
    let a4 = Served::Nvfp4 {
        from_checkpoint: true,
        act: Nvfp4Act::A4,
    };
    f.record(Group::Ffn, 0, a4, 4);
    f.record(Group::Gdn, 1, Served::Bf16, 16);
    assert!(
        f.summary("declared")
            .ends_with("none below the checkpoint's declared weight precision.")
    );
    f.record(Group::Gdn, 1, WIDE, 16);
    assert!(
        f.summary("nvfp4")
            .ends_with("below the checkpoint's declared weight precision: GDN 1.")
    );
}

/// 2026-09-30: The W8A8 install replaces the recorded form and brings the group to declared;
/// an install on a group no arm recorded is refused.
#[test]
fn a_w8a8_install_upgrades_the_recorded_group_and_refuses_an_unrecorded_one() {
    let mut f = ServedFormats::default();
    f.record(Group::Attention, 3, WIDE, 8);
    f.upgrade_w8a8(Group::Attention, 3).unwrap();
    assert_eq!(
        f.summary("declared"),
        "--weight-quantization declared: decode weights by group (layers): \
         attention: 1 FP8 W8A8; none below the checkpoint's declared weight precision."
    );
    assert!(f.upgrade_w8a8(Group::Ffn, 3).is_err());
}

#[test]
fn declared_bits_default_to_sixteen() {
    assert_eq!(declared_weight_bits(LayerPrecision::UNQUANTIZED), 16);
}
