// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Format spellings and byte sizes.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn every_spelling_round_trips() {
    for s in [
        "bf16",
        "f32",
        "i32",
        "fp8/token",
        "fp8/tensor",
        "fp8/channel",
        "fp8/g128",
        "fp8/block128x128",
        "nvfp4/g16",
        "mxfp4/g32",
    ] {
        let f = Format::parse(s).unwrap_or_else(|e| panic!("{s}: {e}"));
        assert_eq!(f.name(), s);
    }
}

#[test]
fn malformed_spellings_are_refused() {
    for s in [
        "",
        "BF16",
        "fp8",
        "fp8/",
        "fp8/g0",
        "fp8/block128",
        "fp8/block0x128",
        "nvfp4/16",
        "nvfp4/g",
        "fp16",
        "mxfp4/g16",
        "mxfp4/g64",
        "mxfp4/g0",
        "fp8/token ",
    ] {
        assert_eq!(Format::parse(s), Err(FormatError(s.to_string())), "{s:?}");
    }
}

#[test]
fn weight_layouts_are_not_edge_formats() {
    assert!(!Format::parse("fp8/channel").unwrap().is_edge_format());
    assert!(!Format::parse("fp8/block128x128").unwrap().is_edge_format());
    assert!(Format::parse("fp8/token").unwrap().is_edge_format());
    assert!(Format::parse("nvfp4/g16").unwrap().is_edge_format());
}

#[test]
fn byte_sizes_count_values_and_scales() {
    assert_eq!(Format::Bf16.bytes(3, 5120), Some(3 * 5120 * 2));
    assert_eq!(Format::F32.bytes(2, 7), Some(56));
    // 2026-09-28: 2560 packed bytes, 320 group scales, one F32 global.
    assert_eq!(
        Format::Nvfp4 { group: 16 }.bytes(1, 5120),
        Some(2560 + 320 + 4)
    );
    let tok = Format::Fp8E4m3 {
        scale: Scale::PerToken,
    };
    assert_eq!(tok.bytes(4, 256), Some(4 * 256 + 4 * 4));
    let g = Format::Fp8E4m3 {
        scale: Scale::Group(128),
    };
    assert_eq!(g.bytes(2, 256), Some(2 * 256 + 2 * 2 * 4));
}

/// 2026-09-30: An NVFP4 activation carries one F32 global per row (`w4a4_quant_rows`), an NVFP4
/// weight one for the tensor; the other formats size alike.
#[test]
fn nvfp4_activations_carry_a_global_per_row_and_weights_one_per_tensor() {
    let f = Format::Nvfp4 { group: 16 };
    assert_eq!(f.bytes(8, 5120), Some(8 * (2560 + 320 + 4)));
    assert_eq!(f.weight_bytes(8, 5120), Some(8 * (2560 + 320) + 4));
    assert_eq!(
        Format::Bf16.weight_bytes(8, 5120),
        Format::Bf16.bytes(8, 5120)
    );
}

#[test]
fn a_dim_off_the_scale_group_has_no_size() {
    assert_eq!(Format::Nvfp4 { group: 16 }.bytes(1, 24), None);
    assert_eq!(
        Format::Fp8E4m3 {
            scale: Scale::Group(128)
        }
        .bytes(1, 200),
        None
    );
    assert_eq!(Format::Bf16.bytes(u64::MAX, 2), None);
}

/// 2026-10-06: MXFP4 has no global multiplier, unlike NVFP4. Catch both scale-size and
/// accidental format-alias regressions with multi-row expert-shaped matrices.
#[test]
fn mxfp4_counts_e8m0_groups_without_an_nvfp4_global() {
    let mx = Format::parse("mxfp4/g32").unwrap();
    assert_ne!(mx, Format::Nvfp4 { group: 32 });
    assert_eq!(
        mx.weight_bytes(32 * 5760, 2880),
        Some(32 * 5760 * (1440 + 90))
    );
    assert_eq!(mx.bytes(2, 32), Some(34));
    assert_eq!(mx.weight_bytes(2, 32), Some(34));
    assert_eq!(Format::Nvfp4 { group: 32 }.weight_bytes(2, 32), Some(38));
    assert_eq!(mx.bytes(1, 31), None);
    assert_eq!(mx.bytes(u64::MAX, 32), None);
    assert_eq!(mx.bytes(0, 32), Some(0));
    assert!(!mx.is_plain());
    assert!(
        !mx.is_edge_format(),
        "no E8M0 activation quantizer is implemented"
    );
}
