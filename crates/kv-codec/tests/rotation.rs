// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `KvCodecSpec::rotates_at`, the predicate the cache write reads to decide which
//! sides it WHT-rotates: true exactly at the head dims a rotation covers.

use metrale_kv_codec::{KvCodecId, KvCodecSpec, Rotation, by_name, spec};

/// 2026-10-18: A rotated codec's WHT covers head dims 128, 256 and 512 and nothing between, which
/// is where the pre-registry write rotated the turbo codecs; an unrotated attention codec rotates
/// nowhere.
#[test]
fn rotated_codecs_rotate_at_128_256_512_and_others_never() {
    let mut rotated = 0;
    for id in KvCodecId::ATTENTION {
        let at: Vec<u32> = (0..=1024).filter(|&hd| spec(id).rotates_at(hd)).collect();
        if spec(id).rotates() {
            rotated += 1;
            assert_eq!(at, [128, 256, 512], "{id:?}");
        } else {
            assert!(at.is_empty(), "{id:?} rotates at {at:?}");
        }
    }
    assert!(rotated > 0, "no rotated codec registered");
}

/// 2026-10-18: The indexer row codec rotates at the one head dim its kernels are built for.
#[test]
fn the_indexer_row_codec_rotates_at_128_only() {
    let id = by_name("fp8_row").expect("registered");
    let at: Vec<u32> = (0..=1024).filter(|&hd| spec(id).rotates_at(hd)).collect();
    assert_eq!(at, [128]);
}

/// 2026-10-18: A block FWHT rotates only whole blocks, and a zero block rotates nothing.
#[test]
fn a_block_fwht_needs_whole_blocks() {
    let with = |block| KvCodecSpec {
        rotation: Rotation::Fwht { block },
        ..*spec(KvCodecId::ALL[0])
    };
    let s = with(64);
    assert!(s.rotates_at(64) && s.rotates_at(128) && s.rotates_at(576));
    assert!(!s.rotates_at(96) && !s.rotates_at(32));
    let zero = with(0);
    assert!((0..=1024).all(|hd| !zero.rotates_at(hd)));
}

/// 2026-10-18: An in-kernel rotation never asks for the WHT bookends, and its codec fits exactly
/// the head dims its kernels are built for (whole 128-value groups at 128, 256, 512); every other
/// attention codec fits every head dim.
#[test]
fn in_kernel_codecs_fit_their_built_head_dims_and_never_take_bookends() {
    let mut in_kernel = 0;
    for id in KvCodecId::ATTENTION {
        let s = spec(id);
        let fits: Vec<u32> = (0..=1024).filter(|&hd| s.fits_head_dim(hd)).collect();
        if matches!(s.rotation, Rotation::InKernel { .. }) {
            in_kernel += 1;
            assert_eq!(fits, [128, 256, 512], "{id:?}");
            assert!((0..=1024).all(|hd| !s.rotates_at(hd)), "{id:?}");
        } else {
            assert_eq!(fits.len(), 1025, "{id:?}");
        }
    }
    assert_eq!(in_kernel, 4, "turbo1/2/3_tcq and turbo4_n128");
    let zero = KvCodecSpec {
        rotation: Rotation::InKernel { block: 0 },
        ..*spec(by_name("turbo3_tcq").expect("registered"))
    };
    assert!((0..=1024).all(|hd| !zero.fits_head_dim(hd)));
}
