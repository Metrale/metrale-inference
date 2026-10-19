// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The spill salt separates every input it folds, and is pinned.

use super::*;
use crate::plan::RungIx;

fn side(codec: KvCodecId, rung: u8) -> KvSidePlan {
    KvSidePlan {
        codec,
        rung: RungIx(rung),
    }
}

fn one_layer(k: KvSidePlan, v: KvSidePlan) -> SpillSalt {
    SpillSalt::new()
        .side(LayerIx(0), KvSide::K, k)
        .side(LayerIx(0), KvSide::V, v)
}

/// 2026-10-18: nvfp4 and turbo4 store the same bits per value and the same scales, which is the
/// collision the salt exists to prevent; every pair of distinct codecs gives distinct salts.
#[test]
fn distinct_codecs_give_distinct_salts() {
    let all = KvCodecId::ALL;
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            let sa = one_layer(side(*a, 0), side(*a, 0)).finish();
            let sb = one_layer(side(*b, 0), side(*b, 0)).finish();
            assert_ne!(sa, sb, "{} vs {}", a.name(), b.name());
        }
    }
    let nvfp4 = one_layer(side(KvCodecId::Nvfp4, 0), side(KvCodecId::Nvfp4, 0));
    let turbo4 = one_layer(side(KvCodecId::Turbo4, 0), side(KvCodecId::Turbo4, 0));
    assert_ne!(nvfp4.finish(), turbo4.finish());
}

/// 2026-10-18: Each field of each record changes the salt on its own.
#[test]
fn every_field_flips_the_salt() {
    let bf = side(KvCodecId::Bf16, 0);
    let base = one_layer(bf, bf)
        .layer_shape(LayerIx(0), 8, 128)
        .indexer(None, None)
        .finish();
    let variants = [
        (
            "layer",
            SpillSalt::new()
                .side(LayerIx(1), KvSide::K, bf)
                .side(LayerIx(0), KvSide::V, bf)
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(None, None),
        ),
        (
            "side",
            SpillSalt::new()
                .side(LayerIx(0), KvSide::V, bf)
                .side(LayerIx(0), KvSide::V, bf)
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(None, None),
        ),
        (
            "codec",
            one_layer(bf, side(KvCodecId::Fp8, 0))
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(None, None),
        ),
        (
            "rung",
            one_layer(bf, side(KvCodecId::Bf16, 1))
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(None, None),
        ),
        (
            "kv heads",
            one_layer(bf, bf)
                .layer_shape(LayerIx(0), 4, 128)
                .indexer(None, None),
        ),
        (
            "head dim",
            one_layer(bf, bf)
                .layer_shape(LayerIx(0), 8, 256)
                .indexer(None, None),
        ),
        (
            "indexer codec",
            one_layer(bf, bf)
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(crate::by_name("fp8_row"), None),
        ),
        (
            "indexer bytes",
            one_layer(bf, bf)
                .layer_shape(LayerIx(0), 8, 128)
                .indexer(None, Some(2064)),
        ),
    ];
    for (name, salt) in variants {
        assert_ne!(salt.finish(), base, "{name} did not change the salt");
    }
}

/// 2026-10-18: The encoding is frozen: a change here strands every block spilled under the old
/// salt, so it must come with a `SPILL_SALT_VERSION` bump.
#[test]
fn the_salt_is_pinned() {
    let bf = side(KvCodecId::Bf16, 0);
    assert_eq!(SPILL_SALT_VERSION, 1);
    assert_eq!(SpillSalt::new().finish(), 0xaf63_bc4c_8601_b62c);
    assert_eq!(one_layer(bf, bf).finish(), 0xa79e_c400_270f_b5eb);
}
