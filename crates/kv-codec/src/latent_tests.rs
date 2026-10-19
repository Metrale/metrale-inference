// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The latent plan's row sizes against the byte counts the checkpoint formats
//! imply, the codecs it accepts for each cache over the whole registry, and its refusals.

use super::*;

fn id(name: &str) -> KvCodecId {
    crate::by_name(name).expect("registered")
}

/// 2026-10-19: MXFP8 window, NVFP4 latent and MXFP4 index-key rows at head dims 512 and 128:
/// `512 + 512/32`, `512/2 + 512/16` and `128/2 + 128/32` bytes.
#[test]
fn the_microscaled_rows_have_the_format_sizes() {
    let plan = LatentKvPlan::new(id("mxfp8"), id("nvfp4"), id("mxfp4")).expect("valid plan");
    let rows = plan.rows(512, 128).expect("whole groups");
    assert_eq!(
        rows,
        LatentRowBytes {
            window: 528,
            latent: 288,
            index_key: 68,
        }
    );
    assert_eq!(rows.layer_bytes(128, None), 128 * 528);
    assert_eq!(rows.layer_bytes(128, Some(64)), 128 * 528 + 64 * (288 + 68));
    // 2026-10-19: Small test geometries: 32 values are one MXFP group and two NVFP4 groups.
    let small = plan.rows(32, 32).expect("whole groups");
    assert_eq!((small.window, small.latent, small.index_key), (33, 18, 17));
}

/// 2026-10-19: Over every registered codec, the window and latent caches accept exactly the
/// unrotated block-scaled non-indexer codecs and the key cache exactly the unrotated
/// block-scaled indexer codecs; every other choice is the latent code.
#[test]
fn each_cache_accepts_exactly_the_plain_block_scaled_codecs() {
    let (w, l, k) = (id("mxfp8"), id("nvfp4"), id("mxfp4"));
    let accepted = |slot: usize| -> Vec<&'static str> {
        KvCodecId::ALL
            .into_iter()
            .filter(|&c| {
                let trio = match slot {
                    0 => (c, l, k),
                    1 => (w, c, k),
                    _ => (w, l, c),
                };
                match LatentKvPlan::new(trio.0, trio.1, trio.2) {
                    Ok(_) => true,
                    Err(e) => {
                        assert_eq!(e.code(), ErrorCode::KvLatentGeometry, "{}", c.name());
                        false
                    }
                }
            })
            .map(KvCodecId::name)
            .collect()
    };
    assert_eq!(accepted(0), ["nvfp4", "mxfp8"]);
    assert_eq!(accepted(1), ["nvfp4", "mxfp8"]);
    assert_eq!(accepted(2), ["mxfp4"]);
}

/// 2026-10-19: A row that is not whole groups is refused for the cache it names.
#[test]
fn a_partial_group_row_is_refused() {
    let plan = LatentKvPlan::new(id("mxfp8"), id("nvfp4"), id("mxfp4")).expect("valid plan");
    for (cache, values) in [
        (LatentCache::Window, 48),
        (LatentCache::Latent, 24),
        (LatentCache::IndexKey, 16),
        (LatentCache::Window, 0),
    ] {
        let e = plan.row_bytes(cache, values).unwrap_err();
        assert_eq!(e.code(), ErrorCode::KvLatentGeometry);
        assert!(e.to_string().contains(cache.name()), "{e}");
    }
    assert!(plan.rows(512, 48).is_err());
}
