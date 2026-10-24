// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: `KvCacheConfig::spill_salt` separates caches whose blocks mean different things
//! and agrees for caches whose blocks mean the same thing.

use metrale_kv_codec::by_name;

use super::KvCacheConfig;

fn config(dtype: &str) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 2,
        head_dim: 256,
        num_layers: 4,
        dtype: dtype.parse().expect("a --kv-cache-dtype spelling"),
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
        v_block_bytes: None,
    }
}

/// 2026-10-18: nvfp4 and turbo4 blocks have the same byte size; only the salt tells them apart.
#[test]
fn same_size_formats_get_different_salts() {
    let (nvfp4, turbo4) = (config("nvfp4"), config("turbo4"));
    assert_eq!(nvfp4.block_bytes(), turbo4.block_bytes());
    assert_ne!(nvfp4.spill_salt(None), turbo4.spill_salt(None));
}

/// 2026-10-18: The same formats and shapes give the same salt, so a restarted serve of one
/// configuration keys its blocks the same way.
#[test]
fn equal_caches_get_equal_salts() {
    assert_eq!(
        config("turbo4k_turbo8v").spill_salt(None),
        config("turbo4k_turbo8v").spill_salt(None)
    );
}

/// 2026-10-18: One layer's format, one layer's shape, the indexer codec and the indexer's block
/// bytes each change the salt.
#[test]
fn each_input_changes_the_salt() {
    let base = config("bf16");
    let salt = base.spill_salt(None);
    let one_layer_fp8 = KvCacheConfig {
        layer_dtypes: vec!["bf16".parse().unwrap(), "fp8".parse().unwrap()],
        ..config("bf16")
    };
    let one_layer_narrow = KvCacheConfig {
        layer_dims: vec![(2, 256), (1, 256)],
        ..config("bf16")
    };
    let resized_v = KvCacheConfig {
        v_block_bytes: Some(2064),
        ..config("bf16")
    };
    assert_ne!(one_layer_fp8.spill_salt(None), salt, "layer format");
    assert_ne!(one_layer_narrow.spill_salt(None), salt, "layer shape");
    assert_ne!(base.spill_salt(by_name("fp8_row")), salt, "indexer codec");
    assert_ne!(resized_v.spill_salt(None), salt, "indexer block bytes");
}
