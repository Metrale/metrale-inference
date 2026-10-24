// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The circuit's KV byte counts against hand-computed layouts, and its refusals.

use std::collections::BTreeMap;

use super::*;

fn pair(s: &str) -> KvPair {
    parse_kv(s).expect("accepted spelling")
}

/// 2026-10-18: One token of 8 KV heads x 256 dims (2048 values per side), per pair, worked by
/// hand from the layouts: bf16 2 B/value; fp8 1 B/value (its scale is per layer, outside the
/// block); turbo4/nvfp4 4 bits/value + one E4M3 byte per 16 values (1024 + 128 per side);
/// turbo3 3 bits + 1 B/16 (768 + 128); turbo2 2 bits + 1 B/16 (512 + 128); turbo8 1 B/value
/// + one bf16 norm per 16 values (2048 + 256).
#[test]
fn token_bytes_match_the_hand_layouts() {
    let cases = [
        ("bf16", 8192),
        ("fp8", 4096),
        ("nvfp4", 2304),
        ("turbo4", 2304),
        ("turbo3", 1792),
        ("turbo2", 1280),
        ("turbo8", 4608),
        ("bf16k_turbo4v", 4096 + 1152),
        ("fp8k_turbo3v", 2048 + 896),
        ("turbo4k_turbo8v", 1152 + 2304),
    ];
    for (s, want) in cases {
        assert_eq!(token_bytes(pair(s), 2048), Some(want), "{s}");
    }
}

/// 2026-10-18: The bf16 and fp8 widths the circuit hard-coded before reading the registry
/// (2.0 and 1.0 bytes per element) hold at every element count, so their estimates do not move.
#[test]
fn bf16_and_fp8_keep_the_old_element_sizes() {
    for e in [1u64, 7, 128, 1024, 4096, 1 << 20] {
        assert_eq!(token_bytes(pair("bf16"), e), Some(2 * e * 2));
        assert_eq!(token_bytes(pair("fp8"), e), Some(2 * e));
    }
}

/// 2026-10-18: A side count that would wrap the spec's multiply is `None`, not a small number.
#[test]
fn an_overflowing_count_is_none() {
    assert_eq!(side_bytes(pair("bf16").k(), u64::MAX / 8), None);
    assert_eq!(token_bytes(pair("turbo4"), u64::MAX / 32), None);
}

/// 2026-10-18: The setting must be present and name a codec; an alias reads as its pair.
#[test]
fn the_setting_is_required_and_parsed() {
    let mut s = BTreeMap::new();
    assert!(
        kv_pair(&s)
            .unwrap_err()
            .contains("states no `kv_cache_dtype`")
    );
    s.insert(KV_SETTING.to_string(), "fp7".to_string());
    assert!(kv_pair(&s).unwrap_err().contains("fp7"));
    s.insert(KV_SETTING.to_string(), "turbo4k_turbo8v".to_string());
    assert_eq!(kv_pair(&s), Ok(pair("turbo4k_turbo8v")));
}
