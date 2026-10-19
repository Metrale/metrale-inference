// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Golden and coverage tests of the Metal kernel table. The golden names and
//! rotation flags are the ones the Metal forward's dtype enum and `Qwen35Kernels::resolve`
//! spelled before the table existed, written out literally.

use super::*;
use metrale_error::ErrorCode;

fn pair(s: &str) -> KvPair {
    s.parse().expect("accepted spelling")
}

fn n(m: &'static str, f: &'static str) -> KernelName {
    KernelName { module: m, func: f }
}

/// 2026-10-18: (dtype, append module, append fn, decode module, decode fn, K rotated,
/// V rotated), as the deleted `MetalKvDtype` and `Qwen35Kernels` had them.
const LEGACY: &[(&str, &str, &str, &str, &str, bool, bool)] = &[
    (
        "bf16",
        "kv_cache_append",
        "kv_cache_append",
        "attention_decode",
        "attention_decode",
        false,
        false,
    ),
    (
        "turbo8",
        "kv_cache_append_turbo8",
        "kv_cache_append_turbo8",
        "attention_decode_turbo8",
        "attention_decode_turbo8",
        true,
        true,
    ),
    (
        "turbo4",
        "kv_cache_append_turbo4",
        "kv_cache_append_turbo4",
        "attention_decode_turbo4",
        "attention_decode_turbo4",
        true,
        true,
    ),
    (
        "turbo3",
        "kv_cache_append_turbo3",
        "kv_cache_append_turbo3",
        "attention_decode_turbo3",
        "attention_decode_turbo3",
        true,
        true,
    ),
    (
        "turbo2",
        "kv_cache_append_turbo2",
        "kv_cache_append_turbo2",
        "attention_decode_turbo2",
        "attention_decode_turbo2",
        true,
        true,
    ),
    (
        "bf16k_turbo4v",
        "kv_cache_append_bf16k_turbov",
        "kv_cache_append_bf16k_turbo4v",
        "attention_decode_bf16k_turbov",
        "attention_decode_bf16k_turbo4v",
        false,
        true,
    ),
    (
        "bf16k_turbo3v",
        "kv_cache_append_bf16k_turbov",
        "kv_cache_append_bf16k_turbo3v",
        "attention_decode_bf16k_turbov",
        "attention_decode_bf16k_turbo3v",
        false,
        true,
    ),
    (
        "bf16k_turbo2v",
        "kv_cache_append_bf16k_turbov",
        "kv_cache_append_bf16k_turbo2v",
        "attention_decode_bf16k_turbov",
        "attention_decode_bf16k_turbo2v",
        false,
        true,
    ),
];

/// 2026-10-18: Each formerly supported Metal dtype resolves the same append and decode kernels,
/// its sides rotate exactly as the enum said, and a rotated cache gets the one WHT pair.
#[test]
fn names_and_rotation_match_the_legacy_enum() {
    for &(s, am, af, dm, df, k_rot, v_rot) in LEGACY {
        let p = pair(s);
        let names = contiguous_kv_kernel_names(p).unwrap_or_else(|e| panic!("{s}: {e}"));
        assert_eq!(names.write, n(am, af), "{s}");
        assert_eq!(names.decode, n(dm, df), "{s}");
        assert_eq!(spec(p.k()).rotates(), k_rot, "{s} K");
        assert_eq!(spec(p.v()).rotates(), v_rot, "{s} V");
        let want_wht = (k_rot || v_rot).then_some((
            n("wht_bf16", "wht_bf16_inplace"),
            n("wht_bf16", "wht_bf16_inplace_inv"),
        ));
        assert_eq!(names.wht, want_wht, "{s}");
        assert_eq!(
            (names.decode_512, names.splitk, names.prefill),
            (None, None, None),
            "{s}"
        );
    }
}

/// 2026-10-18: The pairs Metal serves are exactly the legacy enum's, in its order, so the
/// startup resolution loads the same kernel set.
#[test]
fn pairs_are_the_legacy_set() {
    let got: Vec<&str> = contiguous_kv_pairs().map(KvPair::name).collect();
    let want: Vec<&str> = LEGACY.iter().map(|row| row.0).collect();
    assert_eq!(got, want);
}

/// 2026-10-18: Every accepted pair the legacy enum did not parse is refused with
/// `KvCapabilityMissing`, never served with another pair's kernels.
#[test]
fn other_pairs_are_refused() {
    let accepted = KvCodecId::ALL
        .into_iter()
        .map(KvPair::symmetric)
        .chain(crate::PAIR_ALIASES.iter().map(|a| pair(a.long)));
    let mut refused = 0;
    for p in accepted {
        if LEGACY.iter().any(|row| row.0 == p.name()) {
            continue;
        }
        let err = contiguous_kv_kernel_names(p).expect_err(p.name());
        assert_eq!(err.code(), ErrorCode::KvCapabilityMissing, "{p}");
        refused += 1;
    }
    assert!(refused >= 2, "fp8 and nvfp4 at least");
}

/// 2026-10-18: No key appears twice.
#[test]
fn keys_are_unique() {
    let all: Vec<&KernelRow> = rows().collect();
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a.key, b.key, "duplicate row {:?}", a.key);
        }
    }
}
