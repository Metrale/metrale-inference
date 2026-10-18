// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Golden and coverage tests of the kernel table. The golden names are the ones the
//! engine's per-dtype matches spelled before the table existed, written out literally so a
//! transcription slip in a table macro cannot pass.

use super::*;
use crate::PAIR_ALIASES;

fn pair(s: &str) -> KvPair {
    s.parse().expect("accepted spelling")
}

fn n(m: &'static str, f: &'static str) -> KernelName {
    KernelName { module: m, func: f }
}

/// 2026-10-18: Every pair `--kv-cache-dtype` accepts.
fn all_pairs() -> Vec<KvPair> {
    KvCodecId::ALL
        .into_iter()
        .map(KvPair::symmetric)
        .chain(PAIR_ALIASES.iter().map(|a| pair(a.long)))
        .collect()
}

/// 2026-10-18: (dtype, reshape module, reshape fn, decode module at head_dim 256, decode fn),
/// as `kernel_modules_for_dtype` returned them.
const LEGACY_WRITE_DECODE: &[(&str, &str, &str, &str, &str)] = &[
    (
        "bf16",
        "reshape_and_cache",
        "reshape_and_cache_flash",
        "paged_decode",
        "paged_decode_attn",
    ),
    (
        "fp8",
        "reshape_and_cache",
        "reshape_and_cache_flash_fp8",
        "paged_decode_fp8",
        "paged_decode_attn_fp8",
    ),
    (
        "nvfp4",
        "reshape_and_cache",
        "reshape_and_cache_flash_nvfp4",
        "paged_decode_nvfp4",
        "paged_decode_attn_nvfp4",
    ),
    (
        "turbo4",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo4",
        "paged_decode_turbo4",
        "paged_decode_attn_turbo4",
    ),
    (
        "turbo3",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo3",
        "paged_decode_turbo3",
        "paged_decode_attn_turbo3",
    ),
    (
        "turbo2",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo2",
        "paged_decode_turbo2",
        "paged_decode_attn_turbo2",
    ),
    (
        "turbo8",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo8",
        "paged_decode_turbo8",
        "paged_decode_attn_turbo8",
    ),
    (
        "bf16k_turbo3v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_bf16k_turbo3v",
        "paged_decode_bf16k_turbo3v",
        "paged_decode_attn_bf16k_turbo3v",
    ),
    (
        "bf16k_turbo4v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_bf16k_turbo4v",
        "paged_decode_bf16k_turbo4v",
        "paged_decode_attn_bf16k_turbo4v",
    ),
    (
        "bf16k_turbo2v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_bf16k_turbo2v",
        "paged_decode_bf16k_turbo2v",
        "paged_decode_attn_bf16k_turbo2v",
    ),
    (
        "fp8k_turbo3v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_fp8k_turbo3v",
        "paged_decode_fp8k_turbo3v",
        "paged_decode_attn_fp8k_turbo3v",
    ),
    (
        "fp8k_turbo4v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_fp8k_turbo4v",
        "paged_decode_fp8k_turbo4v",
        "paged_decode_attn_fp8k_turbo4v",
    ),
    (
        "fp8k_turbo2v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_fp8k_turbo2v",
        "paged_decode_fp8k_turbo2v",
        "paged_decode_attn_fp8k_turbo2v",
    ),
    (
        "turbo4k_turbo3v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo4k_turbo3v",
        "paged_decode_turbo4k_turbo3v",
        "paged_decode_attn_turbo4k_turbo3v",
    ),
    (
        "turbo4k_turbo8v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo4k_turbo8v",
        "paged_decode_turbo4k_turbo8v",
        "paged_decode_attn_turbo4k_turbo8v",
    ),
    (
        "turbo3k_turbo8v",
        "reshape_and_cache_turbo",
        "reshape_and_cache_flash_turbo3k_turbo8v",
        "paged_decode_turbo3k_turbo8v",
        "paged_decode_attn_turbo3k_turbo8v",
    ),
];

/// 2026-10-18: (dtype, prefill module, prefill fn), as `required_optional_kernels_for_dtype` and
/// `PrefillKernels` spelled them.
const LEGACY_PREFILL: &[(&str, &str, &str)] = &[
    ("bf16", "prefill_paged", "attn_prefill_paged_64"),
    ("fp8", "prefill_paged_fp8", "attn_prefill_paged_fp8_64"),
    (
        "nvfp4",
        "prefill_paged_nvfp4",
        "attn_prefill_paged_nvfp4_64",
    ),
    (
        "turbo2",
        "prefill_paged_turbo2",
        "attn_prefill_paged_turbo2",
    ),
    (
        "turbo3",
        "prefill_paged_turbo3",
        "attn_prefill_paged_turbo3_64",
    ),
    (
        "turbo4",
        "prefill_paged_turbo4",
        "attn_prefill_paged_turbo4_64",
    ),
    (
        "turbo8",
        "prefill_paged_turbo8",
        "attn_prefill_paged_turbo8_64",
    ),
    (
        "bf16k_turbo3v",
        "prefill_paged_bf16k_turbo3v",
        "attn_prefill_paged_bf16k_turbo3v_64",
    ),
    (
        "bf16k_turbo4v",
        "prefill_paged_bf16k_turbo4v",
        "attn_prefill_paged_bf16k_turbo4v_64",
    ),
    (
        "bf16k_turbo2v",
        "prefill_paged_bf16k_turbo2v",
        "attn_prefill_paged_bf16k_turbo2v_64",
    ),
    (
        "fp8k_turbo3v",
        "prefill_paged_fp8k_turbo3v",
        "attn_prefill_paged_fp8k_turbo3v_64",
    ),
    (
        "fp8k_turbo4v",
        "prefill_paged_fp8k_turbo4v",
        "attn_prefill_paged_fp8k_turbo4v_64",
    ),
    (
        "fp8k_turbo2v",
        "prefill_paged_fp8k_turbo2v",
        "attn_prefill_paged_fp8k_turbo2v_64",
    ),
    (
        "turbo4k_turbo3v",
        "prefill_paged_turbo4k_turbo3v",
        "attn_prefill_paged_turbo4k_turbo3v_64",
    ),
    (
        "turbo4k_turbo8v",
        "prefill_paged_turbo4k_turbo8v",
        "attn_prefill_paged_turbo4k_turbo8v_64",
    ),
    (
        "turbo3k_turbo8v",
        "prefill_paged_turbo3k_turbo8v",
        "attn_prefill_paged_turbo3k_turbo8v_64",
    ),
];

/// 2026-10-18: The write and decode names at head_dim 256 are the legacy ones, and at head_dim
/// 128 a WHT-rotated pair takes the `_128` decode module (Turbo2 and the turbo-K/turbo-V pairs
/// included) while a plain codec keeps its one module.
#[test]
fn write_and_decode_names_match_the_legacy_dispatch() {
    assert_eq!(LEGACY_WRITE_DECODE.len(), all_pairs().len());
    for &(d, rm, rf, dm, df) in LEGACY_WRITE_DECODE {
        let p = pair(d);
        let at256 = kv_kernel_names(p, 256).expect(d);
        assert_eq!(at256.write, n(rm, rf), "{d} write");
        assert_eq!(at256.decode, n(dm, df), "{d} decode at 256");
        let at128 = kv_kernel_names(p, 128).expect(d);
        assert_eq!(at128.write, at256.write, "{d} write at 128");
        assert_eq!(at128.decode.func, df, "{d} decode fn at 128");
        let rotated = spec(p.k()).rotates() || spec(p.v()).rotates();
        if rotated {
            assert_eq!(
                at128.decode.module,
                format!("{dm}_128"),
                "{d} decode module at 128"
            );
        } else {
            assert_eq!(at128.decode.module, dm, "{d} decode module at 128");
        }
        assert_eq!(
            kv_kernel_names(p, 64).expect(d).decode,
            at128.decode,
            "{d} at 64"
        );
    }
}

/// 2026-10-18: Every pair resolves the legacy chunked-prefill kernel.
#[test]
fn prefill_names_match_the_legacy_requirements() {
    assert_eq!(LEGACY_PREFILL.len(), all_pairs().len());
    for &(d, m, f) in LEGACY_PREFILL {
        for hd in [128, 256] {
            assert_eq!(
                kv_kernel_names(pair(d), hd).expect(d).prefill,
                Some(n(m, f)),
                "{d} {hd}"
            );
        }
    }
}

/// 2026-10-18: Only bf16, fp8, turbo4 and turbo8 have a head_dim-512 decode build. Turbo3 and
/// Turbo2 used to load the turbo4 build here, which decodes their blocks with the wrong
/// codebook; nvfp4 and the pairs loaded the fp8 build. Each is now `None`.
#[test]
fn decode_512_is_the_pairs_own_build_or_none() {
    let want: &[(&str, Option<KernelName>)] = &[
        (
            "bf16",
            Some(n("paged_decode_attn_512", "paged_decode_attn")),
        ),
        (
            "fp8",
            Some(n("paged_decode_attn_fp8_512", "paged_decode_attn_fp8")),
        ),
        (
            "turbo4",
            Some(n("paged_decode_turbo4_512", "paged_decode_attn_turbo4")),
        ),
        (
            "turbo8",
            Some(n("paged_decode_turbo8_512", "paged_decode_attn_turbo8")),
        ),
    ];
    for p in all_pairs() {
        let expected = want
            .iter()
            .find(|(d, _)| *d == p.name())
            .and_then(|(_, k)| *k);
        assert_eq!(
            kv_kernel_names(p, 512)
                .unwrap_or_else(|e| panic!("{p}: {e}"))
                .decode_512,
            expected,
            "{p}"
        );
    }
}

/// 2026-10-18: Split-K exists for fp8 and nvfp4 only. bf16 and turbo2 used to load the fp8
/// pair, which nothing on their decode path reads.
#[test]
fn splitk_is_fp8_and_nvfp4_only() {
    for p in all_pairs() {
        let got = kv_kernel_names(p, 256)
            .unwrap_or_else(|e| panic!("{p}: {e}"))
            .splitk;
        let want = match p.name() {
            "fp8" => Some((
                n("paged_decode_fp8", "paged_decode_attn_splitk_fp8"),
                n("paged_decode_fp8", "paged_decode_attn_reduce_fp8"),
            )),
            "nvfp4" => Some((
                n("paged_decode_nvfp4", "paged_decode_attn_splitk_nvfp4"),
                n("paged_decode_nvfp4", "paged_decode_attn_reduce_nvfp4"),
            )),
            _ => None,
        };
        assert_eq!(got, want, "{p}");
    }
}

/// 2026-10-18: The WHT bookends are present exactly when a side is rotated and the head dim is
/// 128, 256 or 512, as the legacy requirement list had it.
#[test]
fn wht_bookends_follow_rotation_and_head_dim() {
    let bookends = Some((
        n("wht_bf16", "wht_bf16_inplace"),
        n("wht_bf16", "wht_bf16_inplace_inv"),
    ));
    for p in all_pairs() {
        let rotated = spec(p.k()).rotates() || spec(p.v()).rotates();
        for hd in [64, 128, 192, 256, 512, 513] {
            let want = if rotated && matches!(hd, 128 | 256 | 512) {
                bookends
            } else {
                None
            };
            assert_eq!(
                kv_kernel_names(p, hd)
                    .unwrap_or_else(|e| panic!("{p}: {e}"))
                    .wht,
                want,
                "{p} {hd}"
            );
        }
    }
}

/// 2026-10-18: No key has two rows, and no two keys share a digest.
#[test]
fn keys_and_digests_are_unique() {
    let all: Vec<&KernelRow> = rows().collect();
    for (i, a) in all.iter().enumerate() {
        for b in &all[i + 1..] {
            assert_ne!(a.key, b.key, "duplicate row {:?}", a.key);
            assert_ne!(a.key.digest(), b.key.digest(), "{:?} vs {:?}", a.key, b.key);
        }
    }
}

/// 2026-10-18: A row keyed by an asymmetric pair names that pair, and every row's key is one
/// `KvKernelKey::new` can produce (a symmetric row has no V codec).
#[test]
fn row_keys_name_accepted_pairs() {
    for row in rows() {
        if let Some(v) = row.key.v {
            assert_ne!(row.key.k, v, "{:?}: symmetric key with a V codec", row.key);
            let p = KvPair::new(row.key.k, v).expect("row names an accepted pair");
            assert!(row.name.func.contains(p.name()) || row.name.module.contains(p.name()));
        }
    }
}

/// 2026-10-18: A pair with no fused kernel is refused with `KvPairUnsupported`, never served
/// with another pair's kernels.
#[test]
fn pair_without_kernels_is_refused() {
    let p = KvPair::from_parts(KvCodecId::Nvfp4, KvCodecId::Turbo3);
    let err = kv_kernel_names(p, 256).expect_err("nvfp4/turbo3 has no fused kernel");
    assert_eq!(err.code(), ErrorCode::KvPairUnsupported);
    assert_eq!(KvKernelKey::new(KvOp::Write, p, Model).kernel_name(), None);
}
