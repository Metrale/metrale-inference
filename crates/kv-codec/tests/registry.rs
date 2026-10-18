// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The round-trip test: `name`, `FromStr`, `ALL`, the `#[repr(u8)]` discriminants
//! and the pair aliases are bijective, and a refused spelling is a coded error that lists every
//! accepted one.

use metrale_error::ErrorCode;
use metrale_kv_codec::{KvCodecId, KvPair, PAIR_ALIASES, SPEC, accepted, spec};

#[test]
fn name_parse_and_discriminant_are_bijective() {
    for (i, id) in KvCodecId::ALL.into_iter().enumerate() {
        assert_eq!(id as usize, i);
        assert_eq!(KvCodecId::from_u8(id as u8), Some(id));
        assert_eq!(id.name().parse::<KvCodecId>().ok(), Some(id));
        assert_eq!(spec(id).id, id);
        assert_eq!(spec(id).name, id.name());
    }
    assert_eq!(SPEC.len(), KvCodecId::ALL.len());
    assert_eq!(KvCodecId::from_u8(u8::try_from(SPEC.len()).unwrap()), None);
    for (i, a) in KvCodecId::ALL.iter().enumerate() {
        for b in &KvCodecId::ALL[i + 1..] {
            assert_ne!(a.name(), b.name());
        }
    }
}

#[test]
fn the_legacy_spellings_parse_to_the_same_pairs() {
    // 2026-10-18: Every `--kv-cache-dtype` value the pre-registry `KvCacheDtype::from_str`
    // accepted, with the (K, V) pair it meant.
    use KvCodecId::*;
    let legacy: [(&str, KvCodecId, KvCodecId); 25] = [
        ("bf16", Bf16, Bf16),
        ("fp8", Fp8, Fp8),
        ("nvfp4", Nvfp4, Nvfp4),
        ("turbo4", Turbo4, Turbo4),
        ("turbo3", Turbo3, Turbo3),
        ("turbo2", Turbo2, Turbo2),
        ("turbo8", Turbo8, Turbo8),
        ("turbo4k_turbo3v", Turbo4, Turbo3),
        ("turbo4k3v", Turbo4, Turbo3),
        ("turbo4k_turbo8v", Turbo4, Turbo8),
        ("turbo4k8v", Turbo4, Turbo8),
        ("turbo3k_turbo8v", Turbo3, Turbo8),
        ("turbo3k8v", Turbo3, Turbo8),
        ("bf16k_turbo4v", Bf16, Turbo4),
        ("bf16k4v", Bf16, Turbo4),
        ("bf16k_turbo3v", Bf16, Turbo3),
        ("bf16k3v", Bf16, Turbo3),
        ("fp8k_turbo4v", Fp8, Turbo4),
        ("fp8k4v", Fp8, Turbo4),
        ("fp8k_turbo3v", Fp8, Turbo3),
        ("fp8k3v", Fp8, Turbo3),
        ("bf16k_turbo2v", Bf16, Turbo2),
        ("bf16k2v", Bf16, Turbo2),
        ("fp8k_turbo2v", Fp8, Turbo2),
        ("fp8k2v", Fp8, Turbo2),
    ];
    for (s, k, v) in legacy {
        let p: KvPair = s.parse().unwrap_or_else(|e| panic!("{s}: {e}"));
        assert_eq!((p.k(), p.v()), (k, v), "{s}");
        assert_eq!(p.name().parse::<KvPair>().ok(), Some(p), "{s}");
    }
    assert_eq!(PAIR_ALIASES.len() * 2 + KvCodecId::ALL.len(), legacy.len());
}

#[test]
fn pair_names_are_unique_and_never_shadow_a_codec() {
    let mut all: Vec<&str> = KvCodecId::ALL.iter().map(|c| c.name()).collect();
    for a in PAIR_ALIASES {
        assert_ne!(a.k, a.v, "{} is symmetric", a.long);
        all.push(a.long);
        all.push(a.short);
    }
    let n = all.len();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), n, "a spelling is listed twice");
}

#[test]
fn an_unknown_spelling_is_a_coded_error_listing_every_accepted_one() {
    let e = "turbo5".parse::<KvPair>().unwrap_err();
    assert_eq!(e.code(), ErrorCode::KvCodecUnknown);
    let text = e.to_string();
    for id in KvCodecId::ALL {
        assert!(text.contains(id.name()), "{text} omits {}", id.name());
    }
    for a in PAIR_ALIASES {
        assert!(text.contains(a.long) && text.contains(a.short), "{text}");
    }
    assert!(text.contains(&accepted()));
    // 2026-10-18: a pair spelling is not a codec.
    let e = "turbo4k3v".parse::<KvCodecId>().unwrap_err();
    assert_eq!(e.code(), ErrorCode::KvCodecUnknown);
}

#[test]
fn an_unaliased_pair_is_refused() {
    use KvCodecId::*;
    let e = KvPair::new(Turbo3, Turbo4).unwrap_err();
    assert_eq!(e.code(), ErrorCode::KvPairUnsupported);
    assert!(KvPair::new(Turbo4, Turbo3).is_ok());
    assert!(KvPair::new(Nvfp4, Nvfp4).is_ok());
    assert!(!KvPair::symmetric(Turbo8).is_asymmetric());
}
