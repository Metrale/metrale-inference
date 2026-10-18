// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The [`KvCacheDtype`] catalogue: every variant and its canonical
//! `--kv-cache-dtype` spelling.
//!
//! The flag is an `Option<String>` in clap, so the list of valid values comes
//! from here: the server's `options_for_flag` builds it from
//! [`KvCacheDtype::ALL`] for the CLI manifest and the TUI option picker.
//!
//! Owner: cache.
//! Invariants:
//! - `name` is read from the kv-codec registry through `pair()`, whose match has no
//!   wildcard arm, so a new variant does not compile until it names its codecs.
//! - `ALL` lists the registry codecs in discriminant order, then the pair aliases in table
//!   order (tested below).
//! - Every `ALL` entry's `name` parses back to that entry (tested below).

use super::KvCacheDtype;
use metrale_kv_codec::KvPair;

impl KvCacheDtype {
    /// 2026-09-25: Every variant, in declaration order. Nothing checks that a
    /// new variant is added here; add it together with its [`KvCacheDtype::name`]
    /// arm, which the compiler does require.
    pub const ALL: [KvCacheDtype; 16] = [
        KvCacheDtype::Bf16,
        KvCacheDtype::Fp8,
        KvCacheDtype::Nvfp4,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo3,
        KvCacheDtype::Turbo2,
        KvCacheDtype::Turbo8,
        KvCacheDtype::Turbo4KTurbo3V,
        KvCacheDtype::Turbo4KTurbo8V,
        KvCacheDtype::Turbo3KTurbo8V,
        KvCacheDtype::Bf16KTurbo4V,
        KvCacheDtype::Bf16KTurbo3V,
        KvCacheDtype::Fp8KTurbo4V,
        KvCacheDtype::Fp8KTurbo3V,
        KvCacheDtype::Bf16KTurbo2V,
        KvCacheDtype::Fp8KTurbo2V,
    ];

    /// 2026-09-25: The canonical `--kv-cache-dtype` spelling. `Display`
    /// delegates here, so a listed value is the string `FromStr` accepts; the
    /// tests below check that round trip. 2026-10-18: Read from the registry: the codec name
    /// for a symmetric variant, the pair alias's long spelling otherwise.
    pub fn name(self) -> &'static str {
        let (k, v) = self.pair();
        KvPair::new(k, v)
            .expect("every legacy pair is a registered pair alias")
            .name()
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheDtype;

    #[test]
    fn every_listed_name_parses_back_to_the_variant_that_produced_it() {
        // 2026-09-25: Each listed name must parse, to the variant that
        // produced it, and `Display` must print that same name.
        for dtype in KvCacheDtype::ALL {
            let parsed: KvCacheDtype = dtype
                .name()
                .parse()
                .unwrap_or_else(|e| panic!("{} does not parse: {e:#}", dtype.name()));
            assert_eq!(
                parsed,
                dtype,
                "{} parses to a different dtype",
                dtype.name()
            );
            assert_eq!(
                dtype.to_string(),
                dtype.name(),
                "Display and the catalogue disagree"
            );
        }
    }

    #[test]
    fn the_catalogue_has_no_duplicates() {
        // 2026-09-25: A duplicate would show one choice twice in a picker.
        for (i, a) in KvCacheDtype::ALL.iter().enumerate() {
            for b in &KvCacheDtype::ALL[i + 1..] {
                assert_ne!(a, b, "{} is listed twice", a.name());
            }
        }
    }
}

#[cfg(test)]
mod registry_tests {
    use super::KvCacheDtype;
    use metrale_kv_codec::{KvCodecId, PAIR_ALIASES};

    #[test]
    fn the_legacy_catalogue_is_the_registry_in_order() {
        // 2026-10-18: The legacy view adds no codec and no pair of its own: its variants are
        // exactly the registry's codecs, then its pair aliases, in table order.
        let want: Vec<(KvCodecId, KvCodecId)> = KvCodecId::ALL
            .iter()
            .map(|&c| (c, c))
            .chain(PAIR_ALIASES.iter().map(|a| (a.k, a.v)))
            .collect();
        let got: Vec<(KvCodecId, KvCodecId)> = KvCacheDtype::ALL.iter().map(|d| d.pair()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn kv_pair_sides_are_the_symmetric_variants_of_the_codecs() {
        for d in KvCacheDtype::ALL {
            let (k, v) = d.kv_pair();
            assert_eq!((k.pair().0, v.pair().0), d.pair(), "{}", d.name());
            assert!(!k.is_asymmetric() && !v.is_asymmetric());
        }
    }
}
