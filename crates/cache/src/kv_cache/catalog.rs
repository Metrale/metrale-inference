// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The [`KvCacheDtype`] catalogue: every format and its canonical
//! `--kv-cache-dtype` spelling.
//!
//! The flag is an `Option<String>` in clap, so the list of valid values comes
//! from here: the server's `options_for_flag` builds it from
//! [`KvCacheDtype::all`] for the CLI manifest and the TUI option picker.
//!
//! Owner: cache.
//! Invariants:
//! - `name` is read from the kv-codec registry pair; this catalogue holds no spelling.
//! - `all` lists the registry attention codecs in discriminant order, then the pair aliases in
//!   table order (tested below), so a codec registered in kv-codec is listed here unedited.
//! - Every `all` entry's `name` parses back to that entry (tested below).

use super::KvCacheDtype;
use metrale_kv_codec::KvPair;

impl KvCacheDtype {
    /// 2026-10-18: Every format the registry accepts: each attention codec on both sides, then
    /// each pair alias, in registry order.
    pub fn all() -> impl Iterator<Item = KvCacheDtype> {
        KvPair::all().map(KvCacheDtype)
    }

    /// 2026-09-25: The canonical `--kv-cache-dtype` spelling. `Display`
    /// delegates here, so a listed value is the string `FromStr` accepts; the
    /// tests below check that round trip. 2026-10-18: Read from the registry: the codec name
    /// for a symmetric format, the pair alias's long spelling otherwise.
    pub fn name(self) -> &'static str {
        self.codec_pair().name()
    }

    /// 2026-10-18: The registry pair this format is.
    pub const fn codec_pair(self) -> KvPair {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheDtype;

    #[test]
    fn every_listed_name_parses_back_to_the_variant_that_produced_it() {
        // 2026-09-25: Each listed name must parse, to the variant that
        // produced it, and `Display` must print that same name.
        for dtype in KvCacheDtype::all() {
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
        let all: Vec<KvCacheDtype> = KvCacheDtype::all().collect();
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
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
        // exactly the registry's attention codecs, then its pair aliases, in table order.
        let want: Vec<(KvCodecId, KvCodecId)> = KvCodecId::ATTENTION
            .iter()
            .map(|&c| (c, c))
            .chain(PAIR_ALIASES.iter().map(|a| (a.k, a.v)))
            .collect();
        let got: Vec<(KvCodecId, KvCodecId)> = KvCacheDtype::all().map(|d| d.pair()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn kv_pair_sides_are_the_symmetric_variants_of_the_codecs() {
        for d in KvCacheDtype::all() {
            let (k, v) = d.kv_pair();
            assert_eq!((k.pair().0, v.pair().0), d.pair(), "{}", d.name());
            assert!(!k.is_asymmetric() && !v.is_asymmetric());
        }
    }
}
