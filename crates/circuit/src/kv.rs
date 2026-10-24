// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: The KV cache as the circuit counts it: the `kv_cache_dtype` setting read as a
//! codec pair, and the bytes one side holds, from that codec's spec in the registry. Estimate,
//! roofline, memory and the requirement check all read the cache through here, so every codec
//! the registry has is sized the same way the engine sizes its blocks.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A side's bytes are its codec's packed values plus its in-block side tensors
//!   (`KvCodecSpec::data_bytes` + `side_bytes`); nothing here restates a codec's width.
//! - A missing or unknown `kv_cache_dtype` is an error, never a default.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use metrale_kv_codec::{KvCodecId, KvPair, spec};

/// 2026-10-18: The policy setting (and the state format key) naming the KV cache's codec pair.
pub const KV_SETTING: &str = "kv_cache_dtype";

/// 2026-10-18: The codec pair `text` names, in any spelling `--kv-cache-dtype` accepts.
pub fn parse_kv(text: &str) -> Result<KvPair, String> {
    text.parse::<KvPair>()
        .map_err(|e| format!("`{KV_SETTING} = {text}`: {e}"))
}

/// 2026-10-18: The codec pair `settings` states.
pub fn kv_pair(settings: &BTreeMap<String, String>) -> Result<KvPair, String> {
    let text = settings
        .get(KV_SETTING)
        .ok_or_else(|| format!("the policy states no `{KV_SETTING}`"))?;
    parse_kv(text)
}

/// 2026-10-18: Bytes of `elements` values of one side stored as `codec`; `None` on overflow.
pub fn side_bytes(codec: KvCodecId, elements: u64) -> Option<u64> {
    // 2026-10-18: The spec multiplies `elements` by at most 64 bits; past this it would wrap.
    elements.checked_mul(64)?;
    let s = spec(codec);
    s.data_bytes(elements).checked_add(s.side_bytes(elements))
}

/// 2026-10-18: Bytes of one token of both sides of a `pair` cache, `elements` values per side.
pub fn token_bytes(pair: KvPair, elements: u64) -> Option<u64> {
    side_bytes(pair.k(), elements)?.checked_add(side_bytes(pair.v(), elements)?)
}

#[cfg(test)]
#[path = "kv_tests.rs"]
mod tests;

/// 2026-10-22: A [`KvPair`] serialized as its canonical `--kv-cache-dtype` spelling and read
/// back through the registry ([`KvPair::from_str`]), so a decoded pair is always one the
/// registry accepts.
pub(crate) mod pair_spelling {
    use core::str::FromStr;

    use metrale_kv_codec::KvPair;
    use serde::{Deserialize, Deserializer, Serializer};

    use crate::prelude::*;

    pub(crate) fn serialize<S: Serializer>(p: &KvPair, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(p.name())
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<KvPair, D::Error> {
        let name = String::deserialize(d)?;
        KvPair::from_str(&name).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod pair_spelling_tests {
    use crate::state::StateDtype;
    use metrale_kv_codec::KvPair;

    /// 2026-10-22: Every pair the registry accepts reads back as itself; an unknown spelling is
    /// refused rather than defaulted.
    #[test]
    fn every_accepted_pair_round_trips_and_an_unknown_one_is_refused() {
        for p in KvPair::all() {
            let text = serde_json::to_string(&StateDtype::Kv(p)).unwrap();
            assert_eq!(
                serde_json::from_str::<StateDtype>(&text).unwrap(),
                StateDtype::Kv(p)
            );
        }
        assert!(serde_json::from_str::<StateDtype>("{\"Kv\":\"not_a_codec\"}").is_err());
    }
}
