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

use std::collections::BTreeMap;

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
