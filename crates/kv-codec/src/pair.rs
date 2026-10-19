// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-18: Parsing `--kv-cache-dtype`: a codec name gives a symmetric [`KvPair`], a
//! [`PAIR_ALIASES`] spelling gives an asymmetric one. Anything else is a coded
//! `KvCodecUnknown` error whose text lists every accepted spelling, read from the registry, so
//! the message cannot drift from the table.
//!
//! Owner: kv-codec.
//! Invariants:
//! - `KvPair::name` parses back to the same pair, for every pair `KvPair::new` accepts.
//! - An asymmetric pair exists only when a pair alias names it.
//! - Only an attention codec (`CodecRole::Attention`) is a side: an indexer codec parses as a
//!   [`KvCodecId`] but never as a [`KvPair`].

use alloc::string::String;
use core::str::FromStr;

use metrale_error::{Error, ErrorCode, error};

use crate::registry::{PAIR_ALIASES, PairAlias, by_name, const_str_eq};
use crate::spec::{AutoBoundary, CodecRole};
use crate::{KvCodecId, spec};

impl FromStr for KvCodecId {
    type Err = Error;
    /// 2026-10-18: The codec, attention or indexer, whose canonical name is `s`. A pair spelling
    /// is not a codec: parse it as a [`KvPair`].
    fn from_str(s: &str) -> Result<Self, Error> {
        by_name(s).ok_or_else(|| error!(ErrorCode::KvCodecUnknown, String::from(s), accepted()))
    }
}

/// 2026-10-18: The codec of each side of an attention layer's cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KvPair {
    k: KvCodecId,
    v: KvCodecId,
}

impl KvPair {
    /// 2026-10-18: The same codec on both sides.
    pub const fn symmetric(id: KvCodecId) -> Self {
        KvPair { k: id, v: id }
    }

    /// 2026-10-18: `(k, v)`, refused with `KvPairUnsupported` when either side is not an
    /// attention codec, or the sides differ and no pair alias (hence no fused kernel) exists for
    /// them.
    pub fn new(k: KvCodecId, v: KvCodecId) -> Result<Self, Error> {
        let sides = is_attention(k) && is_attention(v);
        if sides && (k == v || alias_of(k, v).is_some()) {
            Ok(KvPair { k, v })
        } else {
            Err(error!(
                ErrorCode::KvPairUnsupported,
                k.name(),
                v.name(),
                "any"
            ))
        }
    }

    /// 2026-10-18: A pair already known to be accepted (it came out of a resolved plan).
    pub(crate) const fn from_parts(k: KvCodecId, v: KvCodecId) -> Self {
        KvPair { k, v }
    }

    /// 2026-10-18: The K-side codec.
    pub const fn k(self) -> KvCodecId {
        self.k
    }

    /// 2026-10-18: The V-side codec.
    pub const fn v(self) -> KvCodecId {
        self.v
    }

    /// 2026-10-18: The boundary layers an auto `--kv-high-precision-layers` gives this pair:
    /// the pair alias row's policy when asymmetric, else the codec row's.
    pub fn auto_boundary(self) -> AutoBoundary {
        match alias_of(self.k, self.v) {
            Some(alias) => alias.auto_boundary,
            None => spec(self.k).auto_boundary,
        }
    }

    /// 2026-10-18: True when K and V use different codecs.
    pub fn is_asymmetric(self) -> bool {
        self.k != self.v
    }

    /// 2026-10-18: The canonical spelling: the codec name when symmetric, else the pair alias's
    /// long spelling.
    pub fn name(self) -> &'static str {
        match alias_of(self.k, self.v) {
            Some(alias) => alias.long,
            None => self.k.name(),
        }
    }
}

impl FromStr for KvPair {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self, Error> {
        if let Some(alias) = PAIR_ALIASES
            .iter()
            .find(|a| const_str_eq(a.long, s) || const_str_eq(a.short, s))
        {
            return Ok(KvPair {
                k: alias.k,
                v: alias.v,
            });
        }
        KvCodecId::ATTENTION
            .into_iter()
            .find(|id| id.name() == s)
            .map(KvPair::symmetric)
            .ok_or_else(|| error!(ErrorCode::KvCodecUnknown, String::from(s), accepted()))
    }
}

impl core::fmt::Display for KvPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// 2026-10-18: True when `id` can be a `--kv-cache-dtype` side.
fn is_attention(id: KvCodecId) -> bool {
    spec(id).role == CodecRole::Attention
}

/// 2026-10-18: The alias naming `(k, v)`, when one exists.
fn alias_of(k: KvCodecId, v: KvCodecId) -> Option<&'static PairAlias> {
    PAIR_ALIASES.iter().find(|a| a.k == k && a.v == v)
}

/// 2026-10-18: Every accepted `--kv-cache-dtype` spelling, comma separated: the attention
/// codecs, then each pair's long and short form.
pub fn accepted() -> String {
    let mut out = String::new();
    let mut push = |name: &str| {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(name);
    };
    for id in KvCodecId::ATTENTION {
        push(id.name());
    }
    for alias in PAIR_ALIASES {
        push(alias.long);
        push(alias.short);
    }
    out
}
