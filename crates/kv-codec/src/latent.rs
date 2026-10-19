// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The caches of a compressed-latent sparse-attention layer: a sliding-window ring
//! of per-token rows on every layer, and on the layers that compress, one latent row and one
//! indexer key row per compressed group. Each cache is a registered codec's rows (packed
//! values, then the group scales), so the layer is sized from the specs the way the paged
//! attention cache is.
//!
//! Owner: kv-codec.
//! Invariants:
//! - A plan holds an indexer codec for the key rows and a non-indexer codec for the window and
//!   latent rows, each unrotated and block-scaled, with a host dequantization; anything else
//!   is a coded `KvLatentGeometry` error.
//! - [`LatentKvPlan::row_bytes`] takes only a row that divides into whole groups and whole
//!   bytes.

use alloc::format;
use alloc::string::String;

use metrale_error::{Error, ErrorCode, error};

use crate::spec::{AbiId, CodecRole, Rotation};
use crate::{KvCodecId, spec};

/// 2026-10-19: Which of the three caches a row belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LatentCache {
    /// 2026-10-19: The sliding-window ring, every layer.
    Window = 0,
    /// 2026-10-19: The compressed latent rows.
    Latent = 1,
    /// 2026-10-19: The indexer's compressed key rows.
    IndexKey = 2,
}

impl LatentCache {
    /// 2026-10-19: `"window"`, `"latent"` or `"index key"`.
    pub const fn name(self) -> &'static str {
        match self {
            LatentCache::Window => "window",
            LatentCache::Latent => "latent",
            LatentCache::IndexKey => "index key",
        }
    }
}

/// 2026-10-19: Stored bytes of one row of each cache at one head geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LatentRowBytes {
    /// 2026-10-19: One window ring row.
    pub window: u64,
    /// 2026-10-19: One compressed latent row.
    pub latent: u64,
    /// 2026-10-19: One indexer key row.
    pub index_key: u64,
}

/// 2026-10-19: The codec of each cache of one model's compressed-latent attention, chosen once
/// at load (ROTI).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LatentKvPlan {
    window: KvCodecId,
    latent: KvCodecId,
    index_key: KvCodecId,
}

impl LatentKvPlan {
    /// 2026-10-19: Refuses an index-key codec that is not an indexer codec, a window or latent
    /// codec that is one, and any codec that is rotated, not block-scaled or has no host
    /// dequantization.
    pub fn new(window: KvCodecId, latent: KvCodecId, index_key: KvCodecId) -> Result<Self, Error> {
        for (cache, id) in [
            (LatentCache::Window, window),
            (LatentCache::Latent, latent),
            (LatentCache::IndexKey, index_key),
        ] {
            let s = spec(id);
            let indexer = s.role == CodecRole::Indexer;
            if indexer != (cache == LatentCache::IndexKey) {
                return Err(refuse(cache, format!("{} has role {:?}", s.name, s.role)));
            }
            let plain = matches!(s.rotation, Rotation::None)
                && s.abi == AbiId::BlockScaled
                && s.group > 0
                && s.host_dequant.is_some();
            if !plain {
                return Err(refuse(
                    cache,
                    format!("{} is not an unrotated block-scaled codec", s.name),
                ));
            }
        }
        Ok(Self {
            window,
            latent,
            index_key,
        })
    }

    /// 2026-10-19: The codec of `cache`.
    pub const fn codec(&self, cache: LatentCache) -> KvCodecId {
        match cache {
            LatentCache::Window => self.window,
            LatentCache::Latent => self.latent,
            LatentCache::IndexKey => self.index_key,
        }
    }

    /// 2026-10-19: Bytes of one `cache` row of `values` values: the packed values, then the
    /// group scales. Refuses a row that is not whole groups of whole bytes.
    pub fn row_bytes(&self, cache: LatentCache, values: u32) -> Result<u64, Error> {
        let s = spec(self.codec(cache));
        let v = u64::from(values);
        let whole = values > 0
            && values.is_multiple_of(s.group)
            && (v * u64::from(s.value_bits)).is_multiple_of(8);
        if !whole {
            return Err(refuse(
                cache,
                format!(
                    "a row of {values} values is not whole groups of {} in whole bytes of {}",
                    s.group, s.name
                ),
            ));
        }
        Ok(s.data_bytes(v) + s.side_bytes(v))
    }

    /// 2026-10-19: The three row sizes at `head_dim` (window and latent rows) and `index_dim`
    /// (index-key rows).
    pub fn rows(&self, head_dim: u32, index_dim: u32) -> Result<LatentRowBytes, Error> {
        Ok(LatentRowBytes {
            window: self.row_bytes(LatentCache::Window, head_dim)?,
            latent: self.row_bytes(LatentCache::Latent, head_dim)?,
            index_key: self.row_bytes(LatentCache::IndexKey, index_dim)?,
        })
    }
}

impl LatentRowBytes {
    /// 2026-10-19: One sequence's bytes for one layer: a ring of `window` rows, and on a
    /// compressing layer `groups` latent and index-key rows.
    pub const fn layer_bytes(&self, window: u64, groups: Option<u64>) -> u64 {
        let ring = window * self.window;
        match groups {
            Some(g) => ring + g * (self.latent + self.index_key),
            None => ring,
        }
    }
}

fn refuse(cache: LatentCache, why: String) -> Error {
    error!(ErrorCode::KvLatentGeometry, cache.name(), why)
}

#[cfg(test)]
#[path = "latent_tests.rs"]
mod tests;
