// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-attention-layer KV cache dtypes. 2026-10-18: The boundary-layer rule and the
//! auto boundary count moved into the codec registry (`KvCachePlan::resolve`,
//! `KvPair::auto_boundary`); this file derives the legacy per-layer `KvCacheDtype` list, which
//! the loaders and the cache config still take, from the resolved plan.
//!
//! Owner: server startup (`met serve`).
//! Invariants: the list is empty when every layer holds the requested pair, else it has one
//! entry per plan layer.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype as Legacy;
use metrale_error::{ErrorCode, error};
use metrale_kv_codec::{KvCachePlan, KvPair};

/// 2026-10-18: The legacy dtype of `pair`. Every pair the registry accepts has one
/// (`metrale-cache`'s registry tests); a miss is a coded error, never a guess.
pub(crate) fn legacy(pair: KvPair) -> Result<Legacy> {
    Legacy::from_pair(pair.k(), pair.v()).ok_or_else(|| {
        error!(
            ErrorCode::KvPairUnsupported,
            pair.k().name(),
            pair.v().name(),
            "any"
        )
        .into()
    })
}

/// 2026-10-18: The per-layer legacy dtypes of `plan`, resolved from `requested` with `boundary`
/// layers: empty when every layer holds `requested` (or there are none), else one per layer,
/// logging how many layers hold `boundary`. A plan whose boundary covers every layer is uniform
/// at `boundary` and still gets its list, since the legacy view's base dtype is `requested`.
pub(crate) fn legacy_layer_dtypes(
    plan: &KvCachePlan,
    requested: KvPair,
    boundary: KvPair,
) -> Result<Vec<Legacy>> {
    if plan.layers.is_empty() || plan.uniform() == Some(requested) {
        return Ok(Vec::new());
    }
    let dtypes = plan
        .layers
        .iter()
        .map(|l| legacy(l.pair()))
        .collect::<Result<Vec<_>>>()?;
    let at_boundary = plan.layers.iter().filter(|l| l.pair() == boundary).count();
    tracing::info!(
        "Selective boundary KV cache: {}/{} attention layers at {}, rest at {}",
        at_boundary,
        plan.layers.len(),
        boundary,
        requested,
    );
    Ok(dtypes)
}
