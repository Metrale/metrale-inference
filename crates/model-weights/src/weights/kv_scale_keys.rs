// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The one resolver for a checkpoint's per-layer FP8 KV-cache scales.
//!
//! Exporters spell the pair three ways, relative to the attention prefix `p`
//! (`model.layers.N.self_attn` and the like):
//!
//! | spelling       | K key              | V key              | written by                         |
//! |----------------|--------------------|--------------------|------------------------------------|
//! | `KProjOutput`  | `p.k_proj.k_scale` | `p.v_proj.v_scale` | ModelOpt, older llm-compressor     |
//! | `AttnModule`   | `p.attn.k_scale`   | `p.attn.v_scale`   | vLLM's own `Attention` module name |
//! | `Bare`         | `p.k_scale`        | `p.v_scale`        | llm-compressor `kv_cache_scheme`   |
//!
//! Every spelling carries the same quantity: the dequant multiplier
//! `amax / 448`, which `reshape_and_cache_fp8` divides by on the write and the
//! FP8 attention kernels multiply by on the read.
//!
//! Owner: model-weights (checkpoint key naming).
//! Invariants:
//! - [`resolve_kv_scale_keys`] is the only place a scale key is spelled. The
//!   per-layer loader (`weight_map::load_kv_scales`) and the serve-time census
//!   ([`kv_scale_census`]) both go through it, so the count serve logs is the
//!   number of layers that load a checkpoint scale.
//! - A layer with two spellings, or with K but not V (or V but not K), is an
//!   error, never a silent 1.0.

use std::collections::BTreeMap;

use anyhow::{Result, bail};

/// 2026-09-28: How a checkpoint names one layer's K/V scale pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum KvScaleSpelling {
    KProjOutput,
    AttnModule,
    Bare,
}

impl KvScaleSpelling {
    /// 2026-09-28: Longest suffix first, so a name parses to the most specific
    /// spelling: `p.k_proj.k_scale` is `KProjOutput` of `p`, not `Bare` of
    /// `p.k_proj`.
    pub const PRECEDENCE: [Self; 3] = [Self::KProjOutput, Self::AttnModule, Self::Bare];

    /// 2026-09-28: `(K suffix, V suffix)`, each appended to `"{attn_prefix}."`.
    pub fn suffixes(self) -> (&'static str, &'static str) {
        match self {
            Self::KProjOutput => ("k_proj.k_scale", "v_proj.v_scale"),
            Self::AttnModule => ("attn.k_scale", "attn.v_scale"),
            Self::Bare => ("k_scale", "v_scale"),
        }
    }

    fn keys(self, attn_prefix: &str) -> (String, String) {
        let (k, v) = self.suffixes();
        (format!("{attn_prefix}.{k}"), format!("{attn_prefix}.{v}"))
    }

    /// 2026-09-28: The attention prefix `name` is a K or V scale of, in this
    /// spelling.
    fn prefix_of(self, name: &str) -> Option<&str> {
        let (k, v) = self.suffixes();
        [k, v].into_iter().find_map(|suffix| {
            name.strip_suffix(suffix)
                .and_then(|rest| rest.strip_suffix('.'))
                .filter(|p| !p.is_empty())
        })
    }
}

/// 2026-09-28: The two store keys one attention layer's scales are read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvScaleKeys {
    pub spelling: KvScaleSpelling,
    pub k: String,
    pub v: String,
}

/// 2026-09-28: The scale keys of the attention layer at `attn_prefix`.
///
/// `Ok(None)` when no spelling has either key. An error when the layer has a
/// K or V key in more than one spelling (which one the exporter meant is not
/// ours to guess), or has K without V or V without K in its one spelling.
pub fn resolve_kv_scale_keys(
    has: impl Fn(&str) -> bool,
    attn_prefix: &str,
) -> Result<Option<KvScaleKeys>> {
    let mut found: Option<KvScaleKeys> = None;
    for spelling in KvScaleSpelling::PRECEDENCE {
        let (k, v) = spelling.keys(attn_prefix);
        let (has_k, has_v) = (has(&k), has(&v));
        if !has_k && !has_v {
            continue;
        }
        if has_k != has_v {
            let (present, missing) = if has_k { (&k, &v) } else { (&v, &k) };
            bail!(
                "FP8 KV scales for `{attn_prefix}` are incomplete: the checkpoint has \
                 `{present}` but not `{missing}`"
            );
        }
        if let Some(first) = &found {
            bail!(
                "FP8 KV scales for `{attn_prefix}` are ambiguous: the checkpoint has both \
                 `{}`/`{}` and `{k}`/`{v}`",
                first.k,
                first.v,
            );
        }
        found = Some(KvScaleKeys { spelling, k, v });
    }
    Ok(found)
}

/// 2026-09-28: Every attention layer whose scales [`resolve_kv_scale_keys`]
/// finds, keyed by attention prefix.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct KvScaleCensus {
    pub layers: BTreeMap<String, KvScaleKeys>,
}

impl KvScaleCensus {
    /// 2026-09-28: Layers that load their scales from the checkpoint.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// 2026-09-28: The distinct spellings in use, as their K suffixes, for the
    /// serve log.
    pub fn spellings(&self) -> Vec<&'static str> {
        let mut s: Vec<KvScaleSpelling> = self.layers.values().map(|k| k.spelling).collect();
        s.sort();
        s.dedup();
        s.into_iter().map(|s| s.suffixes().0).collect()
    }
}

/// 2026-09-28: Parse every name that ends in a known scale suffix to its
/// attention prefix (most specific spelling first), then resolve each prefix
/// with [`resolve_kv_scale_keys`]. Any layer that resolver rejects fails the
/// census.
pub fn kv_scale_census<'a>(
    names: impl IntoIterator<Item = &'a str>,
    has: impl Fn(&str) -> bool,
) -> Result<KvScaleCensus> {
    let mut prefixes: Vec<&str> = names
        .into_iter()
        .filter_map(|n| {
            KvScaleSpelling::PRECEDENCE
                .into_iter()
                .find_map(|s| s.prefix_of(n))
        })
        .collect();
    prefixes.sort_unstable();
    prefixes.dedup();
    let mut layers = BTreeMap::new();
    for p in prefixes {
        if let Some(keys) = resolve_kv_scale_keys(&has, p)? {
            layers.insert(p.to_string(), keys);
        }
    }
    Ok(KvScaleCensus { layers })
}

#[cfg(test)]
#[path = "kv_scale_keys_tests.rs"]
mod tests;
