// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `--indexer-cache-dtype`: how a sparse-attention model stores its indexer rows
//! in the paged KV blocks.
//!
//! - `bf16`: one bf16 key row and one bf16 gate row per token, the layout every serve has
//!   always used.
//! - `fp8`: each complete pool of `index_kpool` tokens is stored once, as its compressed key
//!   rotated by a 128-point Walsh-Hadamard transform and quantized to FP8 E4M3 with one
//!   power-of-two scale; the tokens of a pool still being filled sit in a small bf16 ring.
//!   2026-10-18: its rows are the kv-codec registry's `fp8_row` indexer codec
//!   ([`IndexerCacheDtype::codec`]).
//!
//! Owner: config.
//! Invariants:
//! - `bf16` is the default and changes nothing: no loader, planner or kernel reads a field
//!   this module adds unless the value is `fp8`.
//! - [`IndexerCacheDtype::check_serve`] passes `bf16` always and `fp8` only when the model's
//!   loader serves it (2026-10-18: a capability the loader declares, not a model-type list),
//!   with the indexer cache paged. 2026-10-18: peer KV paging is no longer refused: its
//!   namespace folds the cache's codec salt, which includes the indexer codec.

use anyhow::{Result, bail};
use metrale_kv_codec::{KvCodecId, by_name};

/// 2026-10-18: The registry codec of the `fp8` indexer rows, bound at compile time.
const FP8_ROW: KvCodecId = match by_name("fp8_row") {
    Some(id) => id,
    None => panic!("the kv-codec registry has no fp8_row indexer codec"),
};

/// 2026-10-16: The `--indexer-cache-dtype` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IndexerCacheDtype {
    /// 2026-10-16: Per-token bf16 key and gate rows. `Default` only because `ModelConfig` is
    /// deserialized with serve-set fields skipped; serve sets it from the flag.
    #[default]
    Bf16,
    /// 2026-10-16: Pooled FP8 E4M3 keys with a bf16 tail ring.
    Fp8,
}

impl IndexerCacheDtype {
    /// 2026-10-16: Every value, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Bf16, Self::Fp8];

    /// 2026-10-16: The flag and recipe value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }

    /// 2026-10-18: The registry indexer codec the pooled rows are stored as; `None` for `bf16`,
    /// whose per-token rows are not pooled.
    pub const fn codec(self) -> Option<KvCodecId> {
        match self {
            Self::Bf16 => None,
            Self::Fp8 => Some(FP8_ROW),
        }
    }

    /// 2026-10-16: Refuse `fp8` off its scope (see the module invariants). `indexer_paged` is
    /// `glm_dsa_indexer_paged()`.
    /// 2026-10-18: `served` is what the model's loader declares beyond `bf16`
    /// (`ModelWeightLoader::indexer_cache_dtypes`); `model_type` only names the model in the
    /// refusal.
    pub fn check_serve(
        self,
        model_type: &str,
        served: &[IndexerCacheDtype],
        indexer_paged: bool,
    ) -> Result<()> {
        if self == Self::Bf16 {
            return Ok(());
        }
        if !served.contains(&self) {
            bail!(
                "--indexer-cache-dtype {} is not served by model_type {model_type:?} (its loader \
                 stores no {} indexer rows); use --indexer-cache-dtype bf16",
                self.name(),
                self.name()
            );
        }
        if !indexer_paged {
            bail!(
                "--indexer-cache-dtype fp8 stores the indexer rows in the KV blocks, but \
                 METRALE_GLM_DSA_INDEXER_FLAT=1 keeps them in a per-sequence buffer; unset it or \
                 use --indexer-cache-dtype bf16"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metrale_kv_codec::{CodecRole, spec};

    /// 2026-10-16: `bf16` passes every serve; `fp8` passes only a model that serves it, paged,
    /// and each refusal names its cause. 2026-10-18: peer paging is no longer an input.
    /// 2026-10-18: what a model serves is the loader's declared list, not its type string.
    #[test]
    fn fp8_needs_a_paged_serve_of_a_model_that_serves_it() {
        let fp8 = IndexerCacheDtype::Fp8;
        for served in [&[][..], &[fp8][..]] {
            for paged in [true, false] {
                let ok = IndexerCacheDtype::Bf16.check_serve("m", served, paged);
                assert!(ok.is_ok());
            }
        }
        assert!(fp8.check_serve("glm5_next", &[fp8], true).is_ok());
        for m in ["qwen3", "deepseek_v4", "glm5_next"] {
            let e = fp8.check_serve(m, &[], true).unwrap_err();
            assert!(e.to_string().contains("is not served by model_type"), "{e}");
            assert!(e.to_string().contains(m), "{e}");
        }
        let e = fp8.check_serve("m", &[fp8], false).unwrap_err();
        assert!(e.to_string().contains("INDEXER_FLAT"), "{e}");
    }

    /// 2026-10-16: The names are the flag values, and the default is `bf16`.
    #[test]
    fn names_and_default() {
        let names: Vec<_> = IndexerCacheDtype::ALL.iter().map(|d| d.name()).collect();
        assert_eq!(names, ["bf16", "fp8"]);
        assert_eq!(IndexerCacheDtype::default(), IndexerCacheDtype::Bf16);
    }

    /// 2026-10-18: `fp8` rows are a registered indexer codec (never an attention side), and
    /// `bf16` rows are no codec's pooled rows.
    #[test]
    fn fp8_rows_are_the_registry_indexer_codec() {
        assert_eq!(IndexerCacheDtype::Bf16.codec(), None);
        let id = IndexerCacheDtype::Fp8.codec().expect("fp8 has a codec");
        assert_eq!(spec(id).role, CodecRole::Indexer);
        assert!(id.name().parse::<metrale_kv_codec::KvPair>().is_err());
    }
}
