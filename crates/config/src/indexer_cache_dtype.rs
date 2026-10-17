// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: `--indexer-cache-dtype`: how a sparse-attention model stores its indexer rows
//! in the paged KV blocks.
//!
//! - `bf16`: one bf16 key row and one bf16 gate row per token, the layout every serve has
//!   always used.
//! - `fp8`: each complete pool of `index_kpool` tokens is stored once, as its compressed key
//!   rotated by a 128-point Walsh-Hadamard transform and quantized to FP8 E4M3 with one
//!   power-of-two scale; the tokens of a pool still being filled sit in a small bf16 ring.
//!
//! Owner: config.
//! Invariants:
//! - `bf16` is the default and changes nothing: no loader, planner or kernel reads a field
//!   this module adds unless the value is `fp8`.
//! - [`IndexerCacheDtype::check_serve`] passes `bf16` always and `fp8` only for `glm5_next` /
//!   `glm5_next_text` (whose indexer rows live in the KV blocks) with the indexer cache paged
//!   and peer KV paging off (its namespace does not encode the indexer layout).

use anyhow::{Result, bail};

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

    /// 2026-10-16: Refuse `fp8` off its scope (see the module invariants). `indexer_paged` is
    /// `glm_dsa_indexer_paged()`, `kv_paging` whether `METRALE_KV_PAGING=1`.
    pub fn check_serve(self, model_type: &str, indexer_paged: bool, kv_paging: bool) -> Result<()> {
        if self == Self::Bf16 {
            return Ok(());
        }
        if !matches!(model_type, "glm5_next" | "glm5_next_text") {
            bail!(
                "--indexer-cache-dtype fp8 applies only to a glm5_next or glm5_next_text \
                 checkpoint (got model_type {model_type:?}); use --indexer-cache-dtype bf16"
            );
        }
        if !indexer_paged {
            bail!(
                "--indexer-cache-dtype fp8 stores the indexer rows in the KV blocks, but \
                 METRALE_GLM_DSA_INDEXER_FLAT=1 keeps them in a per-sequence buffer; unset it or \
                 use --indexer-cache-dtype bf16"
            );
        }
        if kv_paging {
            bail!(
                "--indexer-cache-dtype fp8 changes the KV block layout, which the \
                 METRALE_KV_PAGING=1 peer namespace does not encode; unset it or use \
                 --indexer-cache-dtype bf16"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-16: `bf16` passes every serve; `fp8` passes only `glm5_next` (and `_text`), paged, without
    /// peer paging, and each refusal names its cause.
    #[test]
    fn fp8_needs_a_paged_glm5_next_serve() {
        for m in ["glm5_next", "glm5_next_text", "qwen3", "deepseek_v4", ""] {
            for (paged, paging) in [(true, false), (false, false), (true, true), (false, true)] {
                assert!(
                    IndexerCacheDtype::Bf16
                        .check_serve(m, paged, paging)
                        .is_ok()
                );
            }
        }
        let fp8 = IndexerCacheDtype::Fp8;
        assert!(fp8.check_serve("glm5_next", true, false).is_ok());
        assert!(fp8.check_serve("glm5_next_text", true, false).is_ok());
        for m in ["qwen3", "deepseek_v4", "glm4_moe"] {
            let e = fp8.check_serve(m, true, false).unwrap_err();
            assert!(e.to_string().contains("glm5_next_text checkpoint"), "{e}");
        }
        let e = fp8.check_serve("glm5_next", false, false).unwrap_err();
        assert!(e.to_string().contains("INDEXER_FLAT"), "{e}");
        let e = fp8.check_serve("glm5_next", true, true).unwrap_err();
        assert!(e.to_string().contains("METRALE_KV_PAGING"), "{e}");
    }

    /// 2026-10-16: The names are the flag values, and the default is `bf16`.
    #[test]
    fn names_and_default() {
        let names: Vec<_> = IndexerCacheDtype::ALL.iter().map(|d| d.name()).collect();
        assert_eq!(names, ["bf16", "fp8"]);
        assert_eq!(IndexerCacheDtype::default(), IndexerCacheDtype::Bf16);
    }
}
