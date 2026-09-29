// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The positional encoding a model's full-attention layers apply to Q and K.
//!
//! Owner: config.
//! Invariants:
//! - A config returned by `parse_config` or `config_from_gguf` has
//!   `attn_position_encoding` set (`resolve_attn_position_encoding`).
//! - A config that declares no positional encoding carries no RoPE variant parameter
//!   (explicit `rotary_dim`, MRoPE, YaRN); such a config is refused, not guessed at.

use anyhow::{Context, Result, bail};

use super::ModelConfig;

/// 2026-09-29: What full attention does to Q and K before the dot product.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnPositionEncoding {
    /// 2026-09-29: Rotary embedding over `rotary_dim()` dims at `rope_theta`, or the layer's
    /// RoPE variant (YaRN, proportional, MRoPE) where the loader configures one.
    Rope,
    /// 2026-09-29: No positional encoding: Q and K enter attention as projected. Nemotron-H
    /// attention has none; position reaches it through the Mamba-2 layers only (HF
    /// `NemotronHAttention.forward` never calls `apply_rotary_pos_emb`).
    None,
}

impl ModelConfig {
    /// 2026-09-29: The resolved positional encoding of full attention. An error for a config
    /// that never went through `parse_config` / `config_from_gguf` and was not built by a
    /// factory that sets it.
    pub fn attn_position_encoding(&self) -> Result<AttnPositionEncoding> {
        self.attn_position_encoding.with_context(|| {
            format!(
                "model_type `{}`: attention positional encoding was never resolved \
                 (config not built by parse_config/config_from_gguf)",
                self.model_type
            )
        })
    }
}

/// 2026-09-29: Settle `attn_position_encoding` after the family parse. A family parser that
/// knows its attention has no positional encoding declares `None`; any other family rotates
/// Q/K by its rope fields, which is what `Rope` records. A `None` declaration next to a RoPE
/// variant parameter is contradictory and refused.
pub(crate) fn resolve_attn_position_encoding(config: &mut ModelConfig) -> Result<()> {
    match config.attn_position_encoding {
        Some(AttnPositionEncoding::None) => {
            let conflicts: Vec<&str> = [
                (config.rotary_dim > 0, "rotary_dim"),
                (config.mrope_interleaved, "mrope_interleaved"),
                (config.mrope_section.iter().any(|&s| s > 0), "mrope_section"),
                (config.yarn_factor > 0.0, "yarn_factor"),
            ]
            .into_iter()
            .filter_map(|(set, name)| set.then_some(name))
            .collect();
            if !conflicts.is_empty() {
                bail!(
                    "model_type `{}` declares no positional encoding for attention, but the \
                     config also sets RoPE parameters: {}. Refusing an ambiguous config.",
                    config.model_type,
                    conflicts.join(", ")
                );
            }
        }
        Some(AttnPositionEncoding::Rope) => {}
        None => config.attn_position_encoding = Some(AttnPositionEncoding::Rope),
    }
    Ok(())
}

#[cfg(test)]
#[path = "position_encoding_tests.rs"]
mod tests;
