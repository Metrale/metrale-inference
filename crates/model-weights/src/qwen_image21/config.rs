// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Strict config boundary for the pinned visual transformer only.

use anyhow::{Result, ensure};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawConfig {
    #[serde(rename = "_class_name")]
    class_name: String,
    #[serde(rename = "_diffusers_version")]
    _diffusers_version: String,
    attention_head_dim: usize,
    axes_dims_rope: [usize; 3],
    context_in_dim: usize,
    in_channels: usize,
    num_attention_heads: usize,
    num_layers: usize,
    out_channels: usize,
    patch_size: usize,
    mlp_ratio: usize,
    eps: f64,
    causal_condition: bool,
}

/// 2026-10-07: Validated pinned shape. No defaults or unchecked constructor.
#[derive(Debug)]
pub struct Config {
    raw: RawConfig,
}

impl Config {
    /// 2026-10-07: Reject unknown or altered math instead of aliasing a decoder.
    pub fn parse(json: &str) -> Result<Self> {
        let raw: RawConfig = serde_json::from_str(json)?;
        ensure!(
            raw.class_name == "QwenImage21Transformer2DModel",
            "wrong transformer class"
        );
        ensure!(
            raw.attention_head_dim == 128
                && raw.axes_dims_rope == [16, 56, 56]
                && raw.context_in_dim == 4096
                && raw.in_channels == 64
                && raw.num_attention_heads == 32
                && raw.num_layers == 32
                && raw.out_channels == 64
                && raw.patch_size == 1
                && raw.mlp_ratio == 3
                && raw.eps == 1e-6
                && raw.causal_condition,
            "unsupported Qwen Image 2.1 transformer math configuration"
        );
        Ok(Self { raw })
    }

    /// 2026-10-07: Validated hidden width.
    pub fn hidden(&self) -> usize {
        self.raw.num_attention_heads * self.raw.attention_head_dim
    }
    /// 2026-10-07: Validated head width.
    pub fn head_dim(&self) -> usize {
        self.raw.attention_head_dim
    }
    /// 2026-10-07: Number of single-stream blocks.
    pub fn layers(&self) -> usize {
        self.raw.num_layers
    }
    /// 2026-10-07: Intermediate SwiGLU width.
    pub fn intermediate(&self) -> usize {
        self.hidden() * self.raw.mlp_ratio
    }
    /// 2026-10-07: Input/output latent width at patch size one.
    pub fn channels(&self) -> usize {
        self.raw.in_channels
    }
}
