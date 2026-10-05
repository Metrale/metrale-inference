// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The GDN prefill block's shape (`GdnDims`) and its step timer (`ssm_prof!`), shared by
//! the block's three parts (`trait_prefill_block.rs`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

/// 2026-10-05: `ctx.profile` timing of one step: synchronise and log the time since `$t0`.
macro_rules! ssm_prof {
    ($ctx:expr, $stream:expr, $k:expr, $label:expr, $t0:expr) => {
        if $ctx.profile {
            if let Some(t0) = $t0 {
                $ctx.gpu.synchronize($stream)?;
                let elapsed = t0.elapsed().as_micros();
                tracing::info!("  SSM prefill [{}] N={}: {}\u{b5}s", $label, $k, elapsed);
            }
        }
    };
}
pub(super) use ssm_prof;

/// 2026-10-05: The GDN block's shape for `num_tokens` rows, from the model config.
pub(super) struct GdnDims {
    pub(super) h: usize,
    pub(super) eps: f32,
    pub(super) k: u32,
    pub(super) bf16: usize,
    pub(super) fp32: usize,
    pub(super) nk: usize,
    pub(super) kd: usize,
    pub(super) nv: usize,
    pub(super) vd: usize,
    pub(super) vpg: usize,
    pub(super) key_dim: usize,
    pub(super) value_dim: usize,
    pub(super) conv_dim: usize,
    pub(super) d_conv: usize,
    pub(super) qkvz_size: usize,
}

impl GdnDims {
    pub(super) fn of(c: &metrale_config::ModelConfig, num_tokens: usize) -> Self {
        let (nk, kd) = (c.linear_num_key_heads, c.linear_key_head_dim);
        let (nv, vd) = (c.linear_num_value_heads, c.linear_value_head_dim);
        Self {
            h: c.hidden_size,
            eps: c.rms_norm_eps as f32,
            k: num_tokens as u32,
            bf16: 2,
            fp32: 4,
            nk,
            kd,
            nv,
            vd,
            vpg: nv / nk,
            key_dim: nk * kd,
            value_dim: nv * vd,
            conv_dim: nk * kd * 2 + nv * vd,
            d_conv: c.linear_conv_kernel_dim,
            qkvz_size: c.ssm_qkvz_size(),
        }
    }
}
