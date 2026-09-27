// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `--moe-nvfp4-experts`: requantize an FP8 checkpoint's routed and shared MoE
//! experts to NVFP4 at load, and run every MoE decode of 1 to 64 rows through the grouped
//! NVFP4 kernels (`moe/forward_nvfp4_grouped_decode.rs`).
//!
//! It lowers the experts' precision, so it is off unless the serve command line asks for it;
//! there is no environment fallback. The loader reads it (`qwen35/load_layers.rs`) together
//! with the diagnostic `METRALE_FORCE_NVFP4_MOE`, which loads the same NVFP4 experts but keeps
//! today's NVFP4 decode arms.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, and anything that reads first fixes it at false.
//! - With it on, a row's MoE output bits do not depend on how many rows share the call
//!   (`FfnComponent::nvfp4_grouped_ok` admits every width the arms below pass it).

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;

static MOE_NVFP4_EXPERTS: OnceLock<bool> = OnceLock::new();

/// 2026-09-27: Publish `--moe-nvfp4-experts`. Returns the value in force; a caller that gets
/// a different value should warn.
pub fn set_moe_nvfp4_experts_from_cli(on: bool) -> bool {
    let _ = MOE_NVFP4_EXPERTS.set(on);
    *MOE_NVFP4_EXPERTS.get().expect("just set")
}

/// 2026-09-27: `--moe-nvfp4-experts` in force? False unless the serve published true.
pub fn moe_nvfp4_experts_enabled() -> bool {
    *MOE_NVFP4_EXPERTS.get_or_init(|| false)
}

impl super::FfnComponent {
    /// 2026-09-27: Whether this is a MoE whose grouped NVFP4 decode serves `m` rows
    /// (`MoeLayer::nvfp4_grouped_decode_ok`).
    pub fn nvfp4_grouped_ok(&self, m: usize, ctx: &ForwardContext) -> bool {
        matches!(self, Self::Moe(moe) if moe.nvfp4_grouped_decode_ok(m, ctx))
    }

    /// 2026-09-27: The grouped NVFP4 MoE decode over `[m, H]` rows into `moe_output()`.
    /// Errors for dense and none; callers gate on [`Self::nvfp4_grouped_ok`] first.
    pub fn forward_nvfp4_grouped(
        &self,
        input: DevicePtr,
        m: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(moe) => moe.forward_nvfp4_grouped_decode(input, m, ctx, stream),
            _ => anyhow::bail!("forward_nvfp4_grouped is MoE-only (m={m})"),
        }
    }
}
