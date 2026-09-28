// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-27: `--expert-quantization`: the precision a native-FP8 checkpoint's routed MoE
//! experts are decoded at. `fp8` (the default) decodes the checkpoint's FP8 experts. The two
//! NVFP4 tiers add an NVFP4 copy of routed-expert projections at load and run every MoE decode
//! of 1 to 64 rows through the grouped decode (`moe/forward_nvfp4_grouped_decode.rs`), whose
//! output for a row does not depend on how many rows share the call; the shared expert and
//! prefill keep the FP8 weights under both.
//!
//! It lowers the routed experts' precision in decode, so it is `fp8` unless the serve command
//! line asks otherwise; there is no environment fallback. The loader reads it
//! (`qwen35/load_layers.rs`) beside the diagnostic `METRALE_FORCE_NVFP4_MOE`, which loads
//! NVFP4 experts only and keeps today's NVFP4 arms.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - The first publication or read wins (`OnceLock`); the serve publishes before the model is
//!   built, and anything that reads first fixes it at `fp8`.
//! - Under an NVFP4 tier a row's MoE output bits do not depend on how many rows share the
//!   call (`FfnComponent::nvfp4_grouped_ok` admits every width the decode arms pass it).

use std::sync::OnceLock;

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;

/// 2026-09-27: The expert precision tiers. Measured on GB10 (Qwen3.6-35B-A3B-FP8, canonical
/// tiers, `--mtp-gate force`):
///
/// | tier | BFCL echolp N=1004 overall/normalized | agentic-webserver | C16 tok/s, J/tok (dgx1) |
/// |---|---|---|---|
/// | fp8 | 84.96/86.03 | pass, 535-546 s | 315.5, 0.227 |
/// | nvfp4-gate-up | 84.86/85.33 | pass 3/3, 504-533 s | 359.6, 0.211 |
/// | nvfp4 | 85.46/86.57 | FAIL: 168 turns, 774 s over the 700 s ceiling | 383.1, 0.201 |
///
/// Weights resident before the KV cache: fp8 38.4 GB, nvfp4-gate-up 49.6 GB, nvfp4 55.2 GB.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ExpertQuantization {
    /// 2026-09-27: The checkpoint's FP8 experts: most stable, still fast.
    #[default]
    Fp8,
    /// 2026-09-27: Routed gate and up projections NVFP4 in decode; routed down, the shared expert and
    /// prefill FP8. The stable speed lever.
    Nvfp4GateUp,
    /// 2026-09-27: Every routed projection NVFP4 in decode; the shared expert and prefill FP8. Fastest,
    /// and it changes the agent's trajectories most.
    Nvfp4,
}

impl ExpertQuantization {
    /// 2026-09-27: Every tier, in the order the flag lists them.
    pub const ALL: [Self; 3] = [Self::Fp8, Self::Nvfp4GateUp, Self::Nvfp4];

    /// 2026-09-27: The tier's flag value and recipe value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Fp8 => "fp8",
            Self::Nvfp4GateUp => "nvfp4-gate-up",
            Self::Nvfp4 => "nvfp4",
        }
    }

    /// 2026-09-27: Whether MoE decode takes the grouped NVFP4 path.
    pub fn nvfp4_decode(self) -> bool {
        self != Self::Fp8
    }

    /// 2026-09-27: Whether the routed down projections decode as NVFP4, so their NVFP4 copy
    /// is built at load; under `nvfp4-gate-up` they decode from the FP8 experts.
    pub fn nvfp4_down(self) -> bool {
        self == Self::Nvfp4
    }
}

static EXPERT_QUANTIZATION: OnceLock<ExpertQuantization> = OnceLock::new();

/// 2026-09-27: Publish `--expert-quantization`. Returns the tier in force; a caller that gets a
/// different one should warn.
pub fn set_expert_quantization_from_cli(q: ExpertQuantization) -> ExpertQuantization {
    let _ = EXPERT_QUANTIZATION.set(q);
    *EXPERT_QUANTIZATION.get().expect("just set")
}

/// 2026-09-27: The `--expert-quantization` tier in force: `fp8` unless the serve published
/// another.
pub fn expert_quantization() -> ExpertQuantization {
    *EXPERT_QUANTIZATION.get_or_init(ExpertQuantization::default)
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

#[cfg(test)]
mod tests {
    use super::ExpertQuantization as Q;

    /// 2026-09-27: The names are distinct and each tier's decode and down precision.
    #[test]
    fn tiers_name_and_split_precision() {
        let names: Vec<_> = Q::ALL.iter().map(|q| q.name()).collect();
        assert_eq!(names, ["fp8", "nvfp4-gate-up", "nvfp4"]);
        assert_eq!(Q::default(), Q::Fp8);
        assert!(!Q::Fp8.nvfp4_decode() && !Q::Fp8.nvfp4_down());
        assert!(Q::Nvfp4GateUp.nvfp4_decode() && !Q::Nvfp4GateUp.nvfp4_down());
        assert!(Q::Nvfp4.nvfp4_decode() && Q::Nvfp4.nvfp4_down());
    }
}
