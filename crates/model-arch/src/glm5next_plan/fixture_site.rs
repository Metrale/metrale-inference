// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: One rank's routed MoE site on the recording device for the CPU launch gates: the
//! served levers' kernels, the rank's MLP config and workspace, and synthetic layer weights.
//! Moved here from `plan_moe_tests.rs` (exact copy) so the install gate shares it.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::recording::RecordingGpu;

use super::fixture::{mlp_config, served_levers};
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Glm5NextMoeWeights, Nvfp4Proj, ScaleLayout, W4a4ActScales,
};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-23: GB10's SM count (the sweep's persistent grid).
pub(crate) const SMS: u32 = 48;
pub(crate) const STREAM: u64 = 7;
/// 2026-10-23: The checkpoint's routed experts, all local under the served `tp` layout (the
/// site check pins it to `cfg.num_experts`).
pub(crate) const EXPERTS: u64 = 288;
pub(crate) const X: DevicePtr = DevicePtr(0x9000_0000);
pub(crate) const OUT: DevicePtr = DevicePtr(0x9100_0000);
pub(crate) const SCALES: W4a4ActScales = W4a4ActScales {
    gate_up: 0.000642,
    down: 0.000115,
};

pub(crate) struct Site {
    pub gpu: RecordingGpu,
    pub k: Glm5NextMlpKernels,
    pub cfg: Glm5NextMlpConfig,
    pub ws: Glm5NextMlpWorkspace,
}

pub(crate) fn site(rank: usize) -> Site {
    let gpu = RecordingGpu::new(SMS);
    let k = Glm5NextMlpKernels::resolve(&gpu, served_levers()).unwrap();
    let cfg = mlp_config(rank);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &cfg, 16, &k.levers.mlp).unwrap();
    gpu.take();
    Site { gpu, k, cfg, ws }
}

/// 2026-10-23: Layer `l`'s weights at distinct synthetic addresses, every expert table carrying
/// a base table (compact scales, as the served load packs them) unless `plain`. The per-expert
/// projections are bound (`forward_moe` checks their count) but no W4A4 launch reads them.
pub(crate) fn weights(l: usize, plain: bool) -> Glm5NextMoeWeights {
    let tag = |s: u64| DevicePtr(0x4_0000_0000 + ((l as u64) << 24) + (s << 12));
    let proj = |s: u64| Nvfp4Proj {
        packed: tag(s),
        scale: tag(s + 1),
        scale_2: 0.5,
        input_scale: None,
        scale_layout: ScaleLayout::Plain,
    };
    let experts = (0..EXPERTS)
        .map(|e| Glm5NextExpertWeights {
            gate_proj: proj(64 + 6 * e),
            up_proj: proj(66 + 6 * e),
            down_proj: proj(68 + 6 * e),
        })
        .collect();
    let table = |s: u64| Glm5NextExpertPtrTable {
        packed_ptrs: tag(s),
        scale_ptrs: tag(s + 1),
        scale2_vals: tag(s + 2),
        scale_bases: if plain { DevicePtr::NULL } else { tag(s + 3) },
    };
    Glm5NextMoeWeights {
        router: tag(1),
        router_bias: tag(2),
        shared: Glm5NextDenseMlpWeights {
            gate_proj: tag(3),
            up_proj: tag(4),
            down_proj: tag(5),
        },
        experts,
        ptrs: Glm5NextMoePtrTables {
            gate: table(16),
            up: table(32),
            down: table(48),
        },
        precision: GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            16,
            true,
        )
        .unwrap(),
        act_scales: Some(SCALES),
    }
}
