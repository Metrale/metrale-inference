// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The Hopper staged point of the tensor-core grouped NVFP4 decode
//! (`kernels/hopper/common/moe_nvfp4_decode_hopper.cu`): each warp issues its whole 16-row weight
//! tile as cp.async copies into shared memory up front (one mbarrier per load group) and runs the
//! split pair's warp body on it, so the expert kernels stop waiting on one 2 KB load group at a
//! time and return the split pair's bytes (`ops/nvfp4_moe_hsm_tests.rs`).
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Off unless `METRALE_MOE_DECODE_HSM=1` (read once per process); then taken only where the
//!   module exists (`try_target_kernel`: Hopper), the tensor-core decode is on, the tables are
//!   row-major (the lean pair keeps priority) and `nvfp4_grouped_hsm_pair_shape_ok` holds.

use super::forward_nvfp4_grouped_decode::{Nvfp4GroupedKernels, Nvfp4GroupedLaunch};
use super::*;
use crate::layers::try_target_kernel;

/// 2026-10-14: `METRALE_MOE_DECODE_HSM` equal to `"1"`, read once per process.
fn moe_decode_hsm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_MOE_DECODE_HSM").as_deref() == Ok("1"))
}

/// 2026-10-14: The (gate+up, down) staged pair, zero handles when the lever is off or the module
/// is absent.
pub(super) fn resolve(gpu: &dyn GpuBackend) -> (KernelHandle, KernelHandle) {
    const MODULE: &str = "moe_nvfp4_decode_hopper";
    if !moe_decode_hsm_enabled() {
        return (KernelHandle(0), KernelHandle(0));
    }
    (
        try_target_kernel(gpu, MODULE, "moe_expert_gate_up_act_nvfp4_grouped_hsm"),
        try_target_kernel(gpu, MODULE, "moe_expert_down_act_nvfp4_grouped_hsm"),
    )
}

impl Nvfp4GroupedKernels {
    /// 2026-10-14: The staged pair for an `inter` x `hidden` expert, when it resolved and admits
    /// the shape.
    pub(super) fn hsm_launch(&self, hidden: u32, inter: u32) -> Option<Nvfp4GroupedLaunch> {
        let ok = super::forward_nvfp4_grouped_decode::nvfp4_grouped_tc_enabled()
            && self.gate_up_hsm.0 != 0
            && self.down_hsm.0 != 0
            && ops::nvfp4_grouped_hsm_pair_shape_ok(hidden, inter);
        ok.then(|| Nvfp4GroupedLaunch {
            gate_up: self.gate_up_hsm,
            gate_up_geometry: ops::NVFP4_GROUPED_GATE_UP_HSM,
            down: self.down_hsm,
            down_geometry: ops::NVFP4_GROUPED_DOWN_HSM,
            gate_up_smem: ops::nvfp4_grouped_hsm_gate_up_smem(hidden),
            down_smem: ops::nvfp4_grouped_hsm_down_smem(inter),
            max_rows: super::NVFP4_GROUPED_DECODE_TC_MAX_ROWS,
        })
    }
}
