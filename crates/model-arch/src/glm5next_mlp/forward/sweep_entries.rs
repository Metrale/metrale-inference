// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_SWEEP_SERIAL` (on unless switched off): which entries of the
//! routed gate/up union sweep the MLP resolves. The plain `w4a4_gemv_mx{8,16}_moe_union_sweep`
//! wait on their grid dependencies, so under PDL (`glm5next_pdl`) the host launches them while the
//! row union still runs and their persistent grid (one CTA per SM) lands beside it; the `_sl`
//! twins hold no wait, so they launch stream-ordered onto an idle device, and they trigger the
//! SwiGLU after their last item (`w4a4_gemv_mx_moe.cu`). Same bytes; on GB10 the MoE layer of the
//! decode chain is 4.9 us faster (`pdl_chain_bench`). Moving only the trigger to the tail (the
//! SwiGLU no longer resident beside the sweep) measured 0.1-0.7 us, and the down (`_k64`) sweep
//! launched stream-ordered gained nothing, so it keeps its PDL edges.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Resolved once at load; an `_sl` entry absent from the target's PTX falls back to its plain
//!   twin with a warning (same bytes, the PDL launch).

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::super::W4A4_MOE_MODULE;

/// 2026-10-21: (plain, serial-launch) names of the 8- and 16-row union sweeps.
const GATE_UP_SWEEPS: [(&str, &str); 2] = [
    (
        "w4a4_gemv_mx8_moe_union_sweep",
        "w4a4_gemv_mx8_moe_union_sweep_sl",
    ),
    (
        "w4a4_gemv_mx16_moe_union_sweep",
        "w4a4_gemv_mx16_moe_union_sweep_sl",
    ),
];

/// 2026-10-21: The entry names the lever selects, in [`GATE_UP_SWEEPS`] order.
fn names(serial: bool) -> [&'static str; 2] {
    GATE_UP_SWEEPS.map(|(plain, sl)| if serial { sl } else { plain })
}

/// 2026-10-21: The 8- and 16-row gate/up union sweeps under the lever. An absent entry resolves
/// to `KernelHandle(0)` (the forward then runs the slot GEMV), an absent `_sl` entry to its plain
/// twin.
pub(crate) fn gate_up_sweeps(gpu: &dyn GpuBackend, serial: bool) -> [KernelHandle; 2] {
    let try_kernel = |e: &str| metrale_model_layers::layers::try_kernel(gpu, W4A4_MOE_MODULE, e);
    let mut i = 0;
    names(serial).map(|name| {
        let plain = GATE_UP_SWEEPS[i].0;
        i += 1;
        let h = try_kernel(name);
        if h.0 != 0 || name == plain {
            return h;
        }
        tracing::warn!(
            "METRALE_GLM_MOE_SWEEP_SERIAL: `{name}` is not in this target's PTX; `{plain}` (PDL \
             launch) runs"
        );
        try_kernel(plain)
    })
}

#[cfg(test)]
#[path = "sweep_entries_tests.rs"]
mod tests;
