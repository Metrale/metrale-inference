// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The `_cs` twins of the routed W4A4 entries (`METRALE_GLM_MOE_SCALE_DICT`,
//! `scale_dict`): every entry a W4A4 routed site can launch, by its plain name, beside the twin
//! that reads compact block scales through one more argument per projection (the per-expert base
//! table, `Glm5NextExpertPtrTable::scale_bases`). A site whose tables carry base tables launches
//! the twins ([`ScaleDictTwins::pick`]); a site without runs the plain entries, unchanged.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Resolved once at load. [`ScaleDictTwins::ready`] (the loader's `scale_dict::plan` input) is
//!   true only when every listed entry the target has also has its twin, so no reachable plain
//!   entry of a packed layer lacks one.
//! - `pick` launches a twin exactly when the table has a base table, and refuses a plain entry it
//!   does not list (an unlisted entry over compact scales would read them as plain).

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{W4A4_GROUPED_MODULE, W4A4_MOE_MODULE};

/// 2026-10-21: The decode entries (`w4a4_gemv_mx_moe.cu`): the union sweeps (mx8, mx16, the
/// serial `_sl`), the one-row own-slots sweep, the slot GEMV, and their `_k64` twins.
const DECODE: [&str; 10] = [
    "w4a4_gemv_mx8_moe_union_sweep",
    "w4a4_gemv_mx16_moe_union_sweep",
    "w4a4_gemv_mx8_moe_union_sweep_sl",
    "w4a4_gemv_mx16_moe_union_sweep_sl",
    "w4a4_gemv_mx8_moe_slots_sweep",
    "w4a4_gemv_mx8_moe_slots",
    "w4a4_gemv_mx8_moe_union_sweep_k64",
    "w4a4_gemv_mx16_moe_union_sweep_k64",
    "w4a4_gemv_mx8_moe_slots_sweep_k64",
    "w4a4_gemv_mx8_moe_slots_k64",
];

/// 2026-10-21: The grouped entries a routed site reaches (`w4a4_grouped_mx.cu`; tile m32 or m128,
/// `forward::w4a4_grouped_tile_index`): one projection plain and `_k64`, the raster `_r` and
/// column-tile `_n` m128 twins, and the fused gate/up entries.
const GROUPED: [&str; 11] = [
    "w4a4_grouped_mx_m32",
    "w4a4_grouped_mx_m32_k64",
    "w4a4_grouped_mx_m128",
    "w4a4_grouped_mx_m128_k64",
    "w4a4_grouped_mx_m128_r",
    "w4a4_grouped_mx_m128_k64_r",
    "w4a4_grouped_mx_m128_n",
    "w4a4_grouped_mx_m128_k64_n",
    "w4a4_grouped_gu_m32",
    "w4a4_grouped_gu_m128",
    "w4a4_grouped_gu_m128_r",
];

const N: usize = DECODE.len() + GROUPED.len();

/// 2026-10-21: (plain, `_cs`) handles of every listed entry; `KernelHandle(0)` where absent.
#[derive(Clone, Copy, Debug)]
pub struct ScaleDictTwins {
    pairs: [(KernelHandle, KernelHandle); N],
}

impl ScaleDictTwins {
    /// 2026-10-21: Resolve every pair from the target's PTX.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let try_kernel = |m: &str, e: &str| metrale_model_layers::layers::try_kernel(gpu, m, e);
        let modules = DECODE.iter().map(|&n| (W4A4_MOE_MODULE, n));
        let all = modules.chain(GROUPED.iter().map(|&n| (W4A4_GROUPED_MODULE, n)));
        let mut pairs = [(KernelHandle(0), KernelHandle(0)); N];
        for (p, (m, name)) in pairs.iter_mut().zip(all) {
            *p = (try_kernel(m, name), try_kernel(m, &format!("{name}_cs")));
        }
        Self { pairs }
    }

    /// 2026-10-21: From handles directly (tests).
    #[cfg(test)]
    pub(crate) fn of(pairs: [(KernelHandle, KernelHandle); N]) -> Self {
        Self { pairs }
    }

    /// 2026-10-21: The first `list.len()` pairs from `(plain, twin)` handle values, the rest
    /// absent (tests that name their own handles).
    #[cfg(test)]
    pub(crate) fn of_list(list: &[(u64, u64)]) -> Self {
        let mut pairs = [(KernelHandle(0), KernelHandle(0)); N];
        for (p, &(a, b)) in pairs.iter_mut().zip(list) {
            *p = (KernelHandle(a), KernelHandle(b));
        }
        assert!(
            list.len() <= N,
            "{} pairs past the {N} listed entries",
            list.len()
        );
        Self { pairs }
    }

    /// 2026-10-21: Whether every listed entry the target has also has its twin (and it has some).
    pub fn ready(&self) -> bool {
        self.pairs.iter().any(|p| p.0.0 != 0) && self.pairs.iter().all(|p| p.0.0 == 0 || p.1.0 != 0)
    }

    /// 2026-10-21: The entry to launch for plain entry `kern` over a table whose base table is
    /// `bases`: `kern` itself and no argument when `bases` is null, else its twin and `bases`.
    /// Refuses a packed table on an unlisted entry or a missing twin.
    pub fn pick(
        &self,
        kern: KernelHandle,
        bases: DevicePtr,
    ) -> Result<(KernelHandle, Option<DevicePtr>)> {
        if bases.is_null() {
            return Ok((kern, None));
        }
        match self.pairs.iter().find(|p| p.0.0 == kern.0 && kern.0 != 0) {
            Some(&(_, twin)) if twin.0 != 0 => Ok((twin, Some(bases))),
            _ => bail!(
                "METRALE_GLM_MOE_SCALE_DICT: a routed launch over compact scales has no `_cs` twin \
                 for kernel {kern:?}; the load's packing decision should have kept plain scales"
            ),
        }
    }
}

#[cfg(test)]
#[path = "scale_dict_kernels_tests.rs"]
mod tests;
