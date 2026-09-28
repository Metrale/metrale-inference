// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSOT for the narrow `w4a16_gemv_batch{M}` tier family (M = 4..8).
//!
//! Owner: model-layers (NVFP4 GEMV dispatch).
//! Invariants:
//! - `select_tier` never returns a tier narrower than `m`, nor one that the
//!   loaded target did not resolve.
//!
//! `w4a16_gemv_batchm_impl<MAX_M>` sizes `acc[]` and `s_vl[]` by `MAX_M` and
//! unrolls its row loop over `MAX_M`, so `MAX_M` also sizes the code. The
//! `t >= M` guard skips a dead row's work at run time but not its
//! instructions, so M=5 on the `MAX_M=8` tier carries three unused rows; the
//! exact-M tiers 5, 6 and 7 avoid that.
//!
//! The width decision lives here once, as a pure function (`select_tier`)
//! over which tiers the loaded target resolved; every consumer holds a
//! `W4a16BatchmTiers` and calls `kernel(m)`.
//!
//! `METRALE_NO_GEMV_EXACT_M_TIERS` (presence-checked: any value, including
//! `0`, disables) hides widths 5/6/7 from the decision, leaving widths 4 and
//! 8. It does not unload the kernels, so an A/B needs no rebuild.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Tier widths in this family, narrowest first. Parallel to the `handles`
/// field of [`W4a16BatchmTiers`] and to the `present` array of
/// [`select_tier`]. `w4a16_gemv_batch16`/`_batch32` are not in this table:
/// they are held separately (`wide`, `wide32`) and handed out only above 8
/// rows, within the row edge ([`W4a16BatchmTiers::edge`]).
pub const W4A16_BATCHM_WIDTHS: [u32; 5] = [4, 5, 6, 7, 8];

/// 2026-09-25: Index into [`W4A16_BATCHM_WIDTHS`] of width 5, the first tier
/// `METRALE_NO_GEMV_EXACT_M_TIERS` hides.
const FIRST_EXACT_M: usize = 1;

/// 2026-09-25: Index of width 7, the last tier `METRALE_NO_GEMV_EXACT_M_TIERS`
/// hides; width 8 stays.
const EXACT_M_LAST: usize = 3;

/// 2026-09-25: Are the exact-M tiers (5/6/7) allowed in the dispatch decision?
///
/// A presence check, read once per process: `kernel` consults it on every
/// launch.
pub fn exact_m_tiers_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_GEMV_EXACT_M_TIERS").is_none())
}

/// 2026-09-25: Pure tier decision: index into [`W4A16_BATCHM_WIDTHS`] of the
/// narrowest tier that covers `m` rows and is present in the loaded target,
/// or `None` when no such tier exists (or `m` is 0).
///
/// `present[i]` is whether `W4A16_BATCHM_WIDTHS[i]` resolved. `exact_m` is
/// [`exact_m_tiers_enabled`], passed in so both values are testable without
/// the process environment or the latched `OnceLock`.
///
/// Narrowest-that-covers, not exact-only: a target missing a tier still
/// dispatches on the next wider one.
pub fn select_tier(
    m: u32,
    present: [bool; W4A16_BATCHM_WIDTHS.len()],
    exact_m: bool,
) -> Option<usize> {
    if m == 0 {
        return None;
    }
    W4A16_BATCHM_WIDTHS
        .iter()
        .enumerate()
        .find(|&(i, &w)| {
            w >= m && present[i] && (exact_m || !(FIRST_EXACT_M..=EXACT_M_LAST).contains(&i))
        })
        .map(|(i, _)| i)
}

/// 2026-09-25: Resolved handles for the narrow `w4a16_gemv_batch{M}` family.
///
/// A zero handle means "this target did not load that tier"; consumers check
/// `.0 != 0`, and [`Self::kernel`] returns a zero handle rather than panicking
/// when nothing in the family covers `m`.
#[derive(Clone, Copy, Debug)]
pub struct W4a16BatchmTiers {
    /// 2026-09-25: Parallel to [`W4A16_BATCHM_WIDTHS`].
    handles: [KernelHandle; W4A16_BATCHM_WIDTHS.len()],
    /// 2026-09-25: `w4a16_gemv_batch16`, handed out for 9..=16 rows when the
    /// row edge exceeds 8: under `gemv_tc::wide_rows_enabled()`, or for a weight
    /// that declares FP4 activations. `ops::w4a16_gemv_batchm` launches the
    /// tensor-core kernel instead when `gemv_tc::tc_kernel` returns one.
    wide: KernelHandle,
    /// 2026-09-25: `w4a16_gemv_batch32`, handed out above 16 rows within the edge.
    wide32: KernelHandle,
    /// 2026-09-28: `w4a4_proj::max_rows` for this backend: the widest the W4A4 kernels serve
    /// (0 when they are absent or the published tier does not prepare them).
    a4_rows: u32,
}

/// 2026-09-25: The "no NVFP4 kernels" state: an all-zero table, for which `kernel`
/// returns a zero handle at every `m`.
impl Default for W4a16BatchmTiers {
    fn default() -> Self {
        Self {
            handles: [KernelHandle(0); W4A16_BATCHM_WIDTHS.len()],
            wide: KernelHandle(0),
            wide32: KernelHandle(0),
            a4_rows: 0,
        }
    }
}

impl W4a16BatchmTiers {
    /// 2026-09-25: Resolve every tier in the family. A missing kernel is a zero
    /// handle, and `select_tier` skips it.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        let mut handles = [KernelHandle(0); W4A16_BATCHM_WIDTHS.len()];
        for (h, w) in handles.iter_mut().zip(W4A16_BATCHM_WIDTHS) {
            // 2026-09-25: Width 8 resolves through `batch8_kernel`, which prefers
            // `w4a16_gemv_batch8_rt2` unless `METRALE_NO_BATCH8_RT=1`.
            *h = if w == 8 {
                super::batch8_kernel(gpu)
            } else {
                super::try_kernel(gpu, "w4a16_gemv", &format!("w4a16_gemv_batch{w}"))
            };
        }
        let wide = super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16");
        // 2026-09-25: The only call of `w4a4_proj::prepare`, which allocates the
        // W4A4 activation scratch; `resolve` runs while layers are built.
        if let Err(e) = crate::layers::ops::w4a4_proj::prepare(gpu) {
            tracing::warn!("w4a4 scratch/kernels unavailable, projections stay W4A16: {e:#}");
        }
        // 2026-09-28: `w4a16_gemv_batch32` is handed out only within a W4A4 edge above 16
        // rows, so it is resolved only when the published tier can take the W4A4 path.
        let wide32 = if crate::layers::weight_quantization().uses_w4a4_decode() {
            super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch32")
        } else {
            KernelHandle(0)
        };
        Self {
            handles,
            wide,
            wide32,
            a4_rows: crate::layers::ops::w4a4_proj::max_rows(gpu),
        }
    }

    /// 2026-09-28: Row edge of the narrow projection arms for a weight stamped `act`: the
    /// W4A4 edge the published tier gives it (`WeightQuantTier::w4a4_rows`), else the W4A16
    /// edge `gemv_tc::narrow_gemv_max_rows`.
    fn edge_of(&self, act: metrale_config::Nvfp4Act) -> u32 {
        let rows = crate::layers::weight_quantization().w4a4_rows(
            act,
            super::ops::w4a4_proj::W4A4_MAX_M,
            super::ops::w4a4_proj::W4A4_WIDE_MAX_M,
            self.a4_rows,
        );
        if rows > 0 {
            rows
        } else {
            super::ops::gemv_tc::narrow_gemv_max_rows()
        }
    }

    /// 2026-09-28: [`Self::edge_of`] for `w`.
    pub fn edge(&self, w: &crate::weight_map::QuantizedWeight) -> u32 {
        self.edge_of(w.act)
    }

    /// 2026-09-28: `w` declares FP4 activations and this backend has the W4A4 kernels. The
    /// published tier decides what that means for a dense FFN's narrow steps
    /// (`WeightQuantTier::ffn_small_batch_w4a4`).
    pub fn declares_a4(&self, w: &crate::weight_map::QuantizedWeight) -> bool {
        w.act == metrale_config::Nvfp4Act::A4 && self.a4_rows > 0
    }

    /// 2026-09-28: [`Self::edge`] capped at `w4a4_proj::W4A4_MAX_M`: the dense FFN's
    /// narrow arms stop at 32 rows, and 33..=64 go to `forward_prefill`.
    pub fn ffn_edge(&self, w: &crate::weight_map::QuantizedWeight) -> u32 {
        self.edge(w).min(super::ops::w4a4_proj::W4A4_MAX_M)
    }

    /// 2026-09-28: [`Self::kernel`] for a projection over `w`, within [`Self::edge`].
    pub fn kernel_for(&self, m: u32, w: &crate::weight_map::QuantizedWeight) -> KernelHandle {
        self.kernel_within(m, self.edge(w))
    }

    /// 2026-09-25: Which tiers this target resolved: the `present` argument of
    /// [`select_tier`].
    fn present(&self) -> [bool; W4A16_BATCHM_WIDTHS.len()] {
        self.handles.map(|h| h.0 != 0)
    }

    /// 2026-09-25: Narrowest resolved tier covering `m` rows, or `KernelHandle(0)` when
    /// this family cannot serve `m`, for a projection without a policy stamp (the lm_head,
    /// the MTP head): within the edge of an unstamped weight (2026-09-28: the W4A16 edge,
    /// or under `nvfp4` with `--w4a4-downcast` the lever's edge, as before the tiers).
    pub fn kernel(&self, m: u32) -> KernelHandle {
        self.kernel_within(m, self.edge_of(metrale_config::Nvfp4Act::Unstamped))
    }

    fn kernel_within(&self, m: u32, edge: u32) -> KernelHandle {
        if m > W4A16_BATCHM_WIDTHS[W4A16_BATCHM_WIDTHS.len() - 1] && m <= edge {
            // 2026-09-25: 9..=16 → batch16, 17.. → batch32. With the 64-row W4A4
            // edge this also covers 33..=64, which the W4A4 path serves;
            // `ops::w4a16_gemv_batchm` refuses more than 32 rows.
            return if m <= 16 { self.wide } else { self.wide32 };
        }
        select_tier(m, self.present(), exact_m_tiers_enabled())
            .map_or(KernelHandle(0), |i| self.handles[i])
    }

    /// 2026-09-25: Width of the tier `select_tier` picks for `m` (the narrow
    /// family only; `kernel` hands out `wide`/`wide32` above 8 rows).
    pub fn width(&self, m: u32) -> Option<u32> {
        select_tier(m, self.present(), exact_m_tiers_enabled()).map(|i| W4A16_BATCHM_WIDTHS[i])
    }

    /// 2026-09-25: Is the base `w4a16_gemv_batch4` tier resolved? Capability probes
    /// for NVFP4 batched decode ask this one.
    pub fn has_base(&self) -> bool {
        self.handles[0].0 != 0
    }
}

#[cfg(test)]
#[path = "w4a16_gemv_tiers_tests.rs"]
mod w4a16_gemv_tiers_tests;
