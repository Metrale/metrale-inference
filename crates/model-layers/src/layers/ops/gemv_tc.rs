// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Routing of the narrow NVFP4 batched GEMV to its tensor-core kernels (`w4a16_gemv_tc8` / `w4a16_gemv_tc16` in kernels/gb10/common/w4a16_gemv_tc.cu).
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The tensor-core entries take the same arguments as the CUDA-core `w4a16_gemv_batch*`
//!   kernels, with their own grid (`ceil(N / cols_per_cta)`) and a 256-thread block.
//! - A launch routes only when `K % 128 == 0` and `1 <= m <= 16`; otherwise the caller keeps its
//!   CUDA-core kernel ([`tc_route`]).
//! - [`super::w4a16_gemv_batchm`] and [`tc_fixed_m`] both route through [`tc_kernel`].
//! - `METRALE_NO_W4A16_TC` set to any non-empty value turns the tensor-core path off
//!   ([`W4a16TcPlan::tc`], resolved once from the startup settings in
//!   `OpsLevers::from_settings`). The bench crate's `PERF_CONTROLS` discloses it, and
//!   `record_env_tests` reads its predicate and the widening's from `ops_levers.rs`.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::weight_map::QuantizedWeight;

/// 2026-09-25: Rows the `w4a16_gemv_tc8` entry covers (its `MT` is 8).
pub const TC8_MAX_M: u32 = 8;
/// 2026-09-25: Rows the `w4a16_gemv_tc16` entry covers (its `MT` is 16).
pub const TC16_MAX_M: u32 = 16;
/// 2026-09-25: Columns per CTA, `8 * NT`: tc8 has `NT = 1`, tc16 has `NT = 2`.
pub const TC8_COLS_PER_CTA: u32 = 8;
pub const TC16_COLS_PER_CTA: u32 = 16;
/// 2026-09-25: Threads per CTA, `TC_WARPS * 32` with `TC_WARPS = 8` in the .cu; the warps split K.
pub const TC_BLOCK: u32 = 256;

/// 2026-09-25: Which tensor-core entry serves a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TcKind {
    M8,
    M16,
}

impl TcKind {
    pub fn cols_per_cta(self) -> u32 {
        match self {
            TcKind::M8 => TC8_COLS_PER_CTA,
            TcKind::M16 => TC16_COLS_PER_CTA,
        }
    }
}

/// 2026-09-25: The routing decision, without I/O. `None` keeps the caller's CUDA-core kernel.
///
/// The kernel walks K in 128-element blocks, so a `K` that is not a multiple of 128 declines.
/// Any `N` routes: in a partial last tile the kernel loads zeros for columns past `N` and never
/// stores them.
/// `tc8` serves `m <= 8`; `tc16` serves `m <= 16`, including `m <= 8` when `tc8` is missing.
pub fn tc_route(
    m: u32,
    n: u32,
    k: u32,
    enabled: bool,
    have8: bool,
    have16: bool,
) -> Option<TcKind> {
    if !enabled || m == 0 || n == 0 || k == 0 || !k.is_multiple_of(128) {
        return None;
    }
    if m <= TC8_MAX_M && have8 {
        Some(TcKind::M8)
    } else if m <= TC16_MAX_M && have16 {
        Some(TcKind::M16)
    } else {
        None
    }
}

/// 2026-09-25: Row edges of the narrow-GEMV dispatch arms: `NARROW_MAX_ROWS` by default, and
/// `WIDE_MAX_ROWS` (the `tc16` reach) when [`W4a16TcPlan::wide`]. Read through
/// [`W4a16TcPlan::narrow_max_rows`].
pub const NARROW_MAX_ROWS: u32 = 8;
pub const WIDE_MAX_ROWS: u32 = TC16_MAX_M;

/// 2026-10-17: The W4A16 tensor-core levers, resolved once from the startup settings and carried
/// on `OpsLevers::w4a16_tc`, so every launch, including one captured in a CUDA graph, sees the
/// same choice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4a16TcPlan {
    /// 2026-09-25: On when `METRALE_NO_W4A16_TC` is unset or empty. Any non-empty value, `0`
    /// included, turns the tensor-core path off.
    pub tc: bool,
    /// 2026-09-25: The opt-in widening: on when `METRALE_W4A16_TC_WIDE` is set to a non-empty
    /// value and [`Self::tc`]. It raises [`Self::narrow_max_rows`] to 16 and lets [`tc_fixed_m`]
    /// route the fixed-M launchers (`w4a16_gemv_batch2/3`, `w4a16_gemv_dual_batch2/3`) to the
    /// tensor-core kernels.
    pub wide: bool,
}

impl W4a16TcPlan {
    /// 2026-10-17: The tensor-core path on and the widening off, as unset variables leave them.
    pub const fn declared() -> Self {
        Self {
            tc: true,
            wide: false,
        }
    }

    /// 2026-09-25: Row edge of the narrow-GEMV dispatch arms (see [`Self::wide`]).
    pub const fn narrow_max_rows(self) -> u32 {
        if self.wide {
            WIDE_MAX_ROWS
        } else {
            NARROW_MAX_ROWS
        }
    }
}

/// 2026-09-25: Resolved handles, cached per backend. A `KernelHandle` names a function in one
/// backend's loaded module, so the cache is keyed by the backend object's address. A missing
/// entry is `KernelHandle(0)`.
#[derive(Clone, Copy)]
struct TcHandles {
    tc8: KernelHandle,
    tc16: KernelHandle,
}

fn tc_handles(gpu: &dyn GpuBackend) -> TcHandles {
    static CACHE: OnceLock<Mutex<Vec<(usize, TcHandles)>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = TcHandles {
        tc8: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc8"),
        tc16: crate::layers::try_kernel(gpu, "w4a16_gemv_tc", "w4a16_gemv_tc16"),
    };
    guard.push((key, h));
    h
}

/// 2026-09-25: The tensor-core kernel and grid x for this launch, or `None` to keep the
/// CUDA-core kernel. It does not consult [`W4a16TcPlan::wide`]; the callers' row edges do.
pub fn tc_kernel(
    gpu: &dyn GpuBackend,
    plan: W4a16TcPlan,
    m: u32,
    n: u32,
    k: u32,
) -> Option<(KernelHandle, u32)> {
    if !plan.tc {
        return None;
    }
    let h = tc_handles(gpu);
    let kind = tc_route(m, n, k, true, h.tc8.0 != 0, h.tc16.0 != 0)?;
    let handle = match kind {
        TcKind::M8 => h.tc8,
        TcKind::M16 => h.tc16,
    };
    Some((handle, n.div_ceil(kind.cols_per_cta())))
}

/// 2026-09-25: The tensor-core route of the fixed-M launchers, taken only when
/// [`W4a16TcPlan::wide`]: `Ok(true)` when it launched, `Ok(false)` to keep the CUDA-core kernel.
#[allow(clippy::too_many_arguments)]
pub fn tc_fixed_m(
    gpu: &dyn GpuBackend,
    plan: W4a16TcPlan,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<bool> {
    if !plan.wide {
        return Ok(false);
    }
    let Some((tc, grid_x)) = tc_kernel(gpu, plan, m, n, k) else {
        return Ok(false);
    };
    KernelLaunch::new(gpu, tc)
        .grid([grid_x, 1, 1])
        .block([TC_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)?;
    Ok(true)
}

#[cfg(test)]
#[path = "gemv_tc_tests.rs"]
mod gemv_tc_tests;
