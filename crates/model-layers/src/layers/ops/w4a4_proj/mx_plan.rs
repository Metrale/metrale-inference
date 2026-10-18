// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which `w4a4_gemv_mx*` entry serves an `m`-row projection, and
//! with what launch shape: the W4A4 levers ([`W4a4Levers`], resolved once from the
//! startup settings) and the pure planning in `mx_plan`.
//!
//! Owner: model-layers ops.
//! Invariants: `W4a4Levers::from_settings` refuses every value its readers panicked on, with
//! the same message, and reads every other value as they did.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_settings::InferenceServerSettings;

use super::{W4A4_MAX_M, W4a4State};

/// 2026-10-17: The W4A4 projection levers. The first [`super::prepare`] on a backend keeps the
/// copy its launches run under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4a4Levers {
    /// 2026-09-25: `METRALE_W4A4_MX_NT` = 4 (default, also unset or empty) | 1. At 4 the
    /// 9..=64-row W4A4 GEMV runs the activation-reuse twins: 17..=32 rows take four 16-row
    /// weight tiles per CTA, 9..=16 and 33..=64 rows take two, and each warp feeds its
    /// activation fragments to every tile from registers, so the activation matrix is read
    /// from L2 that many times less. The twins keep the one-tile kernels' per-warp chunk order
    /// and warp-order reduction. At 1 every width runs the one-tile kernels.
    pub nt: u32,
    /// 2026-09-25: `METRALE_W4A4_MX_PS` = 1 (default, also unset or empty) | 0. At 1, a
    /// 9..=32-row launch whose whole activation stripe fits in [`PS_SMEM_MAX`] and whose weight
    /// has at least [`PS_MIN_TILES_PER_SM`] 16-row tiles per SM runs the persistent
    /// activation-staged entry (`w4a4_gemv_mx{16,32}_ps`): one CTA per SM pulls 16-row tiles
    /// from a counter and reads the activations from shared memory. 0 routes those launches
    /// back to the [`Self::nt`] kernels. It acts only when the tile factor is not 1, so
    /// `METRALE_W4A4_MX_NT=1` selects the one-tile kernels at every width.
    pub ps: bool,
    /// 2026-09-25: `METRALE_W4A4_PROJ_AUDIT`, a diagnostic. Every W4A4 projection launched
    /// outside a graph capture also runs the W4A16 path, synchronises, and accumulates
    /// `||y_w4a4 - y_w4a16|| / ||y_w4a16||` per call site, logged every 64 samples. Captured
    /// launches are not audited, so it needs graphs off (`METRALE_NO_MTP_VERIFY_GRAPHS=1
    /// METRALE_NO_DECODE_GRAPHS_MULTISEQ=1`) to see the decode and verify steps.
    pub audit: bool,
}

impl W4a4Levers {
    /// 2026-10-17: Resolve from the startup settings, refusing a tile factor other than 1 or 4
    /// and a persistent switch other than 0 or 1.
    pub fn from_settings(s: &InferenceServerSettings) -> Result<Self> {
        let nt = match s.kernels.w4a4_mx_nt.as_deref() {
            None | Some("") | Some("4") => 4,
            Some("1") => 1,
            Some(v) => bail!("METRALE_W4A4_MX_NT={v}: expected 1 or 4"),
        };
        let ps = match s.kernels.w4a4_mx_ps.as_deref() {
            None | Some("") | Some("1") => true,
            Some("0") => false,
            Some(v) => bail!("METRALE_W4A4_MX_PS={v}: expected 0 or 1"),
        };
        Ok(Self {
            nt,
            ps,
            audit: s.telemetry.w4a4_proj_audit,
        })
    }

    /// 2026-10-17: What [`Self::from_settings`] returns at the declared defaults.
    pub const fn declared() -> Self {
        Self {
            nt: 4,
            ps: true,
            audit: false,
        }
    }
}

/// 2026-09-25: Largest dynamic shared memory, in bytes, one CTA may opt in to
/// on GB10 (sm_121).
pub const PS_SMEM_MAX: u32 = 101_376;
/// 2026-09-25: The persistent entries run only when the weight has at least
/// this many 16-row tiles per SM; below it `mx_plan` uses the tiled kernels.
pub const PS_MIN_TILES_PER_SM: u32 = 8;

/// 2026-09-25: Pure: 8-token column blocks (`MB`) of the persistent entry
/// serving `m` rows (`w4a4_gemv_mx16_ps` or `w4a4_gemv_mx32_ps`).
pub fn ps_column_blocks(m: u32) -> u32 {
    if m <= 16 { 2 } else { 4 }
}

/// 2026-09-25: Pure: the host side of the persistent entries' launch contract
/// (`w4a4_gemv_mx_ps.cuh`): dynamic shared memory for `mb` column blocks with
/// `sst` k128 chunks staged per warp. 8 warps x sst x mb x (512 B of
/// fragments + 64 B of scales), plus the reduction buffer of `RJ = 2` 4 KiB
/// blocks that both entries instantiate.
pub fn ps_smem_bytes(mb: u32, sst: u32) -> u32 {
    8 * sst * mb * 576 + 2 * 4096
}

/// 2026-09-25: Pure: k128 chunks in one warp's stripe (chunks c = warp mod 8).
pub fn ps_stripe_chunks(k: u32) -> u32 {
    (k / 128).div_ceil(8)
}

/// 2026-09-25: How one W4A4 GEMV launch is shaped.
#[derive(Clone, Copy, Debug)]
pub(super) enum MxLaunch {
    /// 2026-09-25: Grid ceil(N / rows_per_cta).
    Tiles {
        kernel: KernelHandle,
        rows_per_cta: u32,
    },
    /// 2026-09-25: Grid #SMs, `smem` bytes of dynamic shared memory, `sst`
    /// staged chunks.
    Persistent {
        kernel: KernelHandle,
        sst: u32,
        smem: u32,
    },
}

/// 2026-09-25: Pure: the launch for an `m`-row `[n, k]` projection.
pub(super) fn mx_plan(s: &W4a4State, m: u32, n: u32, k: u32, nt: u32, ps: bool) -> MxLaunch {
    if ps && nt != 1 && m > 8 && m <= W4A4_MAX_M {
        let kernel = if m <= 16 { s.mx16_ps } else { s.mx32_ps };
        let sst = ps_stripe_chunks(k);
        let smem = ps_smem_bytes(ps_column_blocks(m), sst);
        if smem <= PS_SMEM_MAX && n.div_ceil(16) >= PS_MIN_TILES_PER_SM * s.sms {
            return MxLaunch::Persistent { kernel, sst, smem };
        }
    }
    let (kernel, rows_per_cta) = mx_pick(s, m, nt);
    MxLaunch::Tiles {
        kernel,
        rows_per_cta,
    }
}

/// 2026-09-25: Pure: (kernel, rows per CTA) for an `m`-row launch at tile
/// factor `nt`.
pub(super) fn mx_pick(s: &W4a4State, m: u32, nt: u32) -> (KernelHandle, u32) {
    match (m, nt) {
        (0..=8, _) => (s.mx8, 16),
        (9..=16, 1) => (s.mx16, 16),
        (9..=16, _) => (s.mx16_nt2, 32),
        (17..=32, 1) => (s.mx32, 16),
        (17..=32, _) => (s.mx32_nt4, 64),
        (_, 1) => (s.mx64, 16),
        _ => (s.mx64_nt2, 32),
    }
}

#[cfg(test)]
#[path = "w4a4_levers_tests.rs"]
mod w4a4_levers_tests;
