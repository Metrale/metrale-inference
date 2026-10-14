// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The decode-width collapse of a low-rank hyper-connection site in one launch
//! (`hc_lowrank_site`, kernels/hopper/common/hc_lowrank_site_hopper.cuh): `hc_pre_stage_bf16`,
//! `hc_lowrank_down_inject` and `hc_lowrank_up_mix` become one kernel whose blocks take the three
//! phases in turn and stream every weight tile to shared memory while the earlier phases run.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Off unless `METRALE_HC_SITE_FUSED=1` (read once per process); off, nothing here runs.
//! - `normed`, `low`, `inj_out` and `y_out` are the bytes of the three-launch route at every T,
//!   and a token's are the same at every T (`hyper_connection_site_tests.rs`): every reduction
//!   uses the tree of the kernel it replaces.
//! - Resolved only on a target that compiles the kernel (Hopper); elsewhere the handle is zero and
//!   [`HcSite::admits`] is false.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::try_target_kernel;

const MODULE: &str = "hc_lowrank_gemv_hopper";
/// 2026-10-14: `HCS_THREADS`.
const THREADS: u32 = 256;
/// 2026-10-14: `HCS_VWARPS`: down blocks per row group (the replaced kernel's warps).
const VWARPS: u32 = 8;
/// 2026-10-14: `HCS_GROUP_ROWS`, `HCS_UP_COLS`, `HCS_DOWN_TOKENS`.
const GROUP_ROWS: u32 = 16;
const UP_COLS: u32 = 16;
const DOWN_TOKENS: u32 = 8;
/// 2026-10-14: `HCS_MAX_NJ`: K = hc * H is at most 8 segments of 2048.
const K_SEGMENT: u32 = 2048;
const MAX_NJ: u32 = 8;
/// 2026-10-14: `HCS_MAX_GROUPS`, `HLG_STREAMS`, `HLG_MAX_RANK`.
const MAX_GROUPS: u32 = 64;
const MAX_STREAMS: u32 = 4;
const MAX_RANK: u32 = 512;
/// 2026-10-14: The largest T: two passes of 8 tokens (`2 * HLG_TOKENS`), the GEMVs' limit.
pub(crate) const HC_SITE_MAX_T: u32 = 16;

/// 2026-10-14: `METRALE_HC_SITE_FUSED` equal to `"1"`, read once per process.
fn hc_site_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_HC_SITE_FUSED").as_deref() == Ok("1"))
}

/// 2026-10-14: One site's device operands.
pub(crate) struct HcSiteArgs {
    /// 2026-10-14: [T, hc, H] FP32.
    pub streams: DevicePtr,
    pub norm_w: DevicePtr,
    pub down_w: DevicePtr,
    /// 2026-10-14: NULL on the model-level mixer; then `inj_out` is not written.
    pub inject_w: DevicePtr,
    pub up_w: DevicePtr,
    /// 2026-10-14: [T, hc * H] BF16 out (the following ops read it).
    pub normed: DevicePtr,
    /// 2026-10-14: FP32 scratch of [`HcSite::partial_bytes`].
    pub partial: DevicePtr,
    pub low: DevicePtr,
    pub y_out: DevicePtr,
    pub inj_out: DevicePtr,
}

/// 2026-10-14: The site kernel; a zero handle when off or not compiled for the target.
#[derive(Clone, Copy)]
pub(crate) struct HcSite {
    kernel: KernelHandle,
}

impl HcSite {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Self {
        let kernel = if hc_site_enabled() {
            try_target_kernel(gpu, MODULE, "hc_lowrank_site")
        } else {
            KernelHandle(0)
        };
        Self { kernel }
    }

    fn rows(a: &HcSiteArgs, hc_mult: u32, rank: u32) -> u32 {
        rank + if a.inject_w.is_null() { 0 } else { hc_mult }
    }

    /// 2026-10-14: Bytes of the down partials, [8, rows, T] FP32.
    pub(crate) fn partial_bytes(a: &HcSiteArgs, num_tokens: u32, hc_mult: u32, rank: u32) -> usize {
        (VWARPS * Self::rows(a, hc_mult, rank) * num_tokens) as usize * 4
    }

    /// 2026-10-14: The shapes the kernel accepts (it traps on the rest): resolved, `1 <= T <= 16`,
    /// `1 <= hc <= 4`, H a multiple of 16, `hc * H` a multiple of 2048 and at most 16384, rank a
    /// multiple of 8 in [8, 512], at most 64 groups of 16 rows, the partials no larger than
    /// `partial_capacity` bytes, and every pointer the kernel reads in 16-byte vectors aligned.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn admits(
        &self,
        a: &HcSiteArgs,
        num_tokens: u32,
        hidden_size: u32,
        hc_mult: u32,
        rank: u32,
        partial_capacity: usize,
    ) -> bool {
        let k = hc_mult * hidden_size;
        let aligned = [a.normed, a.down_w, a.up_w, a.inject_w, a.low]
            .iter()
            .all(|p| p.0.is_multiple_of(16));
        self.kernel.0 != 0
            && (1..=HC_SITE_MAX_T).contains(&num_tokens)
            && (1..=MAX_STREAMS).contains(&hc_mult)
            && hidden_size.is_multiple_of(UP_COLS)
            && k.is_multiple_of(K_SEGMENT)
            && (1..=MAX_NJ).contains(&(k / K_SEGMENT))
            && rank.is_multiple_of(8)
            && (8..=MAX_RANK).contains(&rank)
            && Self::rows(a, hc_mult, rank).div_ceil(GROUP_ROWS) <= MAX_GROUPS
            && Self::partial_bytes(a, num_tokens, hc_mult, rank) <= partial_capacity
            && aligned
    }

    /// 2026-10-14: `hcs_smem_bytes`: the larger weight tile, then the larger staging area.
    fn smem_bytes(num_tokens: u32, hidden_size: u32, hc_mult: u32, rank: u32) -> u32 {
        let nj = hc_mult * hidden_size / K_SEGMENT;
        let r8n = rank / 8;
        let tile = (GROUP_ROWS * nj * 32).max(UP_COLS * hc_mult * r8n);
        let staging = (DOWN_TOKENS * nj * 32).max(num_tokens * r8n);
        16 * (tile + staging)
    }

    /// 2026-10-14: `normed`, `low`, (with `inject_w`) `inj_out`, and `y_out` for `num_tokens`
    /// tokens. The caller checks [`HcSite::admits`] first.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn launch(
        &self,
        gpu: &dyn GpuBackend,
        a: &HcSiteArgs,
        num_tokens: u32,
        hidden_size: u32,
        hc_mult: u32,
        rank: u32,
        norm_eps: f32,
        stream: u64,
    ) -> Result<()> {
        let groups = Self::rows(a, hc_mult, rank).div_ceil(GROUP_ROWS);
        let blocks = num_tokens * hc_mult + VWARPS * groups + hidden_size / UP_COLS;
        KernelLaunch::new(gpu, self.kernel)
            .grid([blocks, 1, 1])
            .block([THREADS, 1, 1])
            .shared_mem(Self::smem_bytes(num_tokens, hidden_size, hc_mult, rank))
            .arg_ptr(a.streams)
            .arg_ptr(a.norm_w)
            .arg_ptr(a.down_w)
            .arg_ptr(a.inject_w)
            .arg_ptr(a.up_w)
            .arg_ptr(a.normed)
            .arg_ptr(a.partial)
            .arg_ptr(a.low)
            .arg_ptr(a.y_out)
            .arg_ptr(a.inj_out)
            .arg_u32(num_tokens)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(rank)
            .arg_f32(norm_eps)
            .arg_f32(1.0f32 / hc_mult as f32)
            .launch(stream)
    }
}
