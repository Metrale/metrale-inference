// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Decode-width GEMVs of the low-rank hyper-connection collapse
//! (kernels/hopper/common/hc_lowrank_gemv_hopper.cu): after `hc_pre_stage_bf16`, two launches
//! replace the small-T cuBLASLt projections and `hc_silu_scale` / `hc_pre_mix` (eight launches at
//! T = 1 with the injection).
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Resolved only on a target that compiles the kernel file (Hopper); elsewhere the handles are
//!   zero and [`HcGemv::admits`] is false, so the cuBLASLt path runs unchanged.
//! - A token's outputs are the same bits at every T and on every run
//!   (`hyper_connection_gemv_tests.rs`); only the projections' summation order differs from the
//!   cuBLASLt path.
//! - `METRALE_HC_GEMV=0` (read once per process) keeps the cuBLASLt path where the kernels exist.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::try_target_kernel;

const MODULE: &str = "hc_lowrank_gemv_hopper";
/// 2026-10-14: The kernels' block width (`HLG_THREADS`).
const THREADS: u32 = 256;
/// 2026-10-14: Weight rows per block of the down GEMV (`HLG_ROWS`).
const DOWN_ROWS: u32 = 4;
/// 2026-10-14: Hidden indices per block of the up GEMV (`HLG_WARPS`).
const UP_COLS: u32 = THREADS / 32;
/// 2026-10-14: Tokens per block in both kernels (`HLG_TOKENS`).
const TOKENS: u32 = 8;
/// 2026-10-14: The largest rank and stream count the kernels hold (`HLG_MAX_RANK`, `HLG_STREAMS`).
const MAX_RANK: u32 = 512;
const MAX_STREAMS: u32 = 4;
/// 2026-10-14: The largest T sent to the GEMVs. They read each weight row once per 8 tokens on
/// CUDA cores, so their cost grows with T while cuBLASLt's tensor-core GEMM stays flat; 16 is the
/// starting point, to be re-cut from a measured crossover.
pub(crate) const HC_GEMV_MAX_T: u32 = 16;

/// 2026-10-14: `METRALE_HC_GEMV` equal to `"0"`, read once per process.
fn hc_gemv_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("METRALE_HC_GEMV").as_deref() == Ok("0"))
}

/// 2026-10-14: The two GEMV kernels; zero handles where the target does not compile them.
#[derive(Clone, Copy)]
pub(crate) struct HcGemv {
    down: KernelHandle,
    up: KernelHandle,
}

/// 2026-10-14: One collapse's device operands. `normed` is `hc_pre_stage_bf16`'s output.
pub(crate) struct HcGemvArgs {
    pub normed: DevicePtr,
    pub down_w: DevicePtr,
    pub up_w: DevicePtr,
    /// 2026-10-14: NULL on the model-level mixer; then `inj_out` is not written.
    pub inject_w: DevicePtr,
    pub low: DevicePtr,
    pub y_out: DevicePtr,
    pub inj_out: DevicePtr,
}

impl HcGemv {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Self {
        if hc_gemv_disabled() {
            return Self {
                down: KernelHandle(0),
                up: KernelHandle(0),
            };
        }
        Self {
            down: try_target_kernel(gpu, MODULE, "hc_lowrank_down_inject"),
            up: try_target_kernel(gpu, MODULE, "hc_lowrank_up_mix"),
        }
    }

    /// 2026-10-14: The shapes and operands the kernels accept (they trap on any other): both
    /// kernels resolved, `1 <= T <= HC_GEMV_MAX_T`, `hc * H` and `rank` multiples of 8, `rank <=
    /// 512`, `1 <= hc <= 4`, and every pointer 16-byte aligned (the kernels load 8-element
    /// vectors; row strides are then aligned by the multiples of 8).
    pub(crate) fn admits(
        &self,
        a: &HcGemvArgs,
        num_tokens: u32,
        hidden_size: u32,
        hc_mult: u32,
        rank: u32,
    ) -> bool {
        let aligned = [a.normed, a.down_w, a.up_w, a.inject_w, a.low]
            .iter()
            .all(|p| p.0.is_multiple_of(16));
        self.down.0 != 0
            && self.up.0 != 0
            && (1..=HC_GEMV_MAX_T).contains(&num_tokens)
            && (1..=MAX_STREAMS).contains(&hc_mult)
            && (hc_mult * hidden_size).is_multiple_of(8)
            && rank.is_multiple_of(8)
            && (8..=MAX_RANK).contains(&rank)
            && aligned
    }

    /// 2026-10-14: `low` and (with `inject_w`) `inj_out`, then `y_out`, for `num_tokens` tokens.
    /// The caller checks [`HcGemv::admits`] first.
    pub(crate) fn launch(
        &self,
        gpu: &dyn GpuBackend,
        a: &HcGemvArgs,
        num_tokens: u32,
        hidden_size: u32,
        hc_mult: u32,
        rank: u32,
        stream: u64,
    ) -> Result<()> {
        let inv_hc = 1.0f32 / hc_mult as f32;
        let k = hc_mult * hidden_size;
        let rows = rank + if a.inject_w.is_null() { 0 } else { hc_mult };
        let token_groups = num_tokens.div_ceil(TOKENS);
        KernelLaunch::new(gpu, self.down)
            .grid([rows.div_ceil(DOWN_ROWS), token_groups, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(a.normed)
            .arg_ptr(a.down_w)
            .arg_ptr(a.inject_w)
            .arg_ptr(a.low)
            .arg_ptr(a.inj_out)
            .arg_u32(num_tokens)
            .arg_u32(k)
            .arg_u32(rank)
            .arg_u32(hc_mult)
            .arg_f32(inv_hc)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.up)
            .grid([hidden_size.div_ceil(UP_COLS), token_groups, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(a.low)
            .arg_ptr(a.up_w)
            .arg_ptr(a.normed)
            .arg_ptr(a.y_out)
            .arg_u32(num_tokens)
            .arg_u32(hidden_size)
            .arg_u32(hc_mult)
            .arg_u32(rank)
            .arg_f32(inv_hc)
            .launch(stream)
    }
}
