// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The route of the attention prefill (`prefill_attn_core.rs`): its projection arm,
//! its attention kernel, the kernels a group of each must list, and the legacy levers that leave
//! it. The projections (Q, K, V and O) run on the transposed NVFP4 twins,
//! as `qwen3_attention/prefill/cache_skip_qkv.rs:315-339`, `paged_qkv.rs:324-348` and
//! `paged_oproj.rs:199-223` run them under the recipe: `ops::w4a16_gemm_n128` (the `_p3` tile)
//! up to 128 rows, then `w4a16_gemm_m128_dispatch` (`prefill_weights.rs:24-100`): the 128-row M
//! tile below `W4A16_VIA_FP8_MIN_M` rows, the FP8 GEMM on the dequantized weight from there.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A plan's projection arm is fixed by its kernels; a pass whose rows another arm serves is
//!   refused at run time, never run on the wrong kernel.
//! - Every legacy switch that would take another arm (a CUTLASS NVFP4 projection, the v2/v3
//!   M128 variants, the BF16-MMA M128 kernel, the non-pipelined tile GEMM) is refused at build.

use anyhow::{Result, bail, ensure};
use metrale_circuit::KernelId;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// 2026-10-03: Which tile a plan's projections take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Proj {
    /// 2026-10-03: `w4a16::w4a16_gemm_t_p3` (`ops::w4a16_gemm_n128`), 1..=128 rows.
    P3,
    /// 2026-10-03: `w4a16::w4a16_gemm_t_m128` (`ops::w4a16_gemm_n128_m128`), 129 rows to below
    /// `W4A16_VIA_FP8_MIN_M`.
    M128,
    /// 2026-10-03: `ops::w4a16_t_via_fp8_ldmab`: the weight dequantized to E4M3, the
    /// activation cast to E4M3, the FP8 GEMM (three launches) from `W4A16_VIA_FP8_MIN_M` rows.
    ViaFp8,
}

impl Proj {
    /// 2026-10-03: The arm whose first kernel is `first`.
    pub fn of(first: &KernelId) -> Result<Self> {
        Ok(match (first.module.as_str(), first.func.as_str()) {
            ("w4a16", "w4a16_gemm_t_p3") => Self::P3,
            ("w4a16", "w4a16_gemm_t_m128") => Self::M128,
            ("w4a16_fp8_ldmab", "fp8_predequant_nvfp4_t") => Self::ViaFp8,
            _ => bail!("`{first}` starts no attention prefill projection arm"),
        })
    }

    /// 2026-10-03: The kernels one projection launches, in order.
    pub fn kernels(self) -> &'static [(&'static str, &'static str)] {
        match self {
            Self::P3 => &[("w4a16", "w4a16_gemm_t_p3")],
            Self::M128 => &[("w4a16", "w4a16_gemm_t_m128")],
            Self::ViaFp8 => &[
                ("w4a16_fp8_ldmab", "fp8_predequant_nvfp4_t"),
                ("w4a16", "bf16_to_fp8"),
                ("w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab"),
            ],
        }
    }

    /// 2026-10-03: Whether the legacy dispatch takes this arm at `m` rows.
    pub fn serves(self, m: u32) -> bool {
        match self {
            Self::P3 => m <= 128,
            Self::M128 => m > 128 && m < ops::W4A16_VIA_FP8_MIN_M,
            Self::ViaFp8 => m >= ops::W4A16_VIA_FP8_MIN_M,
        }
    }
}

/// 2026-10-03: Refuse a build whose environment sends a projection off the arm the rules
/// model: the levers `prefill_weights.rs`, `cache_skip_qkv.rs`, `paged_qkv.rs` and
/// `paged_oproj.rs` read before the transposed-twin arm.
pub(super) fn check_levers(proj: Proj) -> Result<()> {
    let d = ops::GemmDispatch::from_env();
    ensure!(
        !(d.cutlass_nvfp4_gemm
            || d.cutlass_nvfp4_attn_q
            || d.cutlass_nvfp4_attn_kv
            || d.cutlass_nvfp4_attn_o),
        "a METRALE_CUTLASS_NVFP4_* lever routes the attention prefill projections to CUTLASS; \
         the circuit does not model that"
    );
    ensure!(
        std::env::var_os("METRALE_NO_TGEMM_PIPELINE3").is_none(),
        "METRALE_NO_TGEMM_PIPELINE3 takes `w4a16_gemm_t` for the projection tile; the plan runs \
         `w4a16_gemm_t_p3`"
    );
    if proj != Proj::P3 {
        ensure!(
            d.w4a16_variant < 2,
            "METRALE_W4A16_VARIANT selects the v{} M128 kernel; the plan runs `w4a16_gemm_t_m128`",
            d.w4a16_variant
        );
        ensure!(
            !ops::ModelLevers::get().bf16_tc_proj,
            "METRALE_BF16_TC_PROJ takes the BF16-MMA M128 kernel; the plan runs the FP8 tile"
        );
    }
    Ok(())
}

/// 2026-10-03: One projection `[m, k] x W^T -> [m, n]` on the arm, `handle` the arm's GEMM
/// handle (unused by the FP8 arm, whose launcher resolves its own kernels).
#[allow(clippy::too_many_arguments)]
pub(super) fn run(
    gpu: &dyn GpuBackend,
    proj: Proj,
    handle: KernelHandle,
    x: DevicePtr,
    w: &QuantizedWeight,
    y: DevicePtr,
    [m, n, k]: [u32; 3],
    stream: u64,
) -> Result<()> {
    ensure!(
        proj.serves(m),
        "a {m}-row projection does not take the {proj:?} arm this plan was built for"
    );
    match proj {
        Proj::P3 => ops::w4a16_gemm_n128(gpu, handle, x, w, y, m, n, k, stream),
        Proj::M128 => ops::w4a16_gemm_n128_m128(gpu, handle, x, w, y, m, n, k, stream),
        Proj::ViaFp8 => ops::w4a16_t_via_fp8_ldmab(gpu, x, w, y, m, n, k, stream),
    }
}

/// 2026-10-03: Rows from which the paged route takes the 128-row twin (`paged_attn.rs:143`).
pub(super) const PAGED_BR64_ROWS: u32 = 256;

/// 2026-10-03: The attention kernel a plan runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Attn {
    /// 2026-10-03: `attn_prefill_fa128` over the pass's own contiguous Q/K/V.
    Contiguous,
    /// 2026-10-03: `attn_prefill_paged`, below 256 rows.
    PagedSmall,
    /// 2026-10-03: `attn_prefill_fa128_paged`, from 256 rows.
    PagedFa128,
}

impl Attn {
    pub fn kernel(self) -> (&'static str, &'static str) {
        match self {
            Self::Contiguous => ("attn_prefill_fa128", "attn_prefill_fa128"),
            Self::PagedSmall => ("prefill_paged", "attn_prefill_paged"),
            Self::PagedFa128 => ("attn_prefill_fa128", "attn_prefill_fa128_paged"),
        }
    }

    pub fn serves(self, m: u32) -> bool {
        match self {
            Self::Contiguous => true,
            Self::PagedSmall => m < PAGED_BR64_ROWS,
            Self::PagedFa128 => m >= PAGED_BR64_ROWS,
        }
    }
}

/// 2026-10-03: The route's kernels for `proj`, `rope` and `attn`, in launch order.
pub(super) fn expected(proj: Proj, rope: (&str, &str), attn: Attn) -> Vec<(String, String)> {
    let p = || {
        proj.kernels()
            .iter()
            .map(|&(m, f)| (m.to_string(), f.to_string()))
    };
    p().chain(p())
        .chain(p())
        .chain(
            [
                ("ssm_preprocess", "deinterleave_qg_split_qnorm"),
                ("norm", "rms_norm"),
                rope,
                ("reshape_and_cache", "reshape_and_cache_flash"),
                attn.kernel(),
                ("residual_add", "sigmoid_gate_mul_batched"),
            ]
            .into_iter()
            .map(|(m, f)| (m.to_string(), f.to_string())),
        )
        .chain(p())
        .collect()
}

/// 2026-10-03: Refuse the legacy levers that leave the modelled route (`cache_skip.rs:204`,
/// `paged.rs:120-123`, `paged_qkv.rs:30-35`, `paged_oproj.rs:40-45`,
/// `prefill_attn_fa128.rs:36-40`).
pub(super) fn check_route_levers() -> Result<()> {
    for var in ["METRALE_NO_ATTN_FA128", "METRALE_ATTN_W4A4"] {
        ensure!(
            std::env::var_os(var).is_none(),
            "{var} takes an attention prefill arm the circuit does not model"
        );
    }
    for var in ["METRALE_FUSED_KV", "METRALE_ATTN_PREFILL_FUSED_QROPE"] {
        ensure!(
            std::env::var(var).as_deref() != Ok("1"),
            "{var}=1 takes a fused attention prefill arm the circuit does not model"
        );
    }
    Ok(())
}
