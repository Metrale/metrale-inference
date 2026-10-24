// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: The routed W4A4 kernel sets of `Glm5NextMlpKernels` (moved unchanged from
//! `mod.rs`): the grouped W4A4 expert GEMM's entries and the `_k64` twins.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: as `mod.rs`'s.

use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-10-10: `w4a4_grouped_tiles` and `w4a4_grouped_mx_m{32,64,128}[_k64]`
/// (`w4a4_grouped_mx.cu`), indexed by `forward::w4a4_grouped::W4A4_GROUPED_MT`; the `_k64` entries
/// read a routed down projection whose K is a multiple of 64 but not of 128. Each is
/// `KernelHandle(0)` when absent from the target's PTX.
#[derive(Clone, Copy)]
pub struct W4a4GroupedKernels {
    pub tiles: KernelHandle,
    /// 2026-10-13: `w4a4_grouped_one_group`, the offsets and tiles of one group of m rows (the
    /// dense MLP's one-group launch, `forward::w4a4_dense_grouped`).
    pub one_group: KernelHandle,
    pub mx: [KernelHandle; 3],
    pub mx_k64: [KernelHandle; 3],
    /// 2026-10-14: `w4a4_grouped_gu_m{32,64,128}`, gate, up, the clamped SwiGLU and the down
    /// input's static quantization in one launch (`forward::w4a4_grouped_gu`,
    /// `METRALE_GLM_MOE_W4A4_GU_FUSED=1`).
    pub gu: [KernelHandle; 3],
    /// 2026-10-14: `w4a4_grouped_mx_m128_r`, `w4a4_grouped_mx_m128_k64_r` and
    /// `w4a4_grouped_gu_m128_r`: the m128 down (plain and `_k64`) and fused gate/up entries with
    /// a raster-group argument (`forward::w4a4_grouped_raster`, `METRALE_GLM_MOE_W4A4_RASTER`).
    pub raster: [KernelHandle; 3],
    /// 2026-10-15: `w4a4_grouped_mx_m128_n` and `w4a4_grouped_mx_m128_k64_n`: the m128 down with
    /// `nc` column tiles per CTA (`forward::w4a4_grouped_nc`, `METRALE_GLM_MOE_W4A4_NC`).
    pub nc: [KernelHandle; 2],
}

/// 2026-10-10: `w4a4_quant_rows_static_k64`, `w4a4_gemv_mx8_moe_slots_k64` and
/// `w4a4_gemv_mx{8,16}_moe_union_sweep_k64` (`w4a4_gemv_mx_moe.cu`): the quantizer writes rows at
/// `K.next_multiple_of(128)` with zero padding groups, and the GEMVs read the weights at their
/// natural `K % 64 == 0` width. Each is `KernelHandle(0)` when absent from the target's PTX.
#[derive(Clone, Copy)]
pub struct W4a4MoeK64Kernels {
    pub quant: KernelHandle,
    pub slots: KernelHandle,
    /// 2026-10-10: `w4a4_gemv_mx8_moe_slots_sweep_k64`, the one-row own-slots sweep's twin.
    pub own: KernelHandle,
    pub sweep: [KernelHandle; 2],
}
