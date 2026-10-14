// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the dense GEMVs: BF16 for one row, two rows and M
//! rows, and FP8-weight for one row.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`dense_gemv_batchm`] returns an error, without launching, when `m` is
//!   outside `1..=DENSE_GEMV_BATCHM_MAX_M`.

use super::*;

/// 2026-09-25: Dense BF16 GEMV for one row, `C = A @ B^T`: A `[1, K]`, B
/// `[N, K]`, C `[1, N]`. Each 256-thread block computes 4 outputs, 64 threads
/// per output (kernel `dense_gemv_bf16(A, B, C, N, K)`).
pub fn dense_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Dense BF16 GEMV for two rows in one pass over the weight. Each
/// row follows `dense_gemv_bf16`'s K order and reduction (gb10 kernels).
///
/// `input`: `[2, K]` BF16, contiguous; `output`: two rows at
/// `output + t * out_stride` (BF16 elements). The batched SSM decode uses it
/// for a BF16 `in_proj_qkvz` (`qwen3_ssm/trait_decode_batched.rs`).
///
/// Kernel: `dense_gemv_bf16_batch2(A, B, C, N, K, out_stride)`
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: Rows `dense_gemv_bf16_batchm` computes: `MAX_M` in
/// `kernels/gb10/common/dense_gemv_bf16_batchm.cu`. The kernel clamps a
/// larger M, so [`dense_gemv_batchm`] refuses above this value. Each row's
/// accumulation does not depend on M, so a row's result is the same at every
/// batch width.
///
/// Other readers: the row thresholds of the GLM-5.3 DSA, KDA and MLP dispatch
/// (`glm5next_dsa/layer.rs`, `glm5next_kda/mod.rs`, `glm5next_mlp/forward.rs`,
/// `glm5next_layer/levers.rs`), the `verify_k` workspace size
/// (`glm5_next_load/loader.rs`) and the clamp in
/// `target_defaults::resolve_batchm_max`.
pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 16;

/// 2026-09-25: The widest batch that the MTP row dispatch
/// (`mtp_head/row_dispatch.rs`) and the BF16 arm of the model's
/// `lm_head_batched` (`impl_a3_lm_head.rs`) hand to `dense_gemv_bf16_batchm`;
/// `target_defaults::BASELINE_BATCHM_MAX` is set from it. Above it those paths
/// choose other kernels, some of which accumulate in a different order, so
/// this value decides which bits a decode of a given width produces.
pub const DENSE_GEMV_BATCHM_DECODE_MAX_M: u32 = 8;

/// 2026-09-25: Dense BF16 GEMV for M rows, `C[t] = A[t] @ B^T` for `t` in
/// `[0, M)`, reading the weight once for all rows. Each row follows
/// `dense_gemv_bf16`'s K order and reduction (gb10 kernels).
///
/// `input`: `[M, K]` BF16, contiguous; `output`: M rows at
/// `output + t * out_stride` (BF16 elements). Refuses `m` outside
/// `1..=DENSE_GEMV_BATCHM_MAX_M`.
///
/// Kernel: `dense_gemv_bf16_batchm(A, B, C, M, N, K, out_stride)`
pub fn dense_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The kernel clamps M to its MAX_M instead of failing, which
    // would leave the rows past it unwritten.
    ensure!(
        (1..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm: m={m} outside 1..={DENSE_GEMV_BATCHM_MAX_M} \
         (kernel MAX_M clamps silently; use dense_gemm_tc for wider batches)"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-10-14: The direct-schedule entries of `dense_gemv_bf16_batchm.cu`,
/// `dense_gemv_bf16_batchm_r<R>_m<M>(A, B, C, N, K, out_stride)` for every M in
/// `2..=DENSE_GEMV_BATCHM_MAX_M`, compiled only where the target defines
/// `METRALE_BATCHM_DIRECT_R`. Each row keeps the bits of `dense_gemv_bf16`, as
/// in [`dense_gemv_batchm`]; R (outputs per 64-lane group) is read from the
/// entry names, so the build flag is its only source.
#[derive(Debug, Clone)]
pub struct DenseGemvBatchmDirect {
    outputs_per_group: u32,
    /// Entry for `m` at index `m - 2`.
    by_m: Vec<KernelHandle>,
}

impl DenseGemvBatchmDirect {
    /// 2026-10-14: The widest R whose full M set is loaded, or `None` when
    /// the target compiles no direct entries.
    pub fn load(gpu: &dyn GpuBackend) -> Option<Self> {
        [4u32, 2, 1].into_iter().find_map(|r| {
            let by_m = (2..=DENSE_GEMV_BATCHM_MAX_M)
                .map(|m| {
                    let h = crate::layers::try_kernel(
                        gpu,
                        "dense_gemv_bf16_batchm",
                        &format!("dense_gemv_bf16_batchm_r{r}_m{m}"),
                    );
                    (h.0 != 0).then_some(h)
                })
                .collect::<Option<Vec<_>>>()?;
            Some(Self {
                outputs_per_group: r,
                by_m,
            })
        })
    }
}

/// 2026-10-14: [`dense_gemv_batchm`] through the direct-schedule entry for
/// exactly `m` rows: same inputs, same per-row bits. Refuses `m` outside
/// `2..=DENSE_GEMV_BATCHM_MAX_M`.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_direct(
    gpu: &dyn GpuBackend,
    direct: &DenseGemvBatchmDirect,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        (2..=DENSE_GEMV_BATCHM_MAX_M).contains(&m),
        "dense_gemv_batchm_direct: m={m} outside 2..={DENSE_GEMV_BATCHM_MAX_M}"
    );
    KernelLaunch::new(gpu, direct.by_m[(m - 2) as usize])
        .grid([div_ceil(n, 4 * direct.outputs_per_group), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-27: [`dense_gemv_batchm`] over `m` rows split into `y_blocks` block
/// rows of `ceil(m / y_blocks)` rows each (at most `DENSE_GEMV_BATCHM_MAX_M`):
/// more blocks for a narrow weight such as the MoE router, and one launch past
/// 16 rows. Each row's result is the same as in any other split.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_split(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    y_blocks: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(
        m >= 1 && (1..=m).contains(&y_blocks) && m.div_ceil(y_blocks) <= DENSE_GEMV_BATCHM_MAX_M,
        "dense_gemv_batchm_split: m={m} over y_blocks={y_blocks} must give 1..={DENSE_GEMV_BATCHM_MAX_M} rows per block row"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), y_blocks, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-25: FP8-weight GEMV for one row,
/// `C = A @ (dequant(B_fp8) * row_scale)^T`: A `[1, K]` BF16, B `[N, K]` FP8
/// E4M3, row_scale `[N]` f32, C `[1, N]` BF16. One byte per weight, against
/// two for [`dense_gemv`]. Each 256-thread block computes 4 outputs, 64
/// threads per output (kernel `dense_gemv_fp8w(A, B, row_scale, C, N, K)`).
pub fn dense_gemv_fp8w(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[cfg(test)]
#[path = "gemm_quant_gemv_tests.rs"]
mod gemm_quant_gemv_tests;
