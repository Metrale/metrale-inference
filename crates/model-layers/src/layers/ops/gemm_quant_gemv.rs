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

/// 2026-10-14: `#define BATCHM_WIDE_MIN` in `kernels/gb10/common/dense_gemv_bf16_batchm.cu`:
/// the narrowest M the register-resident `dense_gemv_bf16_batchm_wide` entries take.
pub const DENSE_GEMV_BATCHM_WIDE_MIN: u32 = 9;

/// 2026-10-14: The batched BF16 GEMV pair: `narrow` is `dense_gemv_bf16_batchm` (runtime M,
/// `1..=DENSE_GEMV_BATCHM_MAX_M`), `wide` is `dense_gemv_bf16_batchm_wide` (accumulators in
/// registers, `DENSE_GEMV_BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M`; `KernelHandle(0)` when the
/// target lacks it). Both take [`dense_gemv_batchm`]'s arguments and grid, and a wide row has the
/// narrow row's bits (`examples/glm5next_c16_kernel_bench.rs` checks it byte for byte); the wide
/// entry is the faster one at 9..=16 rows (GB10, 51627 x 4096 BF16 at M = 16: 1.95 ms vs 3.04 ms).
#[derive(Clone, Copy, Debug)]
pub struct DenseGemvBatchmPair {
    pub narrow: KernelHandle,
    pub wide: KernelHandle,
}

impl DenseGemvBatchmPair {
    /// 2026-10-14: The entry a launch of `m` rows takes: `wide` inside its band when loaded,
    /// `narrow` otherwise.
    pub fn for_rows(self, m: u32) -> KernelHandle {
        if self.wide.0 != 0 && (DENSE_GEMV_BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M).contains(&m) {
            self.wide
        } else {
            self.narrow
        }
    }
}

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
mod tests {
    use super::*;

    /// 2026-10-14: The wide entry takes exactly 9..=16 rows and only when loaded; every other
    /// width, and a target without it, keeps the runtime-M entry.
    #[test]
    fn batchm_pair_takes_wide_only_inside_its_band() {
        let pair = DenseGemvBatchmPair {
            narrow: KernelHandle(1),
            wide: KernelHandle(2),
        };
        for m in 1..=8 {
            assert_eq!(pair.for_rows(m).0, 1, "m={m}");
        }
        for m in 9..=16 {
            assert_eq!(pair.for_rows(m).0, 2, "m={m}");
        }
        assert_eq!(pair.for_rows(17).0, 1);
        let no_wide = DenseGemvBatchmPair {
            wide: KernelHandle(0),
            ..pair
        };
        for m in 1..=17 {
            assert_eq!(no_wide.for_rows(m).0, 1, "m={m}");
        }
    }
}
