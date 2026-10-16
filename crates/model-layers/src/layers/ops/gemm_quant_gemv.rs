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

/// 2026-10-16: `#define BATCHM_WIDE_MIN` in `kernels/gb10/common/dense_gemv_bf16_batchm.cu`:
/// the narrowest M the register-resident `dense_gemv_bf16_batchm_wide` entries take.
pub const DENSE_GEMV_BATCHM_WIDE_MIN: u32 = 9;

/// 2026-10-16: `#define BATCHM_REG8_MAX` in the same file: the widest M the
/// register-resident `dense_gemv_bf16_batchm_reg8` entries take.
pub const DENSE_GEMV_BATCHM_REG8_MAX: u32 = 8;

/// 2026-10-16: The batched BF16 GEMV entries of one output type. `narrow` is
/// `dense_gemv_bf16_batchm` (runtime M, `1..=DENSE_GEMV_BATCHM_MAX_M`, accumulators in local
/// memory). `reg8` is `dense_gemv_bf16_batchm_reg8` (one register-resident instantiation per
/// M in `1..=DENSE_GEMV_BATCHM_REG8_MAX`) and `wide` is `dense_gemv_bf16_batchm_wide` (the same
/// for `DENSE_GEMV_BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M`); either is `KernelHandle(0)`
/// when its lever is off or the target lacks it. All three take [`dense_gemv_batchm`]'s
/// arguments and grid (never [`dense_gemv_batchm_split`]'s: the register entries ignore
/// `gridDim.y`), and a register row has the narrow row's bits
/// (`examples/glm5next_batchm_reg_gate.rs` checks it byte for byte). GB10, 51627 x 4096 at
/// M = 16: wide 1.95 ms vs narrow 3.04 ms (2026-10-14).
#[derive(Clone, Copy, Debug)]
pub struct DenseGemvBatchmPair {
    pub narrow: KernelHandle,
    pub reg8: KernelHandle,
    pub wide: KernelHandle,
}

impl DenseGemvBatchmPair {
    /// 2026-10-16: The runtime-M entry alone (every row keeps it).
    pub fn narrow_only(narrow: KernelHandle) -> Self {
        Self {
            narrow,
            reg8: KernelHandle(0),
            wide: KernelHandle(0),
        }
    }

    /// 2026-10-16: The entry a launch of `m` rows takes: `reg8` at
    /// `2..=DENSE_GEMV_BATCHM_REG8_MAX`, `wide` at
    /// `DENSE_GEMV_BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M`, each only when loaded, else
    /// `narrow`. One row keeps `narrow` (callers send it to the M = 1 GEMV).
    pub fn for_rows(self, m: u32) -> KernelHandle {
        if self.reg8.0 != 0 && (2..=DENSE_GEMV_BATCHM_REG8_MAX).contains(&m) {
            self.reg8
        } else if self.wide.0 != 0
            && (DENSE_GEMV_BATCHM_WIDE_MIN..=DENSE_GEMV_BATCHM_MAX_M).contains(&m)
        {
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

/// 2026-10-16: The widest row count `METRALE_DENSE_BATCHM_32` adds to the batched BF16 GEMV's
/// band ([`dense_batchm_band`]); those rows run as launches of at most
/// `DENSE_GEMV_BATCHM_MAX_M` ([`dense_gemv_batchm_rows`]).
pub const DENSE_GEMV_BATCHM_32_MAX_M: u32 = 32;

/// 2026-10-16: `METRALE_DENSE_BATCHM_32`: only `1` turns it on.
pub fn dense_batchm_32_from(value: Option<&str>) -> bool {
    value == Some("1")
}

/// 2026-10-16: `METRALE_DENSE_BATCHM_32=1`: launches of 17..=32 rows of the GLM-5.3 BF16
/// projections (`glm_mm`) and of the BF16 head run on the batched GEMV in 16-row launches, so
/// every row keeps the bits it has at 1..=16 rows (a 24/32-row MTP verify otherwise takes
/// cuBLASLt or the tile GEMM). Read once, so the route is the same at CUDA-graph capture and
/// replay.
pub fn dense_batchm_32() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        dense_batchm_32_from(std::env::var("METRALE_DENSE_BATCHM_32").ok().as_deref())
    })
}

/// 2026-10-16: Whether `m` rows take the batched BF16 GEMV: `1..=band`, and with `rows32` also
/// `DENSE_GEMV_BATCHM_MAX_M + 1..=DENSE_GEMV_BATCHM_32_MAX_M`.
pub fn dense_batchm_band(band: u32, rows32: bool, m: u32) -> bool {
    (1..=band).contains(&m)
        || (rows32 && (DENSE_GEMV_BATCHM_MAX_M + 1..=DENSE_GEMV_BATCHM_32_MAX_M).contains(&m))
}

/// 2026-10-16: The pass a dense BF16 projection launch belongs to, named by its caller. Only
/// the batched decode and verify passes (`Decode`, `Verify`) may take the 17..=32-row band of
/// `METRALE_DENSE_BATCHM_32`; a prefill launch keeps its route at every row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenseCallSite {
    /// A prefill sub-chunk or piece.
    Prefill,
    /// The batched decode body: one or more rows per sequence of a decode step.
    Decode,
    /// The batched speculative verify body.
    Verify,
    /// Any other launch (single-sequence k-row forwards, drafter and MTP context rows, one-row
    /// helpers): routed as before `METRALE_DENSE_BATCHM_32`.
    General,
}

/// 2026-10-16: Whether `METRALE_DENSE_BATCHM_32` (`on`) applies at `site`: only at
/// [`DenseCallSite::Decode`] and [`DenseCallSite::Verify`], so a prefill launch of 17..=32 rows
/// never leaves its own route. Callers combine it with [`dense_batchm_band`] for the row count.
pub fn dense_batchm_32_at(on: bool, site: DenseCallSite) -> bool {
    match site {
        DenseCallSite::Decode | DenseCallSite::Verify => on,
        DenseCallSite::Prefill | DenseCallSite::General => false,
    }
}

/// 2026-10-16: [`dense_gemv_batchm`] over `m >= 1` rows in launches of at most
/// `DENSE_GEMV_BATCHM_MAX_M` rows, each on `pair.for_rows(rows)`: input rows `K` elements
/// apart, output rows `out_stride` BF16 elements apart. Each row has the bits it has in a launch
/// of any other width.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_rows(
    gpu: &dyn GpuBackend,
    pair: DenseGemvBatchmPair,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    ensure!(m >= 1, "dense_gemv_batchm_rows: m = 0");
    let mut done = 0u32;
    while done < m {
        let rows = (m - done).min(DENSE_GEMV_BATCHM_MAX_M);
        dense_gemv_batchm(
            gpu,
            pair.for_rows(rows),
            input.offset(done as usize * k as usize * 2),
            weight,
            output.offset(done as usize * out_stride as usize * 2),
            rows,
            n,
            k,
            out_stride,
            stream,
        )?;
        done += rows;
    }
    Ok(())
}

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

    const N: KernelHandle = KernelHandle(1);
    const R8: KernelHandle = KernelHandle(2);
    const W: KernelHandle = KernelHandle(3);

    fn picks(p: DenseGemvBatchmPair) -> Vec<u64> {
        (1..=17).map(|m| p.for_rows(m).0).collect()
    }

    fn band(lo: usize, hi: usize, h: u64) -> Vec<u64> {
        (1..=17)
            .map(|m| if (lo..=hi).contains(&m) { h } else { 1 })
            .collect()
    }

    /// 2026-10-16: Each loaded entry takes exactly its band; one row and 17 rows never take a
    /// register entry; a missing entry leaves its band on the runtime-M entry.
    #[test]
    fn batchm_pair_band_follows_the_loaded_entries() {
        assert_eq!(picks(DenseGemvBatchmPair::narrow_only(N)), vec![1u64; 17]);
        let reg8 = DenseGemvBatchmPair {
            reg8: R8,
            ..DenseGemvBatchmPair::narrow_only(N)
        };
        assert_eq!(picks(reg8), band(2, 8, 2));
        let wide = DenseGemvBatchmPair {
            wide: W,
            ..DenseGemvBatchmPair::narrow_only(N)
        };
        assert_eq!(picks(wide), band(9, 16, 3));
        let both = DenseGemvBatchmPair {
            narrow: N,
            reg8: R8,
            wide: W,
        };
        let mut want = band(2, 8, 2);
        want[8..16].fill(3);
        assert_eq!(picks(both), want);
        assert_eq!(both.for_rows(1).0, N.0);
        assert_eq!(both.for_rows(17).0, N.0);
        assert_eq!(both.for_rows(0).0, N.0);
        assert_eq!(DENSE_GEMV_BATCHM_REG8_MAX + 1, DENSE_GEMV_BATCHM_WIDE_MIN);
    }
}
