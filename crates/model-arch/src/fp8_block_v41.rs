// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The DeepSeek-V4.1 safetensors checkpoint's FP8 projections
//! (`ResidentMat::Fp8Block`: E4M3 weights, one E8M0 scale per 32 x 32 tile) on the bf16 GEMM
//! kernels. A launch dequantizes the weight into a bf16 scratch and, when the projection
//! quantizes its input, rounds a copy of the input to E4M3 in blocks of 32 in a second
//! scratch. Both steps are exact in bf16 (an E4M3 value times a power of two has three
//! mantissa bits), so the product is the checkpoint's FP8 x FP8 linear up to the GEMM's
//! accumulation order. The resident weight stays in its stored precision.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - One projection at a time per stage: the dequantize, the input rounding and the GEMM that
//!   reads them are issued on one stream, so the next launch's writes are ordered after it.
//! - The scratch capacities are explicit ([`Fp8StageSize`]); a projection that does not fit is
//!   an error, never a partial dequantize. A zero size allocates nothing (the GGUF path).

use anyhow::{Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_layers::layers::ops::{FP8_BLOCK_TILE, ResidentMat};

/// 2026-10-19: Input-rounding blocks per launch block: one warp per 32-value block, 8 warps in
/// the 256-thread launch block (`attn_v41_act_quant_fp8`).
pub const ACT_QUANT_BLOCKS_PER_LAUNCH: u32 = 8;
/// 2026-10-19: `dequant_fp8_blockscaled_bf16`'s scale encoding for one E8M0 byte per tile.
const SCALE_KIND_E8M0: u32 = 2;

/// 2026-10-19: Scratch capacities in bf16 elements: the largest FP8 weight (`n * k`) and the
/// largest rounded input (`m * k`) a stage serves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fp8StageSize {
    pub weight_elems: usize,
    pub act_elems: usize,
}

impl Fp8StageSize {
    /// 2026-10-19: No FP8 block projections (the GGUF path): nothing is allocated.
    pub const NONE: Fp8StageSize = Fp8StageSize {
        weight_elems: 0,
        act_elems: 0,
    };
}

/// 2026-10-19: Round `n_values` bf16 values at `x` to E4M3 in place, in blocks of 32 with a
/// power-of-two scale (`attn_v41_act_quant_fp8`).
pub fn act_quant_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    x: DevicePtr,
    n_values: usize,
    stream: u64,
) -> Result<()> {
    let n_blocks = n_values / 32;
    if n_blocks == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([
            (n_blocks as u32).div_ceil(ACT_QUANT_BLOCKS_PER_LAUNCH),
            1,
            1,
        ])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_u32(n_blocks as u32)
        .launch(stream)
}

/// 2026-10-19: Dequantize the `[n, k]` E4M3 weight `w` with its `[n / 32, k / 32]` E8M0 tile
/// scales into bf16 `out` (`dequant_fp8_blockscaled_bf16`, one thread per value).
#[allow(clippy::too_many_arguments)]
pub fn dequant_fp8_block_e8m0(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w: DevicePtr,
    scale: DevicePtr,
    out: DevicePtr,
    n: usize,
    k: usize,
    stream: u64,
) -> Result<()> {
    let t = FP8_BLOCK_TILE;
    ensure!(
        n.is_multiple_of(t) && k.is_multiple_of(t),
        "FP8 block weight [{n}, {k}] is not whole {t} x {t} tiles"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([(k as u32).div_ceil(64), (n as u32).div_ceil(4), 1])
        .block([64, 4, 1])
        .arg_ptr(w)
        .arg_ptr(scale)
        .arg_ptr(out)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .arg_u32(t as u32)
        .arg_u32(t as u32)
        .arg_u32((k / t) as u32)
        .arg_u32(SCALE_KIND_E8M0)
        .launch(stream)
}

/// 2026-10-19: A stage's kernels and its two scratches.
pub struct Fp8BlockStage {
    dequant: KernelHandle,
    act_quant: KernelHandle,
    size: Fp8StageSize,
    w: DevicePtr,
    x: DevicePtr,
}

impl Fp8BlockStage {
    pub fn new(gpu: &dyn GpuBackend, size: Fp8StageSize) -> Result<Self> {
        let alloc = |elems: usize| -> Result<DevicePtr> {
            if elems == 0 {
                Ok(DevicePtr(0))
            } else {
                gpu.alloc(elems * 2)
            }
        };
        Ok(Self {
            dequant: gpu.kernel(
                "dequant_fp8_blockscaled_bf16",
                "dequant_fp8_blockscaled_bf16",
            )?,
            act_quant: gpu.kernel("attn_v41", "attn_v41_act_quant_fp8")?,
            size,
            w: alloc(size.weight_elems)?,
            x: alloc(size.act_elems)?,
        })
    }

    /// 2026-10-19: The input and weight a bf16 GEMM of `[m, k] x [n, k]^T` reads for `w`. An
    /// `Fp8Block` weight comes back as the dequantized bf16 scratch and, when it rounds its
    /// input, `a` as a rounded copy in the input scratch; any other weight passes through.
    #[allow(clippy::too_many_arguments)]
    pub fn lower(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        w: ResidentMat,
        m: usize,
        n: usize,
        k: usize,
        stream: u64,
    ) -> Result<(DevicePtr, ResidentMat)> {
        let ResidentMat::Fp8Block { w, scale, qdq } = w else {
            return Ok((a, w));
        };
        if n * k > self.size.weight_elems {
            bail!(
                "FP8 block weight [{n}, {k}] exceeds the {} bf16 weight scratch",
                self.size.weight_elems
            );
        }
        dequant_fp8_block_e8m0(gpu, self.dequant, w, scale, self.w, n, k, stream)?;
        if !qdq {
            return Ok((a, ResidentMat::Bf16(self.w)));
        }
        if m * k > self.size.act_elems {
            bail!(
                "FP8 block input [{m}, {k}] exceeds the {} bf16 input scratch",
                self.size.act_elems
            );
        }
        gpu.copy_d2d_async(a, self.x, m * k * 2, stream)?;
        act_quant_fp8(gpu, self.act_quant, self.x, m * k, stream)?;
        Ok((self.x, ResidentMat::Bf16(self.w)))
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [self.w, self.x] {
            if p.0 != 0 {
                gpu.free(p)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "fp8_block_v41_tests.rs"]
mod tests;
