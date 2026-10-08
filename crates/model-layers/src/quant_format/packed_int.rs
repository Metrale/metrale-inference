// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Host-side layout of compressed-tensors `pack-quantized` integer weights
//! (INT4 / INT8, symmetric, group 128) and the CPU reference the HIP W4A16 / W8A16 kernels
//! are checked against.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - The byte layout is the one `metrale_config::precision_plan::packed_int` documents and
//!   admits: I32 words little-endian, codes least-significant first, offset binary
//!   (`q = u - 2^(bits-1)`), one scale per 128 K values, no zero points.
//! - [`PackedIntMatrix::new`] refuses a tensor pair whose dtypes or shapes disagree with
//!   that layout, so a decode never reads past a buffer or applies a scale to the wrong group.
//! - The reference accumulates in f64 and dequantizes `q * scale` with the scale widened to
//!   f32 first, the products the kernels form in registers. It shares no code with the kernels.
//!
//! Usage:
//! ```
//! use metrale_config::precision_plan::packed_int::PackedIntScheme;
//! use metrale_model_layers::quant_format::packed_int::PackedIntMatrix;
//! // One INT4 row of K = 128: codes 0..=15 (q = -8..=7) repeated, scale 0.5.
//! let word = |i: u32| (0..8).fold(0u32, |w, j| w | (((8 * i + j) % 16) << (4 * j)));
//! let packed: Vec<u32> = (0..16).map(|i| word(i % 2)).collect();
//! let m = PackedIntMatrix::new(PackedIntScheme::INT4_G128, 1, 128, &packed, &[0.5]).unwrap();
//! assert_eq!(m.weight(0, 0), -4.0);
//! assert_eq!(m.weight(0, 15), 3.5);
//! ```

use anyhow::{Result, ensure};
use metrale_config::precision_plan::packed_int::PackedIntScheme;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-10-07: Module of the strix-hip packed-int GEMVs
/// (kernels/strix-hip/laguna-xs-2.1/int4/packed_int_gemv.cu).
pub const PACKED_INT_GEMV_MODULE: &str = "packed_int_gemv";

/// 2026-10-07: The packed-int GEMV pair for one scheme: the dense decode GEMV
/// (`y[M, N] = x[M, K] W^T`) and the grouped-expert GEMV over a per-expert pointer table.
/// Both are required: a target without them cannot serve packed-int weights, and the
/// lookup error says which entry point is missing. No dispatch calls this yet; it pins the
/// names the strix-hip target compiles (crates/kernels/tests/strix_hip_laguna_int4.rs).
pub fn packed_int_gemv_kernels(
    gpu: &dyn GpuBackend,
    scheme: PackedIntScheme,
) -> Result<(KernelHandle, KernelHandle)> {
    ensure!(
        scheme.group_size == metrale_config::precision_plan::packed_int::PACKED_INT_GROUP_SIZE,
        "no packed-int GEMV for group size {}",
        scheme.group_size
    );
    match scheme.bits {
        4 => Ok((
            gpu.kernel(PACKED_INT_GEMV_MODULE, "packed_int4_gemv_g128")?,
            gpu.kernel(PACKED_INT_GEMV_MODULE, "moe_packed_int4_gemv_ptrtable_g128")?,
        )),
        8 => Ok((
            gpu.kernel(PACKED_INT_GEMV_MODULE, "packed_int8_gemv_g128")?,
            gpu.kernel(PACKED_INT_GEMV_MODULE, "moe_packed_int8_gemv_ptrtable_g128")?,
        )),
        bits => anyhow::bail!("no packed-int GEMV for INT{bits}"),
    }
}

/// 2026-10-07: The stored code of K index `k` in a row of packed words, as a signed value.
pub fn decode_code(scheme: PackedIntScheme, row_words: &[u32], k: usize) -> i32 {
    let per = scheme.codes_per_word();
    let bits = scheme.bits as usize;
    let mask = (1u32 << bits) - 1;
    let unsigned = (row_words[k / per] >> (bits * (k % per))) & mask;
    unsigned as i32 - scheme.code_offset()
}

/// 2026-10-07: Check one `weight_packed` / `weight_scale` tensor pair against the layout of
/// an `[n, k]` weight: I32 `[n, k / codes_per_word]` and a 16-bit float (BF16 or F16) or
/// F32 scale `[n, k / group_size]`. `k` must be a whole number of groups.
pub fn check_tensor_pair(
    scheme: PackedIntScheme,
    n: usize,
    k: usize,
    packed: (&str, &[usize]),
    scale: (&str, &[usize]),
) -> Result<()> {
    let g = scheme.group_size as usize;
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(g),
        "packed int weight [{n}, {k}]: K must be a positive multiple of the group size {g}"
    );
    ensure!(
        packed.0 == "I32",
        "weight_packed dtype {} (expected I32)",
        packed.0
    );
    let words = k / scheme.codes_per_word();
    ensure!(
        packed.1 == [n, words],
        "weight_packed shape {:?} (expected [{n}, {words}] for INT{} [{n}, {k}])",
        packed.1,
        scheme.bits
    );
    ensure!(
        matches!(scale.0, "BF16" | "F16" | "F32"),
        "weight_scale dtype {} (expected BF16, F16 or F32)",
        scale.0
    );
    ensure!(
        scale.1 == [n, k / g],
        "weight_scale shape {:?} (expected [{n}, {}])",
        scale.1,
        k / g
    );
    Ok(())
}

/// 2026-10-07: A borrowed packed-int `[n, k]` weight with its scales already widened to f32.
#[derive(Debug, Clone, Copy)]
pub struct PackedIntMatrix<'a> {
    scheme: PackedIntScheme,
    n: usize,
    k: usize,
    words: &'a [u32],
    scales: &'a [f32],
}

impl<'a> PackedIntMatrix<'a> {
    /// 2026-10-07: `words` row-major `[n, k / codes_per_word]`, `scales` row-major
    /// `[n, k / group_size]`. Lengths that disagree with `[n, k]` are refused.
    pub fn new(
        scheme: PackedIntScheme,
        n: usize,
        k: usize,
        words: &'a [u32],
        scales: &'a [f32],
    ) -> Result<Self> {
        let g = scheme.group_size as usize;
        ensure!(
            n > 0 && k > 0 && k.is_multiple_of(g),
            "packed int weight [{n}, {k}]: K must be a positive multiple of {g}"
        );
        ensure!(
            words.len() == n * (k / scheme.codes_per_word()),
            "packed words {} != {n} x {}",
            words.len(),
            k / scheme.codes_per_word()
        );
        ensure!(
            scales.len() == n * (k / g),
            "scales {} != {n} x {}",
            scales.len(),
            k / g
        );
        Ok(Self {
            scheme,
            n,
            k,
            words,
            scales,
        })
    }

    /// 2026-10-07: Output rows.
    pub fn rows(&self) -> usize {
        self.n
    }

    /// 2026-10-07: Input columns.
    pub fn cols(&self) -> usize {
        self.k
    }

    /// 2026-10-07: The signed code at `(row, k)`.
    pub fn code(&self, row: usize, k: usize) -> i32 {
        let per_row = self.k / self.scheme.codes_per_word();
        decode_code(
            self.scheme,
            &self.words[row * per_row..(row + 1) * per_row],
            k,
        )
    }

    /// 2026-10-07: The dequantized weight at `(row, k)`: `code * scale`, in f32.
    pub fn weight(&self, row: usize, k: usize) -> f32 {
        let g = self.scheme.group_size as usize;
        let scale = self.scales[row * (self.k / g) + k / g];
        self.code(row, k) as f32 * scale
    }

    /// 2026-10-07: `y = W x` for one activation row, f64 accumulation.
    pub fn gemv(&self, x: &[f32]) -> Vec<f32> {
        assert_eq!(x.len(), self.k, "activation length");
        (0..self.n)
            .map(|row| {
                (0..self.k)
                    .map(|k| self.weight(row, k) as f64 * x[k] as f64)
                    .sum::<f64>() as f32
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "packed_int_tests.rs"]
mod tests;
