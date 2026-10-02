// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The configurations of `nvfp4_moe_grouped_microtest` (`Leg`), their row envelopes,
//! and the tensor-core leg's SiLU-row reader.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use half::bf16;

use super::fixture::{HostDot, bf16_to_f64};

/// 2026-09-27: The configurations `forward_nvfp4_grouped_decode` launches. 2026-10-02:
/// `AllNvfp4Tc` is `AllNvfp4` on the tensor-core expert kernels (`moe_nvfp4_grouped_tc.cu`),
/// the declared NVFP4 checkpoint's path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Leg {
    AllNvfp4,
    Nvfp4,
    Nvfp4GateUp,
    AllNvfp4Tc,
    /// 2026-10-02: `AllNvfp4Tc` on MMA-paired weights (`_tc_r`; the weights permuted first).
    AllNvfp4TcPaired,
}

impl Leg {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::AllNvfp4 => "all-nvfp4",
            Self::Nvfp4 => "nvfp4",
            Self::Nvfp4GateUp => "nvfp4-gate-up",
            Self::AllNvfp4Tc => "all-nvfp4-tc",
            Self::AllNvfp4TcPaired => "all-nvfp4-tc-r",
        }
    }
    pub(crate) fn fp8_shared(self) -> bool {
        matches!(self, Self::Nvfp4 | Self::Nvfp4GateUp)
    }
    pub(crate) fn fp8_down(self) -> bool {
        self == Self::Nvfp4GateUp
    }
    pub(crate) fn tc(self) -> bool {
        matches!(self, Self::AllNvfp4Tc | Self::AllNvfp4TcPaired)
    }
    /// 2026-10-02: The widest row count production admits on this leg's kernels.
    pub(crate) fn max_m(self) -> usize {
        if self.tc() {
            ops_rows::TC_MAX
        } else {
            ops_rows::SCALAR_MAX
        }
    }
}

/// 2026-10-02: The row envelopes of `forward_nvfp4_grouped_decode`.
pub(crate) mod ops_rows {
    pub(crate) const SCALAR_MAX: usize =
        metrale_model_layers::layers::moe::NVFP4_GROUPED_DECODE_MAX_ROWS;
    pub(crate) const TC_MAX: usize =
        metrale_model_layers::layers::moe::NVFP4_GROUPED_DECODE_TC_MAX_ROWS;
}

/// 2026-10-02: The tensor-core kernels' SiLU rows: per row, `n` BF16 hi terms then `n` BF16 lo
/// terms in the bytes of `n` FP32 values; each value is hi + lo.
pub(crate) fn hi_lo_rows(b: &[u8], n: usize) -> Vec<f64> {
    let v = bf16_to_f64(b);
    v.chunks_exact(2 * n)
        .flat_map(|r| (0..n).map(move |i| r[i] + r[n + i]))
        .collect()
}

/// 2026-09-27: The SiLU product the gate+up kernels write, from the f64 projections rounded
/// to BF16 as they round them.
pub(crate) fn silu_product(
    gate: &dyn HostDot,
    up: &dyn HostDot,
    x: &[f64],
    inter: usize,
) -> Vec<f64> {
    (0..inter)
        .map(|c| {
            let gv = bf16::from_f64(gate.dot(c, x)).to_f64();
            let uv = bf16::from_f64(up.dot(c, x)).to_f64();
            gv / (1.0 + (-gv).exp()) * uv
        })
        .collect()
}

/// 2026-09-27: `got` within 2 % of the largest |want| of its row, element by element.
pub(crate) fn close(got: &[f64], want: &[f64], what: &str) -> Result<()> {
    let scale = want.iter().fold(1e-30f64, |a, v| a.max(v.abs()));
    for (i, (x, y)) in got.iter().zip(want).enumerate() {
        ensure!(
            x.is_finite() && (x - y).abs() <= 0.02 * scale,
            "{what}: element {i} is {x}, the reference {y} (row max {scale})"
        );
    }
    Ok(())
}
