// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Which kernel family each GLM-5.3 MLP group runs at each row count: the routed
//! experts and the dense MLP of the `first_k_dense_replace` layers, under
//! `--weight-quantization` (the per-module stamp, [`Nvfp4Act`]) and `--activation-quantization`
//! (the per-family ladder: `moe` for the routed experts, `ffn` for the dense MLP).
//!
//! The checkpoint declares W4A4 NVFP4 with static activation scales for both groups, so the
//! default (`declared` / `declared`) runs W4A4 ([`MlpKernel::W4a4Static`]). The shared expert,
//! the router and the MTP layer are declared 16-bit and never reach this choice.
//!
//! | format at the row count | stamp `A4`                         | stamp `Wide` / unstamped |
//! |-------------------------|------------------------------------|--------------------------|
//! | `declared`              | W4A4 up to the W4A4 row cap, then the 16-bit-activation path | 16-bit-activation path |
//! | `nvfp4`                 | W4A4 up to the cap, then 16-bit (refused without kernels or scales) | same |
//! | `bf16`, `adaptive`      | 16-bit-activation path             | 16-bit-activation path   |
//! | `fp8`                   | refused: no FP8-activation MLP kernels for this model         ||
//!
//! The 16-bit-activation path is the one the model ran before this module: W4A16 for the routed
//! experts, and BF16 weights dequantized at load for the dense MLP. Where it serves a row count
//! the checkpoint declares W4A4 for, that row count runs ABOVE the declared precision, and
//! [`Glm5NextMlpPrecision::describe`] says so in the load log.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Pure: no environment, no I/O; the loader passes in the published flags and the kernels
//!   that resolved.
//! - Every row count from 1 has exactly one kernel ([`GroupPrecision::kernel`] is total).

use anyhow::{Result, bail};
use metrale_config::activation_quantization::Ladder;
use metrale_config::{ActQuantFormat, Nvfp4Act};

/// 2026-10-08: One of the two GLM MLP groups whose precision this module decides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpGroup {
    /// 2026-10-08: The routed experts (`mlp.experts.*`), family `moe`.
    RoutedExperts,
    /// 2026-10-08: The dense MLP of layers `0..first_k_dense_replace`, family `ffn`.
    DenseMlp,
}

impl MlpGroup {
    fn name(self) -> &'static str {
        match self {
            Self::RoutedExperts => "routed experts",
            Self::DenseMlp => "dense MLP",
        }
    }

    /// 2026-10-08: The 16-bit-activation kernel of the group (the pre-W4A4 path).
    fn wide_kernel(self) -> MlpKernel {
        match self {
            Self::RoutedExperts => MlpKernel::W4a16,
            Self::DenseMlp => MlpKernel::Bf16,
        }
    }
}

/// 2026-10-08: The kernel family one launch of a group runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpKernel {
    /// 2026-10-08: NVFP4 weights and NVFP4 activations quantized under the checkpoint's static
    /// `input_scale`, on the FP4 block-scale MMA (`w4a4_gemv_mx*`).
    W4a4Static,
    /// 2026-10-08: NVFP4 weights, BF16 activations (the routed experts' pre-W4A4 kernels).
    W4a16,
    /// 2026-10-08: BF16 weights dequantized at load, BF16 activations (the dense MLP's
    /// pre-W4A4 kernels).
    Bf16,
}

/// 2026-10-08: One group's kernel per row count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GroupPrecision {
    group: MlpGroup,
    ladder: Ladder,
    stamp: Nvfp4Act,
    /// 2026-10-08: Widest launch the W4A4 kernels serve for this group; 0 when they did not
    /// resolve on this target.
    w4a4_max_rows: usize,
}

impl GroupPrecision {
    /// 2026-10-08: The group's plan. `ladder` is the family's `--activation-quantization`
    /// ladder, `stamp` what the weight policy answers for the group's modules, `w4a4_max_rows`
    /// the W4A4 kernels' row cap on this target (0: absent), and `has_scales` whether the
    /// checkpoint gave every projection of the group a static activation scale.
    ///
    /// Errors when a rung asks for `fp8`, or for `nvfp4` while the W4A4 kernels or the scales
    /// are absent; and when a `declared` rung resolves to W4A4 but the scales are absent (a
    /// checkpoint that declares static FP4 activations and ships no scales is malformed).
    pub fn resolve(
        group: MlpGroup,
        ladder: Ladder,
        stamp: Nvfp4Act,
        w4a4_max_rows: usize,
        has_scales: bool,
    ) -> Result<Self> {
        for rung in ladder.rungs() {
            match rung.format {
                ActQuantFormat::Fp8 => bail!(
                    "GLM {}: --activation-quantization fp8 has no FP8-activation kernel on this \
                     model; use declared, nvfp4, bf16 or adaptive",
                    group.name()
                ),
                ActQuantFormat::Nvfp4 if w4a4_max_rows == 0 => bail!(
                    "GLM {}: --activation-quantization nvfp4, but this target has no W4A4 \
                     kernels (w4a4_gemv_mx)",
                    group.name()
                ),
                ActQuantFormat::Nvfp4 if !has_scales => bail!(
                    "GLM {}: --activation-quantization nvfp4 needs the checkpoint's static \
                     activation scales (input_scale), and it has none",
                    group.name()
                ),
                ActQuantFormat::Declared
                    if stamp == Nvfp4Act::A4 && w4a4_max_rows > 0 && !has_scales =>
                {
                    bail!(
                        "GLM {}: the checkpoint declares static FP4 activations but ships no \
                         input_scale for them",
                        group.name()
                    )
                }
                _ => {}
            }
        }
        Ok(Self {
            group,
            ladder,
            stamp,
            w4a4_max_rows,
        })
    }

    /// 2026-10-08: The kernel a launch of `rows` rows (at least 1) runs.
    pub fn kernel(&self, rows: usize) -> MlpKernel {
        let r = u32::try_from(rows.max(1)).unwrap_or(u32::MAX);
        let w4a4 = match self.ladder.format(r) {
            ActQuantFormat::Declared => self.stamp == Nvfp4Act::A4,
            ActQuantFormat::Nvfp4 => true,
            ActQuantFormat::Bf16 | ActQuantFormat::Adaptive | ActQuantFormat::Fp8 => false,
        };
        if w4a4 && rows.max(1) <= self.w4a4_max_rows {
            MlpKernel::W4a4Static
        } else {
            self.group.wide_kernel()
        }
    }

    /// 2026-10-08: Whether some row count in `1..=max_rows` runs `kernel`; the loader keeps the
    /// weight forms only the reachable kernels need.
    pub fn reaches(&self, kernel: MlpKernel, max_rows: usize) -> bool {
        (1..=max_rows.max(1)).any(|r| self.kernel(r) == kernel)
    }

    /// 2026-10-08: The row ranges in `1..=max_rows` and their kernels, each marked
    /// `ABOVE declared` where the checkpoint declares W4A4 and the range runs 16-bit
    /// activations. For the load log.
    pub fn describe(&self, max_rows: usize) -> String {
        let mut out = Vec::new();
        let mut start = 1usize;
        let max_rows = max_rows.max(1);
        for r in 1..=max_rows {
            let k = self.kernel(r);
            if r == max_rows || self.kernel(r + 1) != k {
                let above = self.stamp == Nvfp4Act::A4 && k != MlpKernel::W4a4Static;
                out.push(format!(
                    "rows {start}-{r}: {k:?}{}",
                    if above { " (ABOVE declared W4A4)" } else { "" }
                ));
                start = r + 1;
            }
        }
        format!("GLM {} [{}]", self.group.name(), out.join(", "))
    }
}

/// 2026-10-08: Both groups' plans for one model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Glm5NextMlpPrecision {
    pub experts: GroupPrecision,
    pub dense: GroupPrecision,
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod precision_tests;
