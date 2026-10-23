// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: What each tensor format means, as prose and LaTeX: the one source the blueprint
//! exports carry to the store's edge modal. Exhaustive over [`Format`] and [`Scale`], so a new
//! format cannot be added without its semantics.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - `math` states how a stored value decodes to a real number; `encode` how a wider value (FP32,
//!   or BF16 widened exactly) is quantized into the format, with the scale rule the engine's
//!   quantizers use (FP8: $s = \max(\max|x|/448, 10^{-12})$; NVFP4: global $\alpha = \max|x|/(6
//!   \cdot 448)$, per-group E4M3 scales of the group maximum over $6\alpha$). `None` for a format
//!   nothing in the engine quantizes into (`f32`, `i32`, and `mxfp4`, read from checkpoints only).
//! - The same KaTeX rules as a family's explanation (`venn::families::explain`) hold; the
//!   `kernel_math` test renders every entry in strict mode.

use crate::prelude::*;

use crate::format::{Format, Scale};

/// 2026-10-22: A format's explanation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct FormatExplain {
    /// 2026-10-22: Its name in words.
    pub title: String,
    /// 2026-10-22: Bit layout, scale granularity and where it is used.
    pub description: String,
    /// 2026-10-22: Display equations: the decode.
    pub math: Vec<String>,
    /// 2026-10-22: The quantization from a wider value, as one display equation.
    pub encode: Option<String>,
}

const E4M3: &str = r"q = (-1)^{s}\,2^{e-7}\left(1+\tfrac{m}{8}\right),\quad |q| \le 448";
const E2M1: &str = r"q \in \{0,\ \pm 0.5,\ \pm 1,\ \pm 1.5,\ \pm 2,\ \pm 3,\ \pm 4,\ \pm 6\}";

fn scale_words(s: Scale) -> String {
    match s {
        Scale::PerTensor => "one FP32 scale for the whole tensor".into(),
        Scale::PerToken => "one FP32 scale per row (token)".into(),
        Scale::PerChannel => "one FP32 scale per output channel".into(),
        Scale::Group(g) => format!("one FP32 scale per {g} consecutive values of a row"),
        Scale::Block(r, c) => format!("one FP32 scale per {r} x {c} block"),
    }
}

/// 2026-10-22: The scale that applies to element $(r, j)$ of a row-major `[rows, dim]` tensor.
fn scale_index(s: Scale) -> String {
    match s {
        Scale::PerTensor => "s".into(),
        Scale::PerToken => "s_{r}".into(),
        Scale::PerChannel => "s_{j}".into(),
        Scale::Group(g) => format!(r"s_{{r,\lfloor j/{g} \rfloor}}"),
        Scale::Block(r, c) => format!(r"s_{{\lfloor r/{r} \rfloor,\lfloor j/{c} \rfloor}}"),
    }
}

/// 2026-10-22: The set an FP8 scale is the maximum over.
fn scale_set(s: Scale) -> String {
    match s {
        Scale::PerTensor => r"\max_{r,j}|x_{rj}|".into(),
        Scale::PerToken => r"\max_{j}|x_{rj}|".into(),
        Scale::PerChannel => r"\max_{r}|x_{rj}|".into(),
        Scale::Group(g) => {
            format!(r"\max_{{j' \in \text{{group of }} j,\ |\text{{group}}| = {g}}}|x_{{rj'}}|")
        }
        Scale::Block(r, c) => format!(r"\max_{{\text{{the {r} x {c} block of }} (r,j)}}|x|"),
    }
}

impl Format {
    /// 2026-10-22: The format's semantics.
    pub fn explain(&self) -> FormatExplain {
        match *self {
            Format::Bf16 => FormatExplain {
                title: "BF16 (bfloat16)".into(),
                description: "16 bits: 1 sign, 8 exponent and 7 mantissa bits. The exponent range of FP32 with 8 significant bits; the engine's activation and residual format.".into(),
                math: vec![r"x = (-1)^{s}\,2^{e-127}\left(1+\tfrac{m}{2^{7}}\right),\quad e \in [1, 254],\ m \in [0, 127]".into()],
                encode: Some(r"x_{\mathrm{bf16}} = \mathrm{rn}_{\mathrm{bf16}}(x),\ \text{round to nearest, ties to even, of the top 16 bits of the FP32 value}".into()),
            },
            Format::F32 => FormatExplain {
                title: "FP32 (IEEE 754 binary32)".into(),
                description: "32 bits: 1 sign, 8 exponent and 23 mantissa bits. Accumulators, gate values, recurrent state and logits that must not lose precision between kernels.".into(),
                math: vec![r"x = (-1)^{s}\,2^{e-127}\left(1+\tfrac{m}{2^{23}}\right)".into()],
                encode: None,
            },
            Format::I32 => FormatExplain {
                title: "INT32".into(),
                description: "32-bit signed integers: token ids, expert ids and selected indices, never arithmetic values.".into(),
                math: vec![r"t \in \{0, 1, \dots, 2^{31}-1\}".into()],
                encode: None,
            },
            Format::Fp8E4m3 { scale } => FormatExplain {
                title: format!("FP8 E4M3, {}", scale_words(scale)),
                description: format!(
                    "8 bits per value: 1 sign, 4 exponent (bias 7) and 3 mantissa bits, finite only (largest magnitude 448), with {}. A value is the stored code times its scale.",
                    scale_words(scale)
                ),
                math: vec![E4M3.into(), format!(r"\hat{{x}}_{{rj}} = {}\,q_{{rj}}", scale_index(scale))],
                encode: Some(format!(
                    r"{} = \max\!\left(\frac{{{}}}{{448}},\,10^{{-12}}\right),\qquad q_{{rj}} = \mathrm{{sat}}_{{\mathrm{{e4m3}}}}\!\left(\frac{{x_{{rj}}}}{{{}}}\right)",
                    scale_index(scale),
                    scale_set(scale),
                    scale_index(scale)
                )),
            },
            Format::Nvfp4 { group } => FormatExplain {
                title: format!("NVFP4, E2M1 with an E4M3 scale per {group}"),
                description: format!(
                    "4 bits per value (E2M1: 1 sign, 2 exponent, 1 mantissa bit), packed two per byte, one E4M3 scale byte per {group} consecutive values and one FP32 global scale (per tensor for a weight, per row for an activation)."
                ),
                math: vec![
                    E2M1.into(),
                    format!(r"\hat{{x}}_{{j}} = \alpha\; s_{{\lfloor j/{group} \rfloor}}\; q_{{j}},\qquad s_{{b}} \in \mathrm{{E4M3}},\ \alpha \in \mathrm{{FP32}}"),
                ],
                encode: Some(format!(
                    r"\alpha = \frac{{\max_{{j}}|x_{{j}}|}}{{6 \cdot 448}},\qquad s_{{b}} = \mathrm{{sat}}_{{\mathrm{{e4m3}}}}\!\left(\frac{{\max_{{j \in b}}|x_{{j}}|}}{{6\,\alpha}}\right),\qquad q_{{j}} = \mathrm{{rn}}_{{\mathrm{{e2m1}}}}\!\left(\frac{{x_{{j}}}}{{\alpha\, s_{{\lfloor j/{group} \rfloor}}}}\right)"
                )),
            },
            Format::Mxfp4 { group } => FormatExplain {
                title: format!("MXFP4, E2M1 with an E8M0 scale per {group}"),
                description: format!(
                    "OCP Microscaling MXFP4: 4-bit E2M1 values packed two per byte and one E8M0 power-of-two scale byte per {group} consecutive values; no global scale. The engine reads checkpoint-quantized MXFP4 weights and never quantizes into it."
                ),
                math: vec![
                    E2M1.into(),
                    format!(r"\hat{{x}}_{{j}} = 2^{{\,e_{{\lfloor j/{group} \rfloor}} - 127}}\; q_{{j}},\qquad e_{{b}} \in \{{1, \dots, 254\}}"),
                ],
                encode: None,
            },
        }
    }
}

#[cfg(test)]
#[path = "format_explain_tests.rs"]
mod format_explain_tests;
