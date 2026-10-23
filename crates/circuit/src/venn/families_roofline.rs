// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `[roofline]` table of KERNEL_FAMILIES.toml: the time constants (required) and
//! the energy constants (DESIGN-kernel-energy.md §3.3, optional, all-or-none) validated into
//! [`Roofline`] and [`EnergyRoofline`].
//!
//! Owner: metrale-circuit (venn).
//! Invariants: every value present is finite and positive; the energy group is
//! `dram_pj_per_byte`, `static_w`, `energy_label` and `energy_source` together, each per-class
//! pJ/FLOP optional.

use crate::prelude::*;

use serde::Deserialize;

use super::{EnergyRoofline, FamilyError, Roofline};

/// 2026-10-22: The labels an energy constant may carry (owner decision 4).
const LABELS: [&str; 3] = ["preliminary", "measured", "certified"];

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RooflineFile {
    dram_gbps: f64,
    bf16_tflops: f64,
    fp8_tflops: f64,
    nvfp4_tflops: f64,
    context_tokens: u64,
    dram_pj_per_byte: Option<f64>,
    // 2026-10-22: The TOML key keeps the `compute = "cuda_core"` vocabulary of KERNEL_FAMILIES;
    // the Rust name is vendor-neutral (VBOP, `xtask vendor-ban`).
    #[serde(rename = "cuda_core_fp32_pj_per_flop")]
    simt_fp32_pj_per_flop: Option<f64>,
    bf16_mma_pj_per_flop: Option<f64>,
    fp8_mma_pj_per_flop: Option<f64>,
    nvfp4_mma_pj_per_flop: Option<f64>,
    static_w: Option<f64>,
    energy_label: Option<String>,
    energy_source: Option<String>,
}

fn parse_err(msg: &str) -> FamilyError {
    FamilyError::Parse(format!("[roofline]: {msg}"))
}

fn positive(v: f64) -> bool {
    v.is_finite() && v > 0.0
}

/// 2026-10-22: The validated time roofline and, when declared, the energy roofline.
pub(super) fn roofline(r: RooflineFile) -> Result<(Roofline, Option<EnergyRoofline>), FamilyError> {
    let time = [r.dram_gbps, r.bf16_tflops, r.fp8_tflops, r.nvfp4_tflops];
    if !time.iter().all(|v| positive(*v)) || r.context_tokens == 0 {
        return Err(FamilyError::Parse(
            "every [roofline] value must be positive".into(),
        ));
    }
    let time = Roofline {
        dram_gbps: r.dram_gbps,
        bf16_tflops: r.bf16_tflops,
        fp8_tflops: r.fp8_tflops,
        nvfp4_tflops: r.nvfp4_tflops,
        context_tokens: r.context_tokens,
    };
    let per_class = [
        r.simt_fp32_pj_per_flop,
        r.bf16_mma_pj_per_flop,
        r.fp8_mma_pj_per_flop,
        r.nvfp4_mma_pj_per_flop,
    ];
    let energy = match (
        r.dram_pj_per_byte,
        r.static_w,
        r.energy_label,
        r.energy_source,
    ) {
        (None, None, None, None) => {
            if per_class.iter().any(Option::is_some) {
                return Err(parse_err(
                    "a pJ/FLOP constant needs dram_pj_per_byte, static_w, energy_label and \
                     energy_source",
                ));
            }
            None
        }
        (Some(b), Some(s), Some(label), Some(source)) => {
            if !positive(b) || !positive(s) || per_class.iter().flatten().any(|v| !positive(*v)) {
                return Err(parse_err("every energy constant must be positive"));
            }
            if !LABELS.contains(&label.as_str()) || source.trim().is_empty() {
                return Err(parse_err(
                    "energy_label must be preliminary, measured or certified and energy_source \
                     must name the measurement",
                ));
            }
            Some(EnergyRoofline {
                dram_pj_per_byte: b,
                simt_fp32_pj_per_flop: r.simt_fp32_pj_per_flop,
                bf16_mma_pj_per_flop: r.bf16_mma_pj_per_flop,
                fp8_mma_pj_per_flop: r.fp8_mma_pj_per_flop,
                nvfp4_mma_pj_per_flop: r.nvfp4_mma_pj_per_flop,
                static_w: s,
                label,
                source,
            })
        }
        _ => {
            return Err(parse_err(
                "the energy constants dram_pj_per_byte, static_w, energy_label and energy_source \
                 are all-or-none",
            ));
        }
    };
    Ok((time, energy))
}
