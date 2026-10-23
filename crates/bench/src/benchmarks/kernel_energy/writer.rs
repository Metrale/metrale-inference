// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `docs/kernel-perf/measurements.toml` writer for a kernel-energy row: the time
//! fields (`time_us`, `bytes`, `flops`, `bound`, `floor_us`, `pct_of_floor`, as the file header
//! defines them) and the energy group (`metrale_power::row::EnergyRow`, floats in J and W), with
//! the energy floor derived from the hardware's `[roofline]` in KERNEL_FAMILIES.toml (SSOT).
//! `scripts/kernel_perf.py --check` validates what it writes.
//!
//! Owner: bench, kernel-energy.
//! Invariants: a row whose `<kernel> @ <regime>` key is already in the file is refused, never
//! replaced; the appended text parses back as TOML.

use anyhow::{Context, Result, bail};
use metrale_power::row::EnergyRow;

/// 2026-10-22: The time and identity fields of a row.
#[derive(Clone, Debug, PartialEq)]
pub struct BaseRow {
    pub kernel: String,
    pub file: String,
    pub hardware: String,
    pub model: String,
    pub regime: String,
    pub time_us: f64,
    pub bytes: u64,
    pub flops: u64,
    pub source: String,
    pub notes: String,
}

/// 2026-10-22: The hardware's `[roofline]` constants a row needs.
#[derive(Clone, Debug, PartialEq)]
pub struct Roofline {
    pub dram_gbps: f64,
    /// 2026-10-22: The tensor or CUDA-core peak of the row's peak class, TFLOPS.
    pub peak_tflops: f64,
    pub dram_pj_per_byte: Option<f64>,
    /// 2026-10-22: pJ/FLOP of the row's energy floor class; `None` for `dram_only`.
    pub pj_per_flop: Option<f64>,
}

/// 2026-10-22: Read `[roofline]` from a KERNEL_FAMILIES.toml text, for peak class `peak`
/// (`bf16` | `fp8` | `nvfp4`) and energy floor class `floor_class` (`dram_only` |
/// `cuda_core_fp32` | `bf16_mma` | `fp8_mma` | `nvfp4_mma`).
pub fn roofline(families: &str, peak: &str, floor_class: &str) -> Result<Roofline> {
    let doc: toml::Table =
        toml::from_str(families).context("KERNEL_FAMILIES.toml does not parse")?;
    let r = doc
        .get("roofline")
        .and_then(|v| v.as_table())
        .context("KERNEL_FAMILIES.toml has no [roofline]")?;
    let f = |k: &str| r.get(k).and_then(|v| v.as_float());
    let peak_tflops =
        f(&format!("{peak}_tflops")).with_context(|| format!("[roofline] has no {peak}_tflops"))?;
    let pj_per_flop = match floor_class {
        "dram_only" => None,
        c => Some(
            f(&format!("{c}_pj_per_flop"))
                .with_context(|| format!("[roofline] has no {c}_pj_per_flop"))?,
        ),
    };
    Ok(Roofline {
        dram_gbps: f("dram_gbps").context("[roofline] has no dram_gbps")?,
        peak_tflops,
        dram_pj_per_byte: f("dram_pj_per_byte"),
        pj_per_flop,
    })
}

fn quote(s: &str) -> String {
    toml::Value::String(s.to_string()).to_string()
}

/// 2026-10-22: The `[[m]]` block of `base` + `energy`, with the floors derived from `roof` (the
/// energy floor only when `floor_class` is given and the roofline has its constants).
pub fn render(
    base: &BaseRow,
    roof: &Roofline,
    energy: &EnergyRow,
    floor_class: Option<&str>,
) -> Result<String> {
    if !base.time_us.is_finite() || base.time_us <= 0.0 {
        bail!("time_us must be positive");
    }
    let mem_us = base.bytes as f64 / (roof.dram_gbps * 1e3);
    let mma_us = base.flops as f64 / (roof.peak_tflops * 1e6);
    let floor_us = mem_us.max(mma_us);
    let bound = if mem_us >= mma_us {
        "memory"
    } else {
        "compute"
    };
    let mut e = energy.clone();
    if let (Some(class), Some(eb)) = (floor_class, roof.dram_pj_per_byte) {
        let floor_j =
            (base.bytes as f64 * eb + base.flops as f64 * roof.pj_per_flop.unwrap_or(0.0)) * 1e-12;
        e.energy_floor_class = Some(class.to_string());
        e.energy_floor_j = Some(floor_j);
        e.pct_of_energy_floor = Some(floor_j / e.energy_dyn_j * 100.0);
    }
    let mut lines = vec![
        "[[m]]".to_string(),
        format!("kernel = {}", quote(&base.kernel)),
        format!("file = {}", quote(&base.file)),
        format!("hardware = {}", quote(&base.hardware)),
        format!("model = {}", quote(&base.model)),
        format!("regime = {}", quote(&base.regime)),
        format!("time_us = {:.3}", base.time_us),
        format!("bytes = {}", base.bytes),
        format!("flops = {}", base.flops),
        format!("bound = {}", quote(bound)),
        format!("floor_us = {floor_us:.3}"),
        format!("pct_of_floor = {:.2}", floor_us / base.time_us * 100.0),
        format!("source = {}", quote(&base.source)),
        format!("notes = {}", quote(&base.notes)),
    ];
    lines.extend(e.toml_lines());
    let block = lines.join("\n") + "\n";
    toml::from_str::<toml::Table>(&block).context("the rendered row does not parse as TOML")?;
    Ok(block)
}

/// 2026-10-22: `existing` with `block` appended; refused when a row with the same
/// `<kernel> @ <regime>` key is already there.
pub fn append(existing: &str, block: &str) -> Result<String> {
    let doc: toml::Table = toml::from_str(existing).context("measurements.toml does not parse")?;
    let new: toml::Table = toml::from_str(block).context("the new row does not parse")?;
    let key = |m: &toml::Value| {
        let s = |k: &str| m.get(k).and_then(|v| v.as_str()).unwrap_or("").to_string();
        format!("{} @ {}", s("kernel"), s("regime"))
    };
    let row = new
        .get("m")
        .and_then(|v| v.as_array())
        .and_then(|a| a.first())
        .context("the new block holds no [[m]] row")?;
    let k = key(row);
    let rows = doc
        .get("m")
        .and_then(|v| v.as_array())
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    if rows.iter().any(|m| key(m) == k) {
        bail!("measurements.toml already has a row {k}");
    }
    let mut out = existing.trim_end().to_string();
    out.push_str("\n\n");
    out.push_str(block);
    toml::from_str::<toml::Table>(&out)
        .context("measurements.toml would not parse after the append")?;
    Ok(out)
}

#[cfg(test)]
#[path = "writer_tests.rs"]
mod tests;
