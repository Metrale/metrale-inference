// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Roofline estimates on one device: the decode step at a batch, prefill of a prompt,
//! and whether weights plus KV and recurrent state fit in the device's usable memory.
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Every node is costed by `venn::roofline::node_cost` (one cost model for Venn ranking and
//!   hardware plans); the decode step is summed in [`super::gaps`], where a plan group that
//!   loops per row is costed per row, so a missing multi-row kernel shows in the step time.
//! - Prefill of `T` tokens is the main section at `T` rows with a causal half-context (`T/2`).
//! - Datasheet peaks are ceilings, not predictions: the report names the basis of every number.
//! - Not counted: activations and workspaces, conv windows, the CUDA context, and launch
//!   overhead. The fit check says so beside its table.

use std::collections::BTreeMap;

use super::device::Device;
use super::exec::fp4_fallback;
use crate::format::Format;
use crate::fuser::section_of;
use crate::ir::{Circuit, NodeIdx, OpKind, Section};
use crate::rules::Mode;
use crate::venn::families::Roofline;
use crate::venn::roofline::{CostError, node_cost};

/// 2026-09-30: The roofline constants used on a device, and where they come from.
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceRoofline {
    /// 2026-09-30: Constants. A node's own are [`super::plan::Resolved::roofline_of`].
    pub roofline: Roofline,
    /// 2026-09-30: `measured (...)` or `datasheet (...)`.
    pub basis: String,
    /// 2026-10-01: The constants are a measurement on the device's own class; otherwise every
    /// estimate from them is a roofline projection, unmeasured on the device.
    pub measured: bool,
}

/// 2026-09-30: The device's constants: the measured ones when the registry points at them and
/// they are `measured` (the class's own manifest), else the datasheet with `context_tokens`
/// taken from `assumptions`. A device without the FP4 MMA states its NVFP4 slot at the peak of
/// [`fp4_fallback`].
pub fn device_roofline(
    device: &Device,
    measured: Option<&Roofline>,
    assumptions: &Roofline,
) -> DeviceRoofline {
    if let (Some(src), Some(m)) = (&device.measured, measured) {
        return DeviceRoofline {
            roofline: *m,
            basis: format!("measured achievable ({src})"),
            measured: true,
        };
    }
    let p = device.peaks;
    let mut roofline = Roofline {
        dram_gbps: device.bandwidth_gbps,
        bf16_tflops: p.bf16,
        fp8_tflops: p.fp8.unwrap_or(p.bf16),
        nvfp4_tflops: 0.0,
        context_tokens: assumptions.context_tokens,
    };
    roofline.nvfp4_tflops = p
        .fp4_block_scale
        .unwrap_or_else(|| fp4_fallback(device).peak(&roofline).0);
    DeviceRoofline {
        roofline,
        basis: format!(
            "datasheet ceiling (sources {}{})",
            device.sources.join(", "),
            if device.derived.is_empty() {
                String::new()
            } else {
                format!("; derived, not published: {}", device.derived.join(", "))
            }
        ),
        measured: false,
    }
}

/// 2026-09-30: The estimated prefill time of `tokens` tokens, microseconds, each node costed
/// with `roofline_of` its index.
pub fn prefill_us(
    c: &Circuit,
    settings: &BTreeMap<String, String>,
    roofline_of: &dyn Fn(NodeIdx) -> Roofline,
    tokens: u64,
) -> Result<f64, CostError> {
    let main = section_of(Mode::Decode);
    let mut total = 0.0;
    for b in c.blocks.iter().filter(|b| b.section == main) {
        for i in b.first..b.end {
            let causal = Roofline {
                context_tokens: (tokens / 2).max(1),
                ..roofline_of(i)
            };
            total += node_cost(c, &c.nodes[i], Mode::Decode, tokens, settings, &causal)?.time_us;
        }
    }
    Ok(total)
}

/// 2026-09-30: What one sequence and the whole model hold in device memory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Footprint {
    /// 2026-09-30: Every weight of every section (embedding at BF16), bytes.
    pub weights: f64,
    /// 2026-09-30: KV bytes per token of one sequence, over every attention node.
    pub kv_per_token: f64,
    /// 2026-09-30: Recurrent state bytes of one sequence (GatedDeltaNet and Mamba2).
    pub state_per_seq: f64,
}

impl Footprint {
    /// 2026-09-30: Bytes for `seqs` sequences of `context` tokens.
    pub fn bytes(&self, seqs: u64, context: u64) -> f64 {
        self.weights + seqs as f64 * (self.kv_per_token * context as f64 + self.state_per_seq)
    }
}

/// 2026-09-30: The footprint of `c` under `settings` (`kv_cache_dtype`, `ssm_h_dtype`).
pub fn footprint(c: &Circuit, settings: &BTreeMap<String, String>) -> Result<Footprint, String> {
    let dim = |d: &str| {
        c.dims
            .get(d)
            .map(|v| *v as f64)
            .ok_or_else(|| format!("the footprint needs dim `{d}`"))
    };
    // 2026-10-10: The KV element size is read per state (`read_units_bytes`); the policy must
    // still state one this estimate sizes.
    match settings.get("kv_cache_dtype").map(String::as_str) {
        Some("bf16" | "fp8") => {}
        other => return Err(format!("kv_cache_dtype {other:?} has no element size")),
    }
    let h = match settings.get("ssm_h_dtype").map(String::as_str) {
        Some("f32") => 4.0,
        Some("f16" | "f16-pool" | "bf16") => 2.0,
        other => return Err(format!("ssm_h_dtype {other:?} has no element size")),
    };
    let mut weights = 0.0;
    let (mut kv_per_token, mut state) = (0.0, 0.0);
    let main: Vec<bool> = {
        let mut v = vec![false; c.nodes.len()];
        for b in c.blocks.iter().filter(|b| b.section == Section::Main) {
            v[b.first..b.end].iter_mut().for_each(|x| *x = true);
        }
        v
    };
    for (i, n) in c.nodes.iter().enumerate() {
        // 2026-09-30: The draft head reads the main embedding table; count it once.
        if n.op == OpKind::Embed && main[i] {
            weights += dim("vocab")? * dim("hidden")? * 2.0;
        }
        if let Some(w) = n.weight {
            let (out, k) = c
                .weight_shape(n)
                .ok_or_else(|| format!("node `{}`: its weight has no shape", n.id))?;
            let one = w
                .weight_bytes(out, k)
                .ok_or_else(|| format!("node `{}`: weight {} has no size", n.id, w.name()))?
                as f64;
            let copies = match n.op {
                OpKind::ExpertGateUp | OpKind::ExpertDown => dim("experts")?,
                _ => 1.0,
            };
            weights += one * copies;
        }
        match n.op {
            // 2026-10-10: One token of each KV side the node reads, at their declared formats, so
            // a layer kind with its own head geometry (Gemma-4's global layers) counts its own. A
            // sliding-window layer keeps every token too: the engine's paged pool does not drop
            // the tokens behind the window.
            OpKind::PagedAttention => {
                kv_per_token += crate::venn::roofline::read_units_bytes(c, n, settings)
                    .map_err(|e| e.to_string())?;
            }
            OpKind::GdnRecurrence => {
                state += dim("lin_v_heads")? * dim("lin_k_dim")? * dim("lin_v_dim")? * h;
            }
            // 2026-10-08: The latent cache and the indexer's pooled keys, per token, at their
            // declared formats.
            OpKind::MlaAttention | OpKind::IndexSelect => {
                kv_per_token += crate::venn::roofline::read_unit_bytes(c, n, settings)
                    .map_err(|e| e.to_string())?;
            }
            OpKind::SsmUpdate => {
                state += dim("mamba_heads")? * dim("mamba_head_dim")? * dim("ssm_state")? * 4.0;
            }
            _ => {}
        }
    }
    Ok(Footprint {
        weights,
        kv_per_token,
        state_per_seq: state,
    })
}

/// 2026-09-30: Weight bytes a decode step reads at one row: every main-section weight once
/// (routed experts at `top_k` of them). Divided by bandwidth, the C=1 floor.
pub fn weight_floor_bytes(c: &Circuit) -> Result<f64, String> {
    let mut total = 0.0;
    for b in c.blocks.iter().filter(|b| b.section == Section::Main) {
        for n in &c.nodes[b.first..b.end] {
            let Some(w) = n.weight else { continue };
            let (out, k) = c
                .weight_shape(n)
                .ok_or_else(|| format!("node `{}`: its weight has no shape", n.id))?;
            let copies = match n.op {
                OpKind::ExpertGateUp | OpKind::ExpertDown => {
                    *c.dims.get("top_k").ok_or_else(|| {
                        format!("node `{}`: routed experts need dim `top_k`", n.id)
                    })? as f64
                }
                _ => 1.0,
            };
            let one = w
                .weight_bytes(out, k)
                .ok_or_else(|| format!("node `{}`: weight {} has no size", n.id, w.name()))?;
            total += one as f64 * copies;
        }
    }
    Ok(total)
}

/// 2026-09-30: `Format` of a node's first input.
pub fn activation_of(c: &Circuit, n: &crate::ir::Node) -> Option<Format> {
    n.inputs.first().map(|&e| c.edges[e].format)
}
