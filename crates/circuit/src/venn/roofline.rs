// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The roofline estimate of one node's time in one step, from its edge shapes and
//! formats: `max(bytes / DRAM bandwidth, FLOPs / tensor peak)`. It ranks a Venn report's rows
//! until measured profiles replace it.
//!
//! Owner: metrale-circuit (venn).
//! Invariants:
//! - Bytes are every edge the node reads or writes at `rows` rows, plus its weights, plus the
//!   state it reads and writes: the paged KV cache over `context_tokens` per sequence, the
//!   GatedDeltaNet and Mamba2 recurrent states (FP32, read and written once per sequence).
//!   2026-10-08: Latent attention reads `kv_b_proj` and the latent rows it attends (the selected
//!   ones under an indexer), the indexer's selection every pool key; both at the declared state
//!   format.
//! - Routed experts read `E * (1 - (1 - k/E)^T)` distinct experts' weights for `T` tokens
//!   (uniform routing).
//! - The peak is the MMA class of the node's input activation: FP8 or NVFP4 activations run the
//!   FP8 / NVFP4 MMA; anything else (W4A16, W8A16, BF16) runs the BF16 MMA.
//! - Not counted: launch overhead, conv windows, KV writes beyond their input edges. A dim an op
//!   needs that the circuit lacks is an error, never a guess.

use std::collections::BTreeMap;

use super::families::Roofline;
use crate::format::Format;
use crate::ir::{Circuit, Node, OpKind};
use crate::rules::Mode;
use crate::state::{StateAccess, StateDecl, StateDtype, StateFormat};

/// 2026-09-29: One node's estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cost {
    /// 2026-09-29: DRAM bytes.
    pub bytes: f64,
    /// 2026-09-29: FLOPs.
    pub flops: f64,
    /// 2026-09-29: `max(bytes / bandwidth, flops / peak)`, microseconds.
    pub time_us: f64,
}

/// 2026-09-29: Why a node could not be estimated.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CostError {
    /// 2026-09-29: A dim the op's cost needs.
    #[error("node `{node}`: the estimate needs dim `{dim}`, which the circuit does not define")]
    MissingDim {
        /// 2026-09-29: Node id.
        node: String,
        /// 2026-09-29: The dim.
        dim: String,
    },
    /// 2026-09-29: An edge whose rows do not evaluate or whose bytes do not divide.
    #[error("node `{node}`: edge `{edge}` has no byte size at {rows} rows")]
    Edge {
        /// 2026-09-29: Node id.
        node: String,
        /// 2026-09-29: Edge id.
        edge: String,
        /// 2026-09-29: Rows.
        rows: u64,
    },
    /// 2026-09-29: A KV-cache dtype the estimate has no element size for.
    #[error("kv_cache_dtype `{0}` has no element size (bf16 | fp8)")]
    KvDtype(String),
    /// 2026-10-08: A state or parameter an op's estimate reads that the node does not give.
    #[error("node `{node}`: {detail}")]
    Node {
        /// 2026-10-08: Node id.
        node: String,
        /// 2026-10-08: What is missing.
        detail: String,
    },
}

/// 2026-09-29: Estimate `node` at `rows` rows of `mode`.
pub fn node_cost(
    c: &Circuit,
    n: &Node,
    mode: Mode,
    rows: u64,
    settings: &BTreeMap<String, String>,
    r: &Roofline,
) -> Result<Cost, CostError> {
    let dim = |name: &str| {
        c.dims
            .get(name)
            .map(|v| *v as f64)
            .ok_or_else(|| CostError::MissingDim {
                node: n.id.clone(),
                dim: name.to_string(),
            })
    };
    let seqs = if mode == Mode::MultiSeq { rows } else { 1 } as f64;
    let t = rows as f64;
    let mut bytes = 0.0;
    let contiguous_split =
        n.op == OpKind::Split && n.params.get("layout").map(String::as_str) == Some("contiguous");
    if !contiguous_split {
        for &e in n.inputs.iter().chain(&n.outputs) {
            bytes += edge_bytes(c, n, e, rows)?;
        }
    }
    let (out, k) = c.weight_shape(n).ok_or_else(|| CostError::Edge {
        node: n.id.clone(),
        edge: "<weight>".into(),
        rows,
    })?;
    let (out, k) = (out as f64, k as f64);
    let mut flops = 0.0;
    match n.op {
        OpKind::Linear(_) | OpKind::LmHead | OpKind::Router => {
            bytes += weight_bytes(n, out, k)?;
            flops = 2.0 * t * k * out;
        }
        OpKind::ExpertGateUp | OpKind::ExpertDown => {
            let (e, top) = (dim("experts")?, dim("top_k")?);
            let distinct = e * (1.0 - (1.0 - top / e).powf(t));
            bytes += distinct * weight_bytes(n, out, k)?;
            flops = 2.0 * t * top * k * out;
        }
        OpKind::PagedAttention => {
            let kv = match settings.get("kv_cache_dtype").map(String::as_str) {
                Some("bf16") => 2.0,
                Some("fp8") => 1.0,
                other => return Err(CostError::KvDtype(other.unwrap_or("<unset>").to_string())),
            };
            let ctx = r.context_tokens as f64;
            let (kvh, hd, qh) = (dim("kv_heads")?, dim("head_dim")?, dim("q_heads")?);
            bytes += seqs * ctx * kvh * hd * 2.0 * kv;
            flops = 4.0 * t * ctx * qh * hd;
        }
        OpKind::MlaAttention => {
            // 2026-10-08: `kv_b_proj` (both halves, the query absorption and the value
            // projection), then every attended latent row once per sequence.
            bytes += weight_bytes(n, out, k)?;
            let ctx = r.context_tokens as f64;
            let attended = match n.params.get("selection").map(String::as_str) {
                Some("index_topk") => ctx.min(dim("index_topk")? + dim("index_kpool")?),
                Some("all") => ctx,
                other => {
                    return Err(CostError::Node {
                        node: n.id.clone(),
                        detail: format!("selection {other:?} is neither index_topk nor all"),
                    });
                }
            };
            bytes += seqs * attended * read_unit_bytes(c, n, settings)?;
            let (qh, lat) = (dim("q_heads")?, dim("kv_lora")?);
            // 2026-10-10: Scores over the whole cached row (the latent, plus the shared rotary
            // key of a decoupled-RoPE cache), values over the latent.
            let key = read_unit(c, n)?.elements as f64;
            flops = 2.0 * t * qh * lat * (dim("mla_qk")? + dim("mla_v")?)
                + 2.0 * t * attended * qh * (key + lat);
        }
        OpKind::IndexSelect => {
            // 2026-10-08: Every pool key of the context once per sequence (the cache holds one
            // pooled key per `index_kpool` tokens, declared per token), scored by every head.
            let ctx = r.context_tokens as f64;
            bytes += seqs * ctx * read_unit_bytes(c, n, settings)?;
            flops = 2.0
                * t
                * (ctx / dim("index_kpool")?)
                * dim("index_heads")?
                * dim("index_head_dim")?;
        }
        OpKind::GdnRecurrence | OpKind::SsmUpdate => {
            let elems = if n.op == OpKind::GdnRecurrence {
                dim("lin_v_heads")? * dim("lin_k_dim")? * dim("lin_v_dim")?
            } else {
                dim("mamba_heads")? * dim("mamba_head_dim")? * dim("ssm_state")?
            };
            // 2026-09-29: FP32 state, read and written once per sequence; about four FLOPs per
            // state element per row.
            bytes += 2.0 * seqs * elems * 4.0;
            flops = 4.0 * t * elems;
        }
        _ => {}
    }
    let peak = match n.inputs.first().map(|&e| c.edges[e].format) {
        Some(Format::Fp8E4m3 { .. }) if n.op.reads_linear_weight() => r.fp8_tflops,
        _ if nvfp4_mma(c, n) => r.nvfp4_tflops,
        _ => r.bf16_tflops,
    };
    let time_us = (bytes / (r.dram_gbps * 1e3)).max(flops / (peak * 1e6));
    Ok(Cost {
        bytes,
        flops,
        time_us,
    })
}

/// 2026-10-10: The first state `n` reads.
fn read_unit<'c>(c: &'c Circuit, n: &Node) -> Result<&'c StateDecl, CostError> {
    n.state
        .iter()
        .find(|(_, a)| *a == StateAccess::Read)
        .map(|(idx, _)| &c.states[*idx])
        .ok_or_else(|| CostError::Node {
            node: n.id.clone(),
            detail: "the estimate needs the state it reads".into(),
        })
}

/// 2026-10-08: Bytes of one unit (one token) of the first state `n` reads: its elements times
/// its element size, a keyed format read from `settings`.
pub(crate) fn read_unit_bytes(
    c: &Circuit,
    n: &Node,
    settings: &BTreeMap<String, String>,
) -> Result<f64, CostError> {
    let fail = |detail: String| CostError::Node {
        node: n.id.clone(),
        detail,
    };
    let decl = read_unit(c, n)?;
    let dtype = match &decl.format {
        StateFormat::Fixed(d) => *d,
        StateFormat::Keyed(key) => {
            let v = settings
                .get(key)
                .ok_or_else(|| fail(format!("the policy states no `{key}`")))?;
            StateDtype::parse(v).ok_or_else(|| CostError::KvDtype(v.clone()))?
        }
    };
    Ok((decl.elements * dtype.size()) as f64)
}

/// 2026-10-01: `n` multiplies an NVFP4 activation by a linear weight: the node the NVFP4 peak
/// costs.
pub fn nvfp4_mma(c: &Circuit, n: &Node) -> bool {
    n.op.reads_linear_weight()
        && matches!(
            n.inputs.first().map(|&e| c.edges[e].format),
            Some(Format::Nvfp4 { .. })
        )
}

fn edge_bytes(c: &Circuit, n: &Node, e: usize, rows: u64) -> Result<f64, CostError> {
    let edge = &c.edges[e];
    let mut dims = c.dims.clone();
    dims.insert("n".into(), rows);
    let bad = || CostError::Edge {
        node: n.id.clone(),
        edge: edge.id.clone(),
        rows,
    };
    let r = edge.rows.eval(&dims).map_err(|_| bad())?;
    edge.format
        .bytes(r, edge.dim_value)
        .map(|b| b as f64)
        .ok_or_else(bad)
}

fn weight_bytes(n: &Node, out: f64, k: f64) -> Result<f64, CostError> {
    let Some(w) = n.weight else {
        return Ok(0.0);
    };
    w.weight_bytes(out as u64, k as u64)
        .map(|b| b as f64)
        .ok_or_else(|| CostError::Edge {
            node: n.id.clone(),
            edge: "<weight>".into(),
            rows: out as u64,
        })
}
