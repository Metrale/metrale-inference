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
//!   2026-10-18: Latent sparse attention reads its window ring and the compressed rows it
//!   attends (its own state or the nearest earlier layer's); the latent compression reads and
//!   writes its run tail.
//! - Routed experts read `E * (1 - (1 - k/E)^T)` distinct experts' weights for `T` tokens
//!   (uniform routing).
//! - The peak is the MMA class of the node's input activation: FP8 or NVFP4 activations run the
//!   FP8 / NVFP4 MMA; anything else (W4A16, W8A16, BF16) runs the BF16 MMA.
//! - Not counted: launch overhead, conv windows, KV writes beyond their input edges. A dim an op
//!   needs that the circuit lacks is an error, never a guess.

use crate::prelude::*;

use alloc::collections::BTreeMap;

use super::families::Roofline;
use crate::format::Format;
use crate::ir::{Circuit, Node, OpKind};
use crate::kv;
use crate::pipeline::require::keyed_state;
use crate::rules::Mode;
use crate::state::{StateAccess, StateDecl, StateFormat};

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
    /// 2026-09-29: A KV cache the estimate cannot size.
    ///
    /// 2026-10-18: Carries why: the policy states no `kv_cache_dtype`, it names no codec pair,
    /// or a side's byte count overflows.
    #[error("{0}")]
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
    let mut bytes: f64 = edge_traffic(c, n, rows)?.iter().map(|(_, b)| b).sum();
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
            let distinct = e * (1.0 - libm::pow(1.0 - top / e, t));
            bytes += distinct * weight_bytes(n, out, k)?;
            flops = 2.0 * t * top * k * out;
        }
        OpKind::PagedAttention => {
            // 2026-10-18: One token of both sides, from each side's codec spec.
            let pair = kv::kv_pair(settings).map_err(CostError::KvDtype)?;
            let ctx = r.context_tokens as f64;
            let (kvh, hd, qh) = (dim("kv_heads")?, dim("head_dim")?, dim("q_heads")?);
            let per_side = crate::num::sat_u64(kvh * hd);
            let token = kv::token_bytes(pair, per_side).ok_or_else(|| {
                CostError::KvDtype(format!("{pair} KV of {per_side} values per side overflows"))
            })?;
            bytes += seqs * ctx * token as f64;
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
            flops = 2.0 * t * qh * lat * (dim("mla_qk")? + dim("mla_v")?)
                + 4.0 * t * attended * qh * lat;
        }
        OpKind::IndexSelect => {
            // 2026-10-08: Every pool key of the context once per sequence (the cache holds one
            // pooled key per `pool` tokens, declared per token), scored by every head.
            // 2026-10-18: The pool width is the dim the node's `pool` param names; the keys are
            // its own state or, under a `keys` param, the nearest earlier layer's.
            let ctx = r.context_tokens as f64;
            let pool = dim(param(n, "pool")?)?;
            let unit = match n.params.get("keys") {
                Some(local) => named_unit_bytes(c, n, local, settings)?,
                None => read_unit_bytes(c, n, settings)?,
            };
            bytes += seqs * ctx * unit;
            flops = 2.0 * t * (ctx / pool) * dim("index_heads")? * dim("index_head_dim")?;
        }
        OpKind::LatentSparseAttention => {
            // 2026-10-18: The window ring once per sequence, then the selected compressed
            // positions (at most `index_topk` of the `ctx / ratio`), each `ratio` per-token
            // units of the compressed state; one shared latent row is both key and value.
            let ctx = r.context_tokens as f64;
            bytes += seqs * update_state_bytes(c, n, settings)?;
            let compressed = match param(n, "compressed")? {
                "none" => 0.0,
                local => {
                    let ratio = dim(param(n, "ratio")?)?;
                    let pos = (ctx / ratio).min(dim("index_topk")?);
                    bytes += seqs * pos * ratio * named_unit_bytes(c, n, local, settings)?;
                    pos
                }
            };
            let attended = dim("window")?.min(ctx) + compressed;
            flops = 4.0 * t * attended * dim("q_heads")? * dim("head_dim")?;
        }
        OpKind::KvCompress => {
            // 2026-10-18: The run tail read and written once per sequence.
            bytes += 2.0 * seqs * update_state_bytes(c, n, settings)?;
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
    let peak = peak_tflops(c, n, r);
    let time_us = (bytes / (r.dram_gbps * 1e3)).max(flops / (peak * 1e6));
    Ok(Cost {
        bytes,
        flops,
        time_us,
    })
}

/// 2026-10-22: The tensor peak `n` is costed at: the MMA class of its input activation.
pub(crate) fn peak_tflops(c: &Circuit, n: &Node, r: &Roofline) -> f64 {
    match n.inputs.first().map(|&e| c.edges[e].format) {
        Some(Format::Fp8E4m3 { .. }) if n.op.reads_linear_weight() => r.fp8_tflops,
        _ if nvfp4_mma(c, n) => r.nvfp4_tflops,
        _ => r.bf16_tflops,
    }
}

/// 2026-10-22: The edges whose bytes [`node_cost`] counts for `n` at `rows` rows, each with its
/// bytes: every input and output, except a contiguous split's (a view, no traffic).
pub(crate) fn edge_traffic(c: &Circuit, n: &Node, rows: u64) -> Result<Vec<(usize, f64)>, CostError> {
    let contiguous_split =
        n.op == OpKind::Split && n.params.get("layout").map(String::as_str) == Some("contiguous");
    if contiguous_split {
        return Ok(Vec::new());
    }
    n.inputs
        .iter()
        .chain(&n.outputs)
        .map(|&e| Ok((e, edge_bytes(c, n, e, rows)?)))
        .collect()
}

/// 2026-10-08: Bytes of one unit (one token) of the first state `n` reads: its elements times
/// its element size, a keyed format read from `settings`.
pub(crate) fn read_unit_bytes(
    c: &Circuit,
    n: &Node,
    settings: &BTreeMap<String, String>,
) -> Result<f64, CostError> {
    let (idx, _) = n
        .state
        .iter()
        .find(|(_, a)| *a == StateAccess::Read)
        .ok_or_else(|| node_error(n, "the estimate needs the state it reads".into()))?;
    unit_bytes(n, &c.states[*idx], settings)
}

/// 2026-10-18: Bytes of one unit (one slot) of the first state `n` updates.
pub(crate) fn update_state_bytes(
    c: &Circuit,
    n: &Node,
    settings: &BTreeMap<String, String>,
) -> Result<f64, CostError> {
    let (idx, _) = n
        .state
        .iter()
        .find(|(_, a)| *a == StateAccess::Update)
        .ok_or_else(|| node_error(n, "the estimate needs the state it updates".into()))?;
    unit_bytes(n, &c.states[*idx], settings)
}

/// 2026-10-18: Bytes of one unit of the state named `local` that `n` reads: its own read state of
/// that name, else the state of that name of the nearest earlier layer in its section (a layer
/// that attends a cache an earlier layer writes).
pub(crate) fn named_unit_bytes(
    c: &Circuit,
    n: &Node,
    local: &str,
    settings: &BTreeMap<String, String>,
) -> Result<f64, CostError> {
    let own = n
        .state
        .iter()
        .map(|&(i, _)| &c.states[i])
        .find(|s| s.local == local);
    let earlier = || {
        let section = own_section(c, n)?;
        c.states
            .iter()
            .filter(|s| s.local == local && s.section == section)
            .filter(|s| matches!((s.layer, n.layer), (Some(a), Some(b)) if a < b))
            .max_by_key(|s| s.layer)
    };
    let decl = own.or_else(earlier).ok_or_else(|| {
        node_error(
            n,
            format!("no state `{local}` in this layer or an earlier one"),
        )
    })?;
    unit_bytes(n, decl, settings)
}

fn own_section(c: &Circuit, n: &Node) -> Option<crate::ir::Section> {
    let i = c.node(&n.id)?;
    c.blocks
        .iter()
        .find(|b| (b.first..b.end).contains(&i))
        .map(|b| b.section)
}

fn unit_bytes(
    n: &Node,
    decl: &StateDecl,
    settings: &BTreeMap<String, String>,
) -> Result<f64, CostError> {
    let dtype = match &decl.format {
        StateFormat::Fixed(d) => *d,
        StateFormat::Keyed(key) => keyed_state(key, settings).map_err(|e| node_error(n, e))?,
    };
    let unit = decl
        .unit_bytes(dtype)
        .map_err(|e| node_error(n, e.to_string()))?;
    Ok(unit as f64)
}

/// 2026-10-18: The param `key` of `n`, which its estimate needs.
fn param<'n>(n: &'n Node, key: &str) -> Result<&'n str, CostError> {
    n.params
        .get(key)
        .map(String::as_str)
        .ok_or_else(|| node_error(n, format!("the estimate needs param `{key}`")))
}

fn node_error(n: &Node, detail: String) -> CostError {
    CostError::Node {
        node: n.id.clone(),
        detail,
    }
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
    w.weight_bytes(crate::num::sat_u64(out), crate::num::sat_u64(k))
        .map(|b| b as f64)
        .ok_or_else(|| CostError::Edge {
            node: n.id.clone(),
            edge: "<weight>".into(),
            rows: crate::num::sat_u64(out),
        })
}
