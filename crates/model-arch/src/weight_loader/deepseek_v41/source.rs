// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The per-layer tensors of a DeepSeek-V4.1 checkpoint by role ([`T`]), and the port
//! the layer builder reads them through ([`V41Source`]). The GGUF and the safetensors checkpoint
//! name the same tensors differently ([`name`]); each format implements the port once, so the
//! builder never sees a tensor name.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - [`name`], [`expert_name`], [`engram_rows_name`] and [`scale_of`] are the only places a
//!   per-layer tensor name is spelled.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::ResidentMat;

use super::upload_f32;

/// 2026-10-19: Which hyper-connection mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcSite {
    Attn,
    Ffn,
}

/// 2026-10-19: Which tensor of a hyper-connection mix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HcPart {
    Fn,
    Base,
    Scale,
}

/// 2026-10-19: A per-layer tensor by role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T {
    WqA,
    QNorm,
    WqB,
    Wkv,
    KvNorm,
    WoA,
    WoB,
    Sink,
    CompNorm,
    CompWkv,
    CompWgate,
    IdxWqB,
    IdxProj,
    IdxWk,
    IdxKNorm,
    GateW,
    GateBias,
    SharedW1,
    SharedW2,
    SharedW3,
    Hc(HcSite, HcPart),
    AttnNorm,
    FfnNorm,
    EngramQ,
    EngramK,
    EngramWkv,
}

/// 2026-10-19: Every role, in declaration order.
pub const ROLES: [T; 31] = [
    T::WqA,
    T::QNorm,
    T::WqB,
    T::Wkv,
    T::KvNorm,
    T::WoA,
    T::WoB,
    T::Sink,
    T::CompNorm,
    T::CompWkv,
    T::CompWgate,
    T::IdxWqB,
    T::IdxProj,
    T::IdxWk,
    T::IdxKNorm,
    T::GateW,
    T::GateBias,
    T::SharedW1,
    T::SharedW2,
    T::SharedW3,
    T::Hc(HcSite::Attn, HcPart::Fn),
    T::Hc(HcSite::Attn, HcPart::Base),
    T::Hc(HcSite::Attn, HcPart::Scale),
    T::Hc(HcSite::Ffn, HcPart::Fn),
    T::Hc(HcSite::Ffn, HcPart::Base),
    T::Hc(HcSite::Ffn, HcPart::Scale),
    T::AttnNorm,
    T::FfnNorm,
    T::EngramQ,
    T::EngramK,
    T::EngramWkv,
];

/// 2026-10-19: The two checkpoint formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    Gguf,
    Safetensors,
}

/// 2026-10-19: Tensor `t` of layer `l` as `fmt` names it.
pub fn name(fmt: Format, l: usize, t: T) -> String {
    let gguf = fmt == Format::Gguf;
    let pick = |g: &'static str, s: &'static str| if gguf { g } else { s };
    let tail: String = match t {
        T::WqA => "attn.wq_a.weight".into(),
        T::QNorm => "attn.q_norm.weight".into(),
        T::WqB => "attn.wq_b.weight".into(),
        T::Wkv => "attn.wkv.weight".into(),
        T::KvNorm => "attn.kv_norm.weight".into(),
        T::WoA => "attn.wo_a.weight".into(),
        T::WoB => "attn.wo_b.weight".into(),
        T::Sink => "attn.attn_sink".into(),
        T::CompNorm => pick("compressor.norm.weight", "attn.compressor.norm.weight").into(),
        T::CompWkv => pick("compressor.wkv.weight", "attn.compressor.wkv.weight").into(),
        T::CompWgate => pick("compressor.wgate.weight", "attn.compressor.wgate.weight").into(),
        T::IdxWqB => pick("indexer.wq_b.weight", "attn.indexer.wq_b.weight").into(),
        T::IdxProj => pick("indexer.proj.weight", "attn.indexer.weights_proj.weight").into(),
        T::IdxWk => pick("indexer.wk.weight", "attn.indexer.wk.weight").into(),
        T::IdxKNorm => pick("indexer.k_norm.weight", "attn.indexer.k_norm.weight").into(),
        T::GateW => "ffn.gate.weight".into(),
        T::GateBias => pick("ffn.gate.e_score_correction_bias", "ffn.gate.bias").into(),
        T::SharedW1 => pick("ffn.shared_experts.w1", "ffn.shared_experts.w1.weight").into(),
        T::SharedW2 => pick("ffn.shared_experts.w2", "ffn.shared_experts.w2.weight").into(),
        T::SharedW3 => pick("ffn.shared_experts.w3", "ffn.shared_experts.w3.weight").into(),
        T::Hc(site, part) => {
            let s = match site {
                HcSite::Attn => "attn",
                HcSite::Ffn => "ffn",
            };
            let p = match part {
                HcPart::Fn => "fn",
                HcPart::Base => "base",
                HcPart::Scale => "scale",
            };
            format!("hc_{s}_{p}")
        }
        T::AttnNorm => "attn_norm.weight".into(),
        T::FfnNorm => "ffn_norm.weight".into(),
        T::EngramQ => pick("engram.wq", "engram.q_weight").into(),
        T::EngramK => pick("engram.wk", "engram.k_weight").into(),
        T::EngramWkv => pick("engram.wkv", "engram.wkv.weight").into(),
    };
    match fmt {
        Format::Gguf => format!("model.layers.{l}.{tail}"),
        Format::Safetensors => format!("layers.{l}.{tail}"),
    }
}

/// 2026-10-19: The routed-expert projections of the safetensors checkpoint, in the pointer-table
/// order `ResidentExperts::addr` reads (each projection's packed values, then its scales).
pub const EXPERT_TABLES: [&str; 6] = [
    "w1.weight",
    "w1.scale",
    "w3.weight",
    "w3.scale",
    "w2.weight",
    "w2.scale",
];

/// 2026-10-19: Table `table` ([`EXPERT_TABLES`]) of routed expert `e` of layer `l` in the
/// safetensors checkpoint (the GGUF stacks are read by the expert cache).
pub fn expert_name(l: usize, e: usize, table: usize) -> String {
    format!("layers.{l}.ffn.experts.{e}.{}", EXPERT_TABLES[table])
}

/// 2026-10-19: Engram layer `l`'s row table (E4M3 codes) in the safetensors checkpoint; its
/// scales are [`scale_of`] it.
pub fn engram_rows_name(l: usize) -> String {
    format!("layers.{l}.engram.embed.weight")
}

/// 2026-10-19: The E8M0 scale tensor beside the FP8 weight `weight` (`….weight` → `….scale`).
pub fn scale_of(weight: &str) -> Result<String> {
    match weight.strip_suffix(".weight") {
        Some(stem) => Ok(format!("{stem}.scale")),
        None => anyhow::bail!("{weight}: an FP8 weight name ends in .weight"),
    }
}

/// 2026-10-19: An engram layer's projection as the engine holds it.
pub struct EngramWkv {
    /// 2026-10-19: bf16 `[dim * (hc + 1), cols * head_dim]`.
    pub wkv: DevicePtr,
    /// 2026-10-19: The raw Q2_K blocks for the single-token projection, or null.
    pub wkv_q2k: DevicePtr,
    /// 2026-10-19: Round the rows to E4M3 blocks before the projection.
    pub qdq_rows: bool,
}

/// 2026-10-19: Where the layer builder reads a checkpoint's per-layer tensors. Every method reads
/// this rank's part of the tensor: whole on one rank, and on more the slice the format's
/// implementation cuts.
pub trait V41Source {
    /// 2026-10-19: Does layer `l` carry `t`?
    fn has(&self, l: usize, t: T) -> bool;
    /// 2026-10-19: A projection as a [`ResidentMat`] in a precision the GEMMs read.
    fn mat(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<ResidentMat>;
    /// 2026-10-19: A bf16 tensor on the device.
    fn bf16(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<DevicePtr>;
    /// 2026-10-19: A tensor widened to f32 on the host.
    fn host_f32(&self, gpu: &dyn GpuBackend, l: usize, t: T) -> Result<Vec<f32>>;
    /// 2026-10-19: Engram layer `l`'s projection.
    fn engram_wkv(&self, gpu: &dyn GpuBackend, l: usize) -> Result<EngramWkv>;
}

/// 2026-10-19: A new f32 device copy of `t`, which must hold `expect` elements on this rank.
pub fn f32_dev(
    src: &dyn V41Source,
    gpu: &dyn GpuBackend,
    l: usize,
    t: T,
    expect: usize,
) -> Result<DevicePtr> {
    let v = src.host_f32(gpu, l, t)?;
    ensure!(
        v.len() == expect,
        "layer {l} {t:?}: {} elements, expected {expect}",
        v.len()
    );
    upload_f32(gpu, &v)
}

#[cfg(test)]
#[path = "source_tests.rs"]
mod tests;
