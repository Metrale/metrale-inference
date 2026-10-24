// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: Builds the DeepSeek-V4.1 layers from the safetensors checkpoint, one
//! tensor-parallel rank's part: the projections in their stored precision ([`HfSource`]), every
//! layer's routed experts resident on the device as this rank's slice, the E4M3 engram rows read
//! in place, and the token map the serve read from the checkpoint directory. The engines, the
//! runtime and the layers are `build`'s.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every returned layer holds a clone of the same `Arc<V41Runtime>`.
//! - The levers that drive the GGUF expert cache are refused before any device allocation.
//! - Above one rank the experts are sliced over the ranks (`--moe-expert-layout tp`).

use anyhow::{Result, bail, ensure};
use metrale_config::{ModelConfig, MoeExpertLayout};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;
use metrale_settings::InferenceServerSettings;

use crate::attn_v41::AttnV41;
use crate::deepseek_v41_layer::levers::Ds41LoadLevers;
use crate::deepseek_v41_layer::{GraphMode, ResidentSet, V41EngramRows, V41Experts};
use crate::moe_v41::MoeV41;
use crate::moe_v41::resident::ResidentMoe;
use metrale_model_layers::layer::TransformerLayer;

use super::build::{self, Parts, mem_available};
use super::hf::HfSource;
use super::hf_experts::{engram_rows, upload_experts};
use super::tp_cut::Local;

/// 2026-10-19: Refuses the levers that drive the GGUF expert cache or read the GGUF file: the
/// captured-step paths, the routing traces, prefetch and the reader pool. The Q2_K engram
/// (`METRALE_DS41_ENGRAM_Q2K`, on by default) has no Q2_K source here; `load_layers` reports it.
pub fn refuse_cache_levers(levers: &Ds41LoadLevers) -> Result<()> {
    let run = &levers.run;
    let on = [
        (
            "METRALE_DS41_GRAPH / METRALE_DS41_GRAPH_ORACLE",
            run.graph_mode != GraphMode::Off,
        ),
        ("METRALE_DS41_STEP_GRAPH", run.step_graph),
        ("METRALE_DS41_PREFETCH_K", run.prefetch_k > 0),
        ("METRALE_DS41_READER_POOL", levers.reader_pool),
        ("METRALE_DS41_ROUTE_TRACE", levers.route_trace.is_some()),
        ("METRALE_DS41_PREDICT_TRACE", levers.predict_trace.is_some()),
    ];
    let set: Vec<&str> = on.iter().filter(|(_, v)| *v).map(|(k, _)| *k).collect();
    if !set.is_empty() {
        bail!(
            "deepseek-v4.1: {} drive the GGUF expert cache; the safetensors checkpoint keeps its experts resident",
            set.join(", ")
        );
    }
    Ok(())
}

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    settings: &InferenceServerSettings,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let levers = Ds41LoadLevers::from_settings(settings);
    refuse_cache_levers(&levers)?;
    if levers.engram_q2k {
        tracing::info!(
            "DeepSeek-V4.1: METRALE_DS41_ENGRAM_Q2K reads the GGUF engram blocks; the safetensors engram projection is bf16"
        );
    }
    let local = Local::of(config)?;
    ensure!(
        local.world <= 1 || config.moe_expert_layout == MoeExpertLayout::Tp,
        "deepseek-v4.1: the safetensors checkpoint slices its experts over the tensor-parallel ranks; serve with --moe-expert-layout tp"
    );
    ensure!(
        !config.engram_token_map.is_empty(),
        "deepseek-v4.1: no engram token map was read from the checkpoint directory"
    );
    let src = HfSource::new(store, local.clone());
    let plan = build::plan(config, &levers, &src, &local, true)?;
    tracing::info!(
        "DeepSeek-V4.1 (safetensors, rank {} of {}): {} layers, {} heads, {} output groups, expert width {}, kv sources {:?}, index sources {:?}, candidate source {}, window {}, max_seq {}, max_tokens {}",
        config.tp_rank,
        local.world,
        plan.n_layers,
        local.heads.len(),
        local.groups.len(),
        local.inter.len(),
        plan.kv_sources,
        plan.index_sources,
        plan.cand_src,
        plan.attn_cfg.window,
        plan.max_seq,
        plan.max_tokens
    );
    let token_map = config
        .engram_token_map
        .iter()
        .map(|&t| i64::from(t))
        .collect();
    let tables = build::tables(config, token_map)?;
    let experts = upload_experts(gpu, store, config, &local, levers.reader_threads)?;
    tracing::info!(
        "DeepSeek-V4.1: routed experts resident; MemAvailable now {}",
        mem_available()
    );
    let moe = MoeV41::new(gpu, plan.moe_cfg.clone(), levers.run)?;
    let resident = ResidentMoe::new(gpu, &moe, local.world)?;
    let parts = Parts {
        attn: AttnV41::new(gpu, plan.attn_cfg.clone(), levers.run)?,
        engram: build::engram(gpu, config, &src, tables.n_hash_cols(), plan.max_tokens)?,
        moe,
        experts: V41Experts::Resident(ResidentSet {
            moe: resident,
            layers: experts,
        }),
        rows: V41EngramRows::Fp8(engram_rows(store, config)?),
        tables,
    };
    let rt = build::runtime(gpu, config, &levers, &plan, parts)?;
    build::assemble(gpu, config, &src, &plan, &rt)
}

#[cfg(test)]
#[path = "load_layers_hf_tests.rs"]
mod tests;
