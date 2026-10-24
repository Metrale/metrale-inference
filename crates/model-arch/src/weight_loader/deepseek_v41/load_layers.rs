// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: Builds the DeepSeek-V4.1 layers from a GGUF checkpoint: the expert arena and
//! cache over the shard files, the Q2_K engram rows and the token map from the GGUF header. The
//! engines, the runtime and the layers are `build`'s.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every returned layer holds a clone of the same `Arc<V41Runtime>`.
//! - Loading fails unless the expert stacks hold `num_experts` experts, and on more than one
//!   tensor-parallel rank (the GGUF path serves one).

use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;
use metrale_model_weights::weights::expert_stream::{
    EngramRowReader, ExpertArena, ExpertSliceMap, ExpertSource, ShardFiles,
};
use metrale_settings::InferenceServerSettings;

use crate::attn_v41::AttnV41;
use crate::deepseek_v41_layer::levers::Ds41LoadLevers;
use crate::deepseek_v41_layer::{StreamedExperts, V41EngramRows, V41Experts};
use crate::moe_v41::MoeV41;
use metrale_model_layers::layer::TransformerLayer;

use super::build::{self, Parts, mem_available};
use super::gguf_source::GgufSource;
use super::tp_cut::Local;

pub fn load_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    _layer_kv_dtypes: &[metrale_cache::kv_cache::KvCacheDtype],
    settings: &InferenceServerSettings,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let levers = Ds41LoadLevers::from_settings(settings);
    ensure!(
        config.tp_world_size <= 1,
        "deepseek-v4.1: the GGUF checkpoint serves one rank; tensor parallelism needs the safetensors checkpoint"
    );
    let anchor = store
        .deferred("model.layers.0.ffn.experts_stack.gate")
        .context("deepseek-v4.1: the routed expert stacks were not deferred by the GGUF loader")?;
    let model_dir = anchor.path.parent().context("shard path has no parent")?;
    let weights_settings = store
        .weights_settings()
        .context("deepseek-v4.1: the GGUF loader records the weights settings it loaded with")?;
    let files = Arc::new(ShardFiles::open_dir_with(model_dir, weights_settings)?);
    let slices = Arc::new(ExpertSliceMap::new(files.clone())?);
    let rows = EngramRowReader::new(files.clone())?;
    ensure!(
        slices.num_experts() == config.num_experts,
        "expert stacks hold {} experts, config says {}",
        slices.num_experts(),
        config.num_experts
    );
    let src = GgufSource {
        store,
        files: files.clone(),
        engram_q2k: levers.engram_q2k,
    };
    let plan = build::plan(config, &levers, &src, &Local::whole(config), false)?;
    plan.moe_cfg.require_kquant_experts()?;
    let cache_gib = levers.expert_cache_gib;
    let cache_bytes = (cache_gib * (1u64 << 30) as f64) as usize;
    let reader_threads = levers.reader_threads;
    tracing::info!(
        "DeepSeek-V4.1: {} layers, kv sources {:?}, index sources {:?}, candidate source {} ({} x {}), window {}, max_seq {}, max_tokens {}, expert cache {cache_gib} GiB, {reader_threads} readers",
        plan.n_layers,
        plan.kv_sources,
        plan.index_sources,
        plan.cand_src,
        plan.attn_cfg.cand_topk_blocks,
        plan.attn_cfg.cand_block,
        plan.attn_cfg.window,
        plan.max_seq,
        plan.max_tokens
    );

    let token_map = files
        .header(0)
        .get_i64_array("deepseek41.engram.token_map")
        .context("deepseek41.engram.token_map missing from the GGUF metadata")?;
    let tables = build::tables(config, token_map)?;

    let layout = slices.slot_layout();
    // 2026-09-25: The expert slots are device memory behind a page-locked
    // staging ring; `METRALE_DS41_ARENA_DEVICE=0` selects a page-locked arena
    // the GPU reads in place.
    let (arena_device, staging_slots) = (levers.arena_device, levers.staging_slots);
    let (arena, mut lru) = ExpertArena::alloc_with(
        gpu,
        cache_bytes,
        layout,
        arena_device,
        staging_slots,
        weights_settings,
    )?;
    tracing::info!(
        "DeepSeek-V4.1: expert arena {}; MemAvailable now {}",
        arena.describe(),
        mem_available()
    );
    // 2026-09-25: `METRALE_DS41_READER_POOL=1` reads misses and predicted
    // experts on persistent reader threads, which `METRALE_DS41_PREFETCH_K`
    // needs; otherwise each fetch uses scoped threads. The pool is only armed
    // with the page-locked arena.
    if levers.reader_pool {
        if arena_device {
            tracing::warn!(
                "DeepSeek-V4.1: METRALE_DS41_READER_POOL=1 ignored with the device arena (set METRALE_DS41_ARENA_DEVICE=0 for the pool)"
            );
        } else {
            lru.set_pool(slices.clone(), reader_threads);
            tracing::info!("DeepSeek-V4.1: expert reader pool of {reader_threads} threads");
        }
    }
    if let Some(path) = &levers.route_trace {
        lru.set_trace(path)?;
        tracing::info!("DeepSeek-V4.1: expert route trace -> {path}");
    }
    tracing::info!(
        "DeepSeek-V4.1: expert cache {} slots of {:.2} MiB",
        lru.n_slots(),
        layout.bytes as f64 / 1048576.0
    );
    let parts = Parts {
        attn: AttnV41::new(gpu, plan.attn_cfg.clone(), levers.run)?,
        moe: MoeV41::new(gpu, plan.moe_cfg.clone(), levers.run)?,
        engram: build::engram(gpu, config, &src, tables.n_hash_cols(), plan.max_tokens)?,
        experts: V41Experts::Streamed(StreamedExperts {
            lru: Mutex::new(lru),
            arena,
            slices,
        }),
        rows: V41EngramRows::Q2k(rows),
        tables,
    };
    let rt = build::runtime(gpu, config, &levers, &plan, parts)?;
    build::assemble(gpu, config, &src, &plan, &rt)
}
