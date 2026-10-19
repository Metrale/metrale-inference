// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The format-independent half of a DeepSeek-V4.1 load: the engine configs
//! ([`plan`]), the engram engine, the shared [`V41Runtime`] and the layers, all reading the
//! checkpoint through a [`V41Source`]. The GGUF loader (`load_layers`) and the safetensors loader
//! (`hf`) differ only in where the routed experts, the engram rows and the token map come from.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every returned layer holds a clone of the same `Arc<V41Runtime>`.
//! - [`plan`] fails unless `hc_mult` is in 1..=4 and `compress_ratios` has an entry for every
//!   layer, and some layer carries the compressor and the indexer.

use std::sync::{Arc, Mutex};

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerWeights, CompressorWeightsGpu, IndexerWeightsGpu, LayerRole,
    SharedV41,
};
use crate::deepseek_v41_layer::levers::Ds41LoadLevers;
use crate::deepseek_v41_layer::{
    DeepSeekV41Layer, SegState, V41EngramRows, V41Experts, V41Runtime,
};
use crate::engram_v41::{EngramHashTables, EngramHasher, EngramLayerWeights, EngramV41};
use crate::fp8_block_v41::Fp8StageSize;
use crate::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights, RouterWeights};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::qwen3_attention::HcSiteWeights;
use metrale_model_layers::weight_map::DenseWeight;

use super::source::{HcPart, HcSite, T, V41Source, f32_dev};
use super::tp_cut::Local;
use super::{DEFAULT_CANDIDATE_BLOCK, DEFAULT_CANDIDATE_SOURCE, DEFAULT_CANDIDATE_TOPK_BLOCKS};

/// 2026-10-19: What the load decided before any device allocation.
pub struct Plan {
    pub n_layers: usize,
    pub ratios: Vec<usize>,
    pub kv_sources: Vec<usize>,
    pub index_sources: Vec<usize>,
    pub cand_src: usize,
    pub max_seq: usize,
    pub max_tokens: usize,
    pub attn_cfg: AttnV41Cfg,
    pub moe_cfg: MoeV41Cfg,
}

/// 2026-10-19: The FP8 block stage that serves every attention projection of `c` (the largest
/// `n * k` and the largest `max_tokens * k`).
pub fn attn_fp8_stage(c: &AttnV41Cfg) -> Fp8StageSize {
    let sites = [
        (c.q_rank, c.dim),
        (c.n_heads * c.head_dim, c.q_rank),
        (c.head_dim, c.dim),
        (c.groups * c.o_rank, c.n_heads * c.head_dim / c.groups),
        (c.dim, c.groups * c.o_rank),
        (c.index_heads * c.index_hd, c.q_rank),
    ];
    Fp8StageSize {
        weight_elems: sites.iter().map(|&(n, k)| n * k).max().unwrap_or(0),
        act_elems: c.max_tokens * sites.iter().map(|&(_, k)| k).max().unwrap_or(0),
    }
}

/// 2026-10-19: The FP8 block stage of the shared expert of `c` (`[inter, dim]` and `[dim,
/// inter]`).
pub fn moe_fp8_stage(c: &MoeV41Cfg) -> Fp8StageSize {
    Fp8StageSize {
        weight_elems: c.inter * c.dim,
        act_elems: c.max_tokens * c.dim.max(c.inter),
    }
}

/// 2026-10-19: The engine configs for `local`'s rank. `fp8`: the projections are FP8 blocks,
/// which need a stage.
pub fn plan(
    config: &ModelConfig,
    levers: &Ds41LoadLevers,
    src: &dyn V41Source,
    local: &Local,
    fp8: bool,
) -> Result<Plan> {
    let n_layers = config.num_hidden_layers;
    let (dim, hc) = (config.hidden_size, config.hc_mult);
    ensure!(
        (1..=4).contains(&hc),
        "deepseek-v4.1: hc_mult {hc} outside 1..=4"
    );
    // 2026-10-19: The serve's own bounds, when set, cap the levers: no step carries more rows
    // than `serve_max_batch_tokens` and no position reaches `serve_max_seq_len`.
    let cap = |lever: usize, serve: usize| if serve > 0 { lever.min(serve) } else { lever };
    let max_seq =
        cap(levers.max_seq, config.serve_max_seq_len).min(config.max_position_embeddings.max(1));
    let max_tokens = cap(levers.max_tokens, config.serve_max_batch_tokens).min(max_seq);
    let ratios: Vec<usize> = config
        .compress_ratios
        .iter()
        .take(n_layers)
        .copied()
        .collect();
    ensure!(
        ratios.len() == n_layers,
        "compress_ratios has {} entries for {n_layers} layers",
        ratios.len()
    );
    let kv_sources: Vec<usize> = (0..n_layers).filter(|&l| src.has(l, T::CompWkv)).collect();
    let index_sources: Vec<usize> = (0..n_layers).filter(|&l| src.has(l, T::IdxWqB)).collect();
    ensure!(
        !kv_sources.is_empty() && !index_sources.is_empty(),
        "deepseek-v4.1: no compressor / indexer tensors found"
    );
    let or = |v: usize, d: usize| if v > 0 { v } else { d };
    let orf = |v: f32, d: f32| if v > 0.0 { v } else { d };
    let mut attn_cfg = AttnV41Cfg {
        dim,
        n_heads: local.heads.len(),
        head_dim: config.head_dim,
        rope_dim: config.rotary_dim,
        q_rank: config.q_lora_rank,
        o_rank: config.o_lora_rank,
        groups: local.groups.len(),
        window: config.sliding_window as usize,
        eps: config.rms_norm_eps as f32,
        index_heads: config.index_n_heads,
        index_hd: config.index_head_dim,
        index_topk: config.index_topk,
        cand_topk_blocks: or(config.candidate_topk_blocks, DEFAULT_CANDIDATE_TOPK_BLOCKS),
        cand_block: or(config.candidate_block_size, DEFAULT_CANDIDATE_BLOCK),
        max_seq,
        max_tokens,
        rope_theta: config.rope_theta as f32,
        compress_rope_theta: config.compress_rope_theta,
        rope_factor: orf(config.yarn_factor, 16.0),
        orig_seq: or(config.yarn_original_max_position_embeddings, 65536),
        beta_fast: orf(config.yarn_beta_fast, 32.0),
        beta_slow: orf(config.yarn_beta_slow, 1.0),
        fp8_stage: Fp8StageSize::NONE,
        tp_world: local.world,
    };
    let mut moe_cfg = MoeV41Cfg {
        dim,
        inter: local.inter.len(),
        n_routed: config.num_experts,
        topk: config.num_experts_per_tok,
        gate_temp: 1.0,
        norm_topk_prob: config.norm_topk_prob,
        route_scale: config.routed_scaling_factor as f32,
        swiglu_limit: config.swiglu_limit,
        max_tokens,
        fp8_stage: Fp8StageSize::NONE,
    };
    if fp8 {
        attn_cfg.fp8_stage = attn_fp8_stage(&attn_cfg);
        moe_cfg.fp8_stage = moe_fp8_stage(&moe_cfg);
    }
    Ok(Plan {
        n_layers,
        ratios,
        kv_sources,
        index_sources,
        cand_src: config
            .candidate_source_layer_id
            .or(levers.candidate_source)
            .unwrap_or(DEFAULT_CANDIDATE_SOURCE),
        max_seq,
        max_tokens,
        attn_cfg,
        moe_cfg,
    })
}

/// 2026-10-19: The engram hash tables from the config and `token_map`.
pub fn tables(config: &ModelConfig, token_map: Vec<i64>) -> Result<Arc<EngramHashTables>> {
    Ok(Arc::new(EngramHashTables::from_flat(
        config.engram_layer_ids.clone(),
        config.engram_max_ngram_size,
        config.engram_n_heads,
        config.engram_pad_token_id,
        token_map,
        &config.engram_multipliers,
        &config.engram_primes,
        &config.engram_offsets,
    )?))
}

/// 2026-10-19: The engram engine with every engram layer's projections.
pub fn engram(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    src: &dyn V41Source,
    cols: usize,
    max_tokens: usize,
) -> Result<EngramV41> {
    let (dim, hc) = (config.hidden_size, config.hc_mult);
    let mut engram = EngramV41::new(
        gpu,
        dim,
        hc,
        config.engram_head_dim,
        cols,
        config.rms_norm_eps as f32,
        max_tokens,
    )?;
    for &l in &config.engram_layer_ids {
        let q = src.host_f32(gpu, l, T::EngramQ)?;
        let k = src.host_f32(gpu, l, T::EngramK)?;
        ensure!(
            q.len() == hc * dim && k.len() == hc * dim,
            "engram q/k of layer {l}: {} / {} elements, expected {}",
            q.len(),
            k.len(),
            hc * dim
        );
        let w = src.engram_wkv(gpu, l)?;
        engram.add_layer(
            gpu,
            EngramLayerWeights {
                layer: l,
                wkv: w.wkv,
                qk: EngramV41::upload_qk(gpu, &q, &k)?,
                raw: DevicePtr(0),
                rows: DevicePtr(0),
                wkv_q2k: w.wkv_q2k,
                qdq_rows: w.qdq_rows,
            },
        )?;
    }
    Ok(engram)
}

/// 2026-10-19: The engines and the format's expert and engram row sources.
pub struct Parts {
    pub attn: AttnV41,
    pub moe: MoeV41,
    pub engram: EngramV41,
    pub experts: V41Experts,
    pub rows: V41EngramRows,
    pub tables: Arc<EngramHashTables>,
}

/// 2026-10-19: The runtime every layer shares.
pub fn runtime(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    levers: &Ds41LoadLevers,
    plan: &Plan,
    parts: Parts,
) -> Result<Arc<V41Runtime>> {
    let (dim, hc, max_tokens) = (config.hidden_size, config.hc_mult, plan.max_tokens);
    let alloc_f32 = |n: usize| gpu.alloc((n * 4).max(16));
    Ok(Arc::new(V41Runtime {
        levers: levers.run,
        attn: Mutex::new(parts.attn),
        moe: Mutex::new(parts.moe),
        engram: Mutex::new(parts.engram),
        experts: parts.experts,
        rows: parts.rows,
        hasher: Mutex::new(EngramHasher::new(parts.tables.clone(), plan.max_seq)),
        tables: parts.tables,
        shared: Mutex::new(SharedV41::default()),
        step_hashes: Mutex::new(None),
        pre_prev: alloc_f32(max_tokens * hc)?,
        mixes_s: alloc_f32(max_tokens * (2 + hc) * hc)?,
        reader_threads: levers.reader_threads,
        n_layers: plan.n_layers,
        hc_mult: hc,
        hidden: dim,
        sinkhorn_iters: config.hc_sinkhorn_iters.max(1),
        hc_eps: config.hc_eps,
        norm_eps: config.rms_norm_eps as f32,
        seg: Mutex::new(SegState::default()),
        roles: Mutex::new(vec![None; plan.n_layers]),
        engram_layers: Mutex::new(Vec::new()),
        pre_a: alloc_f32(max_tokens * hc)?,
        pre_f: alloc_f32(max_tokens * hc)?,
        post_s: alloc_f32(max_tokens * hc)?,
        comb_s: alloc_f32(max_tokens * hc * hc)?,
        attn_in: gpu.alloc(max_tokens * dim * 2)?,
        attn_cfg: plan.attn_cfg.clone(),
        moe_cfg: plan.moe_cfg.clone(),
        max_tokens,
        step_moe: Mutex::new(Default::default()),
        step_attn_ms: Mutex::new(0.0),
        step_engram_ms: Mutex::new(0.0),
        step_start: Mutex::new(None),
        graph_disabled: std::sync::atomic::AtomicBool::new(false),
        pred_x: gpu.alloc(3 * dim * 2)?,
        pred_trace: Mutex::new(match &levers.predict_trace {
            Some(path) => {
                tracing::info!("DeepSeek-V4.1: routing prediction trace -> {path}");
                Some(std::io::BufWriter::new(std::fs::File::create(path)?))
            }
            None => None,
        }),
    }))
}

pub(super) fn hc_site(
    gpu: &dyn GpuBackend,
    src: &dyn V41Source,
    l: usize,
    site: HcSite,
    c: &ModelConfig,
) -> Result<HcSiteWeights> {
    let hc = c.hc_mult;
    let mix_hc = (2 + hc) * hc;
    let part = |p: HcPart, n: usize| f32_dev(src, gpu, l, T::Hc(site, p), n);
    Ok(HcSiteWeights {
        hc_fn: part(HcPart::Fn, mix_hc * hc * c.hidden_size)?,
        hc_base: part(HcPart::Base, mix_hc)?,
        hc_scale: part(HcPart::Scale, 3)?,
        lowrank: None,
    })
}

fn attn_weights(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    src: &dyn V41Source,
    plan: &Plan,
    l: usize,
    role: LayerRole,
) -> Result<AttnV41LayerWeights> {
    let (head_dim, dim) = (config.head_dim, config.hidden_size);
    let f32d = |t: T, n: usize| f32_dev(src, gpu, l, t, n);
    let comp = if role.is_kv_source {
        let norm = f32d(T::CompNorm, head_dim)?;
        Some(if role.ratio > 1 {
            CompressorWeightsGpu {
                kv: f32d(T::CompWkv, head_dim * dim)?,
                gate: Some(f32d(T::CompWgate, head_dim * dim)?),
                norm,
            }
        } else {
            CompressorWeightsGpu {
                kv: src.bf16(gpu, l, T::CompWkv)?,
                gate: None,
                norm,
            }
        })
    } else {
        None
    };
    let idx = if role.is_index_source {
        Some(IndexerWeightsGpu {
            wq_b: src.mat(gpu, l, T::IdxWqB)?,
            weights_proj: src.bf16(gpu, l, T::IdxProj)?,
            wk: match role.is_kv_source {
                true => Some(src.bf16(gpu, l, T::IdxWk)?),
                false => None,
            },
            k_norm: match role.is_kv_source {
                true => Some(f32d(T::IdxKNorm, config.index_head_dim)?),
                false => None,
            },
        })
    } else {
        None
    };
    Ok(AttnV41LayerWeights {
        role,
        sink: f32d(T::Sink, plan.attn_cfg.n_heads)?,
        wq_a: src.mat(gpu, l, T::WqA)?,
        q_norm: f32d(T::QNorm, config.q_lora_rank)?,
        wq_b: src.mat(gpu, l, T::WqB)?,
        wkv: src.mat(gpu, l, T::Wkv)?,
        kv_norm: f32d(T::KvNorm, head_dim)?,
        wo_a: src.mat(gpu, l, T::WoA)?,
        wo_b: src.mat(gpu, l, T::WoB)?,
        comp,
        idx,
    })
}

fn gate_bias(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    src: &dyn V41Source,
    l: usize,
) -> Result<Vec<f32>> {
    let b = src.host_f32(gpu, l, T::GateBias)?;
    ensure!(
        b.len() == config.num_experts,
        "layer {l}: correction bias has {} entries",
        b.len()
    );
    Ok(b)
}

/// 2026-10-19: Every layer, reading its tensors through `src`.
pub fn assemble(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    src: &dyn V41Source,
    plan: &Plan,
    rt: &Arc<V41Runtime>,
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let k_hc_expand = gpu.kernel("hyper_connection", "hc_expand")?;
    let k_hc_post = gpu.kernel("hyper_connection", "hc_post")?;
    let k_mixes_dot = gpu.kernel("hc_v41", "hc_v41_mixes_dot")?;
    let k_mixes_finish = gpu.kernel("hc_v41", "hc_v41_mixes_finish")?;
    let k_collapse = gpu.kernel("hc_v41", "hc_v41_collapse")?;
    let k_finish_collapse = gpu.kernel("hc_v41", "hc_v41_finish_collapse")?;
    let k_collapse_wide = gpu.kernel("hc_v41", "hc_v41_collapse_wide")?;
    let k_post_wide = gpu.kernel("hc_v41", "hc_v41_post_wide")?;
    // 2026-09-25: Every DeepSeek-V4.1 layer norm scales by `w` through
    // `rms_norm_vanilla`; the shared `rms_norm` kernel scales by `1 + w`. The
    // final norm gets the vanilla kernel through `ships_vanilla_norm_weights`.
    let k_rms_norm = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;

    let n_layers = plan.n_layers;
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(n_layers);
    for l in 0..n_layers {
        let role = LayerRole {
            ratio: plan.ratios[l],
            is_kv_source: plan.kv_sources.contains(&l),
            is_index_source: plan.index_sources.contains(&l),
            is_candidate_source: plan.cand_src == l,
            uses_candidates: plan.cand_src < l,
        };
        let attn_w = attn_weights(gpu, config, src, plan, l, role)?;
        let bias = gate_bias(gpu, config, src, l)?;
        let moe_w = MoeV41LayerWeights {
            layer: l as u32,
            gate_w: src.bf16(gpu, l, T::GateW)?,
            gate_bias_dev: MoeV41::upload_bias(gpu, &bias)?,
            gate_bias: bias,
            shared_w1: src.mat(gpu, l, T::SharedW1)?,
            shared_w2: src.mat(gpu, l, T::SharedW2)?,
            shared_w3: src.mat(gpu, l, T::SharedW3)?,
        };
        let next_router = if l + 1 < n_layers {
            Some(RouterWeights {
                layer: (l + 1) as u32,
                gate_w: src.bf16(gpu, l + 1, T::GateW)?,
                gate_bias: gate_bias(gpu, config, src, l + 1)?,
            })
        } else {
            None
        };
        let engram_index = rt.tables.hash_index(l);
        rt.roles.lock().unwrap()[l] = Some(role);
        if let Some(hi) = engram_index {
            rt.engram_layers.lock().unwrap().push((l, hi));
        }
        layers.push(Box::new(DeepSeekV41Layer {
            idx: l,
            role,
            rt: rt.clone(),
            attn_w,
            moe_w,
            next_router,
            engram_index,
            hc_attn: hc_site(gpu, src, l, HcSite::Attn, config)?,
            hc_ffn: hc_site(gpu, src, l, HcSite::Ffn, config)?,
            attn_norm: DenseWeight {
                weight: src.bf16(gpu, l, T::AttnNorm)?,
            },
            ffn_norm: DenseWeight {
                weight: src.bf16(gpu, l, T::FfnNorm)?,
            },
            k_hc_expand,
            k_hc_post,
            k_mixes_dot,
            k_mixes_finish,
            k_collapse,
            k_finish_collapse,
            k_collapse_wide,
            k_post_wide,
            k_rms_norm,
        }));
    }
    Ok(layers)
}

/// 2026-09-25: `MemAvailable` from `/proc/meminfo` in GiB, or "unknown".
pub fn mem_available() -> String {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("MemAvailable:"))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<f64>().ok())
        })
        .map(|kb| format!("{:.1} GiB", kb / 1048576.0))
        .unwrap_or_else(|| "unknown".to_string())
}

#[cfg(test)]
#[path = "build_tests.rs"]
mod tests;
