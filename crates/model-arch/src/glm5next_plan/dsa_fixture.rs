// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The served-shape DSA mixers of the CPU launch gate (`dsa_tests.rs`), moved out of
//! it (exact copy, items made `pub(crate)`) to keep both files under 500 lines: each rank's DSA
//! stack as `DsaStack::new` builds it at load (the recipe's levers, settings and context), its
//! layers with their projections on the served tiers, a paged KV cache whose V blocks keep the
//! persisted BF16 pool keys, row metadata, and one captured pass from a plan or `decode_k`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::Arc;

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_circuit::Circuit;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::recording::{Op, RecordingGpu};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::circuit_exec::family::RankPlans;
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::layers::{DenseQuantization, DsaQuantization};

use crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use crate::glm5next_dsa::fp8_rows::IndexerRowFormat;
use crate::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use crate::glm5next_dsa::levers::DsaOwnedLevers;
use crate::glm5next_dsa::plan::{DsaPlan, InstalledDsa, entry};
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::glm5next_layer::GlmLevers;
use crate::glm5next_plan::dsa;
use crate::glm5next_plan::fixture::{
    in_child_process, model_config, recipe_default, served_settings,
};
use crate::glm5next_plan::fixture_site::STREAM;

pub(crate) const HIDDEN_IN: DevicePtr = DevicePtr(0x9300_0000);
/// 2026-10-23: The sequence length the passes run at, and the KV block size.
pub(crate) const SEQ_LEN: usize = 70_001;
pub(crate) const BLOCK: usize = 16;

/// 2026-10-23: Run `body` in a child process that publishes the served tiers first.
pub(crate) fn in_served_process(name: &str, body: impl FnOnce()) {
    let publish = || {
        use metrale_model_layers::layers::{
            set_dense_quantization_from_cli, set_dsa_quantization_from_cli,
        };
        assert_eq!(
            set_dense_quantization_from_cli(DenseQuantization::W4a16),
            DenseQuantization::W4a16
        );
        assert_eq!(
            set_dsa_quantization_from_cli(DsaQuantization::W8a8),
            DsaQuantization::W8a8
        );
    };
    in_child_process(&format!("glm5next_plan::dsa::tests::{name}"), publish, body);
}

/// 2026-10-23: The DSA layers of `c`.
pub(crate) fn dsa_layers(c: &Circuit) -> Vec<usize> {
    let mut v: Vec<usize> = c
        .nodes
        .iter()
        .filter(|n| n.block == "dsa")
        .filter_map(|n| n.layer)
        .collect();
    v.dedup();
    v
}

/// 2026-10-23: Rank `rank`'s model config at the recipe's context (`max_model_len`).
pub(crate) fn served_config(rank: usize) -> metrale_config::ModelConfig {
    let mut c = model_config(rank);
    c.serve_max_seq_len = recipe_default("max_model_len").parse().unwrap();
    c
}

/// 2026-10-23: What one rank's DSA layers share, as `DsaStack::new` builds it at load.
pub(crate) struct Stack {
    config: metrale_config::ModelConfig,
    cfg: Glm5NextDsaConfig,
    kernels: Glm5NextDsaKernels,
    layer_kernels: Glm5NextDsaLayerKernels,
    decode: Glm5NextDsaDecodeKernel,
    owned: Arc<DsaOwnedLevers>,
    ws: Arc<Glm5NextDsaWorkspace>,
    format: IndexerRowFormat,
    cache: crate::glm5next_dsa::paged::IndexerCache,
    levers: GlmLevers,
}

pub(crate) fn stack(gpu: &RecordingGpu, rank: usize, levers: GlmLevers) -> Stack {
    let config = served_config(rank);
    let s = served_settings();
    // 2026-10-23: The served stack's KDA layers load the W4A16 tier (its SM count) before any
    // projection runs; a DSA layer alone registers only W8A8 projections.
    crate::glm5next_w4a16_dense::Nvfp4QuantKernels::load(gpu, &levers.dense).unwrap();
    let cfg = Glm5NextDsaConfig::from_config(&config).unwrap();
    let kernels = Glm5NextDsaKernels::resolve(gpu).unwrap();
    let cache = crate::glm5next_dsa::paged::text_stack_indexer_cache(&s);
    let format = IndexerRowFormat::text_stack(
        &config,
        &kernels.fp8,
        cache,
        s.kernels.glm_dsa_pool_keys,
        kernels.rows.pool_keys_ready(),
    )
    .unwrap();
    // 2026-10-23: The loader's `verify_k` (glm5_next_load/loader.rs).
    let verify_k = (metrale_model_layers::layers::ops::DENSE_GEMV_BATCHM_MAX_M as usize)
        .max(crate::glm5next_layer::PREFILL_ROWS)
        .max(levers.prefill_rows)
        .max(levers.row_group);
    Stack {
        decode: Glm5NextDsaDecodeKernel::resolve(gpu, levers.dsa.decode_hb).unwrap(),
        layer_kernels: Glm5NextDsaLayerKernels::resolve(gpu).unwrap(),
        owned: Arc::new(DsaOwnedLevers::from_settings(&s).unwrap()),
        ws: Arc::new(Glm5NextDsaWorkspace::new(gpu, &cfg, verify_k, &levers).unwrap()),
        config,
        cfg,
        kernels,
        format,
        cache,
        levers,
    }
}

/// 2026-10-23: DSA layer `idx` (`attn` among the DSA layers) of `st`, its projections registered
/// on the served tiers as the loader registers them.
pub(crate) fn layer(gpu: &RecordingGpu, st: &Stack, idx: usize, attn: usize) -> Glm5NextDsaLayer {
    let b = || gpu.alloc(64).unwrap();
    let mut w = Glm5NextDsaWeights {
        q_a_proj: b(),
        q_a_layernorm: b(),
        q_absorb: b(),
        kv_a_proj: b(),
        kv_a_layernorm: b(),
        o_absorb: b(),
        wk: b(),
        k_norm_weight: b(),
        k_norm_bias: b(),
        compress_gate: b(),
        wq_b: b(),
        weights_proj: b(),
        ape: b(),
    };
    let mut l = Glm5NextDsaLayer {
        cfg: st.cfg,
        weights: view(&w),
        kernels: st.layer_kernels,
        select_kernels: st.kernels,
        decode_kernel: st.decode,
        workspace: st.ws.clone(),
        layer_idx: idx,
        attn_layer_idx: attn,
        rms_eps: st.config.rms_norm_eps as f32,
        kv_scale: 1.0,
        persist_bt: !st.levers.dsa_alloc_per_step,
        indexer_cache: st.cache,
        indexer_format: st.format,
        levers: st.levers,
        dsa_owned: st.owned.clone(),
    };
    let projs = l.dense_projections();
    {
        let mut slots = [
            &mut w.q_a_proj,
            &mut w.q_absorb,
            &mut w.kv_a_proj,
            &mut w.o_absorb,
            &mut w.wk,
            &mut w.compress_gate,
        ];
        crate::weight_loader::glm5_next_load::dense_tiers::register_projections(
            gpu,
            projs,
            &mut slots,
            DenseQuantization::W4a16,
            DsaQuantization::W8a8,
            &st.levers.dense,
            idx,
        )
        .unwrap();
    }
    l.weights = w;
    gpu.take();
    l
}

pub(crate) fn view(w: &Glm5NextDsaWeights) -> Glm5NextDsaWeights {
    Glm5NextDsaWeights {
        q_a_proj: w.q_a_proj,
        q_a_layernorm: w.q_a_layernorm,
        q_absorb: w.q_absorb,
        kv_a_proj: w.kv_a_proj,
        kv_a_layernorm: w.kv_a_layernorm,
        o_absorb: w.o_absorb,
        wk: w.wk,
        k_norm_weight: w.k_norm_weight,
        k_norm_bias: w.k_norm_bias,
        compress_gate: w.compress_gate,
        wq_b: w.wq_b,
        weights_proj: w.weights_proj,
        ape: w.ape,
    }
}

/// 2026-10-23: What a pass's forward context borrows.
pub(crate) struct Env {
    config: metrale_config::ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    model_levers: ModelLevers,
    settings: metrale_settings::InferenceServerSettings,
    stats: ModelStats,
}

/// 2026-10-23: The forward context of one pass, and the KV cache and metadata it runs over.
pub(crate) struct Rig {
    env: Env,
    kv: PagedKvCache,
    meta: [AttnMetadataDev; 2],
}

pub(crate) fn rig(gpu: &RecordingGpu, st: &Stack) -> Rig {
    let mut small = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    small.hidden_size = 128;
    small.intermediate_size = 128;
    small.num_experts = 1;
    small.num_experts_per_tok = 1;
    small.moe_intermediate_size = 128;
    small.vocab_size = 128;
    let v_block_bytes = IndexerRowFormat::text_stack_v_block_bytes(
        &st.config,
        BLOCK,
        st.cache,
        served_settings().kernels.glm_dsa_pool_keys,
    )
    .unwrap();
    let kv = PagedKvCache::new(
        KvCacheConfig {
            block_size: BLOCK,
            num_kv_heads: 1,
            head_dim: st.cfg.kv_lora_rank,
            num_layers: 12,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
            v_block_bytes,
        },
        8,
        gpu,
    )
    .unwrap();
    let mb = crate::glm5next_dsa::state::max_dsa_context(&st.cfg).div_ceil(BLOCK);
    let meta = |rows: usize| {
        let positions = gpu.alloc(rows * 4).unwrap();
        AttnMetadataDev {
            positions,
            positions_h: positions,
            positions_w: positions,
            slot: gpu.alloc(rows * 8).unwrap(),
            seq_len: gpu.alloc(rows * 4).unwrap(),
            block_table: gpu.alloc(rows * mb * 4).unwrap(),
            max_blocks_per_seq: mb as u32,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    };
    let r = Rig {
        env: Env {
            buffers: BufferArena::new(&small, 8, 16, 16, 8, gpu).unwrap(),
            config: st.config.clone(),
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            model_levers: ModelLevers::defaults(),
            settings: served_settings(),
            stats: ModelStats::new(),
        },
        kv,
        meta: [meta(1), meta(2)],
    };
    gpu.take();
    r
}

impl Env {
    pub(crate) fn ctx<'a>(
        &'a self,
        gpu: &'a RecordingGpu,
        meta: AttnMetadataDev,
        capture: bool,
    ) -> ForwardContext<'a> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.model_levers,
            settings: &self.settings,
            stats: &self.stats,
            attn_metadata: Some(meta),
            profile: false,
            comm: None,
            graph_capture: capture,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

impl Rig {
    pub(crate) fn ctx<'a>(
        &'a self,
        gpu: &'a RecordingGpu,
        rows: usize,
        capture: bool,
    ) -> ForwardContext<'a> {
        self.env.ctx(gpu, self.meta[rows - 1], capture)
    }

    /// 2026-10-23: One captured pass of `rows` rows over a fresh sequence state at [`SEQ_LEN`]:
    /// from `plan`, or `decode_k`.
    pub(crate) fn run(
        &mut self,
        gpu: &RecordingGpu,
        l: &Glm5NextDsaLayer,
        rows: usize,
        plan: Option<&DsaPlan>,
    ) -> anyhow::Result<Vec<Op>> {
        let mut state: Box<dyn LayerState> = Box::new(l.alloc_dsa_state(gpu)?);
        let ctx = self.env.ctx(gpu, self.meta[rows - 1], true);
        let kv = &mut self.kv;
        let r = match plan {
            Some(p) => l.decode_k_plan(
                HIDDEN_IN,
                rows,
                state.as_mut(),
                kv,
                SEQ_LEN,
                &ctx,
                false,
                p,
                STREAM,
            ),
            None => l.decode_k(
                HIDDEN_IN,
                rows,
                state.as_mut(),
                kv,
                SEQ_LEN,
                &mut Vec::new(),
                &ctx,
                STREAM,
                false,
            ),
        };
        // 2026-10-23: The log is emptied whether the pass ran or was refused.
        let ops = gpu.take();
        r.map(|()| ops)
    }
}

/// 2026-10-23: Layer `l`'s DSA plans installed from `plans` (rank circuit `c`).
pub(crate) fn installed(
    gpu: &RecordingGpu,
    c: &Circuit,
    plans: &[metrale_circuit::FusionPlan],
    policy: &metrale_circuit::Policy,
    layer: &Glm5NextDsaLayer,
) -> anyhow::Result<(InstalledDsa, Vec<String>)> {
    let inst = InstalledDsa::default();
    let rp = RankPlans {
        circuit: c,
        plans,
        policy,
    };
    let sites = dsa::install(gpu, layer.layer_idx, Some(rp), layer, &inst)?;
    Ok((inst, sites))
}

pub(crate) fn kernels(ops: &[Op]) -> Vec<String> {
    ops.iter()
        .filter_map(|o| o.kernel().map(str::to_string))
        .collect()
}

pub(crate) fn own(e: entry::Entry) -> String {
    format!("{}::{}", e.0, e.1)
}

/// 2026-10-23: The plan's own launches of a decode row (`rows` 1) or a two-row verify, in order.
pub(crate) fn own_launches(rows: usize) -> Vec<String> {
    let select = |scores| {
        let mut v = vec![own(scores)];
        v.extend(std::iter::repeat_n(own(entry::TOPK_MC_PASS), 4));
        v.extend([own(entry::TOPK_MC_COLLECT), own(entry::EXPAND_ROWS)]);
        v
    };
    let attend = [own(entry::DECODE_HB), own(entry::HB_MERGE)];
    let place = [
        own(entry::STORE),
        own(entry::POOL_KEYS),
        own(entry::WRITE_GEOM),
    ];
    let mut v = Vec::new();
    if rows == 1 {
        v.extend([own(entry::LATENT_WRITE), own(entry::K_NORM)]);
        v.extend(place.clone());
        v.extend(select(entry::SCORES_HEADS_PAGED));
    } else {
        v.push(own(entry::K_NORM));
        for _ in 0..rows {
            v.push(own(entry::LATENT_WRITE));
            v.extend(place.clone());
        }
        v.extend(select(entry::SCORES_HEADS_WINDOW));
    }
    v.extend(attend);
    v
}
