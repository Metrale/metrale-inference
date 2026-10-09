// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The mock-backend rig of the `decode_rows` tests: a small DSA layer whose
//! kernels carry distinct handles, a paged FP8 latent cache, per-row metadata and indexer
//! caches at chosen lengths, and launch filters.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, LayerState, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

use crate::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use crate::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use crate::glm5next_dsa::state::Glm5NextDsaState;
use crate::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};

/// 2026-10-08: Distinct handles, so a launch names its kernel. The decode kernel resolves
/// through the mock to `MOCK_K`.
pub(super) const BATCHM: u64 = 0x105;
pub(super) const LATENT: u64 = 0x107;
pub(super) const KPOOL: u64 = 0x201;
pub(super) const EXPAND: u64 = 0x205;
pub(super) const KNORM: u64 = 0x206;
pub(super) const GEOM: u64 = 0x207;
pub(super) const STORE: u64 = 0x208;
pub(super) const MOCK_K: u64 = 0xDEAD;
/// 2026-10-08: Block-table entries per metadata row.
pub(super) const MB: usize = 4;
pub(super) const HIDDEN: usize = 64;

pub(super) fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: HIDDEN,
        index_heads: 2,
        index_head_dim: 16,
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        local_heads: 1,
        q_lora_rank: 32,
        kv_lora_rank: 512,
        qk_nope_head_dim: 16,
        qk_rope_head_dim: 0,
        v_head_dim: 16,
        max_context: 64,
    }
}

pub(super) struct Rig {
    pub(super) gpu: MockGpuBackend,
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Rig {
    pub(super) fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = 128;
        config.intermediate_size = 128;
        config.num_experts = 1;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 128;
        config.vocab_size = 128;
        let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
        Self {
            gpu,
            config,
            buffers,
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        }
    }

    pub(super) fn ctx(&self, graph_capture: bool) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: &self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture,
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

    pub(super) fn buf(&self, bytes: usize) -> DevicePtr {
        self.gpu.alloc(bytes.max(1)).unwrap()
    }

    pub(super) fn layer(&self) -> Glm5NextDsaLayer {
        let c = cfg();
        let k = KernelHandle;
        Glm5NextDsaLayer {
            cfg: c,
            weights: Glm5NextDsaWeights {
                q_a_proj: self.buf(64),
                q_a_layernorm: self.buf(64),
                q_absorb: self.buf(64),
                kv_a_proj: self.buf(64),
                kv_a_layernorm: self.buf(64),
                o_absorb: self.buf(64),
                wk: self.buf(64),
                k_norm_weight: self.buf(64),
                k_norm_bias: self.buf(64),
                compress_gate: self.buf(64),
                wq_b: self.buf(64),
                weights_proj: self.buf(64),
                ape: self.buf(64),
            },
            kernels: Glm5NextDsaLayerKernels {
                gemm: k(0x101),
                gemm_f32: k(0x102),
                gemv: k(0x103),
                gemv_f32: k(0x104),
                gemv_batchm: k(BATCHM),
                rms_norm: k(0x106),
                latent_write: k(LATENT),
            },
            select_kernels: Glm5NextDsaKernels {
                kpool_compress: k(KPOOL),
                compact_pools: k(0x202),
                index_scores: k(0x203),
                topk_pools: k(0x204),
                expand_selection: k(EXPAND),
                k_norm: k(KNORM),
                write_geom: k(GEOM),
                indexer_store: k(STORE),
                topk_to_mask: k(0x209),
                mla_masked_attn: k(0x20A),
            },
            decode_kernel: Glm5NextDsaDecodeKernel::resolve(&self.gpu).unwrap(),
            workspace: Glm5NextDsaWorkspace::new(&self.gpu, &c, 16).unwrap(),
            layer_idx: 3,
            attn_layer_idx: 0,
            rms_eps: 1e-6,
            kv_scale: 1.0,
            persist_bt: true,
            indexer_cache: crate::glm5next_dsa::paged::IndexerCache::Flat,
        }
    }

    pub(super) fn kv(&self) -> PagedKvCache {
        PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Fp8,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            8,
            &self.gpu,
        )
        .unwrap()
    }

    pub(super) fn meta(&self, rows: usize) -> AttnMetadataDev {
        let positions = self.buf(rows * 4);
        AttnMetadataDev {
            positions,
            positions_h: positions,
            positions_w: positions,
            slot: self.buf(rows * 8),
            seq_len: self.buf(rows * 4),
            block_table: self.buf(rows * MB * 4),
            max_blocks_per_seq: MB as u32,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    }

    /// 2026-10-08: One indexer cache per length, each already holding `len` rows.
    pub(super) fn states(&self, lens: &[usize]) -> Vec<Box<dyn LayerState>> {
        lens.iter()
            .map(|&len| {
                let mut s = Glm5NextDsaState::alloc(&self.gpu, &cfg()).unwrap();
                s.advance(len).unwrap();
                Box::new(s) as Box<dyn LayerState>
            })
            .collect()
    }

    pub(super) fn since(&self, from: usize) -> Vec<MockLaunch> {
        self.gpu.launches_snapshot()[from..].to_vec()
    }
}

pub(super) fn dsa(s: &dyn LayerState) -> &Glm5NextDsaState {
    s.as_any().downcast_ref::<Glm5NextDsaState>().unwrap()
}

pub(super) fn of(launches: &[MockLaunch], func: u64) -> Vec<MockLaunch> {
    launches
        .iter()
        .filter(|l| l.func == func)
        .cloned()
        .collect()
}

pub(super) fn ptr(p: DevicePtr) -> MockArg {
    MockArg::Buffer(p)
}

/// 2026-10-08: Run `decode_rows` over `boxes`, all of them, from metadata row `base`.
#[allow(clippy::too_many_arguments)]
pub(super) fn run_rows(
    rig: &Rig,
    layer: &Glm5NextDsaLayer,
    boxes: &mut [Box<dyn LayerState>],
    lens: &[usize],
    meta: &AttnMetadataDev,
    base: usize,
    capture: bool,
) -> anyhow::Result<()> {
    let mut kv = rig.kv();
    let hidden = rig.buf(boxes.len() * HIDDEN * 2);
    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
        boxes.iter_mut().map(|b| b.as_mut()).collect();
    layer.decode_rows(
        hidden,
        &mut refs,
        lens,
        &mut kv,
        meta,
        base,
        &rig.ctx(capture),
        7,
    )
}
