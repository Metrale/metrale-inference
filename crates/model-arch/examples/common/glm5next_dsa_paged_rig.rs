// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The rig of `glm5next_dsa_paged_parity`: synthetic DSA layer, KV caches,
//! metadata rows and the forward context.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use half::bf16;
use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use metrale_model_arch::glm5next_dsa::paged::IndexerCache;
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};

pub(crate) const HIDDEN: usize = 2048;
pub(crate) const BLOCK: usize = 16;
/// 2026-10-09: Blocks per sequence: 160 tokens, 40 pools against a 4-pool budget.
pub(crate) const MB: usize = 10;
pub(crate) const LEN: usize = MB * BLOCK - 8;
pub(crate) const STREAM: u64 = 0;

pub(crate) struct Lcg(pub(crate) u64);
impl Lcg {
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub(crate) fn bf16_bytes(&mut self, n: usize, scale: f32) -> Vec<u8> {
        (0..n)
            .flat_map(|_| bf16::from_f32(self.next() * scale).to_le_bytes())
            .collect()
    }
}

pub(crate) fn up(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
pub(crate) fn up_i32(gpu: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub(crate) fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(STREAM)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}
pub(crate) fn same(what: &str, a: &[u8], b: &[u8]) -> Result<()> {
    if a != b {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
        bail!("{what}: paged differs from flat at byte {first}");
    }
    Ok(())
}

pub(crate) fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: HIDDEN,
        index_heads: 8,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 16,
        always_select_tail: true,
        local_heads: 4,
        q_lora_rank: 512,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: MB * BLOCK,
    }
}

pub(crate) fn layer(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    rng: &mut Lcg,
) -> Result<Glm5NextDsaLayer> {
    let (h, ql, kvl, d, lh) = (
        cfg.hidden,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.index_head_dim,
        cfg.local_heads,
    );
    let ape: Vec<u8> = (0..cfg.index_kpool * d)
        .flat_map(|_| (rng.next() * 0.1).to_le_bytes())
        .collect();
    let ones = |n: usize| up(gpu, &bf16::from_f32(1.0).to_le_bytes().repeat(n));
    let mut bf = |n: usize, s: f32| up(gpu, &rng.bf16_bytes(n, s));
    let weights = Glm5NextDsaWeights {
        q_a_proj: bf(ql * h, 0.03)?,
        q_a_layernorm: ones(ql)?,
        q_absorb: bf(lh * kvl * ql, 0.03)?,
        kv_a_proj: bf(kvl * h, 0.03)?,
        kv_a_layernorm: ones(kvl)?,
        o_absorb: bf(h * lh * kvl, 0.03)?,
        wk: bf(d * h, 0.03)?,
        k_norm_weight: ones(d)?,
        k_norm_bias: up(gpu, &vec![0u8; d * 2])?,
        compress_gate: bf(d * h, 0.03)?,
        wq_b: bf(cfg.index_heads * d * ql, 0.03)?,
        weights_proj: bf(cfg.index_heads * h, 0.03)?,
        ape: up(gpu, &ape)?,
    };
    Ok(Glm5NextDsaLayer {
        cfg: *cfg,
        weights,
        kernels: Glm5NextDsaLayerKernels::resolve(gpu)?,
        select_kernels: Glm5NextDsaKernels::resolve(gpu)?,
        decode_kernel: Glm5NextDsaDecodeKernel::resolve(gpu)?,
        workspace: Glm5NextDsaWorkspace::new(gpu, cfg, 16)?,
        layer_idx: 1,
        attn_layer_idx: 0,
        rms_eps: 1e-6,
        kv_scale: 1.0,
        persist_bt: true,
        // 2026-10-09: Only `alloc_dsa_state` reads it; every state here is built explicitly.
        indexer_cache: IndexerCache::Paged,
    })
}

pub(crate) fn kv(gpu: &dyn GpuBackend, blocks: usize) -> Result<PagedKvCache> {
    PagedKvCache::new(
        KvCacheConfig {
            block_size: BLOCK,
            num_kv_heads: 1,
            head_dim: 512,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        },
        blocks,
        gpu,
    )
}

/// 2026-10-09: One metadata row per `(position, block table)`, as the serve lays them out.
pub(crate) fn meta(gpu: &dyn GpuBackend, seqs: &[(usize, &[u32])]) -> Result<AttnMetadataDev> {
    let positions: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32).collect();
    let slots: Vec<u8> = seqs
        .iter()
        .flat_map(|(l, bt)| ((bt[l / BLOCK] as usize * BLOCK + l % BLOCK) as i64).to_le_bytes())
        .collect();
    let lens: Vec<i32> = seqs.iter().map(|(l, _)| *l as i32 + 1).collect();
    let bts: Vec<i32> = seqs
        .iter()
        .flat_map(|(_, bt)| bt.iter().map(|b| *b as i32))
        .collect();
    let p = up_i32(gpu, &positions)?;
    Ok(AttnMetadataDev {
        positions: p,
        positions_h: p,
        positions_w: p,
        slot: up(gpu, &slots)?,
        seq_len: up_i32(gpu, &lens)?,
        block_table: up_i32(gpu, &bts)?,
        max_blocks_per_seq: MB as u32,
        num_seqs: seqs.len() as u32,
        seq_slot: DevicePtr::NULL,
        moe_row_adapter: DevicePtr::NULL,
    })
}

pub(crate) struct Fwd {
    pub(crate) config: ModelConfig,
    pub(crate) buffers: BufferArena,
    pub(crate) dispatch: GemmDispatch,
    pub(crate) derived: DerivedWeights,
    pub(crate) levers: ModelLevers,
    pub(crate) stats: ModelStats,
}

impl Fwd {
    pub(crate) fn ctx<'a>(
        &'a self,
        gpu: &'a dyn GpuBackend,
        capture: bool,
        decode_step: bool,
        meta: Option<AttnMetadataDev>,
    ) -> ForwardContext<'a> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: meta,
            profile: false,
            comm: None,
            graph_capture: capture,
            decode_step,
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
