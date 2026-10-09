// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The fixture the GLM-5.3 row-batch gates share (`glm5next_multi_seq_decode_gate`,
//! `glm5next_verify_spans_gate`): synthetic KDA and DSA mixers on the real kernels, the
//! metadata rows and the forward context. Moved here unchanged from the decode gate, so both
//! gates build their layers from the same draws.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
#![allow(dead_code)]

use anyhow::{Result, bail};
use half::bf16;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::attend::Glm5NextDsaDecodeKernel;
use metrale_model_arch::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWeights, Glm5NextDsaWorkspace,
};
use metrale_model_arch::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use metrale_model_arch::glm5next_kda::{
    Glm5NextKdaConfig, Glm5NextKdaKernels, Glm5NextKdaLayer, Glm5NextKdaWeights,
    Glm5NextKdaWorkspace,
};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, MoeLoraRoute};
use metrale_model_layers::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use metrale_model_layers::weight_map::DenseWeight;

pub const HIDDEN: usize = 2048;
pub const BLOCK: usize = 16;
/// 2026-10-08: Block-table entries per sequence: 6 blocks of 16 hold the longest history (79
/// tokens) plus the decoded one.
pub const MB: usize = 6;
pub const STREAM: u64 = 0;

pub struct Lcg(pub u64);
impl Lcg {
    pub fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    pub fn vec(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| self.next() * scale).collect()
    }
}

pub fn up(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}
pub fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter()
            .flat_map(|x| bf16::from_f32(*x).to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
pub fn up_f32(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn up_i32(gpu: &dyn GpuBackend, v: &[i32]) -> Result<DevicePtr> {
    up(
        gpu,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
}
pub fn read(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    gpu.synchronize(STREAM)?;
    let mut b = vec![0u8; n];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b)
}
pub fn copy(gpu: &dyn GpuBackend, src: DevicePtr, n: usize) -> Result<DevicePtr> {
    let p = gpu.alloc(n)?;
    gpu.copy_d2d(src, p, n)?;
    Ok(p)
}

pub fn same(what: &str, a: &[u8], b: &[u8]) -> Result<()> {
    if a != b {
        let first = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
        bail!("{what}: batched differs from single-sequence at byte {first}");
    }
    Ok(())
}

/// 2026-10-09: The synthetic KDA mixer (8 heads of 128) and a 16-row workspace.
pub fn kda_fixture(
    gpu: &dyn GpuBackend,
    rng: &mut Lcg,
) -> Result<(Glm5NextKdaConfig, Glm5NextKdaLayer, Glm5NextKdaWorkspace)> {
    let cfg = Glm5NextKdaConfig {
        hidden: HIDDEN,
        heads: 8,
        head_dim: 128,
        conv_kernel: 4,
        gate_lower_bound: -5.0,
        rms_norm_eps: 1e-6,
        l2_eps: 1e-6,
        chunk: 32,
    };
    let (qkv, hd) = (cfg.qkv_dim(), cfg.head_dim);
    let w = |rng: &mut Lcg, n: usize| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: up_bf16(gpu, &rng.vec(n, 0.03))?,
        })
    };
    let weights = Glm5NextKdaWeights {
        q_proj: w(rng, qkv * HIDDEN)?,
        k_proj: w(rng, qkv * HIDDEN)?,
        v_proj: w(rng, qkv * HIDDEN)?,
        conv: w(rng, cfg.conv_dim() * cfg.conv_kernel)?,
        f_a: w(rng, hd * HIDDEN)?,
        f_b: w(rng, qkv * hd)?,
        dt_bias: up_f32(gpu, &rng.vec(qkv, 0.5))?,
        a_log: up_f32(gpu, &rng.vec(cfg.heads, 0.5))?,
        b_proj: w(rng, cfg.heads * HIDDEN)?,
        g_a: w(rng, hd * HIDDEN)?,
        g_b: w(rng, qkv * hd)?,
        o_norm: DenseWeight {
            weight: up_bf16(gpu, &vec![1.0; hd])?,
        },
        o_proj: w(rng, HIDDEN * qkv)?,
    };
    let layer = Glm5NextKdaLayer::new(0, cfg, weights, Glm5NextKdaKernels::resolve(gpu)?)?;
    let ws = Glm5NextKdaWorkspace::new(gpu, &cfg, 16)?;
    Ok((cfg, layer, ws))
}

/// 2026-10-09: The synthetic DSA mixer's geometry.
pub fn dsa_cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: HIDDEN,
        index_heads: 8,
        index_head_dim: 128,
        index_kpool: 4,
        // 2026-10-08: 4 pools out of up to 19, so the top-k really selects.
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

pub fn dsa_layer(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextDsaConfig,
    rng: &mut Lcg,
) -> Result<Glm5NextDsaLayer> {
    let (h, ql, kvl, d) = (
        cfg.hidden,
        cfg.q_lora_rank,
        cfg.kv_lora_rank,
        cfg.index_head_dim,
    );
    let lh = cfg.local_heads;
    let ape = up_f32(gpu, &rng.vec(cfg.index_kpool * d, 0.1))?;
    let mut bf = |n: usize, s: f32| up_bf16(gpu, &rng.vec(n, s));
    let weights = Glm5NextDsaWeights {
        q_a_proj: bf(ql * h, 0.03)?,
        q_a_layernorm: up_bf16(gpu, &vec![1.0; ql])?,
        q_absorb: bf(lh * kvl * ql, 0.03)?,
        kv_a_proj: bf(kvl * h, 0.03)?,
        kv_a_layernorm: up_bf16(gpu, &vec![1.0; kvl])?,
        o_absorb: bf(h * lh * kvl, 0.03)?,
        wk: bf(d * h, 0.03)?,
        k_norm_weight: up_bf16(gpu, &vec![1.0; d])?,
        k_norm_bias: up_bf16(gpu, &vec![0.0; d])?,
        compress_gate: bf(d * h, 0.03)?,
        wq_b: bf(cfg.index_heads * d * ql, 0.03)?,
        weights_proj: bf(cfg.index_heads * h, 0.03)?,
        ape,
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
        indexer_cache: metrale_model_arch::glm5next_dsa::paged::IndexerCache::Flat,
    })
}

/// 2026-10-08: Metadata rows for `seqs` (position, slot, seq_len, block-table row), at fixed
/// strides, as `upload_batch_metadata_fixed` lays them out.
pub fn meta(gpu: &dyn GpuBackend, seqs: &[(usize, Vec<u32>)]) -> Result<AttnMetadataDev> {
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

pub struct Fwd {
    pub config: ModelConfig,
    pub buffers: BufferArena,
    pub dispatch: GemmDispatch,
    pub derived: DerivedWeights,
    pub levers: ModelLevers,
    pub stats: ModelStats,
}

impl Fwd {
    pub fn ctx<'a>(
        &'a self,
        gpu: &'a dyn GpuBackend,
        capture: bool,
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

impl Fwd {
    /// 2026-10-09: The mixers read only the GPU, the metadata and the flags from the context.
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        // 2026-10-08: The mixers read only the GPU, the metadata and the flags from the context;
        // the arena exists because `ForwardContext` holds one, at the mock tests' small shape.
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = 128;
        config.intermediate_size = 128;
        config.num_experts = 1;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 128;
        config.vocab_size = 128;
        Ok(Fwd {
            buffers: BufferArena::new(&config, 8, 16, 16, 8, gpu)?,
            config,
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        })
    }
}
