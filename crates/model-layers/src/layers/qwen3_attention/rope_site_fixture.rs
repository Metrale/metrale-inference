// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: Test fixture for the RoPE call sites: an ungated attention layer on the mock
//! backend whose plain and strided RoPE kernels carry distinct handles, so a test can count
//! the RoPE launches a phase issued.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{AttnPositionEncoding, ModelConfig};
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{AttentionWeights, DenseWeight};

const ROPE_K: u64 = 0x0B0E;
const ROPE_STRIDED_K: u64 = 0x0B5E;
pub(crate) const HEAD_DIM: usize = 128;

pub(crate) struct Rig {
    pub(crate) gpu: MockGpuBackend,
    pub(crate) config: ModelConfig,
    pub(crate) buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Rig {
    /// 2026-09-29: One Q head and one KV head of `HEAD_DIM`, like a Nemotron-H attention
    /// layer shrunk to one head, under `encoding`.
    pub(crate) fn new(encoding: AttnPositionEncoding) -> Self {
        let gpu = MockGpuBackend::new();
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.hidden_size = HEAD_DIM;
        config.intermediate_size = 128;
        config.num_attention_heads = 1;
        config.num_key_value_heads = 1;
        config.head_dim = HEAD_DIM;
        config.partial_rotary_factor = 1.0;
        config.num_experts = 1;
        config.num_experts_per_tok = 1;
        config.moe_intermediate_size = 128;
        config.vocab_size = 128;
        config.attn_position_encoding = Some(encoding);
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

    /// 2026-09-29: The layer the Nemotron-H loader builds (`new_ungated`, no q/k norm).
    pub(crate) fn layer(&self) -> Qwen3AttentionLayer {
        let dense = DenseWeight {
            weight: self.gpu.alloc(HEAD_DIM * HEAD_DIM * 2).unwrap(),
        };
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };
        let attn = AttentionWeights {
            q_proj: dense,
            k_proj: dense,
            v_proj: dense,
            o_proj: crate::weight_map::QuantizedWeight::null(),
            q_norm: null,
            k_norm: null,
            q_norm_full: None,
            k_norm_full: None,
            k_scale: 1.0,
            v_scale: 1.0,
        };
        let mut layer = Qwen3AttentionLayer::new_ungated(
            dense,
            attn,
            null,
            FfnComponent::None,
            0,
            None,
            None,
            None,
            &self.gpu,
            KvCacheDtype::Bf16,
            0,
            &self.config,
        )
        .unwrap();
        layer.rope_k = KernelHandle(ROPE_K);
        layer.rope_strided_k = KernelHandle(ROPE_STRIDED_K);
        layer
    }

    pub(crate) fn fwd(&self) -> ForwardContext<'_> {
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
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }

    /// 2026-09-29: Metadata whose position, slot and length buffers hold `rows` entries.
    pub(crate) fn meta(&self, rows: usize) -> AttnMetadataDev {
        let positions = self.gpu.alloc(rows * 4).unwrap();
        AttnMetadataDev {
            positions,
            positions_h: positions,
            positions_w: positions,
            slot: self.gpu.alloc(rows * 8).unwrap(),
            seq_len: self.gpu.alloc(rows * 4).unwrap(),
            block_table: self.gpu.alloc(rows * 4).unwrap(),
            max_blocks_per_seq: 1,
            num_seqs: rows as u32,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    }

    /// 2026-09-29: RoPE launches (plain or strided) recorded since launch index `from`.
    pub(crate) fn rope_launches_since(&self, from: usize) -> usize {
        self.gpu.launches_snapshot()[from..]
            .iter()
            .filter(|l| l.func == ROPE_K || l.func == ROPE_STRIDED_K)
            .count()
    }
}
