// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The test fixture of the executor: the real dense circuit (the checked-in
//! instance, its FUSIONS.toml) fused at a mode and row count, compiled over synthetic bindings,
//! and run on the recording mock backend.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::{Circuit, FusionPlan, LayerKind, LinearRole};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::bindings::*;
use super::compile::{DraftFixed, Fixed};
use super::program::Program;
use crate::layer::AttnMetadataDev;
use crate::weight_map::{DenseWeight, QuantizedWeight};

pub(super) const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
pub(super) const WORKSPACE: u64 = 0xC000_0000;

pub(super) fn ptr(tag: u64) -> DevicePtr {
    DevicePtr(tag)
}

pub(super) fn nvfp4(tag: u64) -> BoundWeight {
    BoundWeight::Nvfp4(QuantizedWeight {
        weight: ptr(tag),
        weight_scale: ptr(tag + 1),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
        act: Default::default(),
    })
}

pub(super) fn dense(tag: u64) -> BoundWeight {
    BoundWeight::Dense(DenseWeight { weight: ptr(tag) })
}

pub(super) struct Fixture {
    pub draft: Option<CircuitLayer>,
    pub circuit: Circuit,
    pub plan: FusionPlan,
    pub program: Program,
    pub arena: u64,
    pub fixed: Fixed,
    pub layers: Vec<CircuitLayer>,
    pub head: HeadBinding,
    /// 2026-09-30: The weight and scale pointers of the W8A8 bindings, which a `BoundWeight`
    /// does not expose.
    pub w8a8_ptrs: BTreeSet<u64>,
}

/// 2026-09-30: Layer `i`'s binding (its attention index `attn`) and the W8A8 pointers in it.
pub(super) type Bind = dyn Fn(&Circuit, usize, usize) -> (CircuitLayer, Vec<u64>);

/// 2026-09-28: Weight tags: layer `i`, slot `s` at `0x1_0000_0000 + i << 24 + s << 12`.
pub(super) fn tag(layer: usize, slot: u64) -> u64 {
    0x1_0000_0000 + ((layer as u64) << 24) + (slot << 12)
}

pub(super) fn layer_binding(circuit: &Circuit, i: usize, attn_idx: usize) -> CircuitLayer {
    let d = |k: &str| circuit.dims[k] as u32;
    let mut w = BTreeMap::from([
        (WeightSlot::InputNorm, dense(tag(i, 1))),
        (WeightSlot::PostNorm, dense(tag(i, 2))),
        (WeightSlot::FfnGate, nvfp4(tag(i, 3))),
        (WeightSlot::FfnUp, nvfp4(tag(i, 4))),
        (WeightSlot::Linear(LinearRole::Down), nvfp4(tag(i, 5))),
        (WeightSlot::FfnGateMmq, BoundWeight::Mmq(ptr(tag(i, 25)))),
        (WeightSlot::FfnUpMmq, BoundWeight::Mmq(ptr(tag(i, 26)))),
        (WeightSlot::FfnDownMmq, BoundWeight::Mmq(ptr(tag(i, 27)))),
    ]);
    let mixer = if circuit.layer_kinds[i] == LayerKind::LinearAttention {
        w.insert(WeightSlot::Linear(LinearRole::Qkvz), nvfp4(tag(i, 6)));
        w.insert(WeightSlot::Linear(LinearRole::Ba), dense(tag(i, 7)));
        w.insert(WeightSlot::GdnALog, dense(tag(i, 8)));
        w.insert(WeightSlot::GdnDtBias, dense(tag(i, 9)));
        w.insert(WeightSlot::GdnConv1d, dense(tag(i, 10)));
        w.insert(WeightSlot::GdnNorm, dense(tag(i, 11)));
        w.insert(WeightSlot::Linear(LinearRole::GdnOut), nvfp4(tag(i, 12)));
        w.insert(WeightSlot::Transposed(LinearRole::Qkvz), nvfp4(tag(i, 19)));
        w.insert(
            WeightSlot::Transposed(LinearRole::GdnOut),
            nvfp4(tag(i, 20)),
        );
        // 2026-10-03: The unscaled E4M3 casts the prefill projections read.
        w.insert(WeightSlot::PrefillCast(LinearRole::Qkvz), dense(tag(i, 28)));
        w.insert(
            WeightSlot::PrefillCast(LinearRole::GdnOut),
            dense(tag(i, 29)),
        );
        MixerFacts::Gdn(GdnFacts {
            qkvz_deinterleaved: true,
            h_slot_bytes: STATE_PITCH,
            conv_state_bytes: STATE_PITCH,
            carry: Some(CARRY),
        })
    } else {
        for (s, role) in [
            (13, LinearRole::Q),
            (14, LinearRole::K),
            (15, LinearRole::V),
            (16, LinearRole::O),
        ] {
            w.insert(WeightSlot::Linear(role), nvfp4(tag(i, s)));
            w.insert(WeightSlot::Transposed(role), nvfp4(tag(i, s + 8)));
        }
        w.insert(WeightSlot::QNorm, dense(tag(i, 17)));
        w.insert(WeightSlot::KNorm, dense(tag(i, 18)));
        MixerFacts::Attention(AttnFacts {
            attn_layer_idx: attn_idx,
            kv_dtype: metrale_cache::kv_cache::KvCacheDtype::Bf16,
            num_q_heads: d("q_heads"),
            num_kv_heads: d("kv_heads"),
            head_dim: d("head_dim"),
            gated: true,
            rope: RopeFacts {
                mrope_interleaved: true,
                theta: 1.0e7,
                rotary_dim: 64,
            },
            sliding_window: 0,
            softmax_scale: 0.0625,
            paged_decode_plain_rows: u128::MAX,
        })
    };
    CircuitLayer {
        mixer,
        weights: w,
        unmodelled: Vec::new(),
        moe: None,
    }
}

/// 2026-09-29: A BF16 MTP draft head: weight tags at layer 200.
pub(super) fn draft_binding(circuit: &Circuit) -> CircuitLayer {
    let d = |k: &str| circuit.dims[k] as u32;
    let mut w: BTreeMap<WeightSlot, BoundWeight> = [
        WeightSlot::EmbedNorm,
        WeightSlot::HiddenNorm,
        WeightSlot::InputNorm,
        WeightSlot::PostNorm,
        WeightSlot::FinalNorm,
        WeightSlot::QNorm,
        WeightSlot::KNorm,
        WeightSlot::FfnGate,
        WeightSlot::FfnUp,
        WeightSlot::Linear(LinearRole::MtpFc),
        WeightSlot::Linear(LinearRole::Q),
        WeightSlot::Linear(LinearRole::K),
        WeightSlot::Linear(LinearRole::V),
        WeightSlot::Linear(LinearRole::O),
        WeightSlot::Linear(LinearRole::Down),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, s)| (s, dense(tag(200, 1 + i as u64))))
    .collect();
    w.insert(WeightSlot::LmHead, nvfp4(tag(200, 30)));
    CircuitLayer {
        mixer: MixerFacts::Attention(AttnFacts {
            attn_layer_idx: 0,
            kv_dtype: metrale_cache::kv_cache::KvCacheDtype::Bf16,
            num_q_heads: d("q_heads"),
            num_kv_heads: d("kv_heads"),
            head_dim: d("head_dim"),
            gated: true,
            rope: RopeFacts {
                mrope_interleaved: false,
                theta: 1.0e7,
                rotary_dim: 64,
            },
            sliding_window: 0,
            softmax_scale: 0.0625,
            paged_decode_plain_rows: u128::MAX,
        }),
        weights: w,
        unmodelled: Vec::new(),
        moe: None,
    }
}

pub(super) fn fixed(attn_layers: usize) -> Fixed {
    let meta = 0xA300_0000;
    Fixed {
        hidden: ptr(0xA000_0000),
        residual: ptr(0xA100_0000),
        logits: ptr(0xA200_0000),
        meta: AttnMetadataDev {
            positions: ptr(meta),
            positions_h: ptr(meta),
            positions_w: ptr(meta),
            slot: ptr(meta + 8),
            seq_len: ptr(meta + 16),
            block_table: ptr(meta + 256),
            max_blocks_per_seq: 0,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        k_pools: (0..attn_layers)
            .map(|i| ptr(0xB000_0000 + ((i as u64) << 24)))
            .collect(),
        v_pools: (0..attn_layers)
            .map(|i| ptr(0xB080_0000 + ((i as u64) << 24)))
            .collect(),
        batch_meta: AttnMetadataDev {
            positions: ptr(meta + 0x1_0000),
            positions_h: ptr(meta + 0x1_0000),
            positions_w: ptr(meta + 0x1_0000),
            slot: ptr(meta + 0x1_0000 + 1024),
            seq_len: ptr(meta + 0x1_0000 + 2048),
            block_table: ptr(meta + 0x1_0000 + 3072),
            max_blocks_per_seq: 0,
            num_seqs: 128,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        ffn_act_q8: ptr(0xA400_0000),
        tokens: ptr(0xA500_0000),
        draft: Some(DraftFixed {
            embed: ptr(0xA600_0000),
            meta: crate::layers::mtp_meta::mtp_attn_meta_dev(ptr(meta + 0x3_0000), 0),
            k_pool: ptr(0xB800_0000),
            v_pool: ptr(0xB880_0000),
            block_size: 16,
            cache_stride: 4096,
            vocab: 100_000,
            rows: Some(DraftRows {
                meta: ptr(meta + 0x5_0000),
                lp_offset: 512,
                lm_head_gemv: vec![metrale_gpu_runtime::gpu::KernelHandle(0xDEAD); 129],
                lm_head_twin: false,
            }),
        }),
        verify_batch_meta: AttnMetadataDev {
            positions: ptr(meta + 0x4_0000),
            positions_h: ptr(meta + 0x4_0000),
            positions_w: ptr(meta + 0x4_0000),
            slot: ptr(meta + 0x4_0000 + 1024),
            seq_len: ptr(meta + 0x4_0000 + 3072),
            block_table: ptr(meta + 0x4_0000 + 4096),
            max_blocks_per_seq: 0,
            num_seqs: 0,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        verify_wy_tables: ptr(0xA700_0000),
        verify_batch_tokens: ptr(0xA800_0000),
        verify_meta: AttnMetadataDev {
            positions: ptr(meta + 0x2_0000),
            positions_h: ptr(meta + 0x2_0000),
            positions_w: ptr(meta + 0x2_0000),
            slot: ptr(meta + 0x2_0000 + 256),
            seq_len: ptr(meta + 0x2_0000 + 512),
            block_table: ptr(meta + 0x2_0000 + 768),
            max_blocks_per_seq: 0,
            num_seqs: 4,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        block_size: 16,
        cache_stride: 4096,
        moe: Some(MOE_SCRATCH),
    }
}

/// 2026-10-03: The MoE arena scratch every fixture binds: the sort at its own tag, every
/// arena buffer wide enough for any width.
pub(super) const MOE_SCRATCH: crate::layers::moe::MoeScratch = crate::layers::moe::MoeScratch {
    sort: DevicePtr(0xA900_0000),
    routed_act: DevicePtr(0xAA00_0000),
    shared_act: DevicePtr(0xAB00_0000),
    scratch_bytes: usize::MAX,
    gate_logits_bytes: usize::MAX,
    expert_gate_out_bytes: usize::MAX,
    expert_down_out_bytes: usize::MAX,
    logits_bytes: usize::MAX,
    attn_output_bytes: usize::MAX,
    moe_output_bytes: usize::MAX,
};

pub(super) fn config() -> metrale_config::ModelConfig {
    let mut c = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_conv_kernel_dim = 4;
    c.final_norm_identity = false;
    c
}

pub(super) use super::exec_fixture_build::*;
pub(super) use super::exec_fixture_run::*;
