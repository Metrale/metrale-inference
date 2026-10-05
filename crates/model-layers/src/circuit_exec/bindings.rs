// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `CircuitBindings`, the supertrait of `TransformerLayer` through which a layer hands
//! the circuit executor its weights by role and the facts its kernels need. Layers keep owning
//! their weights; a binding is a copy of pointers.
//!
//! Owner: model-layers circuit executor.
//! Invariants:
//! - A layer lists in `unmodelled` every feature it carries that the circuit does not model
//!   (hyper-connections, LoRA, a projection format the rules do not cover, ...). The executor
//!   refuses a model with any, so a plan never runs over a layer it misdescribes.
//! - A bound weight carries its storage format; the executor checks it against the format the
//!   plan's node was resolved to.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Result, anyhow, ensure};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_circuit::{Circuit, LinearRole};

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::{W8a8Kernels, W8a8Weight};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// 2026-09-28: Which weight of a layer a circuit node reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WeightSlot {
    /// 2026-09-28: The mixer's input RMSNorm.
    InputNorm,
    /// 2026-09-28: The FFN's input RMSNorm (`post_attention_layernorm`).
    PostNorm,
    /// 2026-09-28: A projection; `GateUp` is bound as [`WeightSlot::FfnGate`] and
    /// [`WeightSlot::FfnUp`].
    Linear(LinearRole),
    /// 2026-09-28: The dense FFN's gate projection.
    FfnGate,
    /// 2026-09-28: The dense FFN's up projection.
    FfnUp,
    /// 2026-09-28: Attention's per-head Q RMSNorm.
    QNorm,
    /// 2026-09-28: Attention's per-head K RMSNorm.
    KNorm,
    /// 2026-09-28: GatedDeltaNet `A_log`.
    GdnALog,
    /// 2026-09-28: GatedDeltaNet `dt_bias`.
    GdnDtBias,
    /// 2026-09-28: GatedDeltaNet conv1d taps.
    GdnConv1d,
    /// 2026-09-28: GatedDeltaNet output norm.
    GdnNorm,
    /// 2026-09-28: The transposed twin of a projection, which the tile GEMMs read.
    Transposed(LinearRole),
    /// 2026-09-28: The dense FFN's gate, repacked for the NVFP4 MMQ GEMM.
    FfnGateMmq,
    /// 2026-09-28: The dense FFN's up, repacked for the NVFP4 MMQ GEMM.
    FfnUpMmq,
    /// 2026-09-28: The dense FFN's down, repacked for the NVFP4 MMQ GEMM.
    FfnDownMmq,
    /// 2026-09-29: The MTP draft head's norm of the token embedding (`pre_fc_norm_embedding`).
    EmbedNorm,
    /// 2026-09-29: The MTP draft head's norm of the target hidden (`pre_fc_norm_hidden`).
    HiddenNorm,
    /// 2026-09-29: The MTP draft head's final norm.
    FinalNorm,
    /// 2026-09-29: The MTP draft head's vocabulary projection.
    LmHead,
    // 2026-10-03: Prefill (M6a, GatedDeltaNet).
    /// 2026-10-03: The unscaled E4M3 cast of a projection's NVFP4 weight that its prefill GEMM
    /// reads (`Qwen3SsmLayer::qkvz_fp8`, `out_proj_fp8`), bound as a dense pointer: a derived
    /// serving copy, not the format the node declares.
    PrefillCast(LinearRole),
}

/// 2026-09-28: A bound weight and its storage format.
#[derive(Debug, Clone, Copy)]
pub enum BoundWeight {
    /// 2026-09-28: Unquantized (BF16 for projections; norms and GDN vectors as loaded).
    Dense(DenseWeight),
    /// 2026-09-28: NVFP4, group 16.
    Nvfp4(QuantizedWeight),
    /// 2026-09-28: NVFP4 repacked for the MMQ GEMM (`ops::nvfp4_mmq_repack`); its
    /// `weight_scale_2` stays with the source weight.
    Mmq(DevicePtr),
    /// 2026-09-30: A declared W8A8 projection (`W8a8Mixer`, `W8a8Ffn`): the checkpoint's E4M3
    /// weight with its scales, and the kernels its layer runs it with.
    W8a8(W8a8Weight, W8a8Kernels),
    /// 2026-10-03: An FP8 E4M3 projection with 128 x 128 block scales the layer runs W8A16
    /// (BF16 activations): the checkpoint's own weight.
    Fp8(crate::weight_map::Fp8Weight),
}

impl BoundWeight {
    /// 2026-09-28: The format family, spelled as the precision tables spell weights.
    pub fn family(&self) -> &'static str {
        match self {
            BoundWeight::Dense(_) => "bf16",
            BoundWeight::Nvfp4(_) | BoundWeight::Mmq(_) => "nvfp4",
            BoundWeight::W8a8(..) | BoundWeight::Fp8(_) => "fp8",
        }
    }
}

/// 2026-09-28: What a GatedDeltaNet layer's kernels need beyond its weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GdnFacts {
    /// 2026-09-28: The qkvz projection writes `[Q | K | V | Z]` itself (`sequential_qkvz`), so no
    /// deinterleave follows it.
    pub qkvz_deinterleaved: bool,
    /// 2026-09-30: Bytes from one pool slot's h state to the next (`h_slot_stride_bytes`), the
    /// pitch the batched arm's slot check and strided recurrence use.
    pub h_slot_bytes: u64,
    /// 2026-09-30: Bytes from one pool slot's conv window to the next (`conv_state_bytes`).
    pub conv_state_bytes: u64,
    /// 2026-09-30: The carried-state verify's buffers (`LayerWriteOnAccept::gdn_carry_bind`),
    /// which the batched verify's GDN launches read; `None` until the model binds them.
    pub carry: Option<crate::layer::GdnCarryBinding>,
}

/// 2026-09-28: A full-attention layer's rotary embedding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RopeFacts {
    /// 2026-09-28: Interleaved MRoPE (`mrope_interleaved` with its kernel loaded).
    pub mrope_interleaved: bool,
    /// 2026-09-28: `rope_theta`, after any per-layer override.
    pub theta: f32,
    /// 2026-09-28: Rotated dims per head, after any per-layer override.
    pub rotary_dim: u32,
}

/// 2026-09-28: What a full-attention layer's kernels need beyond its weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttnFacts {
    /// 2026-09-28: This layer's index among the attention layers: its KV-cache layer.
    pub attn_layer_idx: usize,
    /// 2026-09-28: KV-cache dtype of this layer.
    pub kv_dtype: KvCacheDtype,
    /// 2026-09-28: Query heads.
    pub num_q_heads: u32,
    /// 2026-09-28: KV heads.
    pub num_kv_heads: u32,
    /// 2026-09-28: Head dim.
    pub head_dim: u32,
    /// 2026-09-28: Q carries an output gate (`[Q | Gate]` after the split).
    pub gated: bool,
    /// 2026-09-28: Rotary embedding.
    pub rope: RopeFacts,
    /// 2026-09-28: Sliding window in positions; 0 for none.
    pub sliding_window: u32,
    /// 2026-09-28: Softmax scale (`effective_attn_scale`).
    pub softmax_scale: f32,
    /// 2026-09-28: Bit `r - 1` is set when the layer's own decode routing at `r` rows picks
    /// the plain non-split paged kernel (`paged_decode_k`): no split-K pair, no GQA-packed
    /// kernel, no 512-wide head kernel.
    pub paged_decode_plain_rows: u128,
}

impl AttnFacts {
    /// 2026-09-28: The plain paged kernel serves `rows` rows (1..=128).
    pub fn paged_decode_plain(&self, rows: u64) -> bool {
        (1..=128).contains(&rows) && self.paged_decode_plain_rows >> (rows - 1) & 1 == 1
    }
}

/// 2026-09-28: The mixer a layer runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MixerFacts {
    /// 2026-09-28: GatedDeltaNet.
    Gdn(GdnFacts),
    /// 2026-09-28: Full attention.
    Attention(AttnFacts),
}

/// 2026-09-28: One layer, as the executor sees it.
#[derive(Debug, Clone)]
pub struct CircuitLayer {
    /// 2026-09-28: The mixer.
    pub mixer: MixerFacts,
    /// 2026-09-28: Weights by slot.
    pub weights: BTreeMap<WeightSlot, BoundWeight>,
    /// 2026-09-28: Features present on this layer that the circuit does not model.
    pub unmodelled: Vec<String>,
    /// 2026-10-03: The layer's MoE FFN (`MoeLayer::circuit_bind`); `None` for a dense FFN.
    pub moe: Option<crate::layers::moe::MoeBinding>,
}

/// 2026-09-28: A supertrait of `TransformerLayer`; see the module header.
pub trait CircuitBindings {
    /// 2026-09-28: Build the lazily built weights the bound kernels read (the MMQ repacks a
    /// first wide batch would otherwise build), so a binding can hand them out. Queued on
    /// `stream`.
    fn circuit_prepare(
        &self,
        _gpu: &dyn GpuBackend,
        _config: &metrale_config::ModelConfig,
        _levers: &crate::layers::ops::ModelLevers,
        _stream: u64,
    ) -> anyhow::Result<()> {
        Ok(())
    }

    /// 2026-09-28: This layer for the circuit executor; `None` for a layer type it does not bind.
    /// `config` and `levers` are the model's, as its decode reads them.
    fn circuit_layer(
        &self,
        _config: &metrale_config::ModelConfig,
        _levers: &crate::layers::ops::ModelLevers,
    ) -> Option<CircuitLayer> {
        None
    }
}

/// 2026-09-29: The MTP draft head as the executor binds it: its one layer (weights by slot,
/// the attention facts of its own KV cache) and that cache.
#[derive(Debug, Clone)]
pub struct DraftBinding {
    /// 2026-09-29: Weights by slot, the draft attention's facts (`attn_layer_idx` is its
    /// cache's layer), and what the circuit does not model.
    pub layer: CircuitLayer,
    /// 2026-09-29: The draft cache's K pool.
    pub k_pool: DevicePtr,
    /// 2026-09-29: The draft cache's V pool.
    pub v_pool: DevicePtr,
    /// 2026-09-29: Tokens per block of the draft cache.
    pub block_size: u32,
    /// 2026-09-29: `PagedKvCache::cache_stride` of the draft cache.
    pub cache_stride: u64,
    /// 2026-09-29: The vocabulary rows the draft lm_head scores and the argmax reads (the
    /// first `mtp_vocab_size`, or all).
    pub vocab: u32,
    /// 2026-09-30: What the head's n-row draft (`forward_batch_position`) reads beyond the
    /// layer; `None` for a head outside the batched propose's scope.
    pub rows: Option<DraftRows>,
}

/// 2026-09-30: The MTP head's n-row draft facts: where its batched metadata and confidences
/// live, and the LM-head launch it chooses at each width (`MtpHead::lm_head_batch_kernel`,
/// `mtp_head::lm_head_rows_arm`), which the executor checks the plan against.
#[derive(Debug, Clone)]
pub struct DraftRows {
    /// 2026-09-30: The batched propose's metadata allocation (`MtpHead::propose_meta`).
    pub meta: DevicePtr,
    /// 2026-09-30: Byte offset in scratch of the per-row top-1 log-probabilities
    /// (`LP_SCRATCH_OFF`); the ids sit at scratch's start.
    pub lp_offset: usize,
    /// 2026-09-30: `lm_head_batch_kernel(n)` at index `n`; the zero handle where none serves.
    pub lm_head_gemv: Vec<metrale_gpu_runtime::gpu::KernelHandle>,
    /// 2026-09-30: The tile GEMM on the transposed LM-head twin is available to the head.
    pub lm_head_twin: bool,
}

/// 2026-09-28: The model's own weights the head block reads, filled by the model.
#[derive(Debug, Clone)]
pub struct HeadBinding {
    /// 2026-10-04: The token embedding table (the prologue's binding), which a prefill pass
    /// gathers from (`emitters/prefill_ffn.rs` `PrefillEmbed`).
    pub embed: DenseWeight,
    /// 2026-09-28: The final RMSNorm.
    pub final_norm: DenseWeight,
    /// 2026-09-28: The vocabulary projection.
    pub lm_head: BoundWeight,
    /// 2026-09-28: Head features the circuit does not model (a logit overlay, softcapping,
    /// FP32 logits, a vocab-parallel head, ...).
    pub unmodelled: Vec<String>,
    /// 2026-09-28: The widest padded batch the BF16 head serves with the batched GEMV
    /// (`dense_gemv_bf16_batchm`); 0 when that arm is off. Wider batches take the GEMM.
    pub batchm_max_rows: u32,
    /// 2026-10-03: An NVFP4 head's transposed twin and its padded N (`lm_head_nvfp4_t`), which
    /// its tile GEMM reads; `None` for a BF16 head.
    pub nvfp4_twin: Option<(crate::weight_map::QuantizedWeight, u32)>,
    /// 2026-10-05: The declared NVFP4 head runs on the W4A16 row tiles at every row count
    /// (`install_declared_lm_head_w4a16_rows`); `lm_head` is then that NVFP4 head.
    pub nvfp4_rows: bool,
}

/// 2026-09-28: Refuse a model the circuit misdescribes: an unbound layer, a layer or head
/// feature no rule models, or a layer whose mixer is not the circuit's.
pub fn check_bindings(
    circuit: &Circuit,
    layers: &[Option<CircuitLayer>],
    head: &HeadBinding,
) -> Result<Vec<CircuitLayer>> {
    let mut problems = BTreeSet::new();
    let mut out = Vec::with_capacity(layers.len());
    for (i, l) in layers.iter().enumerate() {
        let Some(l) = l else {
            problems.insert(format!("layer {i} has no circuit binding"));
            continue;
        };
        for u in &l.unmodelled {
            problems.insert(format!("layer {i}: {u}"));
        }
        let want_gdn =
            circuit.layer_kinds.get(i) == Some(&metrale_circuit::LayerKind::LinearAttention);
        let is_gdn = matches!(l.mixer, MixerFacts::Gdn(_));
        if want_gdn != is_gdn {
            problems.insert(format!(
                "layer {i}: the circuit and the model disagree on its kind"
            ));
        }
        out.push(l.clone());
    }
    for u in &head.unmodelled {
        problems.insert(format!("head: {u}"));
    }
    if !problems.is_empty() {
        return Err(anyhow!(
            "the circuit does not model this model:\n  {}",
            problems.into_iter().collect::<Vec<_>>().join("\n  ")
        ));
    }
    ensure!(
        out.len() == circuit.layer_kinds.len(),
        "the model has {} layers, the circuit {}",
        out.len(),
        circuit.layer_kinds.len()
    );
    Ok(out)
}
