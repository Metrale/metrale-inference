// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `impl ModelCircuit for TransformerModel`: build the circuit executor from the live
//! layers and buffers, swap it in, and run its decode program in place of the layer loops
//! (`decode_forward_body`). Also the one statement of the single-sequence decode metadata
//! layout, which `decode_dispatch_with` uploads and the executor reads.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - A switch drops every captured decode and verify graph before the previous executor's
//!   workspace is freed, so no graph replays freed memory.
//! - Model features the circuit does not model refuse the build.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::{
    BoundWeight, CircuitExec, Fixed, Fusions, GdnState, HeadBinding, MixerFacts, StepEnv, policy,
};
use metrale_model_layers::layer::{AttnMetadataDev, ForwardContext, SsmLayerState};

use super::super::types::TransformerModel;
use crate::traits::{ForwardDisclosure, ForwardSelect, ModelCircuit, SequenceState};

/// 2026-09-28: Offset of the single-sequence decode metadata in the scratch buffer.
const DECODE_META_OFFSET: usize = 32768;

impl TransformerModel {
    /// 2026-09-28: The single-sequence decode metadata at its fixed upload address.
    pub(super) fn decode_meta(&self, max_blocks: u32, seq_slot: DevicePtr) -> AttnMetadataDev {
        let base = self.buffers.scratch().offset(DECODE_META_OFFSET);
        AttnMetadataDev {
            positions: base,
            positions_h: base,
            positions_w: base,
            slot: base.offset(8),
            seq_len: base.offset(16),
            block_table: base.offset(256),
            max_blocks_per_seq: max_blocks,
            num_seqs: 1,
            seq_slot,
            moe_row_adapter: DevicePtr::NULL,
        }
    }

    /// 2026-09-28: Model features the circuit does not model.
    fn circuit_unmodelled(&self) -> Vec<String> {
        let kv_swap = self.kv_cache.lock().config().cache_blocks_per_seq.is_some();
        [
            (
                self.config.tp_world_size > 1 || self.comm.is_some(),
                "tensor or expert parallelism",
            ),
            (
                !self.dflash_capture_layers.is_empty(),
                "DFlash hidden capture",
            ),
            (self.lora.is_some(), "LoRA adapters"),
            (
                self.config.mamba_num_heads > 0,
                "Mamba-2 state normalisation",
            ),
            (self.config.kv_lora_rank > 0, "latent attention"),
            (kv_swap, "high-speed swap"),
            (self.profile, "profiling"),
        ]
        .into_iter()
        .filter(|(present, _)| *present)
        .map(|(_, what)| what.to_string())
        .collect()
    }

    /// 2026-09-28: The head block's weights, and its unmodelled features.
    fn circuit_head(&self) -> (HeadBinding, &'static str) {
        let (dtype, quantized) = if self.lm_head_q6k.is_some() {
            ("q6k", true)
        } else if self.lm_head_fp8.is_some() {
            ("fp8", true)
        } else if self.lm_head_nvfp4.is_some() {
            ("nvfp4", true)
        } else {
            ("bf16", false)
        };
        let unmodelled = [
            (quantized, "a quantized lm_head"),
            (self.use_fp32_logits, "FP32 logits"),
            (
                self.logit_softcap_kernel.0 != 0 || self.logit_softcap_fp32_kernel.0 != 0,
                "logit softcapping",
            ),
            (self.overlays.is_some(), "token overlays"),
            (
                self.config.final_norm_identity,
                "a checkpoint without a final norm",
            ),
        ]
        .into_iter()
        .filter(|(present, _)| *present)
        .map(|(_, what)| what.to_string())
        .collect();
        let head = HeadBinding {
            final_norm: self.final_norm,
            lm_head: BoundWeight::Dense(self.lm_head_weight),
            unmodelled,
        };
        (head, dtype)
    }

    /// 2026-09-28: Build the executor for `instance`.
    fn build_circuit(
        &self,
        instance: &metrale_circuit::Instance,
        fusions: Fusions,
    ) -> Result<CircuitExec> {
        let unmodelled = self.circuit_unmodelled();
        if !unmodelled.is_empty() {
            bail!(
                "the circuit does not model this model: {}",
                unmodelled.join(", ")
            );
        }
        let layers: Vec<_> = self
            .layers
            .iter()
            .map(|l| l.circuit_layer(&self.config, &self.levers))
            .collect();
        let mut kv_dtypes = layers.iter().flatten().filter_map(|l| match l.mixer {
            MixerFacts::Attention(a) => Some(a.kv_dtype),
            MixerFacts::Gdn(_) => None,
        });
        let kv = kv_dtypes
            .next()
            .context("the model has no attention layer to read a KV dtype from")?;
        ensure!(
            kv_dtypes.all(|d| d == kv),
            "attention layers use different KV-cache dtypes; the circuit states one"
        );
        let (head, lm_head_dtype) = self.circuit_head();
        let fixed = {
            let cache = self.kv_cache.lock();
            let n = cache.num_layers();
            Fixed {
                hidden: self.buffers.hidden_states(),
                residual: self.buffers.residual(),
                logits: self.buffers.logits(),
                meta: self.decode_meta(0, DevicePtr::NULL),
                k_pools: (0..n).map(|i| cache.k_pool_ptr(i)).collect(),
                v_pools: (0..n).map(|i| cache.v_pool_ptr(i)).collect(),
                block_size: u32::try_from(cache.block_size())?,
                cache_stride: cache.cache_stride() as u64,
            }
        };
        CircuitExec::build(metrale_model_layers::circuit_exec::Boot {
            gpu: self.gpu.as_ref(),
            config: &self.config,
            levers: &self.levers,
            instance,
            policy: policy::live_policy(&self.levers, policy::kv_dtype_name(kv)?, lm_head_dtype),
            layers,
            head,
            fixed,
            fusions,
        })
    }

    /// 2026-09-28: Run `exec`'s decode program for `seq` in place of the layer loops, the final
    /// norm and the lm_head.
    pub(super) fn circuit_forward_body(
        &self,
        exec: &CircuitExec,
        seq: &SequenceState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let meta = ctx
            .attn_metadata
            .context("circuit decode needs the step's attention metadata")?;
        let mut gdn = Vec::with_capacity(seq.layer_states.len());
        for (i, st) in seq.layer_states.iter().enumerate() {
            gdn.push(match st.as_any().downcast_ref::<SsmLayerState>() {
                Some(s) => {
                    ensure!(
                        !s.h_is_f16,
                        "layer {i}: FP16 h state under an FP32-state circuit plan"
                    );
                    Some(GdnState {
                        h: s.h_state,
                        conv: s.conv_state,
                    })
                }
                None => None,
            });
        }
        exec.decode.run(&StepEnv {
            gpu: self.gpu.as_ref(),
            stream,
            gdn: &gdn,
            max_blocks_per_seq: meta.max_blocks_per_seq,
        })
    }
}

impl ModelCircuit for TransformerModel {
    fn set_forward(&self, sel: &ForwardSelect) -> Result<()> {
        let next = match sel {
            ForwardSelect::Legacy => None,
            ForwardSelect::Circuit { instance, fusions } => {
                Some(self.build_circuit(instance, *fusions)?)
            }
        };
        self.destroy_lora_decode_graphs();
        let prev = std::mem::replace(&mut *self.circuit.write(), next);
        if let Some(prev) = prev {
            prev.free(self.gpu.as_ref())?;
        }
        Ok(())
    }

    fn forward_disclosure(&self) -> ForwardDisclosure {
        match &*self.circuit.read() {
            None => ForwardDisclosure::legacy(),
            Some(e) => ForwardDisclosure {
                forward: match e.fusions {
                    Fusions::All => "circuit",
                    Fusions::ReferenceOnly => "circuit-reference",
                },
                plan_digest: Some(e.decode_plan.digest.clone()),
                launches_per_step: Some(e.decode.launches.len()),
            },
        }
    }
}
