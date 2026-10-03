// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `impl ModelCircuit for TransformerModel`: build the circuit executor from the live
//! layers and buffers, swap it in, and run its programs in place of the layer loops: decode
//! (`decode_forward_body`) and multi-sequence decode (`decode_batch_compute_main_with`). Also
//! the one statement of the single-sequence decode metadata layout, which
//! `decode_dispatch_with` uploads and the executor reads.
//!
//! Owner: model-engine (decode).
//! Invariants:
//! - A switch drops every captured decode and verify graph before the previous executor's
//!   workspace is freed, so no graph replays freed memory.
//! - Model features the circuit does not model refuse the build.

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::{
    BoundWeight, CircuitExec, DraftFixed, Fixed, Fusions, HeadBinding, MixerFacts, TargetModules,
    policy,
};
use metrale_model_layers::layer::AttnMetadataDev;

use super::super::types::TransformerModel;
use crate::traits::{ForwardDisclosure, ForwardSelect, ModelCircuit};

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
                self.config.ep_world_size > 1 && self.config.tp_world_size > 1,
                "tensor and expert parallelism together",
            ),
            (
                !self.dflash_capture_layers.is_empty(),
                "DFlash hidden capture",
            ),
            (
                self.config.mamba_num_heads > 0,
                "Mamba-2 state normalisation",
            ),
            (self.config.kv_lora_rank > 0, "latent attention"),
            (kv_swap, "high-speed swap"),
            (
                metrale_model_layers::ships_vanilla_norm_weights(&self.config),
                "vanilla RMSNorm weights (the rules launch the 1 + w kernels)",
            ),
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
        let (batchm_max_rows, m16_tc) = self.bf16_head_route();
        let unmodelled = [
            (quantized, "a quantized lm_head"),
            (m16_tc, "the tensor-core BF16 head (lm_head_m16_tc)"),
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
            batchm_max_rows,
        };
        (head, dtype)
    }

    /// 2026-09-28: Build the executor for `instance`.
    fn build_circuit(
        &self,
        instance: &metrale_circuit::Instance,
        fusions: Fusions,
        modules: &TargetModules,
        config_json: &str,
    ) -> Result<CircuitExec> {
        let unmodelled = self.circuit_unmodelled();
        if !unmodelled.is_empty() {
            bail!(
                "the circuit does not model this model: {}",
                unmodelled.join(", ")
            );
        }
        let stream = self.gpu.default_stream();
        for l in &self.layers {
            l.circuit_prepare(self.gpu.as_ref(), &self.config, &self.levers, stream)?;
        }
        // 2026-09-30: The batched MTP verify runs the carried-state GDN verify, whose buffers
        // the layers bind once (`gdn_carry.rs`); binding them now puts them in the layers'
        // circuit facts. Without them the executor compiles no batched verify and each
        // sequence verifies alone.
        // 2026-10-03: The exact MTP verify chain (`--exact-verify`, or a fixed GDN activation
        // format such as `--activation-quantization declared`) has no circuit verify yet: the
        // build compiles no verify program, so every verify runs the legacy layers, said here.
        let exact_verify = metrale_model_layers::layers::qwen3_ssm::verify_exact_enabled();
        if exact_verify && self.proposer.is_some() {
            tracing::info!(
                "circuit: the exact MTP verify chain is on; the verify (single and batched) runs \
                 the legacy layers under the circuit forward"
            );
        }
        // 2026-10-03: LoRA phase 1 verifies each sequence alone: a batched verify's row tables
        // reach the wide FFN arms legacy leaves under an adapter (`impl_circuit_lora.rs`).
        let verify_batch_rows = if !exact_verify
            && self.proposer.is_some()
            && self.lora.is_none()
            && self.gdn_carry_bind_now()?
        {
            Some(
                (4 * metrale_model_layers::speculative::mtp_max_seqs())
                    .min(super::verify_e2::VERIFY_ROW_CAP) as u64,
            )
        } else {
            None
        };
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
        let lora =
            self.circuit_lora(&layers, self.circuit_widths().last().copied().unwrap_or(1))?;
        let (head, lm_head_dtype) = self.circuit_head();
        let draft = self
            .proposer
            .as_ref()
            .and_then(|p| p.circuit_draft(&self.config, &self.levers));
        let fixed = {
            let cache = self.kv_cache.lock();
            let n = cache.num_layers();
            Fixed {
                hidden: self.buffers.hidden_states(),
                residual: self.buffers.residual(),
                logits: self.buffers.logits(),
                meta: self.decode_meta(0, DevicePtr::NULL),
                batch_meta: self.batch_meta_at(
                    self.batch_meta_base(),
                    0,
                    0,
                    DevicePtr::NULL,
                    DevicePtr::NULL,
                ),
                ffn_act_q8: self.buffers.ffn_act_q8(),
                tokens: self.buffers.scratch(),
                draft: draft.as_ref().map(|d| DraftFixed {
                    embed: self.buffers.ssm_qkvz(),
                    meta: metrale_model_layers::layers::mtp_meta::mtp_attn_meta_dev(
                        self.buffers
                            .scratch()
                            .offset(metrale_model_layers::layers::mtp_meta::MTP_META_OFFSET),
                        0,
                    ),
                    k_pool: d.k_pool,
                    v_pool: d.v_pool,
                    block_size: d.block_size,
                    cache_stride: d.cache_stride,
                    vocab: d.vocab,
                    rows: d.rows.clone(),
                }),
                verify_meta: self.verify_meta(0, 0, DevicePtr::NULL),
                verify_batch_meta: self.verify_batch_meta_at(),
                verify_wy_tables: self.verify_wy_tables,
                verify_batch_tokens: super::verify_e::mapped_argmax_host_dev(self.gpu.as_ref())
                    .map_or(self.buffers.scratch(), |(_, d)| d),
                k_pools: (0..n).map(|i| cache.k_pool_ptr(i)).collect(),
                v_pools: (0..n).map(|i| cache.v_pool_ptr(i)).collect(),
                block_size: u32::try_from(cache.block_size())?,
                cache_stride: cache.cache_stride() as u64,
                lora: lora.as_ref().map(|(_, f)| *f),
                comm: self.comm.clone(),
            }
        };
        CircuitExec::build(metrale_model_layers::circuit_exec::Boot {
            gpu: self.gpu.as_ref(),
            config: &self.config,
            config_json,
            levers: &self.levers,
            instance,
            policy: policy::live_policy(
                &self.levers,
                policy::kv_dtype_name(kv)?,
                lm_head_dtype,
                self.lora.is_some(),
            ),
            layers,
            head,
            fixed,
            fusions,
            modules,
            multi_seq_rows: self.circuit_widths(),
            // 2026-09-29: The MTP verify widths the scheduler runs one sequence at
            // (`serial_verify_plan`), when a drafter is loaded.
            draft: draft.map(|d| d.layer),
            verify_rows: if self.proposer.is_some() && !exact_verify {
                vec![2, 3, 4]
            } else {
                Vec::new()
            },
            verify_batch_rows,
            profile: self.profile,
            lora: lora.map(|(b, _)| b),
            swap: Some(self.circuit_swap_boot(policy::kv_dtype_name(kv)?)?),
            // 2026-10-03: The rank's share of the heads and the reduces, or the expert split
            // (`metrale_circuit::parallel`).
            parallel: self.circuit_parallel(),
            // 2026-09-30: The batched propose's widths, up to the sequences MTP runs at once.
            draft_rows: self
                .proposer
                .as_ref()
                .map(|p| p.propose_batch_max(&self.buffers, &self.config))
                .filter(|&w| w >= 2)
                .map(|w| w.min(metrale_model_layers::speculative::mtp_max_seqs()) as u64),
            // 2026-10-03: The pool as allocated; the build refuses a circuit whose state units
            // differ from it (`state_bind`).
            state_pool: metrale_model_layers::circuit_exec::state_bind::StatePool {
                units: metrale_model_layers::circuit_exec::state_bind::PoolUnits {
                    h_stored: self.ssm_pool.h_stored_bytes,
                    conv: self.ssm_pool.conv_bytes,
                },
                h_f16: self.ssm_pool.h_stored_bytes < self.ssm_pool.h_bytes,
            },
            arena: &self.buffers,
            // 2026-10-03: Off until the dense prefill rules cover every prefill node (M6a);
            // until then a pass runs the legacy layers under the circuit forward.
            prefill_max_tokens: None,
        })
    }

    /// 2026-09-28: The padded widths the executor compiles: every rung of the decode ladder up
    /// to the one the serve's widest batch pads to.
    fn circuit_widths(&self) -> Vec<u64> {
        let widest = crate::traits::padded_batch_n(self.levers.max_decode_seqs as usize);
        crate::traits::DECODE_BATCH_LADDER
            .iter()
            .filter(|&&r| r <= widest)
            .map(|&r| r as u64)
            .collect()
    }
}

impl ModelCircuit for TransformerModel {
    fn set_forward(&self, sel: &ForwardSelect) -> Result<()> {
        let next = match sel {
            ForwardSelect::Legacy => None,
            ForwardSelect::Circuit {
                instance,
                fusions,
                modules,
                config_json,
            } => Some(self.build_circuit(instance, *fusions, modules, config_json)?),
        };
        self.destroy_lora_decode_graphs();
        // 2026-09-29: The draft head drops the previous executor's program before its
        // workspace is freed, and takes the next one's after the swap.
        let runner = next.as_ref().and_then(CircuitExec::draft_runner);
        if let Some(p) = self.proposer.as_ref() {
            p.set_circuit_draft(None);
        }
        self.install_state_programs(next.as_ref().map(|e| std::sync::Arc::new(e.state.clone())));
        let prev = std::mem::replace(&mut *self.circuit.write(), next);
        if let Some(prev) = prev {
            prev.free(self.gpu.as_ref())?;
        }
        if let (Some(p), Some(r)) = (self.proposer.as_ref(), runner) {
            p.set_circuit_draft(Some(r));
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
                plan_digest: Some(e.plans_digest()),
                launches_per_step: Some(e.decode.launches.len()),
            },
        }
    }

    fn state_digest(&self, seq: &crate::traits::SequenceState) -> Result<Vec<(String, u64)>> {
        self.state_digest_impl(seq)
    }
}
