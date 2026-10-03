// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: LoRA adapters under `--forward circuit` (LIFECYCLE-DESIGN.md 15.10), the model
//! side: what a build adapts (`circuit_exec::lora::spec_of`), the fixed addresses its launches
//! read, the refusals of LoRA phase 1, and the per-row adapter slots a circuit step uploads.
//!
//! Owner: model-engine (FEATURES workstream).
//! Invariants:
//! - Under the circuit every step uploads its rows' slots, `-1` resolved to the active adapter
//!   (`build_seq_slot_host`), at the address the build bound: the attention fold always reads a
//!   slot. Legacy's one-row decode and verify upload none for an active-adapter request.
//! - A serve LoRA phase 1 cannot run exactly is refused at build, never run partly adapted.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::CircuitLayer;
use metrale_model_layers::circuit_exec::lora::{LoraBoot, LoraFixed, PHASE1_MAX_ROWS, spec_of};

use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-03: The adapters' part of a circuit build over `layers`, whose widest
    /// multi-sequence program is `widest` rows; `None` without a pool.
    pub(super) fn circuit_lora(
        &self,
        layers: &[Option<CircuitLayer>],
        widest: u64,
    ) -> Result<Option<(LoraBoot, LoraFixed)>> {
        let Some(lw) = self.lora.as_ref() else {
            return Ok(None);
        };
        if widest > PHASE1_MAX_ROWS {
            bail!(
                "LoRA under --forward circuit serves steps of at most {PHASE1_MAX_ROWS} rows \
                 (phase 1: the wide FFN arms legacy leaves under an adapter are not ruled yet); \
                 this serve batches {widest}: lower --max-batch-size"
            );
        }
        if self.lora_rotatable {
            bail!(
                "LoRA rotation under --forward circuit (the active adapter's pairs are compiled \
                 into the programs; rotating needs a rebuild, not done yet)"
            );
        }
        let boot = spec_of(lw.max_rank as u64, layers);
        if boot.spec.layers.is_empty() {
            bail!("a LoRA pool that adapts no projection the circuit binds");
        }
        let base = self.batch_meta_base();
        let fixed = LoraFixed {
            xa: self.buffers.lora_xa(),
            delta: self.buffers.lora_delta(),
            // 2026-10-03: The free gap legacy's one-row decode and verify upload a routed slot
            // to (`decode_a.rs`, `verify_b.rs`: `meta_base + 128`), and the batch layout's
            // seq_slot region (`[4R, 8R)`, `DecodeMetaLayout::seq_slot_off`).
            slots_decode: base.offset(128),
            slots_verify: base.offset(128),
            slots_multi_seq: base.offset(self.buffers.decode_meta().seq_slot_off()),
        };
        Ok(Some((boot, fixed)))
    }

    /// 2026-10-03: The slot buffer of a one-row decode or a `count`-row verify of a sequence
    /// whose adapter is `adapter_slot`: under the circuit always uploaded to `dst`, `-1` resolved
    /// to the active adapter; under legacy, legacy's (`upload_seq_slot_uniform`).
    pub(super) fn step_seq_slot(
        &self,
        adapter_slot: i32,
        count: usize,
        dst: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        let (Some(lw), true) = (self.lora.as_ref(), self.circuit.read().is_some()) else {
            return self.upload_seq_slot_uniform(adapter_slot, count, dst, stream);
        };
        let slots = vec![adapter_slot; count];
        let host = metrale_model_layers::lora::build_seq_slot_host(&slots, count, lw.active as i32);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.copy_h2d_async(&bytes, dst, stream)?;
        Ok(dst)
    }
}
