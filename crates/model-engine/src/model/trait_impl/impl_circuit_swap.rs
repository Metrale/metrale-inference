// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The sequence swap under `--forward circuit` (LIFECYCLE-DESIGN.md 15.10), the model
//! side: the runner's binding (the KV pools and strides, the recurrent pool's unit sizes, the
//! stored formats) and the save and restore a scheduler spill calls
//! (`Model::save_sequence_state` / `restore_sequence_state`), run as `kv_swap_out` /
//! `kv_swap_in` on the executor's copy stream. The spill policy stays in the scheduler.
//!
//! Owner: model-engine (FEATURES workstream).
//! Invariants:
//! - The record is legacy's (`sequence/state_io.rs`), so a spill file is readable by either
//!   forward; the restore allocates the blocks in legacy's order before reading.
//! - A restore that fails returns the blocks it allocated and leaves the table empty.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use metrale_circuit::state::StateDtype;
use metrale_config::LayerType;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::circuit_exec::swap::{SwapBinding, SwapBoot};
use metrale_model_layers::layer::SsmLayerState;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// 2026-10-03: The swap runner's binding; `kv_dtype` is the circuit's KV-cache spelling.
    pub(super) fn circuit_swap_boot(&self, kv_dtype: &str) -> Result<SwapBoot> {
        let cache = self.kv_cache.lock();
        let kv = (0..cache.num_layers())
            .map(|i| {
                [
                    (
                        cache.k_pool_ptr(i),
                        cache.k_block_stride_bytes_for_layer(i) as u64,
                    ),
                    (
                        cache.v_pool_ptr(i),
                        cache.v_block_stride_bytes_for_layer(i) as u64,
                    ),
                ]
            })
            .collect();
        let gdn_layers = (0..self.layers.len())
            .filter(|&i| self.config.layer_type(i) == LayerType::LinearAttention)
            .count();
        let unit = [
            self.ssm_pool.h_stored_bytes as u64,
            self.ssm_pool.conv_bytes as u64,
        ];
        // 2026-10-03: The pool's h storage: f16 only under the f16-sized pool (the rule
        // `ssm_reserve::pool_plan` sizes the pool by).
        let h = if metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
            StateDtype::F16
        } else {
            StateDtype::F32
        };
        Ok(SwapBoot {
            bind: SwapBinding {
                kv,
                recurrent_bytes: unit.repeat(gdn_layers),
            },
            formats: BTreeMap::from([
                (
                    "kv_cache_dtype".to_string(),
                    StateDtype::parse(kv_dtype)
                        .with_context(|| format!("KV-cache dtype `{kv_dtype}`"))?,
                ),
                ("ssm_h_storage".to_string(), h),
            ]),
            block_size: cache.block_size() as u64,
        })
    }

    /// 2026-10-03: Each recurrent unit of `seq`, in the record's order (per GatedDeltaNet layer,
    /// h then conv).
    fn swap_units(&self, seq: &SequenceState) -> Result<Vec<DevicePtr>> {
        let mut out = Vec::new();
        for (i, st) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let s = st
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .with_context(|| format!("layer {i}: no GDN state"))?;
                out.extend([s.h_state, s.conv_state]);
            }
        }
        Ok(out)
    }

    /// 2026-10-03: `kv_swap_out` of `seq` to `writer` through the circuit's runner; `false` when
    /// no circuit runner is installed (legacy then saves).
    pub(crate) fn circuit_save(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<bool> {
        let exec = self.circuit.read();
        let Some(runner) = exec.as_ref().and_then(|e| e.swap.as_ref()) else {
            return Ok(false);
        };
        let units = self.swap_units(seq)?;
        let _kv = self.kv_cache.lock();
        runner.swap_out(
            self.gpu.as_ref(),
            self.gpu.default_stream(),
            (&seq.block_table, &units),
            writer,
        )?;
        Ok(true)
    }

    /// 2026-10-03: `kv_swap_in` of a `num_blocks`-block record from `reader` into `seq`, whose
    /// blocks are allocated first; `false` when no circuit runner is installed.
    pub(crate) fn circuit_restore(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<bool> {
        let exec = self.circuit.read();
        let Some(runner) = exec.as_ref().and_then(|e| e.swap.as_ref()) else {
            return Ok(false);
        };
        let units = self.swap_units(seq)?;
        let mut kv = self.kv_cache.lock();
        let mut table = Vec::with_capacity(num_blocks);
        let read = (|| -> Result<()> {
            for _ in 0..num_blocks {
                table.push(kv.alloc_block()?);
            }
            runner.swap_in(
                self.gpu.as_ref(),
                self.gpu.default_stream(),
                (&table, &units),
                reader,
            )
        })();
        match read {
            Ok(()) => {
                seq.block_table = table;
                Ok(true)
            }
            Err(e) => {
                kv.free_blocks(&table);
                seq.block_table.clear();
                Err(e)
            }
        }
    }
}
