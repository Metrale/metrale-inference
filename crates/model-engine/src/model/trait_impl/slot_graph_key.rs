// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The key of the slot-keyed CUDA graphs (one-row decode, the K = 2, 3, 4 verify).
//! A capture bakes the sequence's pool-slot addresses and, besides, two host decisions of the
//! LoRA routing: whether the attention sites fold the installed pair (no slot buffer) or read a
//! per-row slot buffer, and the MoE fold decision (`MoeLoraRoute`). Until 2026-10-03 the key was
//! the pool slot alone, so a graph captured for one adapter replayed its route for the next
//! sequence on that slot whatever that sequence's adapter.
//!
//! Owner: model-engine decode graphs.
//! Invariants:
//! - Two steps share a graph only if they share the slot and both LoRA decisions; the slot
//!   buffer's contents and the adapter slot it names are device data the graph reads.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::MoeLoraRoute;

/// 2026-10-03: A slot-keyed graph's key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SlotGraphKey {
    /// 2026-10-03: The sequence's pool slot (`SequenceState::slot_idx`).
    pub slot: usize,
    /// 2026-10-21: The slot's commit-swap role (`ssm_reserve::commit_swap`): which of its two
    /// buffers the capture baked as the live SSM state.
    pub ssm_swapped: bool,
    /// 2026-10-03: The attention LoRA sites fold the installed pair (the step uploaded no slot
    /// buffer while a pool is loaded, or no pool is loaded).
    pub lora_pair: bool,
    /// 2026-10-03: The MoE LoRA fold decision, as a stable code.
    pub moe_route: u8,
    /// 2026-10-10: The step ends in the sharded device argmax (`device_argmax.rs`) instead of
    /// the gathered logits: a different head tail is captured.
    pub device_argmax: bool,
    /// 2026-10-23: The paired A/B instrument's lever set the capture baked
    /// (`ops::paired_ab`); 0 outside the instrument.
    pub ab_levers: u32,
}

impl SlotGraphKey {
    /// 2026-10-03: The key of a step on `slot` (2026-10-21: in commit-swap role `ssm_swapped`)
    /// whose metadata carries `seq_slot` (NULL when the installed pair folds) under `moe_route`.
    pub(crate) fn new(
        slot: usize,
        ssm_swapped: bool,
        seq_slot: DevicePtr,
        moe_route: MoeLoraRoute,
    ) -> Self {
        SlotGraphKey {
            slot,
            ssm_swapped,
            lora_pair: seq_slot.is_null(),
            moe_route: match moe_route {
                MoeLoraRoute::Fold => 0,
                MoeLoraRoute::Skip => 1,
                MoeLoraRoute::Refuse => 2,
            },
            device_argmax: false,
            ab_levers: 0,
        }
    }

    /// 2026-10-23: This key for a step run with the instrument's lever set `ab`.
    pub(crate) fn with_ab_levers(self, ab: metrale_model_layers::layer::AbLevers) -> Self {
        SlotGraphKey {
            ab_levers: ab.bits(),
            ..self
        }
    }

    /// 2026-10-10: This key for a step that ends in the device argmax when `on`.
    pub(crate) fn with_device_argmax(self, on: bool) -> Self {
        SlotGraphKey {
            device_argmax: on,
            ..self
        }
    }
}

#[cfg(test)]
#[path = "slot_graph_key_tests.rs"]
mod slot_graph_key_tests;
