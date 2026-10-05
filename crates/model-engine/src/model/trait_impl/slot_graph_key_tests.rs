// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The slot-keyed graph key separates every LoRA route a capture bakes.
//!
//! Owner: model-engine decode graphs.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::MoeLoraRoute;

use super::SlotGraphKey;

#[test]
fn a_slot_keys_one_graph_per_lora_route() {
    let pair = SlotGraphKey::new(3, DevicePtr::NULL, MoeLoraRoute::Fold);
    let slots = SlotGraphKey::new(3, DevicePtr(0x1000), MoeLoraRoute::Fold);
    let skip = SlotGraphKey::new(3, DevicePtr::NULL, MoeLoraRoute::Skip);
    let refuse = SlotGraphKey::new(3, DevicePtr::NULL, MoeLoraRoute::Refuse);
    let keys = [pair, slots, skip, refuse];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
    assert_eq!(
        SlotGraphKey::new(3, DevicePtr(0x2000), MoeLoraRoute::Fold),
        slots,
        "the slot buffer's address is not the route: its contents are device data"
    );
    assert_ne!(
        SlotGraphKey::new(4, DevicePtr::NULL, MoeLoraRoute::Fold),
        pair
    );
}
