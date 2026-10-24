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
    let pair = SlotGraphKey::new(3, false, DevicePtr::NULL, MoeLoraRoute::Fold);
    let slots = SlotGraphKey::new(3, false, DevicePtr(0x1000), MoeLoraRoute::Fold);
    let skip = SlotGraphKey::new(3, false, DevicePtr::NULL, MoeLoraRoute::Skip);
    let refuse = SlotGraphKey::new(3, false, DevicePtr::NULL, MoeLoraRoute::Refuse);
    let keys = [pair, slots, skip, refuse];
    for (i, a) in keys.iter().enumerate() {
        for b in &keys[i + 1..] {
            assert_ne!(a, b);
        }
    }
    assert_eq!(
        SlotGraphKey::new(3, false, DevicePtr(0x2000), MoeLoraRoute::Fold),
        slots,
        "the slot buffer's address is not the route: its contents are device data"
    );
    assert_ne!(
        SlotGraphKey::new(4, false, DevicePtr::NULL, MoeLoraRoute::Fold),
        pair
    );
}

/// 2026-10-21: A slot whose live SSM state was swapped into intermediate 0 keys a second graph:
/// the first one baked the home buffers.
#[test]
fn a_swapped_slot_keys_its_own_graph() {
    let home = SlotGraphKey::new(3, false, DevicePtr::NULL, MoeLoraRoute::Fold);
    let swapped = SlotGraphKey::new(3, true, DevicePtr::NULL, MoeLoraRoute::Fold);
    assert_ne!(home, swapped);
    assert_eq!(
        home.slot, swapped.slot,
        "the slot-wide drop still finds both"
    );
}
