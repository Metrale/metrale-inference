// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: `AttnMetadataDev::rows_from`: the per-row arrays of a prefill pass's
//! attention metadata, advanced to a later row.
//!
//! Owner: model-layers.
//! Invariants: none beyond the types.

use super::AttnMetadataDev;
use metrale_gpu_runtime::gpu::DevicePtr;

fn meta(positions_h: DevicePtr, seq_slot: DevicePtr) -> AttnMetadataDev {
    AttnMetadataDev {
        positions: DevicePtr(0x1000),
        positions_h,
        positions_w: positions_h,
        slot: DevicePtr(0x9000),
        seq_len: DevicePtr(0xA000),
        block_table: DevicePtr(0xB000),
        max_blocks_per_seq: 7,
        num_seqs: 1,
        seq_slot,
        moe_row_adapter: DevicePtr::NULL,
    }
}

#[test]
fn per_row_arrays_advance_by_their_element_size_and_the_rest_is_kept() {
    let m = meta(DevicePtr(0x2000), DevicePtr(0x3000)).rows_from(27);
    // 2026-10-01: positions are u32, slots i64, the LoRA slot i32.
    assert_eq!(m.positions, DevicePtr(0x1000 + 27 * 4));
    assert_eq!(m.positions_h, DevicePtr(0x2000 + 27 * 4));
    assert_eq!(m.positions_w, DevicePtr(0x2000 + 27 * 4));
    assert_eq!(m.slot, DevicePtr(0x9000 + 27 * 8));
    assert_eq!(m.seq_slot, DevicePtr(0x3000 + 27 * 4));
    assert_eq!(m.seq_len, DevicePtr(0xA000));
    assert_eq!(m.block_table, DevicePtr(0xB000));
    assert_eq!(m.max_blocks_per_seq, 7);
    assert_eq!(m.num_seqs, 1);
}

#[test]
fn a_null_per_row_array_stays_null() {
    let m = meta(DevicePtr(0x1000), DevicePtr::NULL).rows_from(16);
    assert!(m.seq_slot.is_null(), "no LoRA slot buffer must stay none");
    assert!(m.moe_row_adapter.is_null());
}

#[test]
fn row_zero_is_the_identity() {
    let a = meta(DevicePtr(0x2000), DevicePtr(0x3000));
    let b = a.rows_from(0);
    assert_eq!(
        (b.positions, b.positions_h, b.slot, b.seq_slot),
        (a.positions, a.positions_h, a.slot, a.seq_slot)
    );
}
