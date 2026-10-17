// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Tests of the per-layer expert slab: the slot layout's offsets and stride, and on
//! the mock device one allocation and one adoption per slab, each expert's bytes at its own
//! pointers, and the refusals (outside ids, twice, other sizes).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::{ExpertSlab, SLAB_ALIGN, TP_SLAB_LABEL, slot_layout};

/// 2026-10-14: GLM-5.3 at TP=3: codes and scales of gate, up, down for slice widths 704 (ranks 0
/// and 1) and 640 (rank 2) at hidden 4096.
fn glm_sizes(len: usize) -> [usize; 6] {
    let (codes, scales) = (len * 4096 / 2, len * 4096 / 16);
    [codes, scales, codes, scales, codes, scales]
}

#[test]
fn glm_buffers_pack_with_no_padding() {
    for len in [704, 640] {
        let s = glm_sizes(len);
        let l = slot_layout(&s, SLAB_ALIGN).unwrap();
        assert_eq!(l.stride, s.iter().sum::<usize>(), "len {len}");
        assert_eq!(l.offsets[1], s[0]);
        assert_eq!(l.offsets[5], s[..5].iter().sum::<usize>());
    }
    // 2026-10-14: 288 experts of width 704 make a slab of 1336.5 MiB: one large allocation.
    let l = slot_layout(&glm_sizes(704), SLAB_ALIGN).unwrap();
    assert_eq!(l.stride * 288, 2673 << 19);
}

#[test]
fn offsets_are_aligned_and_buffers_do_not_overlap() {
    let sizes = [1usize, 4095, 4096, 4097, 0, 10, 8191];
    let l = slot_layout(&sizes, SLAB_ALIGN).unwrap();
    assert_eq!(l.sizes, sizes);
    let mut end = 0;
    for (i, (&off, &len)) in l.offsets.iter().zip(&sizes).enumerate() {
        assert_eq!(off % SLAB_ALIGN, 0, "buffer {i} at {off}");
        assert!(
            off >= end,
            "buffer {i} at {off} overlaps the previous end {end}"
        );
        end = off + len;
    }
    assert_eq!(l.offsets, [0, 4096, 8192, 12288, 20480, 20480, 24576]);
    assert_eq!(l.stride, 32768);
    assert_eq!(l.stride % SLAB_ALIGN, 0);
}

#[test]
fn a_zero_alignment_and_an_overflowing_slot_are_refused() {
    assert!(slot_layout(&[16], 0).is_err());
    // 2026-10-14: One overflows aligning up, the other adding the next buffer.
    assert!(slot_layout(&[usize::MAX - 8], SLAB_ALIGN).is_err());
    assert!(slot_layout(&[usize::MAX & !4095, usize::MAX], SLAB_ALIGN).is_err());
}

fn store() -> WeightStore {
    WeightStore::from_map(HashMap::new())
}

fn blobs(id: usize) -> [Vec<u8>; 3] {
    [5000usize, 300, 4096].map(|n| (0..n).map(|i| ((i * 7 + id * 31) % 251) as u8).collect())
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, len: usize) -> Vec<u8> {
    let mut v = vec![0u8; len];
    gpu.copy_d2h(p, &mut v).unwrap();
    v
}

/// 2026-10-14: Experts placed out of order land in their own slots: one allocation of every
/// slot, adopted once, and each expert's bytes read back from its pointers.
#[test]
fn a_layer_is_one_allocation_adopted_once() {
    let gpu = MockGpuBackend::new();
    let st = store();
    let slab = ExpertSlab::new(10..13, TP_SLAB_LABEL);
    let mut got = Vec::new();
    for id in [12, 10, 11] {
        let b = blobs(id);
        let p = slab
            .place(&gpu, &st, id, [&b[0][..], &b[1], &b[2]])
            .unwrap();
        got.push((id, p));
    }
    let stride = slot_layout(&[5000, 300, 4096], SLAB_ALIGN).unwrap().stride;
    assert_eq!(st.derived().len(), 1, "one adoption for the layer");
    assert_eq!(st.derived().bytes(), 3 * stride);
    let base = got.iter().find(|g| g.0 == 10).unwrap().1[0];
    assert!(
        gpu.read_alloc(base).is_some(),
        "slot 0 starts the allocation"
    );
    for (id, p) in got {
        assert_eq!(p[0].0, base.0 + ((id - 10) * stride) as u64);
        for (b, ptr) in blobs(id).iter().zip(p) {
            assert_eq!(read(&gpu, ptr, b.len()), *b, "expert {id}");
        }
    }
    st.derived().release(&gpu).unwrap();
}

#[test]
fn an_expert_outside_the_slab_or_placed_twice_is_refused() {
    let gpu = MockGpuBackend::new();
    let st = store();
    let slab = ExpertSlab::new(4..6, TP_SLAB_LABEL);
    let b = blobs(4);
    let e = slab.place(&gpu, &st, 6, [&b[0][..]]).unwrap_err();
    assert!(e.to_string().contains("outside"), "{e}");
    assert_eq!(st.derived().len(), 0, "nothing allocated for a refused id");
    slab.place(&gpu, &st, 4, [&b[0][..]]).unwrap();
    let e = slab.place(&gpu, &st, 4, [&b[0][..]]).unwrap_err();
    assert!(e.to_string().contains("twice"), "{e}");
}

#[test]
fn an_expert_of_other_sizes_is_refused() {
    let gpu = MockGpuBackend::new();
    let st = store();
    let slab = ExpertSlab::new(0..2, TP_SLAB_LABEL);
    let b = blobs(0);
    slab.place(&gpu, &st, 0, [&b[0][..], &b[1]]).unwrap();
    let e = slab.place(&gpu, &st, 1, [&b[0][..], &b[2]]).unwrap_err();
    assert!(e.to_string().contains("first expert"), "{e}");
}
