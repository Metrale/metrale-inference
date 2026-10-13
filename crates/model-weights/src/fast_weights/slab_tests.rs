// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Which tensors share a shard's slab, and the per-tensor fallback.

use super::{SLAB_MAX_TENSOR_BYTES, alloc_shard_slab, slab_eligible};
use crate::fast_weights::header::TensorMeta;
use crate::weights::WeightDtype;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

fn meta(name: &str, dtype: WeightDtype, len: usize) -> TensorMeta {
    TensorMeta {
        name: name.to_owned(),
        dtype,
        from_f16: false,
        shape: vec![len],
        abs_offset: 0,
        len,
    }
}

fn shard() -> Vec<TensorMeta> {
    vec![
        meta("experts.0.gate_proj.weight", WeightDtype::UInt8, 819_200),
        meta("self_attn.q_proj.weight", WeightDtype::BF16, 4096),
        meta(
            "experts.0.gate_proj.weight_scale",
            WeightDtype::FP8E4M3,
            102_400,
        ),
        meta("experts.0.gate_proj.weight_scale_2", WeightDtype::FP32, 4),
        meta("lm_head.weight", WeightDtype::UInt8, SLAB_MAX_TENSOR_BYTES),
    ]
}

#[test]
fn bf16_and_large_tensors_keep_their_own_allocation() {
    let t = shard();
    let got: Vec<bool> = t.iter().map(slab_eligible).collect();
    assert_eq!(got, [true, false, true, true, false]);
    assert!(slab_eligible(&meta(
        "x",
        WeightDtype::UInt8,
        SLAB_MAX_TENSOR_BYTES - 1
    )));
}

#[test]
fn eligible_tensors_get_distinct_pointers_in_order_and_others_none() {
    let gpu = MockGpuBackend::new();
    let t = shard();
    let ptrs = alloc_shard_slab(&gpu, &t);
    assert_eq!(ptrs.len(), t.len());
    assert!(ptrs[1].is_none() && ptrs[4].is_none());
    let (a, b, c) = (ptrs[0].unwrap(), ptrs[2].unwrap(), ptrs[3].unwrap());
    assert!(a.0 + 819_200 <= b.0, "first member overlaps the second");
    assert!(b.0 + 102_400 <= c.0, "second member overlaps the third");
    assert_eq!(gpu.alloc_count(), 3);
}

#[test]
fn a_failed_slab_falls_back_to_per_tensor_without_leaking() {
    let gpu = MockGpuBackend::new();
    // The mock's default `alloc_slab` allocates per size: the scale fits, then the 819,200 B
    // weight is refused. Every entry must come back None and the scale must be freed again.
    gpu.set_max_allocation_bytes(200_000);
    let t = vec![
        meta("s", WeightDtype::FP8E4M3, 102_400),
        meta("w", WeightDtype::UInt8, 819_200),
    ];
    let ptrs = alloc_shard_slab(&gpu, &t);
    assert!(ptrs.iter().all(Option::is_none));
    assert_eq!(
        gpu.alloc_count(),
        0,
        "the default alloc_slab left an allocation behind"
    );
    assert_eq!(gpu.live_bytes(), Some(0));
}

#[test]
fn a_shard_with_nothing_eligible_allocates_nothing() {
    let gpu = MockGpuBackend::new();
    let ptrs = alloc_shard_slab(&gpu, &[meta("q", WeightDtype::BF16, 4096)]);
    assert_eq!(ptrs, [None]);
    assert_eq!(gpu.alloc_count(), 0);
}
