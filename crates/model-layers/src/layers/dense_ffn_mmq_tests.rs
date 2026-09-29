// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A missing MMQ entry point leaves the transposed weights in place: no repack, allocation, launch or sync.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::weight_map::QuantizedWeight;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

#[test]
fn unavailable_mmq_preserves_transposed_weights_without_repack_or_free() {
    let gpu = MockGpuBackend::new();
    let mut weight = QuantizedWeight::null();
    weight.weight = gpu.alloc(64).unwrap();
    weight.weight_scale = gpu.alloc(8).unwrap();
    let weights = DenseFfnWeights {
        gate_proj: weight,
        up_proj: weight,
        down_proj: weight,
        gate_proj_t: Some(weight),
        up_proj_t: Some(weight),
        down_proj_t: Some(weight),
    };
    let mut layer = DenseFfnLayer::new(weights, &gpu).unwrap();
    // 2026-09-25: `try_kernel` returns handle 0 for an absent entry point. The
    // other handles stay present: one missing handle disables the whole MMQ
    // load path, including the free of the transposed weights.
    layer.nvfp4_mmq_nc_k = KernelHandle(0);
    let allocations = gpu.alloc_count();
    let launches = gpu.launch_count();
    layer.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
    assert_eq!(gpu.alloc_count(), allocations);
    assert_eq!(gpu.launch_count(), launches);
    assert_eq!(gpu.sync_count(), 0);
    for transpose in [
        layer.weights.gate_proj_t,
        layer.weights.up_proj_t,
        layer.weights.down_proj_t,
    ] {
        assert_eq!(transpose.unwrap().weight, weight.weight);
    }
    assert!(layer.fp4mmq_gate.get().is_none());
    assert!(layer.fp4mmq_up.get().is_none());
    assert!(layer.fp4mmq_down.get().is_none());
}

/// 2026-09-28: A layer with gate, up and down stamped `act`, each with its own transposed
/// twin, on the mock (every kernel present).
fn stamped_layer(gpu: &MockGpuBackend, acts: [metrale_config::Nvfp4Act; 3]) -> DenseFfnLayer {
    let w = |act| {
        let mut q = QuantizedWeight::null();
        q.weight = gpu.alloc(64).unwrap();
        q.weight_scale = gpu.alloc(8).unwrap();
        q.act = act;
        q
    };
    let (g, u, d) = (w(acts[0]), w(acts[1]), w(acts[2]));
    let weights = DenseFfnWeights {
        gate_proj: g,
        up_proj: u,
        down_proj: d,
        gate_proj_t: Some(w(acts[0])),
        up_proj_t: Some(w(acts[1])),
        down_proj_t: Some(w(acts[2])),
    };
    DenseFfnLayer::new(weights, gpu).unwrap()
}

/// 2026-09-28: The FP4 MMQ prefill arm quantizes activations to FP4, so it follows the
/// weight-quantization stamps. A `Wide` gate/up (the checkpoint declares wider activations)
/// builds no repack and keeps every transposed twin for the W4A16 GEMMs; a `Wide` down keeps
/// its own twin while an A4 gate/up takes the arm; unstamped weights (the `nvfp4` tier) take
/// it as before the tiers.
#[test]
fn the_fp4_mmq_arm_follows_the_stamps() {
    use metrale_config::Nvfp4Act::{A4, Unstamped, Wide};
    let gpu = MockGpuBackend::new();
    let mut wide = stamped_layer(&gpu, [Wide, Wide, Wide]);
    assert!(!wide.mmq_gate_up_declared());
    wide.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
    assert!(wide.fp4mmq_gate.get().is_none() && wide.fp4mmq_down.get().is_none());
    assert!(wide.weights.gate_proj_t.is_some() && wide.weights.up_proj_t.is_some());
    assert!(wide.weights.down_proj_t.is_some());

    let mut down_wide = stamped_layer(&gpu, [A4, A4, Wide]);
    down_wide.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
    assert!(down_wide.fp4mmq_gate.get().is_some() && down_wide.fp4mmq_up.get().is_some());
    assert!(down_wide.fp4mmq_down.get().is_none());
    assert!(down_wide.weights.gate_proj_t.is_none() && down_wide.weights.up_proj_t.is_none());
    assert!(down_wide.weights.down_proj_t.is_some());

    let mut legacy = stamped_layer(&gpu, [Unstamped, Unstamped, Unstamped]);
    legacy.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
    assert!(legacy.fp4mmq_down.get().is_some());
    assert!(legacy.weights.down_proj_t.is_none() && legacy.weights.gate_proj_t.is_none());
}
