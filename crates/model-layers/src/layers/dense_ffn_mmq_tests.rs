// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A missing MMQ entry point leaves the transposed weights in place: no repack, allocation, launch or sync.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::OpsLevers;
use crate::lever_equivalence::{DefaultLoad, PROBES, settings_from};
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
    let mut layer = DenseFfnLayer::new(
        weights,
        &gpu,
        crate::lever_equivalence::DefaultLoad::at_defaults().lc(),
    )
    .unwrap();
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
    stamped_layer_under(gpu, acts, &DefaultLoad::at_defaults())
}

/// 2026-10-17: [`stamped_layer`] constructed under `load`'s levers.
fn stamped_layer_under(
    gpu: &MockGpuBackend,
    acts: [metrale_config::Nvfp4Act; 3],
    load: &DefaultLoad,
) -> DenseFfnLayer {
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
    DenseFfnLayer::new(weights, gpu, load.lc()).unwrap()
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

/// 2026-10-17: The 16/32/64-row tile handles over every pair of the two kill switches' probes
/// the settings accept, against `mmq_small_tile_enabled` / `mmq_tile64_enabled` (frozen below):
/// the dispatch ran a tile when its handle was non-zero and its readers said on, and now runs it
/// when its handle is non-zero. The settings refuse `garbage`, and `true` now switches a tile
/// off where the old `== "1"` read it as on.
#[test]
fn the_small_mmq_tiles_follow_their_kill_switches_as_before() {
    use metrale_config::Nvfp4Act::A4;
    const SMALL: &str = "METRALE_NO_MMQ_SMALL_TILE";
    const TILE64: &str = "METRALE_NO_MMQ_TILE64";
    let old_on = |v: Option<&str>| v != Some("1");
    let gpu = MockGpuBackend::new();
    for small in PROBES {
        for tile64 in PROBES {
            let mut pairs = Vec::new();
            pairs.extend(small.map(|v| (SMALL, v)));
            pairs.extend(tile64.map(|v| (TILE64, v)));
            let Ok(settings) = settings_from(&pairs) else {
                assert!(small == Some("garbage") || tile64 == Some("garbage"));
                continue;
            };
            let mut load = DefaultLoad::at_defaults();
            load.levers.ops = OpsLevers::from_settings(&settings).unwrap();
            let l = stamped_layer_under(&gpu, [A4, A4, A4], &load);
            let new = [
                l.nvfp4_mmq16_nc_k,
                l.nvfp4_mmq16_wc_k,
                l.nvfp4_mmq32_nc_k,
                l.nvfp4_mmq32_wc_k,
                l.nvfp4_mmq64_nc_k,
                l.nvfp4_mmq64_wc_k,
            ]
            .map(|h| h.0 != 0);
            let switched = |v: Option<&str>| v == Some("true");
            let s = old_on(small) && !switched(small);
            let t = s && old_on(tile64) && !switched(tile64);
            assert_eq!(new, [s, s, s, s, t, t], "{small:?} {tile64:?}");
            assert_ne!(
                l.nvfp4_mmq_nc_k.0, 0,
                "the 128-row tile is never switched off"
            );
        }
    }
}

/// 2026-10-17: `finalize_nvfp4_mmq_load` reads `ffn_nvfp4_mmq` / `ffn_nvfp4_mmq_down` from the
/// levers the layer was built under, where it read their variables' presence.
#[test]
fn the_nvfp4_mmq_finalizer_follows_the_construction_levers() {
    use metrale_config::Nvfp4Act::A4;
    let gpu = MockGpuBackend::new();
    let built = |gate_up: bool, down: bool| {
        let mut load = DefaultLoad::at_defaults();
        load.levers.ffn_nvfp4_mmq = gate_up;
        load.levers.ffn_nvfp4_mmq_down = down;
        let mut l = stamped_layer_under(&gpu, [A4, A4, A4], &load);
        l.finalize_nvfp4_mmq_load(&gpu, 64, 64, 0).unwrap();
        (l.fp4mmq_gate.get().is_some(), l.fp4mmq_down.get().is_some())
    };
    assert_eq!(built(true, true), (true, true));
    assert_eq!(built(true, false), (true, false));
    assert_eq!(built(false, true), (false, false));
    assert_eq!(built(false, false), (false, false));
}

/// 2026-10-17: `finalize_q4k_load` reads `ffn_mmq` / `ffn_mmq_down_q4k` from the levers the layer
/// was built under: off builds nothing; on builds gate/up Q4_K and an int8 down unless
/// `ffn_mmq_down_q4k` keeps down on Q4_K.
#[test]
fn the_q4k_finalizer_follows_the_construction_levers() {
    use metrale_config::Nvfp4Act::Unstamped;
    let gpu = MockGpuBackend::new();
    let built = |mmq: bool, down_q4k: bool| {
        let mut load = DefaultLoad::at_defaults();
        load.levers.ffn_mmq = mmq;
        load.levers.ffn_mmq_down_q4k = down_q4k;
        let mut l = stamped_layer_under(&gpu, [Unstamped; 3], &load);
        l.finalize_q4k_load(&gpu, 64, 64, 0).unwrap();
        (
            l.q4k_gate.get().is_some(),
            l.q4k_down.get().is_some(),
            l.int8_down.get().is_some(),
        )
    };
    assert_eq!(built(false, false), (false, false, false));
    assert_eq!(built(false, true), (false, false, false));
    assert_eq!(built(true, false), (true, false, true));
    assert_eq!(built(true, true), (true, true, false));
}
