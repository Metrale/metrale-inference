// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The per-layer dense-tier registration on a recording mock backend: under `fp8`
//! every projection registers FP8 in order and keeps its BF16; under `w4a16` the W4A16 set is
//! quantized to NVFP4, its weight fields point at the NVFP4 keys and its BF16 is freed, while the
//! rest registers FP8 untouched; an unsupported width is refused before anything is freed.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::*;

/// 2026-10-09: A KDA layer's projections at hidden 512 with two 128-wide heads (K 256 for
/// o_proj), plus a 256-wide shared expert, each in its own allocation.
fn layer(gpu: &MockGpuBackend) -> Vec<Proj> {
    let (hid, ch, hd) = (512, 256, 128);
    [
        (ch, hid, "kda.q_proj"),
        (ch, hid, "kda.k_proj"),
        (ch, hid, "kda.v_proj"),
        (hd, hid, "kda.f_a_proj"),
        (ch, hd, "kda.f_b_proj"),
        (2, hid, "kda.b_proj"),
        (hd, hid, "kda.g_a_proj"),
        (ch, hd, "kda.g_b_proj"),
        (hid, ch, "kda.o_proj"),
        (256, hid, "shared_experts.gate_proj"),
        (256, hid, "shared_experts.up_proj"),
        (hid, 256, "shared_experts.down_proj"),
    ]
    .into_iter()
    .map(|(n, k, name)| (gpu.alloc(n * k * 2).unwrap(), n, k, name))
    .collect()
}

fn is_live(gpu: &MockGpuBackend, p: DevicePtr) -> bool {
    let mut b = [0u8; 1];
    gpu.copy_d2h(p, &mut b).is_ok()
}

/// 2026-10-09: The split is all-FP8 unless the tier is on, and keeps the layer's order.
#[test]
fn the_split_is_all_fp8_unless_w4a16() {
    let gpu = MockGpuBackend::new();
    let projs = layer(&gpu);
    let names = |v: &[Proj]| v.iter().map(|p| p.3).collect::<Vec<_>>();
    let (fp8, w4) = split_by_tier(projs.clone(), false, false).unwrap();
    assert_eq!(names(&fp8), names(&projs));
    assert!(w4.is_empty());
    let (fp8, w4) = split_by_tier(projs, true, false).unwrap();
    assert_eq!(names(&fp8), vec!["kda.f_b_proj", "kda.g_b_proj"]);
    assert_eq!(w4.len(), 10);
    let e = split_by_tier(vec![(DevicePtr(1), 1, 1, "dsa.wq_b")], true, false).unwrap_err();
    assert!(e.to_string().contains("no tier decided"), "{e}");
    assert!(split_by_tier(vec![(DevicePtr(1), 1, 1, "dsa.wq_b")], false, false).is_ok());
}

/// 2026-10-09: Under `w4a16` the ten W4A16 fields hold their NVFP4 keys and their BF16 buffers
/// are gone; f_b and g_b keep their BF16 (registered FP8). Under `fp8` nothing moves or is freed.
#[test]
fn w4a16_retargets_and_frees_exactly_its_set() {
    for on in [false, true] {
        let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
        let gpu = MockGpuBackend::new();
        let projs = layer(&gpu);
        let mut fields: Vec<DevicePtr> = projs.iter().map(|p| p.0).collect();
        let (w4_before, fp8_before) = (
            crate::glm5next_w4a16_dense::registered().0,
            crate::glm5next_fp8_dense::registered().0,
        );
        {
            let mut slots: Vec<&mut DevicePtr> = fields.iter_mut().collect();
            register_projections(&gpu, projs.clone(), &mut slots, on, false, 5).unwrap();
        }
        let w4_new = crate::glm5next_w4a16_dense::registered().0 - w4_before;
        let fp8_new = crate::glm5next_fp8_dense::registered().0 - fp8_before;
        assert_eq!((w4_new, fp8_new), if on { (10, 2) } else { (0, 12) });
        for (p, field) in projs.iter().zip(&fields) {
            let moved = on && !matches!(p.3, "kda.f_b_proj" | "kda.g_b_proj");
            assert_eq!(*field != p.0, moved, "{} field", p.3);
            assert_eq!(is_live(&gpu, p.0), !moved, "{} BF16 buffer", p.3);
            if moved {
                assert!(is_live(&gpu, *field), "{} NVFP4 key is live", p.3);
            }
        }
    }
}

/// 2026-10-09: A W4A16 width the kernel refuses (o_proj at 21 heads: K 2688) fails the layer
/// before its BF16 is freed or its field moved.
#[test]
fn an_unsupported_width_is_refused_before_anything_is_freed() {
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let w = gpu.alloc(64 * 2688 * 2).unwrap();
    let mut field = w;
    let e = register_projections(
        &gpu,
        vec![(w, 64, 2688, "kda.o_proj")],
        &mut [&mut field],
        true,
        false,
        9,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("layer 9 kda.o_proj is [64, 2688]"), "{e}");
    assert_eq!(field, w);
    assert!(is_live(&gpu, w));
}

/// 2026-10-09: Under `METRALE_GLM_DSA_W4A16` the DSA q_a, absorbed q and absorbed o move to
/// W4A16 (retargeted, BF16 freed) and the rest of the DSA layer stays FP8; without it, under the
/// same tier, all six stay FP8 and keep their BF16.
#[test]
fn the_dsa_absorbed_set_moves_only_under_its_lever() {
    for dsa_absorbed in [false, true] {
        let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
        let gpu = MockGpuBackend::new();
        let (hid, ql, lat, kvl) = (512usize, 256usize, 512usize, 256usize);
        let projs: Vec<Proj> = [
            (ql, hid, "dsa.q_a_proj"),
            (lat, ql, "dsa.q_absorb"),
            (kvl, hid, "dsa.kv_a_proj"),
            (hid, lat, "dsa.o_absorb"),
            (128, hid, "dsa.indexer.wk"),
            (128, hid, "dsa.indexer.compress_gate"),
        ]
        .into_iter()
        .map(|(n, k, name)| (gpu.alloc(n * k * 2).unwrap(), n, k, name))
        .collect();
        let mut fields: Vec<DevicePtr> = projs.iter().map(|p| p.0).collect();
        {
            let mut slots: Vec<&mut DevicePtr> = fields.iter_mut().collect();
            register_projections(&gpu, projs.clone(), &mut slots, true, dsa_absorbed, 3).unwrap();
        }
        for (i, p) in projs.iter().enumerate() {
            let moved =
                dsa_absorbed && matches!(p.3, "dsa.q_a_proj" | "dsa.q_absorb" | "dsa.o_absorb");
            assert_eq!(
                fields[i] != p.0,
                moved,
                "{} (dsa_absorbed {dsa_absorbed})",
                p.3
            );
            assert_eq!(is_live(&gpu, p.0), !moved, "{}: BF16 freed iff moved", p.3);
        }
    }
}
