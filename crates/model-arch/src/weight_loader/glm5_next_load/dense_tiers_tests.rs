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

const D: DsaQuantization = DsaQuantization::Declared;

fn both<T: Copy + PartialEq>(t: T) -> Phased<T> {
    Phased::both(t)
}

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

/// 2026-10-09: The split is all-FP8 under `fp8`, by `tier_of` under `w4a16`, empty under
/// `declared`, and keeps the layer's order.
#[test]
fn the_split_is_all_fp8_unless_w4a16() {
    use DenseQuantization as Q;
    let gpu = MockGpuBackend::new();
    let projs = layer(&gpu);
    let names = |v: &[Proj]| v.iter().map(|p| p.3).collect::<Vec<_>>();
    let (fp8, w4) = split_by_tier(projs.clone(), Q::Fp8, D).unwrap();
    assert_eq!(names(&fp8), names(&projs));
    assert!(w4.is_empty());
    let (fp8, w4) = split_by_tier(projs.clone(), Q::W4a16, D).unwrap();
    assert_eq!(names(&fp8), vec!["kda.f_b_proj", "kda.g_b_proj"]);
    assert_eq!(w4.len(), 10);
    let (fp8, w4) = split_by_tier(projs, Q::Declared, D).unwrap();
    assert!(fp8.is_empty() && w4.is_empty());
    let e = split_by_tier(vec![(DevicePtr(1), 1, 1, "dsa.wq_b")], Q::W4a16, D).unwrap_err();
    assert!(e.to_string().contains("no tier decided"), "{e}");
    assert!(split_by_tier(vec![(DevicePtr(1), 1, 1, "dsa.wq_b")], Q::Fp8, D).is_ok());
}

/// 2026-10-09: Under `w4a16` the ten W4A16 fields hold their NVFP4 keys and their BF16 buffers
/// are gone; f_b and g_b keep their BF16 (registered FP8). Under `fp8` nothing moves or is freed.
#[test]
fn w4a16_retargets_and_frees_exactly_its_set() {
    for on in [false, true] {
        let dense = if on {
            DenseQuantization::W4a16
        } else {
            DenseQuantization::Fp8
        };
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
            register_projections(&gpu, projs.clone(), &mut slots, both(dense), both(D), 5).unwrap();
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
        both(DenseQuantization::W4a16),
        both(D),
        9,
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("layer 9 kda.o_proj is [64, 2688]"), "{e}");
    assert_eq!(field, w);
    assert!(is_live(&gpu, w));
}

/// 2026-10-11: `--dsa-quantization` moves exactly the DSA q_a / absorbed q / absorbed o, and
/// `--dense-quantization` everything else, in all nine combinations, on a full DSA layer's six
/// projections plus a shared expert's three: `declared` leaves a projection unregistered (its
/// BF16 field untouched and live), `w8a8`/`fp8` registers it FP8 beside its BF16, `w4a16`
/// retargets its field to an NVFP4 key and frees its BF16.
#[test]
fn each_flag_moves_exactly_its_own_projections() {
    use DenseQuantization as Q;
    use DsaQuantization as S;
    let sites = ["dsa.q_a_proj", "dsa.q_absorb", "dsa.o_absorb"];
    for dense in Q::ALL {
        for dsa in S::ALL {
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
                (256, hid, "shared_experts.gate_proj"),
                (256, hid, "shared_experts.up_proj"),
                (hid, 256, "shared_experts.down_proj"),
            ]
            .into_iter()
            .map(|(n, k, name)| (gpu.alloc(n * k * 2).unwrap(), n, k, name))
            .collect();
            let mut fields: Vec<DevicePtr> = projs.iter().map(|p| p.0).collect();
            let fp8_before = crate::glm5next_fp8_dense::registered().0;
            {
                let mut slots: Vec<&mut DevicePtr> = fields.iter_mut().collect();
                register_projections(&gpu, projs.clone(), &mut slots, both(dense), both(dsa), 3)
                    .unwrap();
            }
            let mut fp8_expected = 0;
            for (i, p) in projs.iter().enumerate() {
                let want = if sites.contains(&p.3) {
                    match dsa {
                        S::Declared => Tier::Bf16,
                        S::W8a8 => Tier::Fp8,
                        S::W4a16 => Tier::W4a16,
                    }
                } else {
                    match dense {
                        Q::Declared => Tier::Bf16,
                        Q::Fp8 => Tier::Fp8,
                        Q::W4a16 if p.3.starts_with("shared_experts.") => Tier::W4a16,
                        Q::W4a16 => Tier::Fp8,
                    }
                };
                assert_eq!(
                    tier_for(p.3, dense, dsa).unwrap(),
                    want,
                    "{} dense {dense:?} dsa {dsa:?}",
                    p.3
                );
                let moved = want == Tier::W4a16;
                fp8_expected += usize::from(want == Tier::Fp8);
                assert_eq!(
                    fields[i] != p.0,
                    moved,
                    "{} dense {dense:?} dsa {dsa:?}",
                    p.3
                );
                assert_eq!(is_live(&gpu, p.0), !moved, "{}: BF16 freed iff moved", p.3);
            }
            assert_eq!(
                crate::glm5next_fp8_dense::registered().0 - fp8_before,
                fp8_expected,
                "FP8 registrations, dense {dense:?} dsa {dsa:?}"
            );
            assert_eq!(
                any_below_declared(dense, dsa),
                dense != Q::Declared || dsa != S::Declared
            );
        }
    }
}

/// 2026-10-14: `--dense-quantization decode=w4a16,prefill=fp8` registers exactly the plain
/// `w4a16` set plus one FP8 copy per W4A16 projection, keyed by its NVFP4 key, which only a
/// prompt pass finds; the BF16 is still freed. `--dsa-quantization decode=w4a16,prefill=w8a8`
/// does the same for the DSA sites alone.
#[test]
fn a_phase_split_adds_one_prompt_pass_copy_per_w4a16_projection() {
    use crate::glm5next_fp8_dense::{has_prefill_copy, registered, registered_prefill};
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    let projs = layer(&gpu);
    let mut fields: Vec<DevicePtr> = projs.iter().map(|p| p.0).collect();
    let split = Phased {
        decode: DenseQuantization::W4a16,
        prefill: DenseQuantization::Fp8,
    };
    {
        let mut slots: Vec<&mut DevicePtr> = fields.iter_mut().collect();
        register_projections(&gpu, projs.clone(), &mut slots, split, both(D), 5).unwrap();
    }
    assert_eq!(crate::glm5next_w4a16_dense::registered().0, 10);
    assert_eq!(registered().0, 2, "f_b and g_b, as under plain w4a16");
    assert_eq!(registered_prefill().0, 10);
    for (p, field) in projs.iter().zip(&fields) {
        let moved = !matches!(p.3, "kda.f_b_proj" | "kda.g_b_proj");
        assert_eq!(*field != p.0, moved, "{}", p.3);
        assert_eq!(is_live(&gpu, p.0), !moved, "{} BF16 freed iff moved", p.3);
        assert!(
            !has_prefill_copy(*field),
            "{}: decode never sees a copy",
            p.3
        );
        let _phase = crate::glm5next_phase::enter_prefill();
        assert_eq!(has_prefill_copy(*field), moved, "{}", p.3);
    }

    // 2026-10-14: From empty registries again.
    drop(_serial);
    let _serial = crate::glm5next_fp8_dense::lock_registries_for_test();
    let w = gpu.alloc(256 * 512 * 2).unwrap();
    let mut field = w;
    let dsa_split = Phased {
        decode: DsaQuantization::W4a16,
        prefill: DsaQuantization::W8a8,
    };
    register_projections(
        &gpu,
        vec![(w, 256, 512, "dsa.q_a_proj")],
        &mut [&mut field],
        both(DenseQuantization::Declared),
        dsa_split,
        2,
    )
    .unwrap();
    assert_ne!(field, w);
    assert_eq!(registered_prefill().0, 1);
    let _phase = crate::glm5next_phase::enter_prefill();
    assert!(has_prefill_copy(field));
}

/// 2026-10-14: A phase pair other than W4A16 decode with an FP8 prompt pass is refused before
/// anything registers.
#[test]
fn an_unserved_phase_pair_is_refused() {
    let gpu = MockGpuBackend::new();
    let projs = layer(&gpu);
    for (dense, bad) in [
        (
            Phased {
                decode: DenseQuantization::Fp8,
                prefill: DenseQuantization::W4a16,
            },
            "kda.q_proj: decode Fp8 with prefill W4a16",
        ),
        (
            Phased {
                decode: DenseQuantization::Declared,
                prefill: DenseQuantization::Fp8,
            },
            "kda.q_proj: decode Bf16 with prefill Fp8",
        ),
    ] {
        let e = prefill_copies(&projs, dense, both(D))
            .unwrap_err()
            .to_string();
        assert!(e.contains(bad), "{e}");
    }
    let names = prefill_copies(
        &projs,
        Phased {
            decode: DenseQuantization::W4a16,
            prefill: DenseQuantization::Fp8,
        },
        both(D),
    )
    .unwrap();
    assert_eq!(names.len(), 10);
    assert!(!names.contains(&"kda.f_b_proj") && !names.contains(&"kda.g_b_proj"));
}
