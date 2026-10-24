// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests of the parallel `O_DIRECT` expert preparation: every prepared projection
//! equals the sequential buffered one and the slicer's cut of the shard bytes, for packed and
//! 16-bit experts on every rank; a moved offset is caught; a resident tensor and a second take are
//! refused.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_model_weights::weights::{DeferredTensor, WeightDtype, WeightStore};

use super::super::expert_tp::host_expert;
use super::super::expert_tp::tests::{HIDDEN, LAYER, bytes_of, packed_expert_id, qualified, shard};
use super::{Buffered, Prepared, WAVE};
use crate::glm5next_mlp::expert_tp::{
    ExpertSlice, expert_slice, slice_expert_cols, slice_expert_rows,
};

type Tensors = Vec<(String, WeightDtype, Vec<usize>, Vec<u8>)>;

/// 2026-10-24: Gate, up and down of expert `id` as the slicer cuts them from the shard bytes in
/// `t`, the oracle both read paths must match.
fn oracle(t: &Tensors, id: usize, s: &ExpertSlice) -> [(Vec<u8>, Vec<u8>); 3] {
    ["gate_proj", "up_proj", "down_proj"].map(|p| {
        let (w, sc) = (
            bytes_of(t, &format!("mlp.experts.{id}.{p}.weight")),
            bytes_of(t, &format!("mlp.experts.{id}.{p}.weight_scale")),
        );
        if p == "down_proj" {
            slice_expert_cols(&w, &sc, HIDDEN, s).unwrap()
        } else {
            slice_expert_rows(&w, &sc, HIDDEN, s).unwrap()
        }
    })
}

/// 2026-10-24: The positive half: on every rank of a padded width, each prepared expert is the
/// sequential path's, scales included, and both are the oracle's cut.
#[test]
fn prepared_packed_experts_equal_the_sequential_reads() {
    let (full, ids) = (336usize, 3usize..7);
    let t: Tensors = ids
        .clone()
        .flat_map(|id| packed_expert_id(full, id))
        .collect();
    let sh = shard("prep-packed", &t);
    let gpu = MockGpuBackend::new();
    for r in 0..3 {
        let s = expert_slice(full, 3, r).unwrap();
        let prepared = Prepared::new(&sh.store, LAYER, ids.clone(), &s);
        for id in ids.clone() {
            let seq = host_expert(
                &sh.store,
                &Buffered {
                    gpu: &gpu,
                    store: &sh.store,
                },
                LAYER,
                id,
                &s,
            )
            .unwrap();
            let par = prepared.take(id).unwrap();
            assert_eq!(par, seq, "r{r} expert {id}");
            for (i, want) in oracle(&t, id, &s).into_iter().enumerate() {
                assert_eq!(
                    (par[i].0.clone(), par[i].1.clone()),
                    want,
                    "r{r} e{id} proj {i}"
                );
            }
            assert_eq!(
                par.map(|h| (h.2, h.3)),
                [(0.5, Some(1.0)), (0.25, Some(1.0)), (0.125, Some(2.0))]
            );
        }
    }
}

/// 2026-10-24: Across wave boundaries, taken in order and in reverse, every expert is still the
/// sequential path's.
#[test]
fn waves_hand_over_every_expert_in_any_order() {
    let (full, ids) = (336usize, 1usize..WAVE + 4);
    let t: Tensors = ids
        .clone()
        .flat_map(|id| packed_expert_id(full, id))
        .collect();
    let sh = shard("prep-waves", &t);
    let gpu = MockGpuBackend::new();
    let s = expert_slice(full, 3, 1).unwrap();
    let seq = |id| {
        let b = Buffered {
            gpu: &gpu,
            store: &sh.store,
        };
        host_expert(&sh.store, &b, LAYER, id, &s).unwrap()
    };
    let forward = Prepared::new(&sh.store, LAYER, ids.clone(), &s);
    for id in ids.clone() {
        assert_eq!(forward.take(id).unwrap(), seq(id), "forward {id}");
    }
    let reverse = Prepared::new(&sh.store, LAYER, ids.clone(), &s);
    for id in ids.clone().rev() {
        assert_eq!(reverse.take(id).unwrap(), seq(id), "reverse {id}");
    }
    assert!(reverse.take(ids.start).is_err(), "taken twice");
}

/// 2026-10-24: A 16-bit expert (the MTP layer's form) quantized on the worker pool is the
/// sequential quantization, bit for bit.
#[test]
fn prepared_bf16_experts_equal_the_sequential_quantization() {
    let full = 256usize;
    let ramp = |n: usize, salt: f32| -> Vec<u8> {
        (0..n)
            .flat_map(|i| {
                half::bf16::from_f32(((i as f32 * 0.37 + salt).sin()) * 3.0).to_le_bytes()
            })
            .collect()
    };
    let t: Tensors = (0..3usize)
        .flat_map(|id| {
            [
                ("gate_proj", vec![full, HIDDEN]),
                ("up_proj", vec![full, HIDDEN]),
                ("down_proj", vec![HIDDEN, full]),
            ]
            .into_iter()
            .enumerate()
            .map(move |(i, (p, shape))| {
                let name = qualified(&format!("mlp.experts.{id}.{p}.weight"));
                (
                    name,
                    WeightDtype::BF16,
                    shape,
                    ramp(full * HIDDEN, (id * 3 + i) as f32),
                )
            })
        })
        .collect();
    let sh = shard("prep-bf16", &t);
    let gpu = MockGpuBackend::new();
    let s = expert_slice(full, 3, 1).unwrap();
    let prepared = Prepared::new(&sh.store, LAYER, 0..3, &s);
    for id in 0..3 {
        let seq = host_expert(
            &sh.store,
            &Buffered {
                gpu: &gpu,
                store: &sh.store,
            },
            LAYER,
            id,
            &s,
        )
        .unwrap();
        assert_eq!(prepared.take(id).unwrap(), seq, "expert {id}");
    }
}

/// 2026-10-24: The negative half: a store whose one tensor points one byte off yields an expert
/// that differs from the oracle, so the equality above would catch a wrong offset.
#[test]
fn a_moved_offset_is_caught() {
    let full = 336usize;
    let t: Tensors = packed_expert_id(full, 5);
    let sh = shard("prep-moved", &t);
    let moved = qualified("mlp.experts.5.up_proj.weight");
    let mut store = WeightStore::from_map(std::collections::HashMap::new());
    for (name, d) in sh.store.deferred_sorted() {
        let off = d.offset + u64::from(*name == moved);
        store.defer(
            name.clone(),
            DeferredTensor::from_shape(d.path.clone(), off, d.shape.clone(), d.dtype).unwrap(),
        );
    }
    let s = expert_slice(full, 3, 0).unwrap();
    let got = Prepared::new(&store, LAYER, 5..6, &s).take(5).unwrap();
    let want = oracle(&t, 5, &s);
    assert_eq!(
        (got[0].0.clone(), got[0].1.clone()),
        want[0],
        "gate untouched"
    );
    assert_ne!(got[1].0, want[1].0, "the moved up_proj codes must differ");
}

/// 2026-10-24: An expert is handed over once, only for the layer's ids; a resident tensor is
/// refused by name with the lever that reads it.
#[test]
fn take_once_in_range_and_resident_is_refused() {
    let full = 336usize;
    let t: Tensors = packed_expert_id(full, 2);
    let sh = shard("prep-take", &t);
    let s = expert_slice(full, 3, 2).unwrap();
    let p = Prepared::new(&sh.store, LAYER, 2..3, &s);
    assert!(p.take(3).unwrap_err().to_string().contains("outside"));
    p.take(2).unwrap();
    assert!(p.take(2).unwrap_err().to_string().contains("twice"));

    let gpu = MockGpuBackend::new();
    let mut map = std::collections::HashMap::new();
    let name = qualified("mlp.experts.2.gate_proj.weight");
    let ptr = metrale_gpu_runtime::gpu::GpuBackend::alloc(&gpu, 16).unwrap();
    map.insert(
        name.clone(),
        metrale_model_weights::weights::WeightTensor {
            ptr,
            shape: vec![full, HIDDEN / 2],
            dtype: WeightDtype::UInt8,
        },
    );
    let mut store = WeightStore::from_map(map);
    for (n, d) in sh.store.deferred_sorted() {
        if *n != name {
            store.defer(n.clone(), d.clone());
        }
    }
    let e = Prepared::new(&store, LAYER, 2..3, &s)
        .take(2)
        .expect_err("resident must be refused");
    let e = format!("{e:#}");
    assert!(e.contains("METRALE_GLM_EXPERT_FAST_LOAD=0"), "{e}");
}

/// 2026-10-24: The digest lines move with any one byte of any buffer and with the scale bits.
#[test]
fn digest_lines_see_one_byte_and_scale_bits() {
    use super::super::expert_digest::lines;
    let device: [Vec<u8>; 6] = std::array::from_fn(|i| vec![i as u8; 64 + i]);
    let sc = [(0.5f32, Some(1.0f32)), (0.25, None), (0.125, Some(2.0))];
    let base = lines(4, 7, &device, &sc);
    assert_eq!(base.len(), 9);
    assert!(
        base[0].starts_with("L4 E7 gate_proj codes 64 "),
        "{}",
        base[0]
    );
    for i in 0..6 {
        let mut d = device.clone();
        d[i][63] ^= 1;
        let l = lines(4, 7, &d, &sc);
        assert_ne!(l[i], base[i], "buffer {i}");
        assert_eq!(l[..i], base[..i]);
    }
    let l = lines(
        4,
        7,
        &device,
        &[(0.5, Some(1.0)), (0.25, Some(1.0)), (0.125, Some(2.0))],
    );
    assert_ne!(l[7], base[7]);
}
