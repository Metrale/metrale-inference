// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The executor under `--weight-quantization declared`: the declared instance's
//! plans (W8A8 attention, GDN projections and the FFN of layers 56-63; W4A4 FFN elsewhere)
//! compiled over bindings of that shape and run on the recording mock backend. Each quantize
//! and projection launch is compared with what the legacy launcher (`ops::w8a8_proj`,
//! `ops::w4a4_proj::nvfp4_proj_small_m`) records for the same projection, and the quantized
//! edge is checked to be where the projection reads it.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{Circuit, LinearRole, Mode};
use metrale_config::Nvfp4Act;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::Fusions;
use super::bindings::*;
use super::exec_fixture::*;
use crate::layers::ops::{self, W8a8Kernels, W8a8Scratch, W8a8Weight};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};

const DECLARED: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth-declared";
/// 2026-09-30: The first layer whose MLP the checkpoint declares FP8 (W8A8).
const W8A8_FFN_FROM: usize = 56;

fn fp8(tag: u64, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: ptr(tag),
        row_scale: ptr(tag + 1),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8PerRow,
    }
}

fn a4(w: BoundWeight) -> BoundWeight {
    match w {
        BoundWeight::Nvfp4(q) => BoundWeight::Nvfp4(QuantizedWeight {
            act: Nvfp4Act::A4,
            ..q
        }),
        other => other,
    }
}

/// 2026-09-30: The loader's declared bindings of unsloth/Qwen3.8-27B-NVFP4 (`w8a8_install.rs`
/// and the layers' `circuit_layer`): FP8 weight tags at slots 40.. of the layer.
fn declared_binding(c: &Circuit, i: usize, attn: usize) -> (CircuitLayer, Vec<u64>) {
    let mut l = layer_binding(c, i, attn);
    let d = |k: &str| c.dims[k] as u32;
    let kernels = W8a8Kernels::load(&MockGpuBackend::new());
    let mut ptrs = Vec::new();
    let mut w8 = |slot: u64, n: u32, k: u32| {
        let f = fp8(tag(i, 40 + slot), n, k);
        ptrs.extend([f.weight.0, f.row_scale.0]);
        f
    };
    let h = d("hidden");
    let bound = |w: W8a8Weight| BoundWeight::W8a8(w, kernels);
    if matches!(l.mixer, MixerFacts::Attention(_)) {
        let (qd, kvd) = (d("q_heads") * d("head_dim"), d("kv_heads") * d("head_dim"));
        let qkv = [w8(0, 2 * qd, h), w8(1, kvd, h), w8(2, kvd, h)];
        let stacked = W8a8Weight::new(&qkv).unwrap();
        for (s, role) in [LinearRole::Q, LinearRole::K, LinearRole::V]
            .into_iter()
            .enumerate()
        {
            let seg = stacked.segment(s).unwrap();
            l.weights.insert(WeightSlot::Linear(role), bound(seg));
        }
        let o = W8a8Weight::new(&[w8(3, h, qd)]).unwrap();
        l.weights
            .insert(WeightSlot::Linear(LinearRole::O), bound(o));
    } else {
        let value = d("lin_v_heads") * d("lin_v_dim");
        let qkv = 2 * d("lin_k_heads") * d("lin_k_dim") + value;
        let input = W8a8Weight::new(&[w8(0, qkv, h), w8(1, value, h)]).unwrap();
        let out = W8a8Weight::new(&[w8(2, h, value)]).unwrap();
        l.weights
            .insert(WeightSlot::Linear(LinearRole::Qkvz), bound(input));
        l.weights
            .insert(WeightSlot::Linear(LinearRole::GdnOut), bound(out));
    }
    let inter = d("inter");
    for (s, slot, n, k) in [
        (4, WeightSlot::FfnGate, inter, h),
        (5, WeightSlot::FfnUp, inter, h),
        (6, WeightSlot::Linear(LinearRole::Down), h, inter),
    ] {
        let w = if i >= W8A8_FFN_FROM {
            bound(W8a8Weight::new(&[w8(s, n, k)]).unwrap())
        } else {
            a4(l.weights[&slot])
        };
        l.weights.insert(slot, w);
    }
    (l, ptrs)
}

fn build(mode: Mode, rows: u64, edit: impl Fn(&mut Vec<CircuitLayer>)) -> anyhow::Result<Fixture> {
    build_for(
        DECLARED,
        &declared_binding,
        Fusions::All,
        mode,
        rows,
        edit,
        |_| {},
    )
}

/// 2026-09-30: The mock launches of the groups whose first member is one of `ids`, in order.
fn launches_of(f: &Fixture, launched: &[MockLaunch], ids: &[&str]) -> Vec<MockLaunch> {
    super::exec_fixture::kernel_launches(f)
        .zip(launched)
        .filter(|(l, _)| {
            let first = f.plan.groups[l.group].nodes[0];
            ids.contains(&f.circuit.nodes[first].id.as_str())
        })
        .map(|(_, m)| m.clone())
        .collect()
}

/// 2026-09-30: A launch with its buffers blanked: grid, block, shared memory and every scalar.
fn shape(l: &MockLaunch) -> ([u32; 3], [u32; 3], u32, Vec<Option<Vec<u8>>>) {
    let args = l
        .args
        .iter()
        .map(|a| match a {
            MockArg::Buffer(_) => None,
            MockArg::Bytes(b) => Some(b.clone()),
        })
        .collect();
    (l.grid, l.block, l.shared_mem, args)
}

fn buf(l: &MockLaunch, i: usize) -> u64 {
    match &l.args[i] {
        MockArg::Buffer(p) => p.0,
        other => panic!("arg {i} is {other:?}"),
    }
}

#[test]
fn every_declared_plan_compiles_over_declared_bindings_and_reads_only_known_buffers() {
    let shapes = [(Mode::Decode, vec![1]), (Mode::Verify, vec![2, 3, 4])]
        .into_iter()
        .chain([(
            Mode::MultiSeq,
            vec![2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128],
        )]);
    for (mode, rows) in shapes {
        for n in rows {
            let f = build(mode, n, |_| {}).unwrap_or_else(|e| panic!("{mode:?} {n}: {e:#}"));
            assert_eq!(
                f.program.launches.len() as u64,
                f.plan.launches() + f.plan.copies(),
                "{mode:?} {n}"
            );
            // 2026-09-30: The verify snapshots copy the conv windows, so those are allocated.
            let gpu = MockGpuBackend::new();
            let gdn = match mode {
                Mode::Verify => super::exec_verify_tests::states_on(&gpu, &f),
                _ => states_rows(&f, 0xD000_0000, n as usize),
            };
            let launched = run_on(&gpu, &f, &gdn, 9);
            assert_pointers_known(&f, &gdn, &launched);
        }
    }
}

/// 2026-09-30: Layer 3's FFN (W4A4) at `m` rows: the circuit's quantize, gate, up, quantize
/// and down launches against `nvfp4_proj_small_m` for gate, up (same input) and down.
#[test]
fn each_w4a4_launch_is_the_one_the_legacy_launcher_makes() {
    for (mode, m) in [
        (Mode::Decode, 1u32),
        (Mode::MultiSeq, 8),
        (Mode::MultiSeq, 16),
        (Mode::MultiSeq, 32),
    ] {
        let f = build(mode, u64::from(m), |_| {}).unwrap();
        let gdn = states_rows(&f, 0xD000_0000, m as usize);
        let got = launches_of(
            &f,
            &run(&f, &gdn, 9),
            &[
                "l3.dense_ffn.xn_quant",
                "l3.dense_ffn.gate_up",
                "l3.dense_ffn.act_quant",
                "l3.dense_ffn.down",
            ],
        );
        let w = |s: WeightSlot| match f.layers[3].weights[&s] {
            BoundWeight::Nvfp4(q) => q,
            other => panic!("{other:?}"),
        };
        let (h, inter) = (5120, 17408);
        let gpu = MockGpuBackend::new();
        ops::w4a4_proj::prepare(&gpu).unwrap();
        let (x, g, u, y) = (
            ptr(0xA000_0000),
            ptr(0xA100_0000),
            ptr(0xA200_0000),
            ptr(0xA300_0000),
        );
        let kh = KernelHandle(0xDEAD);
        ops::w4a4_proj::nvfp4_proj_small_m(&gpu, kh, x, &w(WeightSlot::FfnGate), g, m, inter, h, 7)
            .unwrap();
        ops::w4a4_proj::nvfp4_proj_small_m_same_input(
            &gpu,
            kh,
            x,
            &w(WeightSlot::FfnUp),
            u,
            m,
            inter,
            h,
            7,
        )
        .unwrap();
        ops::w4a4_proj::nvfp4_proj_small_m(
            &gpu,
            kh,
            g,
            &w(WeightSlot::Linear(LinearRole::Down)),
            y,
            m,
            h,
            inter,
            7,
        )
        .unwrap();
        let want = gpu.launches_snapshot();
        assert_eq!(want.len(), 5, "legacy: quantize, gate, up, quantize, down");
        assert_eq!(got.len(), 5, "{mode:?} {m}");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(shape(a), shape(b), "{mode:?} m={m} launch {i}");
        }
        for (q, gemvs) in [(0, &[1, 2][..]), (3, &[4][..])] {
            for &j in gemvs {
                for part in 0..3 {
                    assert_eq!(
                        buf(&got[q], 1 + part),
                        buf(&got[j], part),
                        "launch {j} reads what {q} wrote"
                    );
                }
                assert_eq!(buf(&got[j], 3), buf(&want[j], 3), "launch {j}'s weight");
            }
            let (aq, sc, gs) = (buf(&got[q], 1), buf(&got[q], 2), buf(&got[q], 3));
            // 2026-09-30: Up's rows follow gate's in the gate+up edge (`ffn.rs`).
            if q == 0 {
                let rows_bytes = u64::from(m) * u64::from(inter) * 2;
                assert_eq!(
                    buf(&got[2], 6) - buf(&got[1], 6),
                    rows_bytes,
                    "up after gate"
                );
            }
            let k = if q == 0 { h } else { inter } as u64;
            assert_eq!(
                (sc - aq, gs - sc),
                (u64::from(m) * k / 2, u64::from(m) * k / 16)
            );
        }
    }
}

/// 2026-09-30: Layer 59's O and FFN (W8A8) against `w8a8_proj`, `w8a8_gemv` and the SiLU
/// quantizer as `forward_w8a8` runs them.
#[test]
fn each_w8a8_launch_is_the_one_the_legacy_arm_makes() {
    for (mode, m) in [
        (Mode::Decode, 1usize),
        (Mode::MultiSeq, 8),
        (Mode::MultiSeq, 128),
    ] {
        let f = build(mode, m as u64, |_| {}).unwrap();
        let gdn = states_rows(&f, 0xD000_0000, m);
        let ids = [
            "l59.attn.ag_quant",
            "l59.attn.o",
            "l59.dense_ffn.xn_quant",
            "l59.dense_ffn.gate_up",
            "l59.dense_ffn.act",
            "l59.dense_ffn.down",
        ];
        let got = launches_of(&f, &run(&f, &gdn, 9), &ids);
        let w = |s: WeightSlot| match f.layers[59].weights[&s] {
            BoundWeight::W8a8(w, k) => (w, k),
            other => panic!("{other:?}"),
        };
        let gpu = MockGpuBackend::new();
        let scratch = W8a8Scratch::alloc(&gpu, 17408).unwrap();
        let (o, k) = w(WeightSlot::Linear(LinearRole::O));
        let (gate, up, down) = (
            w(WeightSlot::FfnGate).0,
            w(WeightSlot::FfnUp).0,
            w(WeightSlot::Linear(LinearRole::Down)).0,
        );
        let p = |i: u64| DevicePtr(0xA000_0000 + (i << 24));
        ops::w8a8_proj(&gpu, &k, &o, p(0), o.k(), m, p(1), o.n(), &scratch, 7).unwrap();
        ops::w8a8_proj(
            &gpu,
            &k,
            &gate,
            p(2),
            gate.k(),
            m,
            p(3),
            gate.n(),
            &scratch,
            7,
        )
        .unwrap();
        ops::w8a8_gemv(&gpu, &k, &up, &scratch, m, p(4), up.n(), 7).unwrap();
        let s = ops::W8a8Scale::PerRow;
        ops::w8a8_act_quant_silu(&gpu, &k, s, p(3), p(4), gate.n(), m, gate.n(), &scratch, 7)
            .unwrap();
        ops::w8a8_gemv(&gpu, &k, &down, &scratch, m, p(5), down.n(), 7).unwrap();
        let want = gpu.launches_snapshot();
        assert_eq!((got.len(), want.len()), (7, 7), "{mode:?} {m}");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert_eq!(shape(a), shape(b), "{mode:?} m={m} launch {i}");
        }
        // 2026-09-30: (quantize launch, its output args, GEMV launch): the GEMV's first two
        // arguments are the E4M3 rows and their scales the quantize wrote.
        let rows_bytes = (m * gate.n() as usize * 2) as u64;
        assert_eq!(
            buf(&got[4], 8) - buf(&got[3], 8),
            rows_bytes,
            "up after gate"
        );
        assert_eq!(
            (buf(&got[5], 0), buf(&got[5], 1)),
            (buf(&got[3], 8), buf(&got[4], 8)),
            "the SiLU quantizer reads gate and up"
        );
        for (q, out, g) in [(0, 1, 1), (2, 1, 3), (2, 1, 4), (5, 2, 6)] {
            assert_eq!(
                buf(&got[q], out),
                buf(&got[g], 0),
                "launch {g}'s activation"
            );
            assert_eq!(
                buf(&got[q], out + 1),
                buf(&got[g], 1),
                "launch {g}'s scales"
            );
            assert_eq!(buf(&got[g], 2), buf(&want[g], 2), "launch {g}'s weight");
        }
    }
}

#[test]
fn a_ffn_the_tier_runs_w4a16_is_refused_under_a_w4a4_plan() {
    let e = build(Mode::Decode, 1, |layers| {
        let q = match layers[3].weights[&WeightSlot::FfnGate] {
            BoundWeight::Nvfp4(q) => q,
            _ => unreachable!(),
        };
        layers[3].weights.insert(
            WeightSlot::FfnGate,
            BoundWeight::Nvfp4(QuantizedWeight {
                act: Nvfp4Act::Unstamped,
                ..q
            }),
        );
    })
    .err()
    .map(|e| format!("{e:#}"))
    .unwrap_or_default();
    assert!(e.contains("runs W4A16 under this tier"), "{e}");
}

#[test]
fn an_nvfp4_binding_under_a_w8a8_plan_is_refused() {
    let e = build(Mode::Decode, 1, |layers| {
        layers[59]
            .weights
            .insert(WeightSlot::Linear(LinearRole::O), nvfp4(tag(59, 16)));
    })
    .err()
    .map(|e| format!("{e:#}"))
    .unwrap_or_default();
    assert!(
        e.contains("l59.attn.o") && e.contains("expected a W8A8 weight, the layer holds nvfp4"),
        "{e}"
    );
}
