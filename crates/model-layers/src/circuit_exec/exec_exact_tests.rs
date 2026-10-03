// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The executor's verify programs under the exact MTP verify chain
//! (`gdn_verify_exact = on`, the dense recipe's `exact_verify` variant), compiled from the real
//! dense circuit over synthetic bindings and run on the recording mock backend: the launches per
//! GDN layer, each chain's arguments against the legacy call shape
//! (`trait_decode_batched_conv_gdn_exact_chain.rs`, `trait_decode_batched_conv_gdn_multi_exact.rs`),
//! the batched runs' carried twins, exact fold and per-sequence chains, and the refusals.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_circuit::{Circuit, Mode, RowTable};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::Fusions;
use super::bindings::CircuitLayer;
use super::exec_fixture::*;
use super::exec_verify_batch_tests::{TABLES, buffer, seqs, states_batch_on};
use super::exec_verify_tests::states_on;
use super::program::{GdnState, StepEnv};

const EXACT: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth-exact-verify";

fn bind(c: &Circuit, i: usize, attn: usize) -> (CircuitLayer, Vec<u64>) {
    (layer_binding(c, i, attn), Vec::new())
}

fn exact_on(
    env: (&MockGpuBackend, &metrale_config::ModelConfig),
    mode: Mode,
    rows: u64,
    arm: Arm,
) -> anyhow::Result<Fixture> {
    build_on(env, EXACT, &bind, Fusions::All, (mode, rows, arm), |_| {}, |_| {})
}

fn verify(k: u64) -> Fixture {
    exact_on((&MockGpuBackend::new(), &config()), Mode::Verify, k, Arm::Primary)
        .unwrap_or_else(|e| panic!("exact verify K={k}: {e:#}"))
}

fn table(t: &'static str) -> Fixture {
    let rows = RowTable::parse(t).unwrap().rows();
    exact_on(
        (&MockGpuBackend::new(), &config()),
        Mode::VerifyBatch,
        rows,
        Arm::Table(t),
    )
    .unwrap_or_else(|e| panic!("exact `{t}`: {e:#}"))
}

fn word(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn ptr_of(a: &MockArg) -> DevicePtr {
    match a {
        MockArg::Buffer(p) => *p,
        other => panic!("{other:?} is not a buffer"),
    }
}

/// 2026-10-03: `(launch, layer)` of every launch of `name` (`module::func`), in order.
fn launches_of<'a>(f: &'a Fixture, name: &'a str) -> impl Iterator<Item = (usize, usize)> + 'a {
    kernel_launches(f)
        .enumerate()
        .filter(move |(_, l)| l.kernel == name)
        .map(|(j, l)| (j, f.circuit.nodes[f.plan.groups[l.group].nodes[0]].layer.unwrap()))
}

fn gdn_layers(gdn: &[Vec<GdnState>]) -> usize {
    gdn.iter().filter(|l| !l.is_empty()).count()
}

/// 2026-10-03: A kernel the exact chain replaces in the verify: the BF16 conv, the WY
/// recurrences (per sequence and carried), the BF16 fold and the BF16-input norm.
fn legacy_verify_kernel(name: &str) -> bool {
    name.ends_with("::causal_conv1d_update_l2norm")
        || name.contains("::gated_delta_rule_wy")
        || name.starts_with("gated_delta_rule_carry::gdn_carry_wy")
        || name == "gated_delta_rule_carry::gdn_carry_conv"
        || name == "gated_delta_rule_carry::gdn_carry_flush"
        || name.ends_with("::gated_rms_norm_prefill")
        || name.contains("gdn_verify_fused")
}

/// 2026-10-03: `(key_dim, nv, conv_dim)` of the circuit.
fn dims(f: &Fixture) -> (u32, u32, u32) {
    let d = |n: &str| f.circuit.dims[n] as u32;
    let key = d("lin_k_heads") * d("lin_k_dim");
    let nv = d("lin_v_heads");
    (key, nv, key * 2 + nv * d("lin_v_dim"))
}

// 2026-10-03: Each width is three launches per GDN layer (conv chain, exact chain, strided norm)
// and no copy. Mutation: an emitter that also launches the BF16 conv or the WY, or a rule set that
// leaves an `off` verify rule live under `on`, fails.
#[test]
fn each_exact_width_runs_three_gdn_launches_per_layer_and_no_legacy_verify_kernel() {
    for k in 2..=4u64 {
        let f = verify(k);
        let gpu = MockGpuBackend::new();
        let gdn = states_on(&gpu, &f);
        let launched = run_on(&gpu, &f, &gdn, 9);
        assert_eq!(launched.len() as u64, f.plan.launches(), "K={k}");
        assert_eq!(f.plan.copies(), 0, "K={k}: the chains write their slots inline");
        assert_eq!(gpu.d2d_count(), 0, "K={k}");
        let layers = gdn_layers(&gdn);
        let chain = format!("gdn_exact_carry::gdn_exact_chain{k}");
        for name in [
            "gated_delta_rule_carry::gdn_conv_chain_f32",
            chain.as_str(),
            "norm::gated_rms_norm_f32_input_strided",
        ] {
            assert_eq!(launches_of(&f, name).count(), layers, "K={k} {name}");
        }
        let legacy: Vec<&str> = kernel_launches(&f)
            .map(|l| l.kernel.as_str())
            .filter(|n| legacy_verify_kernel(n))
            .collect();
        assert!(legacy.is_empty(), "K={k}: {legacy:?}");
        assert_pointers_known(&f, &gdn, &launched);
    }
}

// 2026-10-03: The three launches pass what the legacy chain passes: the sequence's conv and h
// state, the rollback slots of rows 0..K-2 and the state itself for the rest, K, the conv dims,
// q/k/v at their offsets in the conv rows, `[decay | beta]` rows, and the chain's rows into the
// norm, K rows of it. Mutation: a slot shifted by one row, a placeholder of NULL, or the norm
// reading another buffer fails.
#[test]
fn the_exact_chains_mirror_the_legacy_call_shape() {
    for k in 2..=4u64 {
        let f = verify(k);
        let gpu = MockGpuBackend::new();
        let gdn = states_on(&gpu, &f);
        let launched: Vec<MockLaunch> = run_on(&gpu, &f, &gdn, 9);
        let (key, nv, conv_dim) = dims(&f);
        let (mut conv_rows, mut core_rows) = (BTreeMap::new(), BTreeMap::new());
        for (j, l) in kernel_launches(&f).enumerate() {
            let layer = f.circuit.nodes[f.plan.groups[l.group].nodes[0]].layer.unwrap();
            let (a, grid) = (&launched[j].args, launched[j].grid);
            let ku = k as usize;
            if l.kernel == "gated_delta_rule_carry::gdn_conv_chain_f32" {
                let st = gdn[layer][0];
                assert_eq!(a[0], buffer(st.conv), "K={k} layer {layer}");
                for t in 0..3 {
                    let want = if t + 1 < ku { st.conv_steps[t] } else { st.conv };
                    assert_eq!(a[4 + t], buffer(want), "K={k} layer {layer} window {t}");
                }
                assert_eq!(
                    a[7..12],
                    [word(k as u32), word(conv_dim), word(4), word(key * 2), word(128)],
                    "K={k} layer {layer}"
                );
                assert_eq!(a[12], MockArg::Bytes(1e-6f32.to_le_bytes().to_vec()));
                assert_eq!(grid, [conv_dim.div_ceil(256), 1, 1]);
                conv_rows.insert(layer, (ptr_of(&a[3]), a[14].clone()));
            } else if l.kernel == format!("gdn_exact_carry::gdn_exact_chain{k}") {
                let st = gdn[layer][0];
                let (q, stride) = conv_rows[&layer].clone();
                assert_eq!(a[0], buffer(st.h), "K={k} layer {layer}");
                assert_eq!(
                    a[1..4],
                    [
                        buffer(q),
                        buffer(q.offset(key as usize * 4)),
                        buffer(q.offset(key as usize * 8))
                    ],
                    "K={k} layer {layer}: q/k/v in the FP32 conv rows"
                );
                assert_eq!(a[5], buffer(ptr_of(&a[4]).offset(nv as usize * 4)));
                for t in 0..3 {
                    let want = if t + 1 < ku { st.h_steps[t] } else { st.h };
                    assert_eq!(a[7 + t], buffer(want), "K={k} layer {layer} h slot {t}");
                }
                assert_eq!(a[10..13], [word(16), word(nv), word(128)]);
                assert_eq!(a[13..16], [stride.clone(), stride, word(nv * 2)]);
                assert_eq!(grid, [nv, 1, 1]);
                core_rows.insert(layer, (ptr_of(&a[6]), a[16].clone()));
            } else if l.kernel == "norm::gated_rms_norm_f32_input_strided" {
                let (core, stride) = core_rows[&layer].clone();
                assert_eq!((a[0].clone(), a[8].clone()), (buffer(core), stride));
                assert_eq!(grid, [nv, k as u32, 1], "K={k}: one launch over the K rows");
            }
        }
        let layers = gdn_layers(&gdn);
        assert_eq!((conv_rows.len(), core_rows.len()), (layers, layers));
    }
}

// 2026-10-03: A step whose sequence lacks a rollback slot a row writes fails before launching;
// the slot after the last row is never required. Mutation: dropping the check, or requiring all
// three slots at K = 2, fails.
#[test]
fn the_chains_require_exactly_the_rollback_slots_of_their_width() {
    let f = verify(3);
    for (what, conv) in [("conv", true), ("h", false)] {
        let gpu = MockGpuBackend::new();
        let mut gdn = states_on(&gpu, &f);
        let first = gdn.iter().position(|l| !l.is_empty()).unwrap();
        if conv {
            gdn[first][0].conv_steps[1] = DevicePtr::NULL;
        } else {
            gdn[first][0].h_steps[1] = DevicePtr::NULL;
        }
        let err = f
            .program
            .run(&StepEnv {
                gpu: &gpu,
                stream: 7,
                gdn: &gdn,
                max_blocks_per_seq: 9,
                prefill: None,
            })
            .unwrap_err();
        assert!(
            format!("{err:#}").contains(&format!("lacks the {what} rollback slots")),
            "{err:#}"
        );
    }
    let f = verify(2);
    let gpu = MockGpuBackend::new();
    let mut gdn = states_on(&gpu, &f);
    for l in gdn.iter_mut().filter(|l| !l.is_empty()) {
        l[0].h_steps[1] = DevicePtr::NULL;
        l[0].conv_steps[2] = DevicePtr::NULL;
    }
    let launched = run_on(&gpu, &f, &gdn, 9);
    assert_eq!(launched.len() as u64, f.plan.launches());
}

// 2026-10-03: The build refuses where the legacy chain declines to a per-row arm the circuit
// does not model: a parent kernel unlinked, a sigmoid-gated norm; and a twin missing is refused
// by the kernel table. Mutation: dropping any check in `chain_ready` builds a plan legacy never
// runs.
#[test]
fn the_exact_build_refuses_where_the_legacy_chain_declines() {
    let refused = |gpu: &MockGpuBackend, cfg: &metrale_config::ModelConfig| {
        match exact_on((gpu, cfg), Mode::Verify, 3, Arm::Primary) {
            Ok(_) => panic!("the exact verify built"),
            Err(e) => format!("{e:#}"),
        }
    };
    for (m, func) in [
        ("causal_conv1d", "causal_conv1d_update_l2norm_f32"),
        ("gated_delta_rule", "gated_delta_rule_decode_f32"),
        ("norm", "gated_rms_norm_f32_input"),
        ("norm", "gated_rms_norm_f32_input_strided"),
    ] {
        let gpu = MockGpuBackend::new();
        gpu.deny_kernel(m, func);
        let err = refused(&gpu, &config());
        assert!(err.contains(&format!("`{m}::{func}` is not linked")), "{err}");
    }
    let gpu = MockGpuBackend::new();
    gpu.deny_kernel("gdn_exact_carry", "gdn_exact_chain3");
    let err = refused(&gpu, &config());
    assert!(err.contains("gdn_exact_carry::gdn_exact_chain3"), "{err}");
    let mut cfg = config();
    cfg.gdn_norm_sigmoid = true;
    let err = refused(&MockGpuBackend::new(), &cfg);
    assert!(err.contains("sigmoid-gated"), "{err}");
}

// 2026-10-03: Every golden table compiles under the exact chain to exactly its plan's launches,
// no copies, one strided norm per GDN layer and no kernel the chain replaces, reading only known
// memory. Mutation: an emitter that keeps the BF16 fold, or a per-sequence norm, fails.
#[test]
fn every_golden_table_under_the_exact_chain_launches_its_plan_and_no_legacy_verify_kernel() {
    for t in TABLES {
        let f = table(t);
        assert_eq!(
            (f.program.launches.len() as u64, f.plan.copies()),
            (f.plan.launches(), 0),
            "`{t}`"
        );
        let gpu = MockGpuBackend::new();
        gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
        let gdn = states_batch_on(&gpu, &f, seqs(t));
        let launched = run_on(&gpu, &f, &gdn, 9);
        assert_eq!(launched.len() as u64, f.plan.launches(), "`{t}`");
        let norms = launches_of(&f, "norm::gated_rms_norm_f32_input_strided").count();
        assert_eq!(norms, gdn_layers(&gdn), "`{t}`");
        let legacy: Vec<&str> = kernel_launches(&f)
            .map(|l| l.kernel.as_str())
            .filter(|n| legacy_verify_kernel(n))
            .collect();
        assert!(legacy.is_empty(), "`{t}`: {legacy:?}");
        assert_pointers_known(&f, &gdn, &launched);
    }
}

// 2026-10-03: A contiguous run is the carried FP32 conv and the exact twin over the run's slices
// (`decode_batched_conv_gdn_multi_exact_carry`): the run's first conv state and slot slice, K and
// n, the lazy twin and flag from 8 sequences, the layer's WY table slice and engaged words, and
// the conv's rows as the twin's q. Mutation: the batch's first sequence for the run's, the eager
// twin at 8 sequences, or a row-indexed slice fails.
#[test]
fn a_contiguous_exact_run_launches_the_carried_twins_on_its_own_slices() {
    let f = table("4x2 2x8");
    let gdn = states_rows(&f, 0xD000_0000, 10);
    let launched = run(&f, &gdn, 9);
    let layers = gdn_layers(&gdn);
    let (_, nv, conv_dim) = dims(&f);
    let conv: Vec<_> = launches_of(&f, "gated_delta_rule_carry::gdn_carry_conv_f32").collect();
    assert_eq!(conv.len(), 2 * layers, "one carried conv per run per GDN layer");
    let mut q_of = BTreeMap::new();
    for (i, &(j, layer)) in conv.iter().enumerate() {
        let (first, k, n) = if i % 2 == 0 { (0, 4, 2) } else { (2, 2, 8) };
        let a = &launched[j].args;
        assert_eq!(a[0], buffer(gdn[layer][first].conv), "run {i}: conv state");
        assert_eq!(a[5], buffer(CARRY.slot_tab.offset(first * 4)), "run {i}: slots");
        assert_eq!(a[8], word(k), "run {i}: K");
        assert_eq!(a[16], word(u32::from(n >= 8)), "run {i}: lazy");
        assert_eq!(launched[j].grid, [conv_dim.div_ceil(256), n, 1]);
        q_of.insert((layer, first), ptr_of(&a[3]));
    }
    let eager: Vec<_> = launches_of(&f, "gdn_exact_carry::gdn_exact_carry4").collect();
    let lazy: Vec<_> = launches_of(&f, "gdn_exact_carry::gdn_exact_carry2_lazy").collect();
    assert_eq!((eager.len(), lazy.len()), (layers, layers));
    for (runs, first, n) in [(&eager, 0usize, 2u32), (&lazy, 2, 8)] {
        for &(j, layer) in runs {
            let a = &launched[j].args;
            let ssm = gdn[..layer].iter().filter(|l| !l.is_empty()).count();
            let tables = f
                .fixed
                .verify_wy_tables
                .offset(ssm * crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES);
            assert_eq!(a[0], buffer(tables.offset(first * 8)), "layer {layer}: WY slice");
            assert_eq!(a[1], buffer(q_of[&(layer, first)]), "layer {layer}: q");
            assert_eq!(a[8], buffer(CARRY.slot_tab.offset(first * 4)));
            assert_eq!(a[11], word(n));
            assert_eq!(a[17], word(nv * 2));
            assert_eq!(a[19], buffer(CARRY.flag.offset(first * 4)), "engaged words");
        }
    }
}

// 2026-10-03: A fragmented carried run folds with the model directory's exact fold and the conv
// fold, then runs each sequence's chain on its own state; an uncarried table folds nothing.
// Mutation: the BF16 fold under the chain, a fold on an uncarried run, or one sequence's state
// read for another fails.
#[test]
fn a_fragmented_exact_run_folds_exactly_then_chains_each_sequence_on_its_own_state() {
    for (t, folds, ks) in [
        ("4x2! 2x2", 1, &[4u32, 4][..]),
        ("uncarried: 4x2! 2x2!", 0, &[4, 4, 2, 2][..]),
    ] {
        let f = table(t);
        let gpu = MockGpuBackend::new();
        gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
        let gdn = states_batch_on(&gpu, &f, 4);
        let launched = run_on(&gpu, &f, &gdn, 9);
        let layers = gdn_layers(&gdn);
        for (name, want) in [
            ("gdn_exact_carry::gdn_exact_carry_flush", folds * layers),
            ("gated_delta_rule_carry::gdn_carry_conv_flush", folds * layers),
            ("gated_delta_rule_carry::gdn_carry_flush", 0),
        ] {
            assert_eq!(launches_of(&f, name).count(), want, "`{t}` {name}");
        }
        let mut seq = BTreeMap::<usize, usize>::new();
        for (j, layer) in launches_of(&f, "gated_delta_rule_carry::gdn_conv_chain_f32") {
            let s = seq.entry(layer).or_default();
            let st = gdn[layer][*s];
            let a = &launched[j].args;
            assert_eq!(a[0], buffer(st.conv), "`{t}` layer {layer} sequence {s}");
            assert_eq!(a[7], word(ks[*s]), "`{t}` layer {layer} sequence {s}: K");
            *s += 1;
        }
        assert!(seq.values().all(|&n| n == ks.len()), "`{t}`: {seq:?}");
        let mut seq = BTreeMap::<usize, usize>::new();
        let chains = kernel_launches(&f)
            .enumerate()
            .filter(|(_, l)| l.kernel.starts_with("gdn_exact_carry::gdn_exact_chain"));
        for (j, l) in chains {
            let layer = f.circuit.nodes[f.plan.groups[l.group].nodes[0]].layer.unwrap();
            let s = seq.entry(layer).or_default();
            assert_eq!(
                launched[j].args[0],
                buffer(gdn[layer][*s].h),
                "`{t}` layer {layer} sequence {s}: h"
            );
            assert!(l.kernel.ends_with(&ks[*s].to_string()), "`{t}`: {}", l.kernel);
            *s += 1;
        }
        assert!(seq.values().all(|&n| n == ks.len()), "`{t}`: {seq:?}");
    }
}

// 2026-10-03: A contiguous exact run whose slots are not consecutive at run time fails before it
// launches. Mutation: dropping the conv-slot check reads another sequence's window.
#[test]
fn a_contiguous_exact_run_refuses_out_of_place_slots() {
    let f = table("2x4");
    let mut gdn = states_rows(&f, 0xD000_0000, 4);
    for layer in gdn.iter_mut().filter(|l| !l.is_empty()) {
        layer[3] = GdnState {
            conv: layer[3].conv.offset(STATE_PITCH as usize),
            ..layer[3]
        };
    }
    let gpu = MockGpuBackend::new();
    let e = f
        .program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
            prefill: None,
        })
        .unwrap_err();
    assert!(format!("{e:#}").contains("conv slots on"), "{e:#}");
}
