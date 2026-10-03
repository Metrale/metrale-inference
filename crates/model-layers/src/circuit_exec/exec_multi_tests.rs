// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The executor's multi-sequence programs, compiled from the real dense circuit at
//! every padded width of the decode graph ladder over synthetic bindings and run on the
//! recording mock backend: launch counts, the pointers read, each row's own GDN state, the
//! Q/gate row stride, and the refusals.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::{LinearRole, Mode};
use metrale_gpu_runtime::gpu::mock::MockArg;

use super::Fusions;
use super::bindings::*;
use super::exec_fixture::*;
use super::program::GdnState;

/// 2026-09-28: The padded widths `model-engine` `traits::padded_batch_n` pads a batch to.
const LADDER: [u64; 11] = [2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128];

fn at(rows: u64) -> Fixture {
    build_at(Fusions::All, Mode::MultiSeq, rows, |_| {}, |_| {}).unwrap()
}

fn err_at(
    rows: u64,
    edit: impl Fn(&mut Vec<CircuitLayer>),
    head: impl Fn(&mut HeadBinding),
) -> String {
    build_at(Fusions::All, Mode::MultiSeq, rows, edit, head)
        .err()
        .map(|e| format!("{e:#}"))
        .unwrap_or_default()
}

fn u32_arg(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

#[test]
fn every_width_launches_what_its_plan_counts_and_reads_only_known_buffers() {
    for rows in LADDER {
        for fusions in [Fusions::All, Fusions::ReferenceOnly] {
            let f = build_at(fusions, Mode::MultiSeq, rows, |_| {}, |_| {})
                .unwrap_or_else(|e| panic!("{rows} rows, {fusions:?}: {e:#}"));
            assert_eq!(f.program.launches.len() as u64, f.plan.launches(), "{rows}");
            let gdn = states_rows(&f, 0xD000_0000, rows as usize);
            let launched = run(&f, &gdn, 9);
            assert_eq!(launched.len(), f.program.launches.len(), "{rows}");
            assert_pointers_known(&f, &gdn, &launched);
        }
    }
}

/// 2026-09-30: Row `r` of every GDN layer moved `r + 1` extra slots out, so no two rows are
/// contiguous: the `gdn_state_slots_fragmented` route's condition.
fn fragmented(f: &Fixture, rows: usize) -> Vec<Vec<GdnState>> {
    let mut gdn = states_rows(f, 0xD000_0000, rows);
    for layer in gdn.iter_mut() {
        for (r, s) in layer.iter_mut().enumerate() {
            let shift = (r * (r + 1) / 2) * STATE_PITCH as usize;
            s.h = s.h.offset(shift);
            s.conv = s.conv.offset(shift);
        }
    }
    gdn
}

/// 2026-09-30: The launches of `kernel`, with the GDN layer each belongs to.
fn gdn_launches<'a>(f: &'a Fixture, kernel: &'a str) -> impl Iterator<Item = (usize, usize)> + 'a {
    f.program
        .launches
        .iter()
        .enumerate()
        .filter(move |(_, l)| l.kernel == kernel)
        .map(|(j, l)| {
            (
                j,
                f.circuit.nodes[f.plan.groups[l.group].nodes[0]]
                    .layer
                    .unwrap(),
            )
        })
}

// 2026-09-30: The primary arm under `ssm_batched_recurrent = on` launches each strided kernel
// once per GDN layer for every row, addressed at row 0's state. Mutation: passing row `n - 1`'s
// state, or launching per row, fails.
#[test]
fn the_batched_arm_reads_row_zeros_state_once_per_layer() {
    for rows in [2u64, 16, 128] {
        let f = at(rows);
        let gdn = states_rows(&f, 0xD000_0000, rows as usize);
        let launched = run(&f, &gdn, 9);
        let gdn_layers = gdn.iter().filter(|l| !l.is_empty()).count();
        for (kernel, state) in [
            (
                "causal_conv1d::causal_conv1d_update_l2norm_f32_strided",
                (|s: GdnState| s.conv) as fn(GdnState) -> _,
            ),
            (
                "gated_delta_rule::gated_delta_rule_decode_f32_strided",
                |s: GdnState| s.h,
            ),
        ] {
            let mut n = 0;
            for (j, layer) in gdn_launches(&f, kernel) {
                let args = &launched[j].args;
                assert!(
                    args.contains(&MockArg::Buffer(state(gdn[layer][0]))),
                    "{rows}: {kernel}"
                );
                assert!(
                    args.contains(&u32_arg(rows as u32)),
                    "{rows}: {kernel} batch"
                );
                n += 1;
            }
            assert_eq!(n, gdn_layers, "{rows} rows: {kernel} once per GDN layer");
        }
        for per_row in [
            "causal_conv1d::causal_conv1d_update_l2norm_f32",
            "gated_delta_rule::gated_delta_rule_decode_f32",
        ] {
            assert_eq!(gdn_launches(&f, per_row).count(), 0, "{rows}: {per_row}");
        }
    }
}

// 2026-09-30: A batched launch that finds a row's state out of place fails instead of reading
// another sequence's state. Mutation: dropping `contiguous_base`'s check runs it.
#[test]
fn a_batched_launch_refuses_out_of_place_slots() {
    let f = at(16);
    let gpu = metrale_gpu_runtime::gpu::mock::MockGpuBackend::new();
    gpu.set_kernel_n_tile(metrale_gpu_runtime::gpu::KernelHandle(0xDEAD), 128);
    let e = f
        .program
        .run(&super::program::StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &fragmented(&f, 16),
            max_blocks_per_seq: 9,
            prefill: None,
        })
        .unwrap_err();
    assert!(format!("{e:#}").contains("slots past row 0"), "{e:#}");
}

// 2026-09-30: The BA-gates kernel a plan names must be the one `ops::dense_gemm_ba_gates_prefill`
// picks for its rows on this device (the mock is a 48-SM GB10, so the twin from 96 rows). A plan
// that names the base kernel where the engine picks the twin is refused. Mutation: dropping the
// emitter's pick check compiles the disowned plan.
#[test]
fn the_ba_gates_kernel_is_the_one_the_engine_picks() {
    let twin = "ssm_ba_gates_hopper::dense_gemm_ba_gates_prefill_hopper";
    let base = "ssm_preprocess::dense_gemm_ba_gates_prefill";
    for (rows, want) in [(64u64, base), (96, twin), (128, twin)] {
        let f = at(rows);
        let ks: Vec<&str> = f
            .program
            .launches
            .iter()
            .map(|l| l.kernel.as_str())
            .filter(|k| k.contains("dense_gemm_ba_gates_prefill"))
            .collect();
        assert!(
            !ks.is_empty() && ks.iter().all(|k| *k == want),
            "{rows}: {ks:?}"
        );
    }
    let bind = |c: &metrale_circuit::Circuit, i: usize, attn: usize| {
        (layer_binding(c, i, attn), Vec::new())
    };
    let off = Arm::Setting("ssm_ba_gates_hopper", "off");
    let e = build_for(
        RECIPE,
        &bind,
        Fusions::All,
        (Mode::MultiSeq, 96, off),
        |_| {},
        |_| {},
    )
    .err()
    .map(|e| format!("{e:#}"))
    .unwrap_or_default();
    assert!(e.contains("where the engine picks the twin"), "{e}");
}

// 2026-09-30: The fragmented-slots route runs the per-row arm, each row on its own state.
#[test]
fn the_fragmented_route_reads_each_rows_own_state() {
    for rows in [2u64, 16, 128] {
        let f = build_route_at("gdn_state_slots_fragmented", Mode::MultiSeq, rows)
            .unwrap_or_else(|e| panic!("{rows}: {e:#}"));
        let gdn = fragmented(&f, rows as usize);
        let launched = run(&f, &gdn, 9);
        let mut seen = std::collections::BTreeMap::<(usize, &str), usize>::new();
        for (j, l) in f.program.launches.iter().enumerate() {
            let state = match l.kernel.as_str() {
                "causal_conv1d::causal_conv1d_update_l2norm_f32" => |s: GdnState| s.conv,
                "gated_delta_rule::gated_delta_rule_decode_f32" => |s: GdnState| s.h,
                _ => continue,
            };
            let layer = f.circuit.nodes[f.plan.groups[l.group].nodes[0]]
                .layer
                .unwrap();
            let row = seen.entry((layer, l.kernel.as_str())).or_default();
            let want = state(gdn[layer][*row]);
            assert!(
                launched[j].args.contains(&MockArg::Buffer(want)),
                "{rows} rows: layer {layer} launch {row} of {} misses its state",
                l.kernel
            );
            *row += 1;
        }
        assert!(seen.values().all(|&n| n == rows as usize), "{seen:?}");
        assert!(!seen.is_empty());
    }
}

#[test]
fn q_and_its_gate_are_read_at_their_shared_row_stride() {
    for rows in [2u64, 4, 16, 128] {
        let f = at(rows);
        let d = |k: &str| f.circuit.dims[k] as u32;
        let (q_row, kv) = (
            2 * d("q_heads") * d("head_dim"),
            d("kv_heads") * d("head_dim"),
        );
        let launched = run(&f, &states_rows(&f, 0xD000_0000, rows as usize), 9);
        let mut checked = 0;
        for (j, l) in f.program.launches.iter().enumerate() {
            let want: &[u32] = match l.kernel.as_str() {
                "rope::rope_forward_strided" => &[q_row, kv],
                "paged_decode::paged_decode_attn" => &[q_row],
                "residual_add::sigmoid_gate_mul_batched" => &[q_row],
                _ => continue,
            };
            for &v in want {
                assert!(
                    launched[j].args.contains(&u32_arg(v)),
                    "{rows} rows: {} lacks stride {v}",
                    l.kernel
                );
            }
            checked += 1;
        }
        let attn = f
            .layers
            .iter()
            .filter(|l| matches!(l.mixer, MixerFacts::Attention(_)))
            .count();
        assert_eq!(
            checked,
            3 * attn,
            "{rows} rows: three strided launches per attention layer"
        );
    }
}

#[test]
fn a_width_the_bindings_route_differently_is_refused() {
    let no_plain_16 = |l: &mut Vec<CircuitLayer>| {
        for x in l.iter_mut() {
            if let MixerFacts::Attention(a) = &mut x.mixer {
                a.paged_decode_plain_rows &= !(1 << 15);
            }
        }
    };
    assert!(err_at(16, no_plain_16, |_| {}).contains("at 16 rows"));
    assert_eq!(err_at(12, no_plain_16, |_| {}), "");
    let narrow_head = |h: &mut HeadBinding| h.batchm_max_rows = 4;
    assert!(err_at(8, |_| {}, narrow_head).contains("with the GEMM"));
    assert_eq!(err_at(4, |_| {}, narrow_head), "");
    let wide_head = |h: &mut HeadBinding| h.batchm_max_rows = 12;
    assert!(err_at(12, |_| {}, wide_head).contains("with the batched GEMV"));
    let no_twin = |l: &mut Vec<CircuitLayer>| {
        l[3].weights.remove(&WeightSlot::Transposed(LinearRole::Q));
    };
    assert!(err_at(12, no_twin, |_| {}).contains("Transposed(Q)"));
    assert_eq!(err_at(8, no_twin, |_| {}), "");
    let no_repack = |l: &mut Vec<CircuitLayer>| {
        l[0].weights.remove(&WeightSlot::FfnDownMmq);
    };
    assert!(err_at(16, no_repack, |_| {}).contains("FfnDownMmq"));
    assert_eq!(err_at(2, no_repack, |_| {}), "");
}
