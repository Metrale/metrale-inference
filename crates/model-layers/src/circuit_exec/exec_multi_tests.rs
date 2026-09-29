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
            assert_pointers_known(&f, &gdn);
        }
    }
}

#[test]
fn each_row_of_a_gdn_layer_reads_its_own_sequence_state() {
    for rows in [2u64, 16, 128] {
        let f = at(rows);
        let gdn = states_rows(&f, 0xD000_0000, rows as usize);
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
