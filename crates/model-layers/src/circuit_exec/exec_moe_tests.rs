// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The executor over the Qwen3.6-35B-A3B-FP8 instance (the MoE flagship recipe):
//! its decode, multi-sequence, verify and batched-verify plans compiled over bindings of the
//! checkpoint's shape (FP8 W8A16 attention and GDN projections, the grouped FP8 MoE with its
//! W8A8 expert step) and run on the recording mock backend. The MoE launches are checked
//! against the grouped decode's own argument wiring (`forward_fp8_grouped_decode_routed`,
//! `run_fp8_grouped_w8a8`), and a binding whose dispatch differs from the plan is refused.
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants: none beyond the types.

use metrale_circuit::{Circuit, LayerKind, LinearRole, Mode};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::Fusions;
use super::bindings::*;
use super::exec_fixture::*;
use crate::layers::moe::{Fp8Tables, MoeBinding, MoeFacts, MoeKernels};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, Fp8Weight, WeightQuantFormat};

const MOE: &str = "qwen3.6/qwen3.6-35b-a3b-fp8-bf16head";
const K: KernelHandle = KernelHandle(0xDEAD);

fn fp8(tag: u64, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: ptr(tag),
        row_scale: ptr(tag + 1),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// 2026-10-03: Layer `i`'s MoE: expert tables at slots 50.., the mock's one kernel handle for
/// every kernel the layer would launch.
fn moe_binding(c: &Circuit, i: usize, w8a8: bool, tensor_core: bool) -> MoeBinding {
    let d = |k: &str| c.dims[k] as u32;
    let (h, inter) = (d("hidden"), d("moe_inter"));
    let tables = |s: u64| Fp8Tables {
        weights: ptr(tag(i, s)),
        scales: ptr(tag(i, s + 1)),
    };
    let geometry = |tc: bool, gate_up: bool| match (tc, gate_up) {
        (true, true) => ops::FP8_GROUPED_GATE_UP_TC,
        (true, false) => ops::FP8_GROUPED_DOWN_TC,
        (false, true) => ops::FP8_GROUPED_GATE_UP_SCALAR,
        (false, false) => ops::FP8_GROUPED_DOWN_SCALAR,
    };
    MoeBinding {
        router: DenseWeight {
            weight: ptr(tag(i, 50)),
        },
        shared_gate: DenseWeight {
            weight: ptr(tag(i, 51)),
        },
        gate: tables(52),
        up: tables(54),
        down: tables(56),
        shared: Fp8ExpertWeight {
            gate_proj: fp8(tag(i, 58), d("shared_inter"), h),
            up_proj: fp8(tag(i, 60), d("shared_inter"), h),
            down_proj: fp8(tag(i, 62), h, d("shared_inter")),
        },
        facts: MoeFacts {
            num_experts: d("experts"),
            top_k: d("top_k"),
            hidden: h,
            inter,
            norm_topk_prob: true,
            tensor_core,
            w8a8,
        },
        kernels: MoeKernels {
            router_rows: K,
            router_gemm: K,
            topk_rows: K,
            topk_batched: K,
            sort: K,
            gate_up: K,
            gate_up_geometry: geometry(tensor_core, true),
            down: K,
            down_geometry: geometry(tensor_core, false),
            quant_w8a8: K,
            gate_up_w8a8: K,
            down_w8a8: K,
            blend: K,
        },
    }
}

/// 2026-10-03: The 35B's bindings: FP8 block-scaled mixer projections at slots 40.., no dense
/// FFN, and the MoE.
fn moe_layer(c: &Circuit, i: usize, attn: usize, w8a8: bool, tc: bool) -> CircuitLayer {
    let mut l = layer_binding(c, i, attn);
    let d = |k: &str| c.dims[k] as u32;
    let h = d("hidden");
    l.weights.retain(|s, _| {
        !matches!(
            s,
            WeightSlot::FfnGate
                | WeightSlot::FfnUp
                | WeightSlot::Linear(LinearRole::Down)
                | WeightSlot::FfnGateMmq
                | WeightSlot::FfnUpMmq
                | WeightSlot::FfnDownMmq
                | WeightSlot::Transposed(_)
        )
    });
    let mut put = |role: LinearRole, slot: u64, n: u32, k: u32| {
        l.weights.insert(
            WeightSlot::Linear(role),
            BoundWeight::Fp8(fp8(tag(i, slot), n, k)),
        );
    };
    if c.layer_kinds[i] == LayerKind::LinearAttention {
        let (kd, vd) = (
            d("lin_k_heads") * d("lin_k_dim"),
            d("lin_v_heads") * d("lin_v_dim"),
        );
        put(LinearRole::Qkvz, 40, 2 * kd + 2 * vd, h);
        put(LinearRole::GdnOut, 42, h, vd);
    } else {
        let (q, kv) = (d("q_heads") * d("head_dim"), d("kv_heads") * d("head_dim"));
        put(LinearRole::Q, 40, 2 * q, h);
        put(LinearRole::K, 42, kv, h);
        put(LinearRole::V, 44, kv, h);
        put(LinearRole::O, 46, h, q);
    }
    l.moe = Some(moe_binding(c, i, w8a8, tc));
    l
}

fn build_moe(mode: Mode, rows: u64, w8a8: bool, tc: bool) -> anyhow::Result<Fixture> {
    let bind =
        move |c: &Circuit, i: usize, attn: usize| (moe_layer(c, i, attn, w8a8, tc), Vec::new());
    build_for(
        MOE,
        &bind,
        Fusions::All,
        (mode, rows, Arm::Primary),
        |_| {},
        |_| {},
    )
}

/// 2026-10-03: Every primary plan the instance checks in (decode, the multi-sequence ladder,
/// verify K = 2..4) compiles to exactly its planned launches, and runs.
#[test]
fn every_35b_plan_compiles_to_the_launches_it_counts() {
    let inst = super::sources::instance(MOE).unwrap();
    let shapes: Vec<(Mode, u64)> = inst
        .plans
        .iter()
        .filter(|(m, _)| **m != Mode::Draft)
        .flat_map(|(m, rows)| rows.iter().map(move |&r| (*m, r)))
        .collect();
    for (mode, rows) in shapes {
        let f = build_moe(mode, rows, true, true)
            .unwrap_or_else(|e| panic!("{mode:?} at {rows}: {e:#}"));
        assert_eq!(
            f.program.launches.len() as u64,
            f.plan.launches() + f.plan.copies(),
            "{mode:?} {rows}"
        );
        // 2026-10-03: A verify's state copies read allocated states (exec_verify_tests.rs runs
        // them); the other modes run on the mock here.
        if mode != Mode::Verify {
            let launched = run(&f, &states_rows(&f, 0xD000_0000, rows as usize), 9);
            assert_eq!(launched.len() as u64, f.plan.launches(), "{mode:?} {rows}");
        }
    }
}

/// 2026-10-03: The batched-verify tables compile too.
#[test]
fn every_35b_verify_table_compiles() {
    let inst = super::sources::instance(MOE).unwrap();
    for table in &inst.verify_batch {
        // 2026-10-03: The fixture names a table by its spelling, which `Display` round-trips.
        let t: &'static str = Box::leak(table.to_string().into_boxed_str());
        let bind =
            |c: &Circuit, i: usize, attn: usize| (moe_layer(c, i, attn, true, true), Vec::new());
        let arm = (Mode::VerifyBatch, table.rows(), Arm::Table(t));
        let f = build_for(MOE, &bind, Fusions::All, arm, |_| {}, |_| {})
            .unwrap_or_else(|e| panic!("table {t}: {e:#}"));
        assert_eq!(
            f.program.launches.len() as u64,
            f.plan.launches() + f.plan.copies(),
            "table {t}"
        );
    }
}

fn ptr_arg(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a buffer"),
    }
}

fn u32_arg(l: &MockLaunch, i: usize) -> u32 {
    match &l.args[i] {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("arg {i} is {other:?}, not bytes"),
    }
}

/// 2026-10-03: One layer's W8A8 MoE at 16 rows, as `run_fp8_grouped_w8a8` wires it: the input
/// quantizer's outputs are what gate+up reads, the sort's outputs (at the arena scratch, by
/// `grouped_sort_out`) are what the expert and blend kernels read, gate+up's E4M3 products and
/// scales are what down reads, and the grids are the grouped kernels' for the active-expert
/// cap of 16 rows.
#[test]
fn the_w8a8_moe_reads_what_its_previous_launch_wrote() {
    let rows = 16u64;
    let f = build_moe(Mode::MultiSeq, rows, true, true).unwrap();
    let launched = run(&f, &states_rows(&f, 0xD000_0000, rows as usize), 9);
    let names: Vec<&str> = f
        .program
        .launches
        .iter()
        .map(|l| l.kernel.as_str())
        .collect();
    let at = |k: &str| {
        names
            .iter()
            .position(|n| *n == k)
            .unwrap_or_else(|| panic!("no {k}"))
    };
    let topk = &launched[at("moe_topk::moe_topk_softmax_rows")];
    let sort = &launched[at("moe_fp8_grouped_sort::moe_fp8_grouped_sort")];
    let quant = &launched[at("moe_fp8_grouped_tc_w8a8::moe_act_quant_e4m3")];
    let gu = &launched[at("moe_fp8_grouped_tc_w8a8::moe_expert_gate_up_act_fp8_grouped_tc_w8a8")];
    let down = &launched[at("moe_fp8_grouped_tc_w8a8::moe_expert_down_act_fp8_grouped_tc_w8a8")];
    let blend = &launched[at("moe_fp8_grouped_blend::moe_weighted_sum_blend_fp8_grouped")];
    let (m, top_k, experts) = (rows as u32, 8u32, 256u32);
    let (sorted, cap) = crate::layers::moe::MoeBinding::sort_out(
        &moe_binding(&f.circuit, 0, true, true),
        &MOE_SCRATCH,
        m,
    );
    assert_eq!(cap, (m * top_k).min(experts));
    // 2026-10-03: top-k writes the ids the sort reads.
    assert_eq!(ptr_arg(sort, 0), ptr_arg(topk, 1));
    assert_eq!(ptr_arg(sort, 1), sorted.sorted_token_ids);
    assert_eq!(u32_arg(sort, 7), m * top_k);
    // 2026-10-03: The quantized input and the sort outputs feed gate+up.
    assert_eq!(
        (ptr_arg(gu, 0), ptr_arg(gu, 1)),
        (ptr_arg(quant, 1), ptr_arg(quant, 2))
    );
    assert_eq!(
        [8, 9, 10, 11].map(|i| ptr_arg(gu, i)),
        [
            sorted.expert_offsets,
            sorted.sorted_token_ids,
            sorted.active_experts,
            sorted.active_count
        ]
    );
    assert_eq!(gu.grid, [512 / 128, cap + m.div_ceil(8), 1]);
    // 2026-10-03: down reads gate+up's routed and shared products with their scales.
    assert_eq!(
        (ptr_arg(down, 0), ptr_arg(down, 1)),
        (ptr_arg(gu, 6), ptr_arg(gu, 7))
    );
    assert_eq!(
        (ptr_arg(down, 8), ptr_arg(down, 9)),
        (ptr_arg(gu, 16), ptr_arg(gu, 17))
    );
    assert_eq!(
        ptr_arg(gu, 7),
        ptr_arg(gu, 6).offset((m * top_k * 512) as usize),
        "the routed scales follow the [te, inter] E4M3 products"
    );
    assert_eq!(down.grid, [2048 / 256, cap + m.div_ceil(8), 1]);
    // 2026-10-03: the blend maps slots back through token_to_perm and weighs by top-k's weights.
    assert_eq!(ptr_arg(blend, 1), ptr_arg(down, 4));
    assert_eq!(ptr_arg(blend, 2), ptr_arg(topk, 2));
    assert_eq!(ptr_arg(blend, 3), sorted.token_to_perm);
    assert_eq!(ptr_arg(blend, 4), ptr_arg(down, 12));
    assert_eq!(blend.grid, [2048 / 256, m, 1]);
}

/// 2026-10-03: A layer that runs W8A16 experts (no FP8 expert activations published) refuses
/// the instance's W8A8 plan, and one off the tensor-core experts refuses the tensor-core
/// plan at one row.
#[test]
fn a_binding_whose_dispatch_differs_from_the_plan_is_refused() {
    let e = build_moe(Mode::Decode, 1, false, true)
        .err()
        .expect("W8A16 experts under a W8A8 plan");
    assert!(format!("{e:#}").contains("runs W8A16 experts"), "{e:#}");
    let e = build_moe(Mode::Decode, 1, true, false)
        .err()
        .expect("scalar experts at one row");
    assert!(format!("{e:#}").contains("does not take 1 rows"), "{e:#}");
}
