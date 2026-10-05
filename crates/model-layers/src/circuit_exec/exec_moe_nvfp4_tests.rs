// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The executor over the nvidia/Qwen3.6-35B-A3B-NVFP4 instance (recipe
//! `qwen3.6-35b-a3b-nvfp4-declared`): every plan it checks in, compiled over bindings of the
//! checkpoint's shape (W8A8 attention and GDN projections on the adopted block-scaled FP8 weights,
//! the grouped NVFP4 MoE, the declared NVFP4 head on its row tiles, a BF16 MoE drafter) and run on
//! the recording mock backend. The expert step's launches are compared with what the legacy
//! launchers record for the same buffers, and a binding whose expert kernels differ from the
//! plan's is refused.
//!
//! Owner: model-layers (MoE) circuit emitters.
//! Invariants: none beyond the types.

use metrale_circuit::{Circuit, LayerKind, LinearRole, Mode};
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, MockLaunch};

use super::Fusions;
use super::bindings::*;
use super::exec_fixture::*;
use crate::layers::moe::{ExpertKind, MoeBinding, MoeExperts, MoeFacts, MoeKernels};
use crate::layers::ops::{self, W8a8Kernels, W8a8Weight};
use crate::weight_map::{DenseWeight, Fp8Weight, QuantizedWeight, WeightQuantFormat};

const NV: &str = "qwen3.6/qwen3.6-35b-a3b-nvfp4-declared";
/// 2026-10-05: The mock backend's one kernel handle.
const K: KernelHandle = KernelHandle(0xDEAD);

fn q4(tag: u64) -> QuantizedWeight {
    match nvfp4(tag) {
        BoundWeight::Nvfp4(q) => q,
        other => panic!("{other:?}"),
    }
}

fn tables(tag: u64) -> ops::Nvfp4ExpertTables {
    ops::Nvfp4ExpertTables {
        packed_ptrs: ptr(tag),
        scale_ptrs: ptr(tag + 1),
        scale2_vals: ptr(tag + 2),
    }
}

/// 2026-10-05: The checkpoint's MoE dims (config.json).
fn dim(k: &str) -> u32 {
    match k {
        "experts" => 256,
        "top_k" => 8,
        "hidden" => 2048,
        "moe_inter" => 512,
        other => panic!("no MoE dim {other}"),
    }
}

/// 2026-10-05: A MoE binding of `kind` for layer `i` (tags at slots 50..; the drafter is 200).
fn moe(i: usize, kind: ExpertKind) -> MoeBinding {
    let d = dim;
    let bf16 = kind == ExpertKind::Bf16;
    let experts = if bf16 {
        MoeExperts::Bf16 {
            gate: ptr(tag(i, 52)),
            up: ptr(tag(i, 53)),
            down: ptr(tag(i, 54)),
            shared: [55, 56, 57].map(|s| DenseWeight {
                weight: ptr(tag(i, s)),
            }),
        }
    } else {
        MoeExperts::Nvfp4 {
            gate: tables(tag(i, 52)),
            up: tables(tag(i, 55)),
            down: tables(tag(i, 58)),
            shared: [61, 64, 67].map(|s| q4(tag(i, s))),
        }
    };
    let (gu, dn) = if bf16 {
        (ops::BF16_GROUPED_GATE_UP_TC, ops::BF16_GROUPED_DOWN_TC)
    } else {
        (ops::NVFP4_GROUPED_GATE_UP_TC, ops::NVFP4_GROUPED_DOWN_TC)
    };
    MoeBinding {
        router: DenseWeight {
            weight: ptr(tag(i, 50)),
        },
        shared_gate: DenseWeight {
            weight: ptr(tag(i, 51)),
        },
        experts,
        facts: MoeFacts {
            num_experts: d("experts"),
            top_k: d("top_k"),
            hidden: d("hidden"),
            inter: d("moe_inter"),
            norm_topk_prob: true,
            tensor_core: true,
            w8a8: false,
            kind,
        },
        kernels: MoeKernels {
            router_rows: K,
            router_gemm: K,
            topk_rows: K,
            topk_batched: K,
            sort: K,
            gate_up: K,
            gate_up_geometry: gu,
            down: K,
            down_geometry: dn,
            quant_w8a8: K,
            gate_up_w8a8: K,
            down_w8a8: K,
            blend: K,
            router_gemv: K,
            topk_one_row: K,
            fused_gate_up_bf16: K,
            fused_down_bf16: K,
            blend_one_row: K,
        },
    }
}

/// 2026-10-05: The loader's bindings: the attention and GDN projections W8A8 on block-scaled
/// FP8 weights at slots 40.. (`w8a8_decode_arm.rs`, `qwen3_ssm/w8a8_decode.rs`), no dense FFN,
/// the lean NVFP4 MoE.
fn layer(c: &Circuit, i: usize, attn: usize) -> (CircuitLayer, Vec<u64>) {
    let mut l = layer_binding(c, i, attn);
    let d = |k: &str| c.dims[k] as u32;
    let h = d("hidden");
    let kernels = W8a8Kernels::load(&MockGpuBackend::new());
    let mut ptrs = Vec::new();
    let mut w8 = |slot: u64, n: u32, k: u32| {
        let f = Fp8Weight {
            weight: ptr(tag(i, 40 + slot)),
            row_scale: ptr(tag(i, 41 + slot)),
            n,
            k,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        };
        ptrs.extend([f.weight.0, f.row_scale.0]);
        f
    };
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
    let mut put = |role: LinearRole, w: W8a8Weight| {
        l.weights
            .insert(WeightSlot::Linear(role), BoundWeight::W8a8(w, kernels));
    };
    if c.layer_kinds[i] == LayerKind::LinearAttention {
        let (kd, vd) = (
            d("lin_k_heads") * d("lin_k_dim"),
            d("lin_v_heads") * d("lin_v_dim"),
        );
        put(
            LinearRole::Qkvz,
            W8a8Weight::new(&[w8(0, 2 * kd + 2 * vd, h)]).unwrap(),
        );
        put(
            LinearRole::GdnOut,
            W8a8Weight::new(&[w8(2, h, vd)]).unwrap(),
        );
    } else {
        let (q, kv) = (d("q_heads") * d("head_dim"), d("kv_heads") * d("head_dim"));
        let qkv = W8a8Weight::new(&[w8(0, 2 * q, h), w8(2, kv, h), w8(4, kv, h)]).unwrap();
        for (s, role) in [LinearRole::Q, LinearRole::K, LinearRole::V]
            .into_iter()
            .enumerate()
        {
            put(role, qkv.segment(s).unwrap());
        }
        put(LinearRole::O, W8a8Weight::new(&[w8(6, h, q)]).unwrap());
    }
    assert_eq!(c.dims["moe_inter"] as u32, dim("moe_inter"));
    l.moe = Some(moe(i, ExpertKind::Nvfp4Lean));
    (l, ptrs)
}

fn build(mode: Mode, rows: u64, arm: Arm, kind: ExpertKind) -> anyhow::Result<Fixture> {
    let gpu = MockGpuBackend::new();
    build_drafting(
        (&gpu, &config()),
        NV,
        &layer,
        Fusions::All,
        (mode, rows, arm),
        (
            move |ls: &mut Vec<CircuitLayer>| {
                for l in ls.iter_mut() {
                    if let Some(m) = l.moe.as_mut() {
                        m.facts.kind = kind;
                    }
                }
            },
            |h: &mut HeadBinding| {
                h.lm_head = nvfp4(0x9100_0000);
                h.nvfp4_rows = true;
            },
            |d: &mut CircuitLayer| {
                d.weights
                    .retain(|s, _| !matches!(s, WeightSlot::FfnGate | WeightSlot::FfnUp));
                d.weights.remove(&WeightSlot::Linear(LinearRole::Down));
                d.moe = Some(moe(200, ExpertKind::Bf16));
            },
        ),
    )
}

/// 2026-10-05: Every primary plan the instance checks in (decode, the multi-sequence ladder,
/// verify K = 2..4, the drafter's one- and n-row propose) and every batched-verify table compiles
/// to exactly its planned launches, and the non-verify ones run.
#[test]
fn every_nvfp4_35b_plan_compiles_to_the_launches_it_counts() {
    let inst = super::sources::instance(NV).unwrap();
    let primary = inst
        .plans
        .iter()
        .flat_map(|(m, rows)| rows.iter().map(move |&r| (*m, r, Arm::Primary)));
    let tables = inst.verify_batch.iter().map(|t| {
        let s: &'static str = Box::leak(t.to_string().into_boxed_str());
        (Mode::VerifyBatch, t.rows(), Arm::Table(s))
    });
    for (mode, rows, arm) in primary.chain(tables) {
        let f = build(mode, rows, arm, ExpertKind::Nvfp4Lean)
            .unwrap_or_else(|e| panic!("{mode:?} at {rows} ({arm:?}): {e:#}"));
        assert_eq!(
            f.program.launches.len() as u64,
            f.plan.launches() + f.plan.copies(),
            "{mode:?} {rows} {arm:?}"
        );
        if matches!(mode, Mode::Decode | Mode::MultiSeq) {
            let launched = run(&f, &states_rows(&f, 0xD000_0000, rows as usize), 9);
            assert_eq!(launched.len() as u64, f.plan.launches(), "{mode:?} {rows}");
        }
    }
}

fn group_launches(f: &Fixture, launched: &[MockLaunch], first: &str) -> Vec<MockLaunch> {
    kernel_launches(f)
        .zip(launched)
        .filter(|(l, _)| f.circuit.nodes[f.plan.groups[l.group].nodes[0]].id == first)
        .map(|(_, m)| m.clone())
        .collect()
}

/// 2026-10-05: Layer 0's expert step at 24 rows is, launch for launch, what
/// `forward_nvfp4_grouped_decode` makes for the same buffers: gate+up over the post-norm rows with
/// the sort's outputs, its SiLU products at the arena's `expert_gate_out` (routed) and `logits`
/// (shared), down from there into the edges the blend reads.
#[test]
fn the_nvfp4_expert_step_is_the_launches_legacy_makes() {
    let m = 24u32;
    let f = build(
        Mode::MultiSeq,
        m.into(),
        Arm::Primary,
        ExpertKind::Nvfp4Lean,
    )
    .unwrap();
    let launched = run(&f, &states_rows(&f, 0xD000_0000, m as usize), 9);
    let got = group_launches(&f, &launched, "l0.moe_ffn.experts_gate_up");
    let blend = &group_launches(&f, &launched, "l0.moe_ffn.shared_gate")[0];
    let b = f.layers[0].moe.clone().unwrap();
    let MoeExperts::Nvfp4 {
        gate,
        up,
        down,
        shared: [sg, su, sd],
    } = b.experts
    else {
        unreachable!()
    };
    let (sort, cap) = b.sort_out(&MOE_SCRATCH, m);
    let buf = |l: &MockLaunch, i: usize| match &l.args[i] {
        metrale_gpu_runtime::gpu::mock::MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}"),
    };
    let (x, edown, sdown) = (buf(&got[0], 0), buf(&got[1], 4), buf(&got[1], 12));
    let stream = got[0].stream;
    let (act, sh_act) = (MOE_SCRATCH.routed_act, MOE_SCRATCH.shared_act);
    let gpu = MockGpuBackend::new();
    let (h, inter) = (2048, 512);
    ops::moe_expert_gate_up_act_nvfp4_grouped(
        &gpu,
        K,
        ops::NVFP4_GROUPED_GATE_UP_TC,
        x,
        gate,
        up,
        act,
        sort.expert_offsets,
        sort.sorted_token_ids,
        sort.active_experts,
        sort.active_count,
        &sg,
        &su,
        sh_act,
        inter,
        h,
        cap,
        m,
        stream,
    )
    .unwrap();
    ops::moe_expert_down_act_nvfp4_grouped(
        &gpu,
        K,
        ops::NVFP4_GROUPED_DOWN_TC,
        act,
        down,
        edown,
        sort.expert_offsets,
        sort.active_experts,
        sort.active_count,
        sh_act,
        &sd,
        sdown,
        h,
        inter,
        cap,
        m,
        stream,
    )
    .unwrap();
    assert_eq!(
        format!("{got:?}"),
        format!("{:?}", gpu.launches_snapshot()),
        "the expert step's two launches"
    );
    // 2026-10-05: The step reads the rows the post-norm wrote, and the blend reads its outputs.
    let post = &group_launches(&f, &launched, "l0.gdn.add")[0];
    assert!(
        post.args.contains(&got[0].args[0]),
        "gate+up reads the post-norm rows"
    );
    assert_eq!((buf(blend, 1), buf(blend, 4)), (edown, sdown));
}

/// 2026-10-05: The plan names the pair the layer's dispatch picks: the lean plan over a layer
/// whose tables stayed row-major, or the reverse, is refused.
#[test]
fn a_layer_whose_tables_are_in_the_other_layout_is_refused() {
    for (planned, bound) in [
        ("lean", ExpertKind::Nvfp4TensorCore),
        ("tensor_core", ExpertKind::Nvfp4Lean),
    ] {
        let e = build(
            Mode::Decode,
            1,
            Arm::Setting("moe_nvfp4_kernels", planned),
            bound,
        )
        .map(|_| ())
        .expect_err("a plan over the other layout");
        assert!(
            format!("{e:#}").contains("the experts' tables are"),
            "{e:#}"
        );
    }
}
