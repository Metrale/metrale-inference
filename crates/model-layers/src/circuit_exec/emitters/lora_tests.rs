// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The LoRA emitters compiled from the real dense circuit adapted on every adaptable
//! projection (`metrale_circuit::lora::adapt`), over the fixture's synthetic bindings plus
//! adapters, run on the recording mock: each fold lands on the buffer its projection wrote and
//! its readers read, with the mode's slot upload, the right row count and, for the pair, legacy's
//! per-row and per-half offsets; and the refusals.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::Result;
use metrale_circuit::lora::{ADAPTABLE, LoraSpec};
use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{AvailableKernels, Circuit, FusionPlan, LinearRole, Mode};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::super::bindings::{CircuitLayer, HeadBinding, MixerFacts};
use super::super::super::compile::{self, Inputs};
use super::super::super::exec_fixture::{
    RECIPE, WORKSPACE, config, dense, fixed, layer_binding, ptr,
};
use super::super::super::kernels::KernelTable;
use super::super::super::program::{GdnState, LaunchKind, Program, StepEnv};
use super::super::super::sources;
use super::{LoraFixed, LoraLayer};
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair, LoraRoute};
use crate::weight_map::DenseWeight;

const RANK: u32 = 16;

const LORA: LoraFixed = LoraFixed {
    xa: DevicePtr(0xA900_0000),
    delta: DevicePtr(0xAA00_0000),
    slots_decode: DevicePtr(0xAB00_0000),
    slots_multi_seq: DevicePtr(0xAB10_0000),
    slots_verify: DevicePtr(0xAB20_0000),
};

fn pair(tag: u64, k_in: u32, n_out: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight { weight: ptr(tag) },
        b: DenseWeight {
            weight: ptr(tag + 0x100),
        },
        rank: RANK,
        k_in,
        n_out,
        scale: 0.5,
        max_rank: RANK,
    }
}

fn route(tag: u64, k_in: u32, n_out: u32) -> LoraRoute {
    LoraRoute {
        a_table: ptr(tag),
        b_table: ptr(tag + 0x10),
        scale_table: ptr(tag + 0x20),
        k_in,
        n_out,
        max_rank: RANK,
    }
}

/// 2026-10-03: Layer `i`'s adapters: routes for its attention projections, the active pairs
/// for its FFN and its GatedDeltaNet out_proj; tags at `0x7_0000_0000 + i << 20`.
fn adapters(c: &Circuit, i: usize, kernels: LoraKernels) -> LoraLayer {
    let d = |k: &str| c.dims[k] as u32;
    let t = 0x7_0000_0000 + ((i as u64) << 20);
    let (h, inter) = (d("hidden"), d("inter"));
    let q = d("q_heads") * d("head_dim");
    let kv = d("kv_heads") * d("head_dim");
    let mut routes = BTreeMap::new();
    let mut pairs = BTreeMap::new();
    if c.layer_kinds[i] == metrale_circuit::LayerKind::FullAttention {
        routes.insert(LinearRole::Q, route(t, h, q * 2));
        routes.insert(LinearRole::K, route(t + 0x1000, h, kv));
        routes.insert(LinearRole::V, route(t + 0x2000, h, kv));
        routes.insert(LinearRole::O, route(t + 0x3000, q, h));
    } else {
        let gdn_v = d("lin_v_heads") * d("lin_v_dim");
        pairs.insert(LinearRole::GdnOut, vec![pair(t + 0x4000, gdn_v, h)]);
    }
    pairs.insert(
        LinearRole::GateUp,
        vec![pair(t + 0x5000, h, inter), pair(t + 0x6000, h, inter)],
    );
    pairs.insert(LinearRole::Down, vec![pair(t + 0x7000, inter, h)]);
    LoraLayer {
        kernels,
        routes,
        pairs,
    }
}

struct Adapted {
    circuit: Circuit,
    plan: FusionPlan,
    program: Program,
    layers: Vec<CircuitLayer>,
}

/// 2026-10-03: The adapted `mode` plan at `rows`, compiled over the fixture bindings with
/// adapters on every layer, after `edit`.
fn build(mode: Mode, rows: u64, edit: impl Fn(&mut Vec<CircuitLayer>)) -> Result<Adapted> {
    let mut inst = sources::instance(RECIPE)?;
    // 2026-10-03: As `policy::live_policy` states it for a model holding a LoRA pool.
    inst.policy.settings.insert(
        metrale_circuit::lora::ACTIVE_SETTING.to_string(),
        "on".to_string(),
    );
    let mut loaded = metrale_circuit::load(&inst, sources::sources(&inst)?)?;
    let c = &loaded.circuit;
    let spec = LoraSpec {
        rank: u64::from(RANK),
        layers: (0..c.layer_kinds.len())
            .map(|l| {
                let roles = ADAPTABLE
                    .into_iter()
                    .filter(|r| {
                        c.nodes.iter().any(|n| {
                            n.layer == Some(l) && n.op == metrale_circuit::OpKind::Linear(*r)
                        })
                    })
                    .collect();
                (l, roles)
            })
            .collect(),
    };
    loaded.circuit = metrale_circuit::lora::adapt(&loaded.circuit, &spec)?;
    let avail = AvailableKernels::all_named_by(&loaded.rules);
    let plan = metrale_circuit::fuse(
        &loaded.circuit,
        &loaded.rules,
        &avail,
        &inst.policy,
        mode,
        rows,
    )?;
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
    let gpu = MockGpuBackend::new();
    crate::layers::ops::w4a4_proj::prepare(&gpu)?;
    let kernels = LoraKernels::new(&gpu)?;
    let mut attn = 0;
    let mut layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let mut l = layer_binding(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            l.lora = Some(adapters(&loaded.circuit, i, kernels));
            l
        })
        .collect();
    edit(&mut layers);
    let mut fx = fixed(attn);
    fx.lora = Some(LORA);
    let head = HeadBinding {
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
    };
    let cfg = config();
    let table = KernelTable::resolve(&gpu, &avail);
    let program = compile::compile(
        &loaded.circuit,
        &plan,
        &layout,
        &buffers,
        ptr(WORKSPACE),
        &Inputs {
            gpu: &gpu,
            config: &cfg,
            kernels: &table,
            fixed: &fx,
            layers: &layers,
            head: &head,
            draft: None,
        },
    )?;
    Ok(Adapted {
        circuit: loaded.circuit,
        plan,
        program,
        layers,
    })
}

/// 2026-10-03: Each kernel launch with the member locals and layer of its group.
fn launched(a: &Adapted, rows: usize) -> Vec<(MockLaunch, String, Vec<String>, Option<usize>)> {
    let verify = a.plan.mode == Mode::Verify;
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(KernelHandle(0xDEAD), 128);
    // 2026-10-03: Per GDN layer, each sequence's state: contiguous at the fixture's slot pitch in a
    // multi-sequence step (the batched arm checks it); a verify's one sequence with its conv
    // window and rollback slots allocated on the mock (its snapshots copy them).
    let d = |k: &str| a.circuit.dims[k] as usize;
    let window =
        (d("lin_k_heads") * d("lin_k_dim") * 2 + d("lin_v_heads") * d("lin_v_dim")) * 4 * 4;
    let alloc = || gpu.alloc(window).unwrap();
    let gdn: Vec<Vec<GdnState>> = a
        .layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => (0..if verify { 1 } else { rows })
                .map(|r| {
                    let at = 0xD000_0000 + ((i as u64) << 24) + ((r as u64) << 16);
                    let conv = |t: u64| {
                        if verify {
                            alloc()
                        } else {
                            ptr(at + 0x8000 + 0x1000 * t)
                        }
                    };
                    GdnState {
                        h: ptr(at),
                        conv: conv(0),
                        h_steps: [1, 2, 3].map(|t| ptr(at + 0x1000 * t)),
                        conv_steps: [1, 2, 3].map(conv),
                    }
                })
                .collect(),
            MixerFacts::Attention(_) => Vec::new(),
        })
        .collect();
    a.program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
        })
        .unwrap();
    let kernels = a
        .program
        .launches
        .iter()
        .filter(|l| l.kind == LaunchKind::Kernel);
    gpu.launches_snapshot()
        .into_iter()
        .zip(kernels)
        .map(|(m, l)| {
            let g = &a.plan.groups[l.group];
            let locals = g
                .nodes
                .iter()
                .map(|&n| a.circuit.nodes[n].local.clone())
                .collect();
            let layer = a.circuit.nodes[g.nodes[0]].layer;
            (m, l.kernel.clone(), locals, layer)
        })
        .collect()
}

fn ptr_arg(m: &MockLaunch, i: usize) -> u64 {
    match &m.args[i] {
        MockArg::Buffer(p) => p.0,
        other => panic!("arg {i} is {other:?}, not a buffer"),
    }
}

fn has_ptr(m: &MockLaunch, p: u64) -> bool {
    m.args.contains(&MockArg::Buffer(DevicePtr(p)))
}

type Launched = Vec<(MockLaunch, String, Vec<String>, Option<usize>)>;

/// 2026-10-03: Index of the launch of `kernel` in layer `layer` whose group holds `local`.
fn find(l: &Launched, kernel: &str, local: &str, layer: usize) -> Vec<usize> {
    l.iter()
        .enumerate()
        .filter(|(_, (_, k, ls, at))| {
            k == kernel && ls.iter().any(|x| x == local) && *at == Some(layer)
        })
        .map(|(i, _)| i)
        .collect()
}

/// 2026-10-03: For every attention projection of every attention layer: the bgmv fold reads the
/// mode's slots over `rows` rows, folds into the buffer the projection wrote, and a later launch
/// reads that buffer.
fn assert_attention_folds(a: &Adapted, rows: usize, slots: DevicePtr) {
    let l = launched(a, rows);
    let mut checked = 0;
    for (layer, kind) in a.circuit.layer_kinds.iter().enumerate() {
        if *kind != metrale_circuit::LayerKind::FullAttention {
            continue;
        }
        for r in ["q", "k", "v", "o"] {
            let lb = format!("{r}_lora_b");
            let fold = find(&l, "lora_bgmv::lora_bgmv_expand_fold", &lb, layer);
            let shrink = find(
                &l,
                "lora_bgmv::lora_bgmv_shrink",
                &format!("{r}_lora_a"),
                layer,
            );
            assert_eq!((fold.len(), shrink.len()), (1, 1), "layer {layer} {r}");
            let (m, f) = (&l[fold[0]].0, fold[0]);
            assert_eq!(
                m.grid[1] as usize, rows,
                "layer {layer} {r}: one block row per row"
            );
            assert_eq!(
                ptr_arg(m, 1),
                slots.0,
                "layer {layer} {r}: the mode's slot upload"
            );
            assert_eq!(ptr_arg(&l[shrink[0]].0, 3), LORA.xa.0);
            assert!(shrink[0] < f, "layer {layer} {r}: shrink before fold");
            let base = ptr_arg(m, 4);
            let wrote = l[..shrink[0]].iter().rposition(|(x, k, ls, at)| {
                *at == Some(layer)
                    && ls.iter().any(|y| y == r)
                    && !k.starts_with("lora")
                    && has_ptr(x, base)
            });
            assert!(
                wrote.is_some(),
                "layer {layer} {r}: no projection wrote {base:#x}"
            );
            let read = l[f + 1..]
                .iter()
                .any(|(x, k, _, _)| !k.starts_with("lora") && has_ptr(x, base));
            assert!(
                read,
                "layer {layer} {r}: nothing reads the folded {base:#x}"
            );
            checked += 1;
        }
    }
    assert!(checked >= 4, "no attention layer was checked");
}

#[test]
fn one_row_attention_folds_land_on_the_projection_output_with_the_decode_slots() {
    let a = build(Mode::Decode, 1, |_| {}).unwrap();
    assert_attention_folds(&a, 1, LORA.slots_decode);
}

#[test]
fn multi_row_attention_folds_read_the_multi_sequence_slots_over_every_row() {
    let a = build(Mode::MultiSeq, 3, |_| {}).unwrap();
    assert_attention_folds(&a, 3, LORA.slots_multi_seq);
    let l = launched(&a, 3);
    let q = l.iter().filter(|(_, k, ls, _)| {
        k == "w4a16_gemv::w4a16_gemv_batch3" && ls == &vec!["q".to_string()]
    });
    assert!(
        q.count() > 0,
        "the adapted q runs the plain batch3 GEMV (legacy's adapter route)"
    );
}

#[test]
fn verify_attention_folds_read_the_verify_slots() {
    let a = build(Mode::Verify, 4, |_| {}).unwrap();
    assert_attention_folds(&a, 4, LORA.slots_verify);
}

#[test]
fn the_pair_folds_each_row_of_each_half_at_legacys_offsets() {
    let rows = 3;
    let a = build(Mode::MultiSeq, rows as u64, |_| {}).unwrap();
    let l = launched(&a, rows);
    let gdn = a
        .circuit
        .layer_kinds
        .iter()
        .position(|k| *k == metrale_circuit::LayerKind::LinearAttention)
        .unwrap();
    let inter = a.circuit.dims["inter"] as u64;
    let h = a.circuit.dims["hidden"] as u64;
    let adds = |local: &str| -> Vec<u64> {
        find(&l, "residual_add::bf16_scaled_add", local, gdn)
            .into_iter()
            .map(|i| ptr_arg(&l[i].0, 0))
            .collect()
    };
    let gu = adds("gate_up_lora_b");
    assert_eq!(gu.len(), 2 * rows, "gate then up, each row");
    let base = gu[0];
    let want: Vec<u64> = (0..2 * rows as u64).map(|i| base + i * inter * 2).collect();
    assert_eq!(gu, want, "gate rows, then up rows, n_out apart");
    let out = adds("out_lora_b");
    assert_eq!(out.len(), rows);
    assert!(out.windows(2).all(|w| w[1] == w[0] + h * 2));
    let gemvs = find(&l, "gemv::dense_gemv_bf16", "gate_up_lora_a", gdn);
    assert_eq!(
        gemvs.len(),
        2 * 2 * rows,
        "shrink and expand per row per half"
    );
    assert!(gemvs.iter().all(|&i| has_ptr(&l[i].0, LORA.xa.0)));
}

#[test]
fn a_layer_without_adapters_and_a_gate_up_with_one_pair_are_refused() {
    let e = build(Mode::Decode, 1, |ls| ls[3].lora = None)
        .err()
        .unwrap();
    assert!(format!("{e:#}").contains("binds no adapters"), "{e:#}");
    let e = build(Mode::Decode, 1, |ls| {
        for l in ls.iter_mut() {
            if let Some(ad) = l.lora.as_mut() {
                ad.pairs.get_mut(&LinearRole::GateUp).unwrap().truncate(1);
            }
        }
    })
    .err()
    .unwrap();
    assert!(format!("{e:#}").contains("binds 1 pairs"), "{e:#}");
}
