// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The executor compiled from the real dense circuit (the checked-in instance, its
//! FUSIONS.toml and its decode plan) over synthetic bindings, run on the recording mock backend:
//! launch counts, the pointers every launch reads, what a step may change, and the refusals.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::{BTreeMap, BTreeSet};

use metrale_circuit::planner::plan_buffers_with;
use metrale_circuit::{
    AvailableKernels, Circuit, FusionPlan, LayerKind, LinearRole, Mode, Numerics,
};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::bindings::*;
use super::compile::{self, Fixed, Inputs};
use super::kernels::KernelTable;
use super::program::{GdnState, Program, StepEnv};
use super::{Fusions, sources};
use crate::layer::AttnMetadataDev;
use crate::weight_map::{DenseWeight, QuantizedWeight};

const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
const WORKSPACE: u64 = 0xC000_0000;

fn ptr(tag: u64) -> DevicePtr {
    DevicePtr(tag)
}

fn nvfp4(tag: u64) -> BoundWeight {
    BoundWeight::Nvfp4(QuantizedWeight {
        weight: ptr(tag),
        weight_scale: ptr(tag + 1),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    })
}

fn dense(tag: u64) -> BoundWeight {
    BoundWeight::Dense(DenseWeight { weight: ptr(tag) })
}

struct Fixture {
    circuit: Circuit,
    plan: FusionPlan,
    program: Program,
    arena: u64,
    fixed: Fixed,
    layers: Vec<CircuitLayer>,
    head: HeadBinding,
}

/// 2026-09-28: Weight tags: layer `i`, slot `s` at `0x1_0000_0000 + i << 24 + s << 12`.
fn tag(layer: usize, slot: u64) -> u64 {
    0x1_0000_0000 + ((layer as u64) << 24) + (slot << 12)
}

fn layer_binding(circuit: &Circuit, i: usize, attn_idx: usize) -> CircuitLayer {
    let d = |k: &str| circuit.dims[k] as u32;
    let mut w = BTreeMap::from([
        (WeightSlot::InputNorm, dense(tag(i, 1))),
        (WeightSlot::PostNorm, dense(tag(i, 2))),
        (WeightSlot::FfnGate, nvfp4(tag(i, 3))),
        (WeightSlot::FfnUp, nvfp4(tag(i, 4))),
        (WeightSlot::Linear(LinearRole::Down), nvfp4(tag(i, 5))),
    ]);
    let mixer = if circuit.layer_kinds[i] == LayerKind::LinearAttention {
        w.insert(WeightSlot::Linear(LinearRole::Qkvz), nvfp4(tag(i, 6)));
        w.insert(WeightSlot::Linear(LinearRole::Ba), dense(tag(i, 7)));
        w.insert(WeightSlot::GdnALog, dense(tag(i, 8)));
        w.insert(WeightSlot::GdnDtBias, dense(tag(i, 9)));
        w.insert(WeightSlot::GdnConv1d, dense(tag(i, 10)));
        w.insert(WeightSlot::GdnNorm, dense(tag(i, 11)));
        w.insert(WeightSlot::Linear(LinearRole::GdnOut), nvfp4(tag(i, 12)));
        MixerFacts::Gdn(GdnFacts {
            qkvz_deinterleaved: true,
        })
    } else {
        for (s, role) in [
            (13, LinearRole::Q),
            (14, LinearRole::K),
            (15, LinearRole::V),
            (16, LinearRole::O),
        ] {
            w.insert(WeightSlot::Linear(role), nvfp4(tag(i, s)));
        }
        w.insert(WeightSlot::QNorm, dense(tag(i, 17)));
        w.insert(WeightSlot::KNorm, dense(tag(i, 18)));
        MixerFacts::Attention(AttnFacts {
            attn_layer_idx: attn_idx,
            kv_dtype: metrale_cache::kv_cache::KvCacheDtype::Bf16,
            num_q_heads: d("q_heads"),
            num_kv_heads: d("kv_heads"),
            head_dim: d("head_dim"),
            gated: true,
            rope: RopeFacts {
                mrope_interleaved: true,
                theta: 1.0e7,
                rotary_dim: 64,
            },
            sliding_window: 0,
            softmax_scale: 0.0625,
            paged_decode_plain: true,
        })
    };
    CircuitLayer {
        mixer,
        weights: w,
        unmodelled: Vec::new(),
    }
}

fn fixed(attn_layers: usize) -> Fixed {
    let meta = 0xA300_0000;
    Fixed {
        hidden: ptr(0xA000_0000),
        residual: ptr(0xA100_0000),
        logits: ptr(0xA200_0000),
        meta: AttnMetadataDev {
            positions: ptr(meta),
            positions_h: ptr(meta),
            positions_w: ptr(meta),
            slot: ptr(meta + 8),
            seq_len: ptr(meta + 16),
            block_table: ptr(meta + 256),
            max_blocks_per_seq: 0,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        k_pools: (0..attn_layers)
            .map(|i| ptr(0xB000_0000 + ((i as u64) << 24)))
            .collect(),
        v_pools: (0..attn_layers)
            .map(|i| ptr(0xB080_0000 + ((i as u64) << 24)))
            .collect(),
        block_size: 16,
        cache_stride: 4096,
    }
}

fn config() -> metrale_config::ModelConfig {
    let mut c = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_conv_kernel_dim = 4;
    c.final_norm_identity = false;
    c
}

fn build(fusions: Fusions, edit: impl Fn(&mut Vec<CircuitLayer>)) -> anyhow::Result<Fixture> {
    let inst = sources::instance(RECIPE)?;
    let loaded = metrale_circuit::load(&inst, sources::sources(&inst)?)?;
    let mut avail = AvailableKernels::all_named_by(&loaded.rules);
    if fusions == Fusions::ReferenceOnly {
        for r in loaded
            .rules
            .iter()
            .filter(|r| matches!(r.numerics, Numerics::BitIdentical { .. }))
        {
            for k in &r.kernels {
                avail.kernels.remove(k);
            }
        }
    }
    let plan = metrale_circuit::fuse(
        &loaded.circuit,
        &loaded.rules,
        &avail,
        &inst.policy,
        Mode::Decode,
        1,
    )?;
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, 1, &layout)?;
    let mut attn = 0;
    let mut layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let l = layer_binding(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            l
        })
        .collect();
    edit(&mut layers);
    let head = HeadBinding {
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
    };
    let fixed = fixed(attn);
    let gpu = MockGpuBackend::new();
    let cfg = config();
    let table = KernelTable::resolve(&gpu, &AvailableKernels::all_named_by(&loaded.rules));
    let program = compile::compile(
        &loaded.circuit,
        &plan,
        &layout,
        &buffers,
        ptr(WORKSPACE),
        &Inputs {
            config: &cfg,
            kernels: &table,
            fixed: &fixed,
            layers: &layers,
            head: &head,
        },
    )?;
    Ok(Fixture {
        circuit: loaded.circuit,
        plan,
        program,
        arena: buffers.arena_bytes,
        fixed,
        layers,
        head,
    })
}

fn states(f: &Fixture, base: u64) -> Vec<Option<GdnState>> {
    f.layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => Some(GdnState {
                h: ptr(base + ((i as u64) << 20)),
                conv: ptr(base + ((i as u64) << 20) + 0x8_0000),
            }),
            MixerFacts::Attention(_) => None,
        })
        .collect()
}

fn run(f: &Fixture, gdn: &[Option<GdnState>], max_blocks: u32) -> Vec<MockLaunch> {
    let gpu = MockGpuBackend::new();
    f.program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn,
            max_blocks_per_seq: max_blocks,
        })
        .unwrap();
    gpu.launches_snapshot()
}

#[test]
fn a_program_launches_exactly_what_its_plan_counts() {
    let full = build(Fusions::All, |_| {}).unwrap();
    let reference = build(Fusions::ReferenceOnly, |_| {}).unwrap();
    for f in [&full, &reference] {
        assert_eq!(f.program.launches.len() as u64, f.plan.launches());
        let launched = run(f, &states(f, 0xD000_0000), 9);
        assert_eq!(launched.len(), f.program.launches.len());
        assert!(launched.iter().all(|l| l.stream == 7));
    }
    let boundaries = full.circuit.layer_kinds.len() - 1;
    assert_eq!(
        reference.program.launches.len() - full.program.launches.len(),
        boundaries,
        "the cross-layer fusion saves one launch per layer boundary"
    );
}

#[test]
fn every_pointer_a_launch_reads_is_bound_placed_or_the_steps() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let gdn = states(&f, 0xD000_0000);
    let mut known: BTreeSet<u64> = BTreeSet::new();
    for l in &f.layers {
        for w in l.weights.values() {
            match w {
                BoundWeight::Dense(d) => known.insert(d.weight.0),
                BoundWeight::Nvfp4(q) => known.insert(q.weight.0) && known.insert(q.weight_scale.0),
            };
        }
    }
    known.extend([f.head.final_norm.weight.0, 0x9100_0000]);
    let m = f.fixed.meta;
    known.extend([f.fixed.hidden.0, f.fixed.residual.0, f.fixed.logits.0]);
    known.extend([m.positions.0, m.slot.0, m.seq_len.0, m.block_table.0]);
    known.extend(f.fixed.k_pools.iter().chain(&f.fixed.v_pools).map(|p| p.0));
    known.extend(gdn.iter().flatten().flat_map(|s| [s.h.0, s.conv.0]));
    for (i, l) in run(&f, &gdn, 9).iter().enumerate() {
        for a in &l.args {
            if let MockArg::Buffer(p) = a {
                let in_ws = (WORKSPACE..WORKSPACE + f.arena).contains(&p.0);
                assert!(
                    in_ws || known.contains(&p.0) || p.0 == 0,
                    "launch {i} ({}) reads {:#x}, which is neither bound nor placed",
                    f.program.launches[i].kernel,
                    p.0
                );
            }
        }
    }
}

#[test]
fn a_step_changes_only_the_state_and_width_arguments_that_read_them() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let a = run(&f, &states(&f, 0xD000_0000), 9);
    let b = run(&f, &states(&f, 0xE000_0000), 40);
    let mut changed = BTreeMap::<String, usize>::new();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            (x.grid, x.block, x.shared_mem),
            (y.grid, y.block, y.shared_mem)
        );
        if x.args != y.args {
            *changed
                .entry(f.program.launches[i].kernel.clone())
                .or_default() += 1;
        }
    }
    let gdn_layers = f
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, MixerFacts::Gdn(_)))
        .count();
    let attn_layers = f.layers.len() - gdn_layers;
    assert_eq!(
        changed,
        BTreeMap::from([
            (
                "causal_conv1d::causal_conv1d_update_l2norm_f32".to_string(),
                gdn_layers
            ),
            (
                "gated_delta_rule::gated_delta_rule_decode_f32".to_string(),
                gdn_layers
            ),
            ("paged_decode::paged_decode_attn".to_string(), attn_layers),
        ])
    );
}

#[test]
fn the_cross_layer_norm_reads_the_next_layers_input_norm_weight() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let launched = run(&f, &states(&f, 0xD000_0000), 9);
    let mut seen = 0;
    for (i, l) in f.program.launches.iter().enumerate() {
        if l.kernel != "residual_add_rms_norm_exact::residual_add_rms_norm_exact" {
            continue;
        }
        let group = &f.plan.groups[l.group];
        let next = f.circuit.nodes[group.nodes[1]].layer.unwrap();
        let want = tag(next, 1);
        assert!(
            launched[i].args.contains(&MockArg::Buffer(ptr(want))),
            "group {} does not read layer {next}'s input norm",
            l.group
        );
        seen += 1;
    }
    assert_eq!(seen, f.layers.len() - 1);
}

#[test]
fn a_binding_the_plan_does_not_describe_is_refused() {
    let err = |edit: &dyn Fn(&mut Vec<CircuitLayer>)| {
        build(Fusions::All, edit)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
    };
    let e = err(&|l| {
        l[0].mixer = MixerFacts::Gdn(GdnFacts {
            qkvz_deinterleaved: false,
        })
    });
    assert!(e.contains("deinterleaved"), "{e}");
    let e = err(&|l| {
        l[1].weights
            .insert(WeightSlot::Linear(LinearRole::GdnOut), dense(1));
    });
    assert!(e.contains("resolved") || e.contains("NVFP4"), "{e}");
    let e = err(&|l| {
        if let MixerFacts::Attention(a) = &mut l[3].mixer {
            a.paged_decode_plain = false;
        }
    });
    assert!(e.contains("plain paged kernel"), "{e}");
    let mut layers: Vec<Option<CircuitLayer>> = build(Fusions::All, |_| {})
        .unwrap()
        .layers
        .into_iter()
        .map(Some)
        .collect();
    let inst = sources::instance(RECIPE).unwrap();
    let circuit = metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .circuit;
    let head = HeadBinding {
        final_norm: DenseWeight { weight: ptr(1) },
        lm_head: dense(2),
        unmodelled: Vec::new(),
    };
    layers[5]
        .as_mut()
        .unwrap()
        .unmodelled
        .push("an out_proj LoRA adapter".into());
    let e = compile::check_bindings(&circuit, &layers, &head)
        .unwrap_err()
        .to_string();
    assert!(e.contains("layer 5: an out_proj LoRA adapter"), "{e}");
    layers[5] = None;
    let e = compile::check_bindings(&circuit, &layers, &head)
        .unwrap_err()
        .to_string();
    assert!(e.contains("layer 5 has no circuit binding"), "{e}");
}

#[test]
fn ptx_availability_reads_entry_points_not_names() {
    let ptx: &[u8] =
        b".visible .entry rope_forward(\n.param .u64 a\n)\n// rope_forward_strided is a comment\n";
    assert!(super::kernels::ptx_defines(ptx, "rope_forward"));
    assert!(!super::kernels::ptx_defines(ptx, "rope_forward_strided"));
    assert!(!super::kernels::ptx_defines(ptx, "rope"));
    let inst = sources::instance(RECIPE).unwrap();
    let rules = metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .rules;
    let avail = super::kernels::available_in(&rules, &[("rope", ptx)]).unwrap();
    assert!(
        avail
            .kernels
            .iter()
            .all(|k| k.module == "rope" && k.func == "rope_forward")
    );
    let gpu = MockGpuBackend::new();
    let table = KernelTable::resolve(&gpu, &avail);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup is issued for a kernel the target does not define"
    );
    assert!(
        table
            .handle(&metrale_circuit::KernelId {
                module: "norm".into(),
                func: "rms_norm".into()
            })
            .is_err()
    );
}
