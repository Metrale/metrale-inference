// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The test fixture of the executor: the real dense circuit (the checked-in
//! instance, its FUSIONS.toml) fused at a mode and row count, compiled over synthetic bindings,
//! and run on the recording mock backend.
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

pub(super) const RECIPE: &str = "qwen3.8/qwen3.8-27b-nvfp4-unsloth";
pub(super) const WORKSPACE: u64 = 0xC000_0000;

pub(super) fn ptr(tag: u64) -> DevicePtr {
    DevicePtr(tag)
}

pub(super) fn nvfp4(tag: u64) -> BoundWeight {
    BoundWeight::Nvfp4(QuantizedWeight {
        weight: ptr(tag),
        weight_scale: ptr(tag + 1),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
        act: Default::default(),
    })
}

pub(super) fn dense(tag: u64) -> BoundWeight {
    BoundWeight::Dense(DenseWeight { weight: ptr(tag) })
}

pub(super) struct Fixture {
    pub circuit: Circuit,
    pub plan: FusionPlan,
    pub program: Program,
    pub arena: u64,
    pub fixed: Fixed,
    pub layers: Vec<CircuitLayer>,
    pub head: HeadBinding,
}

/// 2026-09-28: Weight tags: layer `i`, slot `s` at `0x1_0000_0000 + i << 24 + s << 12`.
pub(super) fn tag(layer: usize, slot: u64) -> u64 {
    0x1_0000_0000 + ((layer as u64) << 24) + (slot << 12)
}

pub(super) fn layer_binding(circuit: &Circuit, i: usize, attn_idx: usize) -> CircuitLayer {
    let d = |k: &str| circuit.dims[k] as u32;
    let mut w = BTreeMap::from([
        (WeightSlot::InputNorm, dense(tag(i, 1))),
        (WeightSlot::PostNorm, dense(tag(i, 2))),
        (WeightSlot::FfnGate, nvfp4(tag(i, 3))),
        (WeightSlot::FfnUp, nvfp4(tag(i, 4))),
        (WeightSlot::Linear(LinearRole::Down), nvfp4(tag(i, 5))),
        (WeightSlot::FfnGateMmq, BoundWeight::Mmq(ptr(tag(i, 25)))),
        (WeightSlot::FfnUpMmq, BoundWeight::Mmq(ptr(tag(i, 26)))),
        (WeightSlot::FfnDownMmq, BoundWeight::Mmq(ptr(tag(i, 27)))),
    ]);
    let mixer = if circuit.layer_kinds[i] == LayerKind::LinearAttention {
        w.insert(WeightSlot::Linear(LinearRole::Qkvz), nvfp4(tag(i, 6)));
        w.insert(WeightSlot::Linear(LinearRole::Ba), dense(tag(i, 7)));
        w.insert(WeightSlot::GdnALog, dense(tag(i, 8)));
        w.insert(WeightSlot::GdnDtBias, dense(tag(i, 9)));
        w.insert(WeightSlot::GdnConv1d, dense(tag(i, 10)));
        w.insert(WeightSlot::GdnNorm, dense(tag(i, 11)));
        w.insert(WeightSlot::Linear(LinearRole::GdnOut), nvfp4(tag(i, 12)));
        w.insert(WeightSlot::Transposed(LinearRole::Qkvz), nvfp4(tag(i, 19)));
        w.insert(
            WeightSlot::Transposed(LinearRole::GdnOut),
            nvfp4(tag(i, 20)),
        );
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
            w.insert(WeightSlot::Transposed(role), nvfp4(tag(i, s + 8)));
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
            paged_decode_plain_rows: u128::MAX,
        })
    };
    CircuitLayer {
        mixer,
        weights: w,
        unmodelled: Vec::new(),
    }
}

pub(super) fn fixed(attn_layers: usize) -> Fixed {
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
        batch_meta: AttnMetadataDev {
            positions: ptr(meta + 0x1_0000),
            positions_h: ptr(meta + 0x1_0000),
            positions_w: ptr(meta + 0x1_0000),
            slot: ptr(meta + 0x1_0000 + 1024),
            seq_len: ptr(meta + 0x1_0000 + 2048),
            block_table: ptr(meta + 0x1_0000 + 3072),
            max_blocks_per_seq: 0,
            num_seqs: 128,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        ffn_act_q8: ptr(0xA400_0000),
        block_size: 16,
        cache_stride: 4096,
    }
}

pub(super) fn config() -> metrale_config::ModelConfig {
    let mut c = metrale_config::ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_conv_kernel_dim = 4;
    c.final_norm_identity = false;
    c
}

pub(super) fn build(
    fusions: Fusions,
    edit: impl Fn(&mut Vec<CircuitLayer>),
) -> anyhow::Result<Fixture> {
    build_at(fusions, Mode::Decode, 1, edit, |_| {})
}

/// 2026-09-28: Compile the `mode` plan at `rows` rows over the synthetic bindings, after `edit`
/// and `edit_head`.
pub(super) fn build_at(
    fusions: Fusions,
    mode: Mode,
    rows: u64,
    edit: impl Fn(&mut Vec<CircuitLayer>),
    edit_head: impl Fn(&mut HeadBinding),
) -> anyhow::Result<Fixture> {
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
        mode,
        rows,
    )?;
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
    let mut attn = 0;
    let mut layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let l = layer_binding(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            l
        })
        .collect();
    edit(&mut layers);
    let mut head = HeadBinding {
        final_norm: DenseWeight {
            weight: ptr(0x9000_0000),
        },
        lm_head: dense(0x9100_0000),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
    };
    edit_head(&mut head);
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
            gpu: &gpu,
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

pub(super) fn states(f: &Fixture, base: u64) -> Vec<Vec<GdnState>> {
    states_rows(f, base, 1)
}

/// 2026-09-28: `rows` distinct GDN states per GDN layer, from `base`.
pub(super) fn states_rows(f: &Fixture, base: u64, rows: usize) -> Vec<Vec<GdnState>> {
    f.layers
        .iter()
        .enumerate()
        .map(|(i, l)| match l.mixer {
            MixerFacts::Gdn(_) => (0..rows)
                .map(|r| {
                    let at = base + ((i as u64) << 24) + ((r as u64) << 16);
                    GdnState {
                        h: ptr(at),
                        conv: ptr(at + 0x8000),
                    }
                })
                .collect(),
            MixerFacts::Attention(_) => Vec::new(),
        })
        .collect()
}

pub(super) fn run(f: &Fixture, gdn: &[Vec<GdnState>], max_blocks: u32) -> Vec<MockLaunch> {
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

/// 2026-09-28: Every buffer a launch of `f` reads is a bound weight, a row of a fixed buffer,
/// the step's metadata, one of `gdn`'s states, or in the workspace.
pub(super) fn assert_pointers_known(f: &Fixture, gdn: &[Vec<GdnState>]) {
    let mut known: BTreeSet<u64> = BTreeSet::new();
    for l in &f.layers {
        for w in l.weights.values() {
            match w {
                BoundWeight::Dense(d) => known.insert(d.weight.0),
                BoundWeight::Nvfp4(q) => known.insert(q.weight.0) && known.insert(q.weight_scale.0),
                BoundWeight::Mmq(p) => known.insert(p.0),
            };
        }
    }
    known.extend([f.head.final_norm.weight.0, 0x9100_0000]);
    for m in [f.fixed.meta, f.fixed.batch_meta] {
        known.extend([m.positions.0, m.slot.0, m.seq_len.0, m.block_table.0]);
    }
    known.insert(f.fixed.ffn_act_q8.0);
    let row = |dim: &str| f.circuit.dims[dim] * 2;
    let rows_of = [
        (f.fixed.hidden.0, row("hidden")),
        (f.fixed.residual.0, row("hidden")),
        (f.fixed.logits.0, row("vocab")),
    ];
    let in_fixed = |p: u64| {
        rows_of
            .iter()
            .any(|&(base, w)| (0..f.plan.rows).any(|r| p == base + r * w))
    };
    known.extend(f.fixed.k_pools.iter().chain(&f.fixed.v_pools).map(|p| p.0));
    known.extend(gdn.iter().flatten().flat_map(|s| [s.h.0, s.conv.0]));
    for (i, l) in run(f, gdn, 9).iter().enumerate() {
        for a in &l.args {
            if let MockArg::Buffer(p) = a {
                let in_ws = (WORKSPACE..WORKSPACE + f.arena).contains(&p.0);
                assert!(
                    in_ws || known.contains(&p.0) || in_fixed(p.0) || p.0 == 0,
                    "launch {i} ({}) reads {:#x}, which is neither bound nor placed",
                    f.program.launches[i].kernel,
                    p.0
                );
            }
        }
    }
}
