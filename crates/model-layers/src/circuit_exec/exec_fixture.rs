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
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::bindings::*;
use super::compile::{self, DraftFixed, Fixed, Inputs};
use super::kernels::KernelTable;
use super::program::Program;
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
    pub draft: Option<CircuitLayer>,
    pub circuit: Circuit,
    pub plan: FusionPlan,
    pub program: Program,
    pub arena: u64,
    pub fixed: Fixed,
    pub layers: Vec<CircuitLayer>,
    pub head: HeadBinding,
    /// 2026-09-30: The weight and scale pointers of the W8A8 bindings, which a `BoundWeight`
    /// does not expose.
    pub w8a8_ptrs: BTreeSet<u64>,
}

/// 2026-09-30: Layer `i`'s binding (its attention index `attn`) and the W8A8 pointers in it.
pub(super) type Bind = dyn Fn(&Circuit, usize, usize) -> (CircuitLayer, Vec<u64>);

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
        // 2026-10-03: The unscaled E4M3 casts the prefill projections read.
        w.insert(WeightSlot::PrefillCast(LinearRole::Qkvz), dense(tag(i, 28)));
        w.insert(
            WeightSlot::PrefillCast(LinearRole::GdnOut),
            dense(tag(i, 29)),
        );
        MixerFacts::Gdn(GdnFacts {
            qkvz_deinterleaved: true,
            h_slot_bytes: STATE_PITCH,
            conv_state_bytes: STATE_PITCH,
            carry: Some(CARRY),
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

/// 2026-09-29: A BF16 MTP draft head: weight tags at layer 200.
pub(super) fn draft_binding(circuit: &Circuit) -> CircuitLayer {
    let d = |k: &str| circuit.dims[k] as u32;
    let mut w: BTreeMap<WeightSlot, BoundWeight> = [
        WeightSlot::EmbedNorm,
        WeightSlot::HiddenNorm,
        WeightSlot::InputNorm,
        WeightSlot::PostNorm,
        WeightSlot::FinalNorm,
        WeightSlot::QNorm,
        WeightSlot::KNorm,
        WeightSlot::FfnGate,
        WeightSlot::FfnUp,
        WeightSlot::Linear(LinearRole::MtpFc),
        WeightSlot::Linear(LinearRole::Q),
        WeightSlot::Linear(LinearRole::K),
        WeightSlot::Linear(LinearRole::V),
        WeightSlot::Linear(LinearRole::O),
        WeightSlot::Linear(LinearRole::Down),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, s)| (s, dense(tag(200, 1 + i as u64))))
    .collect();
    w.insert(WeightSlot::LmHead, nvfp4(tag(200, 30)));
    CircuitLayer {
        mixer: MixerFacts::Attention(AttnFacts {
            attn_layer_idx: 0,
            kv_dtype: metrale_cache::kv_cache::KvCacheDtype::Bf16,
            num_q_heads: d("q_heads"),
            num_kv_heads: d("kv_heads"),
            head_dim: d("head_dim"),
            gated: true,
            rope: RopeFacts {
                mrope_interleaved: false,
                theta: 1.0e7,
                rotary_dim: 64,
            },
            sliding_window: 0,
            softmax_scale: 0.0625,
            paged_decode_plain_rows: u128::MAX,
        }),
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
        tokens: ptr(0xA500_0000),
        draft: Some(DraftFixed {
            embed: ptr(0xA600_0000),
            meta: crate::layers::mtp_meta::mtp_attn_meta_dev(ptr(meta + 0x3_0000), 0),
            k_pool: ptr(0xB800_0000),
            v_pool: ptr(0xB880_0000),
            block_size: 16,
            cache_stride: 4096,
            vocab: 100_000,
            rows: Some(DraftRows {
                meta: ptr(meta + 0x5_0000),
                lp_offset: 512,
                lm_head_gemv: vec![metrale_gpu_runtime::gpu::KernelHandle(0xDEAD); 129],
                lm_head_twin: false,
            }),
        }),
        verify_batch_meta: AttnMetadataDev {
            positions: ptr(meta + 0x4_0000),
            positions_h: ptr(meta + 0x4_0000),
            positions_w: ptr(meta + 0x4_0000),
            slot: ptr(meta + 0x4_0000 + 1024),
            seq_len: ptr(meta + 0x4_0000 + 3072),
            block_table: ptr(meta + 0x4_0000 + 4096),
            max_blocks_per_seq: 0,
            num_seqs: 0,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
        verify_wy_tables: ptr(0xA700_0000),
        verify_batch_tokens: ptr(0xA800_0000),
        verify_meta: AttnMetadataDev {
            positions: ptr(meta + 0x2_0000),
            positions_h: ptr(meta + 0x2_0000),
            positions_w: ptr(meta + 0x2_0000),
            slot: ptr(meta + 0x2_0000 + 256),
            seq_len: ptr(meta + 0x2_0000 + 512),
            block_table: ptr(meta + 0x2_0000 + 768),
            max_blocks_per_seq: 0,
            num_seqs: 4,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        },
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
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    build_for(
        RECIPE,
        &bind,
        fusions,
        (mode, rows, Arm::Primary),
        edit,
        edit_head,
    )
}

/// 2026-09-30: Which arm of a plan a fixture compiles.
#[derive(Debug, Clone, Copy)]
pub(super) enum Arm {
    /// 2026-09-30: The plan the policy selects.
    Primary,
    /// 2026-09-30: The arm of the runtime route with this id; an error when it does not apply.
    Route(&'static str),
    /// 2026-09-30: The plan under the instance's policy with one setting changed.
    Setting(&'static str, &'static str),
    /// 2026-09-30: The batched-verify plan of this row table (`mode` is `VerifyBatch`).
    Table(&'static str),
}

/// 2026-09-30: The carried-verify buffers every fixture GDN layer binds.
pub(super) const CARRY: crate::layer::GdnCarryBinding = crate::layer::GdnCarryBinding {
    flag: DevicePtr(0xE000_0000),
    stash: DevicePtr(0xE100_0000),
    pend: DevicePtr(0xE200_0000),
    slot_tab: DevicePtr(0xE300_0000),
    seq_floats: 4096,
    conv_stash: DevicePtr(0xE400_0000),
    conv_seq_elems: 1024,
    conv_tab: DevicePtr(0xE500_0000),
};

/// 2026-09-30: [`build_at`] for the batched-verify plan of `table`.
pub(super) fn build_table(table: &'static str) -> anyhow::Result<Fixture> {
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    let rows = metrale_circuit::RowTable::parse(table)?.rows();
    build_for(
        RECIPE,
        &bind,
        Fusions::All,
        (Mode::VerifyBatch, rows, Arm::Table(table)),
        |_| {},
        |_| {},
    )
}

/// 2026-09-30: [`build_at`] for the arm of runtime route `route`.
pub(super) fn build_route_at(
    route: &'static str,
    mode: Mode,
    rows: u64,
) -> anyhow::Result<Fixture> {
    let bind = |c: &Circuit, i: usize, attn: usize| (layer_binding(c, i, attn), Vec::new());
    build_for(
        RECIPE,
        &bind,
        Fusions::All,
        (mode, rows, Arm::Route(route)),
        |_| {},
        |_| {},
    )
}

/// 2026-09-30: [`build_at`] for the instance `recipe`, its layers bound by `bind`. The W4A4
/// kernels are prepared on the build's backend, as the model build does.
pub(super) fn build_for(
    recipe: &str,
    bind: &Bind,
    fusions: Fusions,
    (mode, rows, arm): (Mode, u64, Arm),
    edit: impl Fn(&mut Vec<CircuitLayer>),
    edit_head: impl Fn(&mut HeadBinding),
) -> anyhow::Result<Fixture> {
    let inst = sources::instance(recipe)?;
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
    let mut policy = inst.policy.clone();
    if let Arm::Setting(k, v) = arm {
        policy.settings.insert(k.into(), v.into());
    }
    let mut plan = match arm {
        Arm::Table(t) => metrale_circuit::fuse_table(
            &loaded.circuit,
            &loaded.rules,
            &avail,
            &policy,
            &metrale_circuit::RowTable::parse(t)?,
        )?,
        _ => metrale_circuit::fuse(&loaded.circuit, &loaded.rules, &avail, &policy, mode, rows)?,
    };
    if let Arm::Route(id) = arm {
        let set = (loaded.rules.as_slice(), loaded.runtime.as_slice());
        let arms = metrale_circuit::runtime::route_arms(
            &loaded.circuit,
            set,
            &avail,
            &inst.policy,
            &plan,
        )?;
        plan = arms
            .into_iter()
            .find(|(r, _)| r.id == id)
            .map(|(_, p)| p)
            .ok_or_else(|| anyhow::anyhow!("route `{id}` does not apply at {mode:?} {rows}"))?;
    }
    let layout = compile::layout(&loaded.circuit, &plan)?;
    let buffers = plan_buffers_with(&loaded.circuit, &plan, rows, &layout)?;
    let mut attn = 0;
    let mut w8a8_ptrs = BTreeSet::new();
    let mut layers: Vec<CircuitLayer> = (0..loaded.circuit.layer_kinds.len())
        .map(|i| {
            let (l, ptrs) = bind(&loaded.circuit, i, attn);
            attn += usize::from(matches!(l.mixer, MixerFacts::Attention(_)));
            w8a8_ptrs.extend(ptrs);
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
    let draft = (mode == Mode::Draft).then(|| draft_binding(&loaded.circuit));
    let fixed = fixed(attn);
    let gpu = MockGpuBackend::new();
    crate::layers::ops::w4a4_proj::prepare(&gpu)?;
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
            draft: draft.as_ref(),
            arena: None,
        },
    )?;
    Ok(Fixture {
        draft,
        circuit: loaded.circuit,
        plan,
        program,
        arena: buffers.arena_bytes,
        fixed,
        layers,
        head,
        w8a8_ptrs,
    })
}

// 2026-09-30: The run helpers live in exec_fixture_run.rs (split for the file-size cap).
pub(super) use super::exec_fixture_run::*;
