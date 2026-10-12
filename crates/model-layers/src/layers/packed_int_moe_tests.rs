// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: `PackedIntMoeLayer` on the mock backend: the launch sequence, grids and
//! arguments of one forward, the pointer tables it uploads, the INT4/INT8 kernel choice, and
//! the refusals (missing kernels, null or missing experts, rows the arena cannot hold).
//!
//! Owner: model-layers (packed-int MoE).
//! Invariants: none beyond the types.

use super::{PackedIntExpert, PackedIntMoeLayer, PackedIntMoeWeights, PackedIntTensor};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::DenseWeight;
use metrale_config::ModelConfig;
use metrale_config::precision_plan::packed_int::PackedIntScheme;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

const H: usize = 256;
const INTER: usize = 128;
const E: usize = 4;
const TOP_K: usize = 2;
/// 2026-10-07: Rows the test arena holds (`BufferArena::new` max_batch_tokens).
const ARENA_ROWS: usize = 8;

const GEMM: u64 = 0xD0;
const TOPK: u64 = 0x70;
const SILU: u64 = 0x51;
const GROUPED4: u64 = 0x41;
const GROUPED8: u64 = 0x81;
const COMBINE: u64 = 0xC0;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = H;
    c.moe_intermediate_size = INTER;
    c.shared_expert_intermediate_size = INTER;
    c.num_experts = E;
    c.num_experts_per_tok = TOP_K;
    c.norm_topk_prob = true;
    c.routed_scaling_factor = 2.5;
    c.vocab_size = 128;
    c
}

fn mock() -> MockGpuBackend {
    let gpu = MockGpuBackend::new();
    for (module, func, h) in [
        ("gemm", "dense_gemm_bf16", GEMM),
        ("moe_topk_sig", "moe_topk_sigmoid_batched", TOPK),
        ("moe_silu_mul", "moe_silu_mul", SILU),
        (
            "packed_int_gemv",
            "moe_packed_int4_gemv_ptrtable_g128",
            GROUPED4,
        ),
        (
            "packed_int_gemv",
            "moe_packed_int8_gemv_ptrtable_g128",
            GROUPED8,
        ),
        ("packed_int_gemv", "moe_packed_int_combine", COMBINE),
    ] {
        gpu.set_kernel_handle(module, func, KernelHandle(h));
    }
    gpu
}

fn tensor(gpu: &dyn GpuBackend) -> PackedIntTensor {
    PackedIntTensor {
        words: gpu.alloc(64).unwrap(),
        scales: gpu.alloc(64).unwrap(),
    }
}

fn weights(gpu: &dyn GpuBackend, scheme: PackedIntScheme, experts: usize) -> PackedIntMoeWeights {
    let dense = |gpu: &dyn GpuBackend| DenseWeight {
        weight: gpu.alloc(64).unwrap(),
    };
    PackedIntMoeWeights {
        scheme,
        gate: dense(gpu),
        correction_bias: dense(gpu),
        shared_gate: dense(gpu),
        shared_up: dense(gpu),
        shared_down: dense(gpu),
        experts: (0..experts)
            .map(|_| PackedIntExpert {
                gate_proj: tensor(gpu),
                up_proj: tensor(gpu),
                down_proj: tensor(gpu),
            })
            .collect(),
    }
}

struct Fixture {
    gpu: MockGpuBackend,
    config: ModelConfig,
    buffers: BufferArena,
    dispatch: GemmDispatch,
    derived: DerivedWeights,
    levers: ModelLevers,
    stats: ModelStats,
}

impl Fixture {
    fn new() -> Self {
        let gpu = mock();
        let config = config();
        let buffers = BufferArena::new(&config, ARENA_ROWS, 256, 16, 8, &gpu).unwrap();
        Self {
            gpu,
            config,
            buffers,
            dispatch: GemmDispatch::defaults(),
            derived: DerivedWeights::new(),
            levers: ModelLevers::defaults(),
            stats: ModelStats::new(),
        }
    }

    fn ctx(&self) -> ForwardContext<'_> {
        ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: &self.gpu,
            config: &self.config,
            dispatch: &self.dispatch,
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    }
}

fn i32a(v: i32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn u32a(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

fn read_u64s(gpu: &MockGpuBackend, table: DevicePtr) -> Vec<u64> {
    gpu.read_alloc(table)
        .unwrap()
        .chunks(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn forward_launches_router_shared_routed_and_combine_in_order() {
    let f = Fixture::new();
    let w = weights(&f.gpu, PackedIntScheme::INT4_G128, E);
    let (bias, gate_w) = (w.correction_bias.weight, w.gate.weight);
    let layer = PackedIntMoeLayer::new(w, &f.config, &f.gpu).unwrap();
    let ctx = f.ctx();
    let input = f.buffers.norm_output();
    let start = f.gpu.launches_snapshot().len();
    let out = layer.forward_rows(input, 3, &ctx, 7).unwrap();
    assert_eq!(out, f.buffers.moe_output());
    let launches = &f.gpu.launches_snapshot()[start..];
    let funcs: Vec<u64> = launches.iter().map(|l| l.func).collect();
    assert_eq!(
        funcs,
        [
            GEMM, TOPK, GEMM, GEMM, SILU, GEMM, GROUPED4, GROUPED4, SILU, GROUPED4, COMBINE
        ]
    );
    assert!(launches.iter().all(|l| l.stream == 7));

    let b = &f.buffers;
    let ids = b.scratch();
    let topk_w = ids.offset(3 * TOP_K * 4);
    // 2026-10-07: Router GEMM [3, H] x [E, H]^T into gate_logits.
    assert_eq!(launches[0].args[1], MockArg::Buffer(gate_w));
    assert_eq!(launches[0].args[2], MockArg::Buffer(b.gate_logits()));
    // 2026-10-07: Top-k: bias, ids, weights, E, k, normalize, scaling 2.5, one block per row.
    assert_eq!(launches[1].grid, [3, 1, 1]);
    assert_eq!(
        launches[1].args[1..],
        [
            MockArg::Buffer(bias),
            MockArg::Buffer(ids),
            MockArg::Buffer(topk_w),
            u32a(E as u32),
            u32a(TOP_K as u32),
            u32a(1),
            MockArg::Bytes(2.5f32.to_le_bytes().to_vec()),
        ]
    );
    // 2026-10-07: Shared down writes attn_output, which the combine reads.
    assert_eq!(launches[5].args[2], MockArg::Buffer(b.attn_output()));

    let grouped = |l: &metrale_gpu_runtime::gpu::mock::MockLaunch,
                   x: DevicePtr,
                   words: DevicePtr,
                   scales: DevicePtr,
                   y: DevicePtr,
                   div: i32,
                   n: i32,
                   k: i32| {
        assert_eq!(l.grid, [(n as u32).div_ceil(8), (3 * TOP_K) as u32, 1]);
        assert_eq!(l.block, [256, 1, 1]);
        assert_eq!(
            l.args,
            [
                MockArg::Buffer(x),
                MockArg::Buffer(words),
                MockArg::Buffer(scales),
                MockArg::Buffer(ids),
                MockArg::Buffer(y),
                i32a(E as i32),
                i32a(div),
                i32a(n),
                i32a(k),
            ]
        );
    };
    let (g, u, d) = (layer.gate_table, layer.up_table, layer.down_table);
    let (h, i) = (H as i32, INTER as i32);
    let k = TOP_K as i32;
    grouped(
        &launches[6],
        input,
        g.words,
        g.scales,
        b.expert_gate_out(),
        k,
        i,
        h,
    );
    grouped(
        &launches[7],
        input,
        u.words,
        u.scales,
        b.expert_up_out(),
        k,
        i,
        h,
    );
    // 2026-10-07: SiLU over every routed slot.
    assert_eq!(launches[8].args[3], u32a((3 * TOP_K * INTER) as u32));
    let gate_out = b.expert_gate_out();
    grouped(
        &launches[9],
        gate_out,
        d.words,
        d.scales,
        b.expert_down_out(),
        1,
        h,
        i,
    );
    assert_eq!(launches[10].grid, [3, 1, 1]);
    assert_eq!(
        launches[10].args,
        [
            MockArg::Buffer(b.expert_down_out()),
            MockArg::Buffer(topk_w),
            MockArg::Buffer(b.attn_output()),
            MockArg::Buffer(b.moe_output()),
            i32a(h),
            i32a(k),
        ]
    );
}

#[test]
fn pointer_tables_hold_each_expert_in_id_order() {
    let f = Fixture::new();
    let w = weights(&f.gpu, PackedIntScheme::INT4_G128, E);
    let want = |pick: fn(&PackedIntExpert) -> PackedIntTensor| {
        let words: Vec<u64> = w.experts.iter().map(|e| pick(e).words.0).collect();
        let scales: Vec<u64> = w.experts.iter().map(|e| pick(e).scales.0).collect();
        (words, scales)
    };
    let gate = want(|e| e.gate_proj);
    let up = want(|e| e.up_proj);
    let down = want(|e| e.down_proj);
    let layer = PackedIntMoeLayer::new(w, &f.config, &f.gpu).unwrap();
    for (table, (words, scales)) in [
        (layer.gate_table, gate),
        (layer.up_table, up),
        (layer.down_table, down),
    ] {
        assert_eq!(read_u64s(&f.gpu, table.words), words);
        assert_eq!(read_u64s(&f.gpu, table.scales), scales);
    }
}

#[test]
fn int8_scheme_launches_the_int8_grouped_kernel() {
    let f = Fixture::new();
    let layer = PackedIntMoeLayer::new(
        weights(&f.gpu, PackedIntScheme::INT8_G128, E),
        &f.config,
        &f.gpu,
    )
    .unwrap();
    assert_eq!(layer.scheme(), PackedIntScheme::INT8_G128);
    let start = f.gpu.launches_snapshot().len();
    layer
        .forward_rows(f.buffers.norm_output(), 1, &f.ctx(), 0)
        .unwrap();
    let grouped: Vec<u64> = f.gpu.launches_snapshot()[start..]
        .iter()
        .map(|l| l.func)
        .filter(|&h| h == GROUPED4 || h == GROUPED8)
        .collect();
    assert_eq!(grouped, [GROUPED8; 3]);
}

#[test]
fn ffn_component_runs_every_row_count_through_forward_rows() {
    let f = Fixture::new();
    let layer = PackedIntMoeLayer::new(
        weights(&f.gpu, PackedIntScheme::INT4_G128, E),
        &f.config,
        &f.gpu,
    )
    .unwrap();
    let ffn = FfnComponent::PackedIntMoe(layer);
    let ctx = f.ctx();
    let input = f.buffers.norm_output();
    assert_eq!(ffn.forward(input, &ctx, 0).unwrap(), f.buffers.moe_output());
    ffn.forward_prefill(input, 5, &ctx, 0).unwrap();
    let launches = f.gpu.launches_snapshot();
    let combines: Vec<[u32; 3]> = launches
        .iter()
        .filter(|l| l.func == COMBINE)
        .map(|l| l.grid)
        .collect();
    assert_eq!(combines, [[1, 1, 1], [5, 1, 1]]);
    assert!(!ffn.moe_grouped_decode_ok());
    assert!(!ffn.fp32_routing_active(&f.levers));
}

#[test]
fn missing_kernels_fail_construction() {
    for (module, func) in [
        ("packed_int_gemv", "moe_packed_int4_gemv_ptrtable_g128"),
        ("packed_int_gemv", "packed_int4_gemv_g128"),
        ("packed_int_gemv", "moe_packed_int_combine"),
        ("moe_topk_sig", "moe_topk_sigmoid_batched"),
    ] {
        let f = Fixture::new();
        f.gpu.deny_kernel(module, func);
        let w = weights(&f.gpu, PackedIntScheme::INT4_G128, E);
        let err = PackedIntMoeLayer::new(w, &f.config, &f.gpu).err();
        assert!(
            err.is_some(),
            "{module}::{func} denied but construction succeeded"
        );
    }
}

#[test]
fn null_or_missing_experts_are_refused() {
    let f = Fixture::new();
    let w = weights(&f.gpu, PackedIntScheme::INT4_G128, E - 1);
    let err = PackedIntMoeLayer::new(w, &f.config, &f.gpu).err().unwrap();
    assert!(err.to_string().contains("3 experts loaded"), "{err}");

    let mut w = weights(&f.gpu, PackedIntScheme::INT4_G128, E);
    w.experts[2].down_proj.scales = DevicePtr::NULL;
    let err = PackedIntMoeLayer::new(w, &f.config, &f.gpu).err().unwrap();
    assert!(err.to_string().contains("expert 2 down_proj"), "{err}");

    let mut w = weights(&f.gpu, PackedIntScheme::INT4_G128, E);
    w.correction_bias.weight = DevicePtr::NULL;
    let err = PackedIntMoeLayer::new(w, &f.config, &f.gpu).err().unwrap();
    assert!(err.to_string().contains("correction bias"), "{err}");
}

#[test]
fn rows_beyond_the_arena_launch_nothing() {
    let f = Fixture::new();
    let layer = PackedIntMoeLayer::new(
        weights(&f.gpu, PackedIntScheme::INT4_G128, E),
        &f.config,
        &f.gpu,
    )
    .unwrap();
    let ctx = f.ctx();
    let input = f.buffers.norm_output();
    let start = f.gpu.launches_snapshot().len();
    assert!(layer.forward_rows(input, 0, &ctx, 0).is_err());
    let err = layer.forward_rows(input, 4096, &ctx, 0).unwrap_err();
    assert!(err.to_string().contains("4096 rows need"), "{err}");
    assert_eq!(f.gpu.launches_snapshot().len(), start);
    layer.forward_rows(input, ARENA_ROWS, &ctx, 0).unwrap();
}
