// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: The launch plan of the GLM W4A4 MLP on a recording mock backend: which kernel
//! runs in which order, over how many rows, with which K, N, slot count, activation row divisor
//! and static scale. The numerics are the GPU tests' job (`tests/glm5next_w4a4_cuda.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::Glm5NextMlpWorkspace;
use super::super::moe_experts::MoeSite;
use super::{forward_dense_w4a4, w4a4_experts};
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextDenseNvfp4Weights, Glm5NextExpertPtrTable,
    Glm5NextMoePtrTables, Glm5NextMoeWeights, Nvfp4Proj, W4a4ActScales,
};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const QUANT: u64 = 0x51;
const SLOTS: u64 = 0x52;
const MX8: u64 = 0x58;
const MX16: u64 = 0x5A;
const MX32: u64 = 0x5C;
const SWIGLU: u64 = 0x5E;
const SWEEP8: u64 = 0x60;
const SWEEP16: u64 = 0x62;
const ROW_UNION: u64 = 0x64;
const OWN_SWEEP: u64 = 0x66;

/// 2026-10-08: GLM-5.3-Flash at TP=3, EP=3, rank 0.
fn cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: 288,
        local_experts: 96,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
        expert_shard: crate::glm5next_mlp::ExpertShard::Whole,
    }
}

fn kernels(gpu: &MockGpuBackend) -> Glm5NextMlpKernels {
    let mut k = Glm5NextMlpKernels::resolve(gpu).expect("mock resolves every kernel");
    k.w4a4_quant_static = KernelHandle(QUANT);
    k.w4a4_moe_slots = KernelHandle(SLOTS);
    k.w4a4_mx = [KernelHandle(MX8), KernelHandle(MX16), KernelHandle(MX32)];
    k.swiglu = KernelHandle(SWIGLU);
    k.w4a4_moe_sweep = [KernelHandle(SWEEP8), KernelHandle(SWEEP16)];
    k.moe_row_union = KernelHandle(ROW_UNION);
    k.w4a4_moe_slots_sweep = KernelHandle(OWN_SWEEP);
    k
}

fn u32_arg(l: &MockLaunch, i: usize) -> u32 {
    match &l.args[i] {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("arg {i} is {other:?}, not 4 bytes"),
    }
}

fn f32_arg(l: &MockLaunch, i: usize) -> f32 {
    f32::from_bits(u32_arg(l, i))
}

fn ptr_arg(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn proj(tag: u64) -> Nvfp4Proj {
    Nvfp4Proj {
        packed: DevicePtr(tag),
        scale: DevicePtr(tag + 1),
        scale_2: 0.5,
        input_scale: None,
    }
}

const SCALES: W4a4ActScales = W4a4ActScales {
    gate_up: 0.000642,
    down: 0.000115,
};

/// 2026-10-08: 40 dense rows run as a 32-row and an 8-row chunk on mx32 and mx8: per chunk one
/// quantization under the gate/up scale feeding both gate and up, one SwiGLU over all rows,
/// then per chunk a quantization of the SwiGLU product under the down scale and the down GEMV.
#[test]
fn dense_rows_run_in_chunks_with_the_right_scales_and_shapes() {
    let gpu = MockGpuBackend::new();
    let (c, k) = (cfg(), kernels(&gpu));
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 64).unwrap();
    let w = Glm5NextDenseNvfp4Weights {
        gate_proj: proj(0x1000),
        up_proj: proj(0x2000),
        down_proj: proj(0x3000),
    };
    let (x, out) = (DevicePtr(0x9000_0000), DevicePtr(0xA000_0000));
    forward_dense_w4a4(&gpu, &k, &c, &w, SCALES, 4096, x, out, 40, &ws, 7).unwrap();
    let l = gpu.launches_snapshot();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(
        funcs,
        vec![
            QUANT, MX32, MX32, QUANT, MX8, MX8, SWIGLU, QUANT, MX32, QUANT, MX8
        ]
    );
    // 2026-10-08: Quantizations: rows (grid.x), K, static scale; the second chunk reads row 32.
    for (i, rows, kk, gs, input) in [
        (0, 32, 4096, SCALES.gate_up, x),
        (3, 8, 4096, SCALES.gate_up, x.offset(32 * 4096 * 2)),
        (7, 32, 4096, SCALES.down, ws.a_act),
        (9, 8, 4096, SCALES.down, ws.a_act.offset(32 * 4096 * 2)),
    ] {
        assert_eq!(l[i].grid[0], rows, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 0), input, "launch {i}");
        assert_eq!(u32_arg(&l[i], 4), kk, "launch {i}");
        assert_eq!(f32_arg(&l[i], 5), gs, "launch {i}");
    }
    // 2026-10-08: GEMVs: weight, scale_2, output row offset, M, N, K, grid ceil(N / 16).
    for (i, w_tag, dst, m) in [
        (1, 0x1000, ws.a_gate, 32),
        (2, 0x2000, ws.a_up, 32),
        (4, 0x1000, ws.a_gate.offset(32 * 4096 * 2), 8),
        (8, 0x3000, out, 32),
        (10, 0x3000, out.offset(32 * 4096 * 2), 8),
    ] {
        assert_eq!(ptr_arg(&l[i], 3), DevicePtr(w_tag), "launch {i}");
        assert_eq!(f32_arg(&l[i], 5), 0.5, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 6), dst, "launch {i}");
        assert_eq!(
            [u32_arg(&l[i], 7), u32_arg(&l[i], 8), u32_arg(&l[i], 9)],
            [m, 4096, 4096]
        );
        assert_eq!(l[i].grid, [256, 1, 1], "launch {i}");
    }
    assert!(l.iter().all(|l| l.stream == 7));
}

fn table(tag: u64) -> Glm5NextExpertPtrTable {
    Glm5NextExpertPtrTable {
        packed_ptrs: DevicePtr(tag),
        scale_ptrs: DevicePtr(tag + 1),
        scale2_vals: DevicePtr(tag + 2),
    }
}

fn moe_weights() -> Glm5NextMoeWeights {
    let null = DevicePtr::NULL;
    Glm5NextMoeWeights {
        router: null,
        router_bias: null,
        shared: Glm5NextDenseMlpWeights {
            gate_proj: null,
            up_proj: null,
            down_proj: null,
        },
        experts: Vec::new(),
        ptrs: Glm5NextMoePtrTables {
            gate: table(0x1000),
            up: table(0x2000),
            down: table(0x3000),
        },
        precision: GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            16,
            true,
        )
        .unwrap(),
        act_scales: Some(SCALES),
    }
}

fn run_experts(rows: usize) -> (MockGpuBackend, Vec<MockLaunch>, Glm5NextMlpWorkspace) {
    run_experts_with(rows, |_| {})
}

fn run_experts_with(
    rows: usize,
    edit: impl FnOnce(&mut Glm5NextMlpKernels),
) -> (MockGpuBackend, Vec<MockLaunch>, Glm5NextMlpWorkspace) {
    let gpu = MockGpuBackend::new();
    let c = cfg();
    let mut k = kernels(&gpu);
    edit(&mut k);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 16).unwrap();
    let w = moe_weights();
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: &c,
        w: &w,
        x: X,
        rows,
        ws: &ws,
        stream: 5,
    };
    w4a4_experts(&site, SCALES).unwrap();
    let l = gpu.launches_snapshot();
    (gpu, l, ws)
}

const X: DevicePtr = DevicePtr(0x9000_0000);

/// 2026-10-09: The two sweep launches of a routed W4A4 run: gate and up in one launch (two tables,
/// two outputs), down in another (its table in both slots), each reading entry list `u_eid`
/// and `u_slot`, on one CTA per SM (the mock reports GB10's 48 SMs).
fn assert_sweep_launches(
    l: &[MockLaunch],
    at: [usize; 2],
    rows: usize,
    u_eid: DevicePtr,
    ws: &Glm5NextMlpWorkspace,
) {
    // 2026-10-09: (launch, [table, output] per slot, nproj, N, K, act_div).
    for (i, projs, nproj, n, kk, div) in [
        (
            at[0],
            [(0x1000, ws.a_gate), (0x2000, ws.a_up)],
            2,
            2048,
            4096,
            8,
        ),
        (
            at[1],
            [(0x3000, ws.expert_out), (0x3000, ws.expert_out)],
            1,
            4096,
            2048,
            1,
        ),
    ] {
        assert_eq!(ptr_arg(&l[i], 0), ws.w4a4_aq, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 3), u_eid, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 4), ws.u_slot, "launch {i}");
        for (j, (t, dst)) in projs.into_iter().enumerate() {
            let a = 5 + 4 * j;
            assert_eq!(ptr_arg(&l[i], a), DevicePtr(t), "launch {i} table {j}");
            assert_eq!(
                ptr_arg(&l[i], a + 1),
                DevicePtr(t + 1),
                "launch {i} table {j}"
            );
            assert_eq!(
                ptr_arg(&l[i], a + 2),
                DevicePtr(t + 2),
                "launch {i} table {j}"
            );
            assert_eq!(ptr_arg(&l[i], a + 3), dst, "launch {i} output {j}");
        }
        let args: Vec<u32> = (13..20).map(|a| u32_arg(&l[i], a)).collect();
        assert_eq!(
            args,
            vec![nproj, n, kk, rows as u32, 8, div, 288],
            "launch {i}"
        );
        assert_eq!(l[i].grid, [48, 1, 1], "launch {i}");
        assert_eq!(l[i].block, [256, 1, 1], "launch {i}");
    }
}

/// 2026-10-09: One row of routed experts runs the own-slots sweep with no union build: the row
/// quantized under the gate/up scale, gate and up in one sweep whose entry list is the ids row,
/// the SwiGLU, the 8 slot products quantized under the down scale, and down in a second sweep.
#[test]
fn one_routed_row_sweeps_its_own_slots() {
    let (_gpu, l, ws) = run_experts(1);
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(funcs, vec![QUANT, OWN_SWEEP, SWIGLU, QUANT, OWN_SWEEP]);
    assert_eq!((l[0].grid[0], u32_arg(&l[0], 4)), (1, 4096));
    assert_eq!((l[3].grid[0], u32_arg(&l[3], 4)), (8, 2048));
    assert_eq!(ptr_arg(&l[3], 0), ws.a_act);
    assert_sweep_launches(&l, [1, 4], 1, ws.ids, &ws);
}

/// 2026-10-08: One row of routed experts where the own-slots sweep did not resolve: the row
/// quantized under the gate/up scale, gate and up over its 8 slots on the slot GEMV reading row
/// `slot / 8` (act_div = top_k), the SwiGLU over 8 x 2048, the 8 slot products quantized under
/// the down scale, and down over the 8 slots reading their own rows (act_div = 1) into
/// `expert_out`.
#[test]
fn one_routed_row_without_the_own_sweep_runs_the_slot_gemv() {
    let (_gpu, l, ws) = run_experts_with(1, |k| k.w4a4_moe_slots_sweep = KernelHandle(0));
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(funcs, vec![QUANT, SLOTS, SLOTS, SWIGLU, QUANT, SLOTS]);
    assert_eq!((l[0].grid[0], u32_arg(&l[0], 4)), (1, 4096));
    assert_eq!(f32_arg(&l[0], 5), SCALES.gate_up);
    assert_eq!(ptr_arg(&l[0], 0), X);
    assert_eq!((l[4].grid[0], u32_arg(&l[4], 4)), (8, 2048));
    assert_eq!(f32_arg(&l[4], 5), SCALES.down);
    assert_eq!(ptr_arg(&l[4], 0), ws.a_act);
    // 2026-10-08: Slot GEMVs: table, output, N, K, act_div, num_experts, grid (ceil(N/16), 8).
    for (i, t, dst, n, kk, div) in [
        (1, 0x1000, ws.a_gate, 2048, 4096, 8),
        (2, 0x2000, ws.a_up, 2048, 4096, 8),
        (5, 0x3000, ws.expert_out, 4096, 2048, 1),
    ] {
        assert_eq!(ptr_arg(&l[i], 3), ws.ids, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 4), DevicePtr(t), "launch {i}");
        assert_eq!(ptr_arg(&l[i], 6), DevicePtr(t + 2), "launch {i}");
        assert_eq!(ptr_arg(&l[i], 7), dst, "launch {i}");
        let args: Vec<u32> = (8..12).map(|a| u32_arg(&l[i], a)).collect();
        assert_eq!(args, vec![n, kk, div, 288], "launch {i}");
        assert_eq!(l[i].grid, [n / 16, 8, 1], "launch {i}");
    }
}

/// 2026-10-09: 3 and 12 rows build the experts' union once (one block of rows x top_k threads)
/// and run the persistent union sweep (the 8-token entry up to 8 rows, the 16-token one above)
/// over the union tables; the quantizations are those of the slot path (rows, then rows x
/// top_k slot products).
#[test]
fn several_routed_rows_sweep_each_union_expert_once() {
    for (rows, sweep) in [(3usize, SWEEP8), (12, SWEEP16)] {
        let (_gpu, l, ws) = run_experts(rows);
        let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
        assert_eq!(
            funcs,
            vec![QUANT, ROW_UNION, sweep, SWIGLU, QUANT, sweep],
            "{rows} rows"
        );
        let slots = (rows * 8) as u32;
        assert_eq!(l[0].grid[0], rows as u32);
        assert_eq!(
            (l[1].block[0], u32_arg(&l[1], 3), u32_arg(&l[1], 4)),
            (slots, rows as u32, 8)
        );
        assert_eq!(l[4].grid[0], slots);
        assert_sweep_launches(&l, [2, 5], rows, ws.u_eid, &ws);
    }
}

/// 2026-10-08: More rows than the activation scratch holds is an error before any launch.
#[test]
fn rows_past_the_scratch_are_refused_before_launching() {
    let gpu = MockGpuBackend::new();
    let (c, k) = (cfg(), kernels(&gpu));
    // 2026-10-08: A 2-row workspace: its scratch holds 2 x 8 = 16 rows, so 3 routed rows (24
    // slots) do not fit.
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 2).unwrap();
    let w = moe_weights();
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: &c,
        w: &w,
        x: DevicePtr(0x9000_0000),
        rows: 3,
        ws: &ws,
        stream: 0,
    };
    assert!(w4a4_experts(&site, SCALES).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}
