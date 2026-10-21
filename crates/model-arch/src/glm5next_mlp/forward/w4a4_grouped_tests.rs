// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The launch plan of the grouped W4A4 routed experts on a recording mock backend:
//! the order of the sort, tile pass, quantizations, grouped GEMMs and SwiGLU, each with its
//! tile entry, grid, table, gather map, N, K and static scale. The bytes are the GPU bench's job
//! (`examples/glm5next_moe_prefill_w4a4_bench.rs`, against the slot GEMV).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::Glm5NextMlpWorkspace;
use super::super::moe_experts::MoeSite;
use super::super::w4a4::w4a4_experts;
use super::super::w4a4_grouped_raster::RasterGroups;
use super::{takes_grouped, tile_index, w4a4_grouped_experts_with, zeroes_expert_out};
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup, MlpKernel};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextMoePtrTables, Glm5NextMoeWeights,
    W4a4ActScales,
};
use crate::glm5next_mlp::{
    Glm5NextMlpConfig, Glm5NextMlpKernels, W4a4GroupedKernels, W4a4MoeK64Kernels,
};

const QUANT: u64 = 0x51;
const QUANT_K64: u64 = 0x53;
const SWIGLU: u64 = 0x5E;
const SORT: u64 = 0x61;
const TILES: u64 = 0x63;
const ONE_GROUP: u64 = 0x64;
const G32: u64 = 0x65;
const G64: u64 = 0x67;
const G32_K64: u64 = 0x69;
const G64_K64: u64 = 0x6B;
const G128: u64 = 0x6D;
const G128_K64: u64 = 0x6F;
const GU32: u64 = 0x71;
const GU64: u64 = 0x73;
const GU128: u64 = 0x75;
const R128: u64 = 0x77;
const R128_K64: u64 = 0x79;
const RGU128: u64 = 0x7B;
const N128: u64 = 0x7D;
const N128_K64: u64 = 0x7F;

const SCALES: W4a4ActScales = W4a4ActScales {
    gate_up: 0.000642,
    down: 0.000115,
};
const X: DevicePtr = DevicePtr(0x9000_0000);

/// 2026-10-10: Rank 0 of GLM-5.3's tp expert layout at TP=3: every expert, 704 columns each.
fn cfg_tp() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 704,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: 288,
        local_experts: 288,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 1,
        expert_shard: crate::glm5next_mlp::ExpertShard::Sliced(
            crate::glm5next_mlp::expert_tp::expert_slice(2048, 3, 0).unwrap(),
        ),
    }
}

/// 2026-10-10: Eight whole 2048-wide experts, so 48 rows reach the 64-slot tile cheaply.
fn cfg_small() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        moe_intermediate: 2048,
        num_experts: 8,
        local_experts: 8,
        expert_shard: crate::glm5next_mlp::ExpertShard::Whole,
        ..cfg_tp()
    }
}

fn kernels(gpu: &MockGpuBackend) -> Glm5NextMlpKernels {
    let mut k = Glm5NextMlpKernels::resolve(gpu, crate::glm5next_layer::GlmLevers::declared())
        .expect("mock resolves every kernel");
    k.w4a4_quant_static = KernelHandle(QUANT);
    k.swiglu = KernelHandle(SWIGLU);
    k.moe_sort_by_expert = KernelHandle(SORT);
    k.w4a4_moe_k64 = W4a4MoeK64Kernels {
        quant: KernelHandle(QUANT_K64),
        ..k.w4a4_moe_k64
    };
    k.w4a4_grouped = W4a4GroupedKernels {
        tiles: KernelHandle(TILES),
        one_group: KernelHandle(ONE_GROUP),
        mx: [KernelHandle(G32), KernelHandle(G64), KernelHandle(G128)],
        mx_k64: [
            KernelHandle(G32_K64),
            KernelHandle(G64_K64),
            KernelHandle(G128_K64),
        ],
        gu: [KernelHandle(GU32), KernelHandle(GU64), KernelHandle(GU128)],
        raster: [
            KernelHandle(R128),
            KernelHandle(R128_K64),
            KernelHandle(RGU128),
        ],
        nc: [KernelHandle(N128), KernelHandle(N128_K64)],
    };
    k
}

fn table(tag: u64) -> Glm5NextExpertPtrTable {
    Glm5NextExpertPtrTable {
        packed_ptrs: DevicePtr(tag),
        scale_ptrs: DevicePtr(tag + 1),
        scale2_vals: DevicePtr(tag + 2),
        scale_bases: DevicePtr::NULL,
    }
}

fn moe_weights(cap: usize) -> Glm5NextMoeWeights {
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
            cap,
            true,
        )
        .unwrap(),
        act_scales: Some(SCALES),
    }
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

/// 2026-10-10: `w4a4_experts` over `rows` rows on a `grouped` workspace of `rows` rows.
fn run(
    c: Glm5NextMlpConfig,
    rows: usize,
    grouped: bool,
) -> (anyhow::Result<()>, Vec<MockLaunch>, Glm5NextMlpWorkspace) {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, rows, grouped, 0).unwrap();
    let w = moe_weights(usize::MAX);
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
    let r = w4a4_experts(&site, SCALES);
    (r, gpu.launches_snapshot(), ws)
}

/// 2026-10-10: One grouped GEMM launch: tables, output, gather map, the tile offsets, N, K and
/// the grid (128-column tiles, 64 for the 128-slot entry; slots / MT + experts).
#[allow(clippy::too_many_arguments)]
fn assert_grouped(
    l: &MockLaunch,
    ws: &Glm5NextMlpWorkspace,
    t: u64,
    c: DevicePtr,
    gather: DevicePtr,
    nk: (u32, u32),
    tiles: (u32, u32, u32),
) {
    let act = [ws.w4a4_aq, ws.w4a4_as, ws.w4a4_ag];
    assert_grouped_act(l, ws, act, t, c, gather, nk, tiles);
}

/// 2026-10-14: [`assert_grouped`] reading its activation from `act`.
#[allow(clippy::too_many_arguments)]
fn assert_grouped_act(
    l: &MockLaunch,
    ws: &Glm5NextMlpWorkspace,
    act: [DevicePtr; 3],
    t: u64,
    c: DevicePtr,
    gather: DevicePtr,
    (n, k): (u32, u32),
    (slots, mt, experts): (u32, u32, u32),
) {
    for (i, want) in act.into_iter().enumerate() {
        assert_eq!(ptr_arg(l, i), want, "activation arg {i}");
    }
    for j in 0..3u64 {
        assert_eq!(
            ptr_arg(l, 3 + j as usize),
            DevicePtr(t + j),
            "table arg {j}"
        );
    }
    assert_eq!(ptr_arg(l, 6), c);
    assert_eq!(ptr_arg(l, 7), ws.expert_offsets());
    assert_eq!(ptr_arg(l, 8), ws.w4a4_tile_off);
    assert_eq!(ptr_arg(l, 9), gather);
    assert_eq!(
        (u32_arg(l, 10), u32_arg(l, 11), u32_arg(l, 12)),
        (experts, n, k)
    );
    let cols = if mt == 128 { 64 } else { 128 };
    assert_eq!(l.grid, [n.div_ceil(cols), slots.div_ceil(mt) + experts, 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(l.stream, 5);
}

/// 2026-10-10: 32 rows of the tp layout's 704-wide slices: the sort over 256 slots, the tile
/// pass at 32 slots per tile, the rows quantized under the gate/up scale, gate and up on the
/// 32-slot entry gathering through `sorted_token_ids` (N 704, K 4096), the SwiGLU over all slots,
/// the slot products quantized by the `_k64` quantizer under the down scale, and down on the
/// 32-slot `_k64` entry over the sorted rows (N 4096, K 704) into `expert_out`.
#[test]
fn a_wide_k64_slice_runs_sort_tiles_and_three_grouped_gemms() {
    let (r, l, ws) = run(cfg_tp(), 32, true);
    r.unwrap();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(
        funcs,
        vec![SORT, TILES, QUANT, G32, G32, SWIGLU, QUANT_K64, G32_K64]
    );
    assert_eq!(
        (u32_arg(&l[0], 5), u32_arg(&l[0], 6), u32_arg(&l[0], 7)),
        (256, 288, 8)
    );
    assert_eq!(ptr_arg(&l[0], 0), ws.ids);
    assert_eq!(ptr_arg(&l[1], 1), ws.w4a4_tile_off);
    assert_eq!((u32_arg(&l[1], 2), u32_arg(&l[1], 3)), (288, 32));
    assert_eq!(
        (ptr_arg(&l[2], 0), l[2].grid[0], u32_arg(&l[2], 4)),
        (X, 32, 4096)
    );
    assert_eq!(f32_arg(&l[2], 5), SCALES.gate_up);
    let g = ws.sorted_token_ids();
    assert_grouped(
        &l[3],
        &ws,
        0x1000,
        ws.a_gate,
        g,
        (704, 4096),
        (256, 32, 288),
    );
    assert_grouped(&l[4], &ws, 0x2000, ws.a_up, g, (704, 4096), (256, 32, 288));
    assert_eq!(u32_arg(&l[5], 3), 256 * 704);
    assert_eq!(
        (ptr_arg(&l[6], 0), l[6].grid[0], u32_arg(&l[6], 4)),
        (ws.a_act, 256, 704)
    );
    assert_eq!(f32_arg(&l[6], 5), SCALES.down);
    let null = DevicePtr::NULL;
    assert_grouped(
        &l[7],
        &ws,
        0x3000,
        ws.expert_out,
        null,
        (4096, 704),
        (256, 32, 288),
    );
}

/// 2026-10-10: At 48 routed slots per expert on average the 128-slot entries run (plain down at
/// K 2048: the plain quantizer and entry); at 47 the 32-slot ones.
/// 2026-10-14: The m128 down is its raster twin under the lever's default (`0,4`).
#[test]
fn the_tile_follows_the_average_slots_per_expert() {
    for (rows, mt, gu, dn) in [(48usize, 128u32, G128, R128), (47, 32, G32, G32)] {
        let (r, l, ws) = run(cfg_small(), rows, true);
        r.unwrap();
        let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
        assert_eq!(
            funcs,
            vec![SORT, TILES, QUANT, gu, gu, SWIGLU, QUANT, dn],
            "{rows}"
        );
        assert_eq!(u32_arg(&l[1], 3), mt, "{rows}");
        let slots = (rows * 8) as u32;
        let null = DevicePtr::NULL;
        assert_grouped(
            &l[7],
            &ws,
            0x3000,
            ws.expert_out,
            null,
            (4096, 2048),
            (slots, mt, 8),
        );
    }
}

/// 2026-10-10: A workspace sized without the lever holds 16 routed rows of quantized slots, so
/// a wide launch is refused before anything runs, never written past the scratch.
#[test]
fn a_workspace_sized_without_the_lever_is_refused_before_launching() {
    let (r, l, _ws) = run(cfg_tp(), 32, false);
    let e = r.unwrap_err().to_string();
    assert!(e.contains("METRALE_GLM_MOE_W4A4_GROUPED"), "{e}");
    assert!(l.is_empty());
}

/// 2026-10-10: Without a `_k64` grouped entry a 704-wide slice is refused before launching.
#[test]
fn a_missing_entry_is_refused_before_launching() {
    let gpu = MockGpuBackend::new();
    let mut k = kernels(&gpu);
    k.w4a4_grouped.mx_k64[0] = KernelHandle(0);
    let c = cfg_tp();
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, 32, true, 0).unwrap();
    let w = moe_weights(usize::MAX);
    let site = MoeSite {
        gpu: &gpu,
        k: &k,
        cfg: &c,
        w: &w,
        x: X,
        rows: 32,
        ws: &ws,
        stream: 0,
    };
    assert!(w4a4_experts(&site, SCALES).is_err());
    assert!(gpu.launches_snapshot().is_empty());
}

/// 2026-10-10: 16 rows stay on the sweeps; 17 take the grouped GEMM.
#[test]
fn the_grouped_gemm_starts_past_the_slot_gemv_rows() {
    assert!(!takes_grouped(16));
    assert!(takes_grouped(17));
    let (r, l, _ws) = run(cfg_tp(), 16, true);
    r.unwrap();
    assert!(
        l.iter()
            .all(|l| ![SORT, TILES, G32, G32_K64].contains(&l.func))
    );
}

/// 2026-10-10: The loader's cap: unbounded only when every grouped entry, the sort and the
/// indexed combine resolved; the plan then runs W4A4 at every row count.
#[test]
fn the_grouped_row_cap_needs_every_kernel() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    assert_eq!(k.w4a4_expert_rows_grouped(), usize::MAX);
    let p = moe_weights(k.w4a4_expert_rows_grouped()).precision;
    assert!(
        [1usize, 16, 17, 2048, 8192]
            .iter()
            .all(|&r| p.kernel(r) == MlpKernel::W4a4Static)
    );
    let mut k2 = k;
    k2.w4a4_grouped.tiles = KernelHandle(0);
    assert_eq!(k2.w4a4_expert_rows_grouped(), k.w4a4_expert_rows());
    let mut k3 = k;
    k3.combine_indexed = KernelHandle(0);
    assert_eq!(k3.w4a4_expert_rows_grouped(), k.w4a4_expert_rows());
}

/// 2026-10-10: `expert_out` is zeroed first unless the grouped W4A4 GEMM runs on a rank that
/// holds every expert.
#[test]
fn expert_out_is_zeroed_unless_every_slot_is_written() {
    let tp = cfg_tp();
    let ep = Glm5NextMlpConfig {
        local_experts: 96,
        ..tp
    };
    assert!(!zeroes_expert_out(true, &tp));
    assert!(zeroes_expert_out(false, &tp));
    assert!(zeroes_expert_out(true, &ep));
}

/// 2026-10-10: The tile from the routed slots per expert (the lever's parse moved to
/// `levers_tests.rs` on 2026-10-17).
#[test]
fn the_tile_follows_the_slots_per_expert() {
    assert_eq!(tile_index(16384, 288), 2);
    assert_eq!(tile_index(8192, 288), 0);
}

/// 2026-10-14: 1728 rows x top 8 over 288 experts = 48 slots per expert, the first m128 launch.
const M128_ROWS: usize = 1728;

/// 2026-10-14: The routed experts' launches over `rows` rows on a fresh mock (so two runs see
/// the same buffer addresses), the fused and raster levers as given. 2026-10-15: and the nc lever.
fn plan(
    c: Glm5NextMlpConfig,
    rows: usize,
    fused: bool,
    raster: Option<RasterGroups>,
    nc: Option<u32>,
) -> Vec<MockLaunch> {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, rows, true, 0).unwrap();
    let w = moe_weights(usize::MAX);
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
    w4a4_grouped_experts_with(&site, SCALES, fused, raster, nc).unwrap();
    gpu.launches_snapshot()
}

/// 2026-10-14: Two launch lists equal field for field (`MockLaunch` has no `PartialEq`).
fn assert_same(got: &[MockLaunch], want: &[MockLaunch]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert_eq!(
            (g.func, g.grid, g.block, g.shared_mem, g.stream, &g.args),
            (w.func, w.grid, w.block, w.shared_mem, w.stream, &w.args)
        );
    }
}

fn funcs(l: &[MockLaunch]) -> Vec<u64> {
    l.iter().map(|l| l.func).collect()
}

#[path = "w4a4_grouped_gu_tests.rs"]
mod gu;
#[path = "w4a4_grouped_nc_tests.rs"]
mod nc;
#[path = "w4a4_grouped_raster_tests.rs"]
mod raster;
#[path = "w4a4_grouped_scale_dict_tests.rs"]
mod scale_dict;
