// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The launch plan of the dense MLP's one-group W4A4 launch
//! (`METRALE_GLM_DENSE_W4A4_GROUPED_MIN_ROWS`) on a recording mock backend: from the threshold
//! up, the offsets pass, one quantization per input over all rows and one 128-slot grouped GEMM
//! per projection with its table, output, N, K and grid; below it, exactly the chunk path; past
//! the quantized scratch, an error before any launch. The bytes are the GPU bench's job
//! (`examples/glm5next_moe_prefill_w4a4_bench`, part 3, against the chunk path).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::Glm5NextMlpWorkspace;
use super::super::w4a4::tests::{
    MX32, QUANT, SCALES, SWIGLU, cfg, f32_arg, kernels, proj, ptr_arg, u32_arg,
};
use super::super::w4a4_dense::forward_dense_w4a4;
use super::super::w4a4_grouped::smem_bytes_of;
use super::{DenseGroupedTables, TILE};
use crate::glm5next_mlp::Glm5NextMlpKernels;
use crate::glm5next_mlp::weights::{Glm5NextDenseNvfp4Weights, Glm5NextExpertPtrTable};

const ONE_GROUP: u64 = 0x80;
const G128: u64 = 0x82;
const X: DevicePtr = DevicePtr(0x9000_0000);
const OUT: DevicePtr = DevicePtr(0xA000_0000);
const OFFSETS: DevicePtr = DevicePtr(0xB000_0000);
const TILE_OFF: DevicePtr = DevicePtr(0xB000_1000);

fn grouped_kernels(gpu: &MockGpuBackend) -> Glm5NextMlpKernels {
    let mut k = kernels(gpu);
    k.w4a4_grouped.one_group = KernelHandle(ONE_GROUP);
    k.w4a4_grouped.mx[TILE] = KernelHandle(G128);
    k
}

fn table(tag: u64) -> Glm5NextExpertPtrTable {
    Glm5NextExpertPtrTable {
        packed_ptrs: DevicePtr(tag),
        scale_ptrs: DevicePtr(tag + 8),
        scale2_vals: DevicePtr(tag + 16),
    }
}

fn tables(min_rows: usize) -> DenseGroupedTables {
    DenseGroupedTables {
        min_rows,
        gate: table(0x4000),
        up: table(0x5000),
        down: table(0x6000),
        offsets: OFFSETS,
        tile_off: TILE_OFF,
    }
}

fn weights() -> Glm5NextDenseNvfp4Weights {
    Glm5NextDenseNvfp4Weights {
        gate_proj: proj(0x1000),
        up_proj: proj(0x2000),
        down_proj: proj(0x3000),
    }
}

/// 2026-10-13: `m` dense rows of GLM-5.3 rank 0 (hidden 4096, width 4096) under `t`, on a
/// workspace of `max_rows` rows whose W4A4 scratch is sized `grouped`.
fn run(
    m: usize,
    t: Option<DenseGroupedTables>,
    max_rows: usize,
    grouped: bool,
) -> (
    anyhow::Result<()>,
    Vec<metrale_gpu_runtime::gpu::mock::MockLaunch>,
    Glm5NextMlpWorkspace,
) {
    let gpu = MockGpuBackend::new();
    let (c, k) = (cfg(), grouped_kernels(&gpu));
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, max_rows, grouped, 0).unwrap();
    let r = forward_dense_w4a4(
        &gpu,
        &k,
        &c,
        &weights(),
        SCALES,
        4096,
        X,
        OUT,
        m,
        &ws,
        t.as_ref(),
        7,
    );
    (r, gpu.launches_snapshot(), ws)
}

/// 2026-10-13: 300 rows at a threshold of 33: one offsets pass with m by value, a quantization
/// of all 300 rows under each static scale, and gate, up and down each one 128-slot grouped
/// GEMM over one group (offsets, tile offsets, no gather, one expert) of its own table into the
/// chunk path's buffers; no chunk GEMV.
#[test]
fn rows_from_the_threshold_run_one_group_per_projection() {
    let (r, l, ws) = run(300, Some(tables(33)), 512, true);
    r.unwrap();
    let funcs: Vec<u64> = l.iter().map(|l| l.func).collect();
    assert_eq!(
        funcs,
        vec![ONE_GROUP, QUANT, G128, G128, SWIGLU, QUANT, G128]
    );
    assert_eq!((l[0].grid, l[0].block), ([1, 1, 1], [32, 1, 1]));
    assert_eq!((ptr_arg(&l[0], 0), ptr_arg(&l[0], 1)), (OFFSETS, TILE_OFF));
    assert_eq!((u32_arg(&l[0], 2), u32_arg(&l[0], 3)), (300, 128));
    for (i, input, gs) in [(1, X, SCALES.gate_up), (5, ws.a_act, SCALES.down)] {
        assert_eq!(l[i].grid[0], 300, "launch {i}");
        assert_eq!(ptr_arg(&l[i], 0), input, "launch {i}");
        assert_eq!(u32_arg(&l[i], 4), 4096, "launch {i}");
        assert_eq!(f32_arg(&l[i], 5), gs, "launch {i}");
    }
    for (i, tag, dst) in [
        (2, 0x4000, ws.a_gate),
        (3, 0x5000, ws.a_up),
        (6, 0x6000, OUT),
    ] {
        let g = &l[i];
        assert_eq!(g.grid, [4096 / 64, 300usize.div_ceil(128) as u32 + 1, 1]);
        assert_eq!((g.block, g.shared_mem), ([256, 1, 1], smem_bytes_of(TILE)));
        let ptrs: Vec<DevicePtr> = (0..10).map(|a| ptr_arg(g, a)).collect();
        assert_eq!(
            ptrs,
            vec![
                ws.w4a4_aq,
                ws.w4a4_as,
                ws.w4a4_ag,
                DevicePtr(tag),
                DevicePtr(tag + 8),
                DevicePtr(tag + 16),
                dst,
                OFFSETS,
                TILE_OFF,
                DevicePtr::NULL,
            ],
            "launch {i}"
        );
        assert_eq!(
            [u32_arg(g, 10), u32_arg(g, 11), u32_arg(g, 12)],
            [1, 4096, 4096]
        );
    }
    assert!(l.iter().all(|l| l.stream == 7));
}

/// 2026-10-13: The threshold itself takes the one-group launch; one row below it, and the lever
/// off, run exactly the chunk path (the negative controls of the test above).
#[test]
fn rows_below_the_threshold_keep_the_chunks() {
    let funcs = |m, t| -> Vec<u64> {
        let (r, l, _) = run(m, t, 512, true);
        r.unwrap();
        l.iter().map(|l| l.func).collect()
    };
    assert_eq!(funcs(64, Some(tables(64)))[0], ONE_GROUP);
    let chunks = vec![
        QUANT, MX32, MX32, QUANT, MX32, MX32, SWIGLU, QUANT, MX32, QUANT, MX32,
    ];
    assert_eq!(funcs(63, Some(tables(64)))[..], chunks[..]);
    assert_eq!(funcs(64, None)[..], chunks[..]);
}

/// 2026-10-13: A launch wider than the quantized scratch (a workspace sized without
/// `METRALE_GLM_MOE_W4A4_GROUPED`) is refused before anything launches.
#[test]
fn rows_past_the_quantized_scratch_are_refused_before_launching() {
    let (_, _, ws) = run(1, None, 512, false);
    let m = ws.w4a4_rows + 1;
    assert!(m <= 512, "the scratch must be narrower than the workspace");
    let (r, l, _) = run(m, Some(tables(33)), 512, false);
    let e = r.unwrap_err().to_string();
    assert!(e.contains("METRALE_GLM_MOE_W4A4_GROUPED=1"), "{e}");
    assert!(l.is_empty(), "{} launches before the refusal", l.len());
}
