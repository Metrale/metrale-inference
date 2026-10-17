// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The launch plan of `router_logits` on a recording mock backend: which GEMV
//! entry the rows take, over how many rows, reading and writing which rows. The bits of the
//! batched entries against the M = 1 GEMV are the GPU microtest's job
//! (`examples/glm5next_rowbatch_microtest.rs`).
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::router_logits;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const GEMV_F32: u64 = 0x71;
const NARROW_F32: u64 = 0x72;
const WIDE_F32: u64 = 0x73;
const ROUTER: DevicePtr = DevicePtr(0x1000);
const X: DevicePtr = DevicePtr(0x9000_0000);
const LOGITS: DevicePtr = DevicePtr(0xA000_0000);
const HIDDEN: usize = 4096;
const EXPERTS: usize = 288;

fn cfg() -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: EXPERTS,
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
    k.gemv_f32 = KernelHandle(GEMV_F32);
    k.gemv_batchm_f32 = KernelHandle(NARROW_F32);
    k.gemv_batchm_wide_f32 = KernelHandle(WIDE_F32);
    k
}

fn ptr(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn u32_arg(l: &MockLaunch, i: usize) -> u32 {
    match &l.args[i] {
        MockArg::Bytes(b) => u32::from_le_bytes(b[..4].try_into().unwrap()),
        other => panic!("arg {i} is {other:?}, not 4 bytes"),
    }
}

fn run(k: &Glm5NextMlpKernels, rows: usize, batched: bool) -> Vec<MockLaunch> {
    let _g = crate::glm5next_fp8_dense::lock_registries_for_test();
    let gpu = MockGpuBackend::new();
    router_logits(&gpu, k, &cfg(), ROUTER, X, LOGITS, rows, batched, 3).unwrap();
    gpu.launches_snapshot()
}

/// 2026-10-09: Batched, 16 rows are one wide FP32-out launch over all rows (M = 16, N = 288,
/// K = 4096, contiguous output); 4 rows one narrow launch.
#[test]
fn batched_rows_are_one_launch_on_the_entry_for_their_width() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    for (rows, func) in [
        (16usize, WIDE_F32),
        (9, WIDE_F32),
        (8, NARROW_F32),
        (4, NARROW_F32),
    ] {
        let l = run(&k, rows, true);
        assert_eq!(l.len(), 1, "{rows} rows");
        assert_eq!(l[0].func, func, "{rows} rows");
        assert_eq!(
            [ptr(&l[0], 0), ptr(&l[0], 1), ptr(&l[0], 2)],
            [X, ROUTER, LOGITS]
        );
        assert_eq!(
            [
                u32_arg(&l[0], 3),
                u32_arg(&l[0], 4),
                u32_arg(&l[0], 5),
                u32_arg(&l[0], 6)
            ],
            [rows as u32, EXPERTS as u32, HIDDEN as u32, EXPERTS as u32]
        );
        assert_eq!(l[0].grid, [72, 1, 1]);
        assert_eq!(l[0].stream, 3);
    }
}

/// 2026-10-09: Off (the default), with one row, or without a batched entry, every row is its
/// own M = 1 GEMV reading row `r` of `x` and writing row `r` of the logits.
#[test]
fn per_row_gemvs_otherwise() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let mut no_wide = k;
    no_wide.gemv_batchm_wide_f32 = KernelHandle(0);
    for (kk, rows, batched) in [(&k, 4usize, false), (&k, 1, true), (&no_wide, 4, true)] {
        let l = run(kk, rows, batched);
        assert_eq!(l.len(), rows);
        for (r, l) in l.iter().enumerate() {
            assert_eq!(l.func, GEMV_F32);
            assert_eq!(ptr(l, 0), X.offset(r * HIDDEN * 2));
            assert_eq!(ptr(l, 2), LOGITS.offset(r * EXPERTS * 4));
            assert_eq!(
                [u32_arg(l, 3), u32_arg(l, 4)],
                [EXPERTS as u32, HIDDEN as u32]
            );
        }
    }
}
