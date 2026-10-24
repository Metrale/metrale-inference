// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The launch shapes `topk_combine` picks. The bytes of the rank kernel against the
//! base kernel, and of the split combine against the one-block combine, are the GPU microtest's
//! job (`examples/glm5next_c1_fuse_microtest.rs`).
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::{
    COMBINE_VEC8_MAX_K, COMBINE_VEC8_MIN_ROWS, ROUTER_RANK_MAX_EXPERTS, combine_blocks_per_row,
    combine_launch, combine_with, topk_launch, vec8_for,
};
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};
use metrale_gpu_runtime::gpu::DevicePtr;

const BASE: u64 = 0x51;
const RANK: u64 = 0x52;

fn cfg(num_experts: usize) -> Glm5NextMlpConfig {
    Glm5NextMlpConfig {
        hidden: 4096,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: 2048,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts,
        local_experts: num_experts,
        ep_rank: 0,
        top_k: 8,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 1,
        expert_shard: crate::glm5next_mlp::ExpertShard::Whole,
    }
}

fn kernels(rank: u64) -> Glm5NextMlpKernels {
    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextMlpKernels::resolve(&gpu, crate::glm5next_layer::GlmLevers::declared())
        .expect("mock resolves every kernel");
    k.router = KernelHandle(BASE);
    k.router_rank = KernelHandle(rank);
    k
}

/// 2026-10-10: Off, the base kernel on 256 threads whatever the expert count; on, the rank
/// kernel on one thread per expert rounded up to whole warps (its first pass ranks each run of
/// 32 experts with full-warp shuffles).
#[test]
fn rank_kernel_gets_one_thread_per_expert_in_whole_warps() {
    let k = kernels(RANK);
    for n in [288usize, 8, 64] {
        let (f, block) = topk_launch(&k, &cfg(n), false).unwrap();
        assert_eq!((f.0, block), (BASE, 256));
    }
    for (n, block) in [(288usize, 288u32), (289, 320), (31, 32), (1024, 1024)] {
        let (f, got) = topk_launch(&k, &cfg(n), true).unwrap();
        assert_eq!((f.0, got), (RANK, block), "{n} experts");
    }
}

/// 2026-10-10: The lever on without the kernel, or with more experts than one block of threads
/// holds, is refused rather than quietly served by the base kernel.
#[test]
fn rank_lever_that_cannot_run_is_an_error() {
    let missing = topk_launch(&kernels(0), &cfg(288), true).unwrap_err();
    assert!(
        missing.to_string().contains("not in this target"),
        "{missing}"
    );
    let wide = topk_launch(&kernels(RANK), &cfg(ROUTER_RANK_MAX_EXPERTS + 1), true).unwrap_err();
    assert!(wide.to_string().contains("1..=1024"), "{wide}");
}

/// 2026-10-10: Split, one block per 256 elements of a row (rounded up, so every element has a
/// thread); otherwise one block per row.
#[test]
fn split_combine_covers_hidden_one_element_per_thread() {
    assert_eq!(combine_blocks_per_row(false, 4096), 1);
    assert_eq!(combine_blocks_per_row(true, 4096), 16);
    assert_eq!(combine_blocks_per_row(true, 4097), 17);
    assert_eq!(combine_blocks_per_row(true, 100), 1);
}

const COMBINE: u64 = 0x61;
const COMBINE_IDX: u64 = 0x62;
const VEC8: u64 = 0x63;
const VEC8_IDX: u64 = 0x64;
const ALIGNED: [DevicePtr; 3] = [DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000)];

fn combine_kernels(vec8: [u64; 2]) -> Glm5NextMlpKernels {
    let mut k = kernels(RANK);
    k.combine = KernelHandle(COMBINE);
    k.combine_indexed = KernelHandle(COMBINE_IDX);
    k.combine_vec8 = vec8.map(KernelHandle);
    k
}

/// 2026-10-14: Off, the vec8 lever leaves the scalar kernels and their grids exactly as they
/// were, split or not.
#[test]
fn vec8_off_keeps_the_scalar_combine() {
    let k = combine_kernels([VEC8, VEC8_IDX]);
    for (indexed, func) in [(false, COMBINE), (true, COMBINE_IDX)] {
        for split in [false, true] {
            let (f, y) = combine_launch(&k, &cfg(288), indexed, split, false, ALIGNED).unwrap();
            assert_eq!((f.0, y), (func, combine_blocks_per_row(split, 4096)));
        }
    }
}

/// 2026-10-14: On, the `_vec8` twin of the order in use, one thread per eight elements of a row
/// rounded up to whole blocks of 256, whatever the split lever says.
#[test]
fn vec8_takes_one_thread_per_eight_elements() {
    let k = combine_kernels([VEC8, VEC8_IDX]);
    for (hidden, y) in [(4096usize, 2u32), (2048, 1), (6144, 3), (8, 1), (2056, 2)] {
        let c = Glm5NextMlpConfig { hidden, ..cfg(288) };
        for (indexed, func) in [(false, VEC8), (true, VEC8_IDX)] {
            for split in [false, true] {
                let got = combine_launch(&k, &c, indexed, split, true, ALIGNED).unwrap();
                assert_eq!((got.0.0, got.1), (func, y), "hidden {hidden}");
            }
        }
    }
}

/// 2026-10-14: The lever on where the `_vec8` kernel cannot run is refused, not served by the
/// scalar kernel: the kernel missing, a row not a whole number of 16-byte groups, more slots
/// than its registers hold, or an unaligned buffer.
#[test]
fn vec8_that_cannot_run_is_an_error() {
    let err = |k: &Glm5NextMlpKernels, c: &Glm5NextMlpConfig, bufs, indexed| {
        combine_launch(k, c, indexed, false, true, bufs)
            .unwrap_err()
            .to_string()
    };
    let k = combine_kernels([VEC8, VEC8_IDX]);
    assert!(err(&combine_kernels([VEC8, 0]), &cfg(288), ALIGNED, true).contains("not in this"));
    assert!(err(&combine_kernels([0, VEC8_IDX]), &cfg(288), ALIGNED, false).contains("not in"));
    let odd = Glm5NextMlpConfig {
        hidden: 4100,
        ..cfg(288)
    };
    assert!(err(&k, &odd, ALIGNED, true).contains("multiple of 8"));
    let wide = Glm5NextMlpConfig {
        top_k: COMBINE_VEC8_MAX_K + 1,
        ..cfg(288)
    };
    assert!(err(&k, &wide, ALIGNED, false).contains("top_k 17"));
    let max = Glm5NextMlpConfig {
        top_k: COMBINE_VEC8_MAX_K,
        ..cfg(288)
    };
    assert!(combine_launch(&k, &max, false, false, true, ALIGNED).is_ok());
    for i in 0..3 {
        let mut bufs = ALIGNED;
        bufs[i] = DevicePtr(bufs[i].0 + 8);
        assert!(
            err(&k, &cfg(288), bufs, true).contains("16-byte aligned"),
            "buffer {i}"
        );
    }
}

/// 2026-10-14: The `_vec8` launch passes exactly the scalar launch's arguments (the kernels
/// share a signature), on the same rows and block; only the kernel and the grid's y differ.
#[test]
fn vec8_launch_passes_the_scalar_arguments() {
    let gpu = MockGpuBackend::new();
    let c = cfg(288);
    let k = combine_kernels([VEC8, VEC8_IDX]);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &c, 4, &k.levers.mlp).unwrap();
    let out = gpu.alloc(4 * c.hidden * 2).unwrap();
    for indexed in [false, true] {
        let before = gpu.launches_snapshot().len();
        combine_with(&gpu, &k, &c, &ws, out, 3, indexed, (true, false), 7).unwrap();
        combine_with(&gpu, &k, &c, &ws, out, 3, indexed, (true, true), 7).unwrap();
        let l = gpu.launches_snapshot();
        let (base, vec) = (&l[before], &l[before + 1]);
        let want = if indexed {
            (COMBINE_IDX, VEC8_IDX)
        } else {
            (COMBINE, VEC8)
        };
        assert_eq!((base.func, vec.func), want);
        assert_eq!((base.grid, vec.grid), ([3, 16, 1], [3, 2, 1]));
        assert_eq!((base.block, base.stream), (vec.block, vec.stream));
        assert_eq!(base.args, vec.args);
        assert_eq!(base.args.len(), if indexed { 7 } else { 6 });
    }
}

/// 2026-10-15: The `_vec8` kernel takes only combines of [`COMBINE_VEC8_MIN_ROWS`] rows or more,
/// and never with the lever off.
#[test]
fn vec8_only_from_the_row_floor() {
    let floor = COMBINE_VEC8_MIN_ROWS;
    assert!(!vec8_for(true, 1) && !vec8_for(true, 16) && !vec8_for(true, floor - 1));
    assert!(vec8_for(true, floor) && vec8_for(true, 8192));
    assert!(!vec8_for(false, floor) && !vec8_for(false, 8192));
}

/// 2026-10-22: The ids the layer dump reads (`Glm5NextMlpWorkspace::ids`, `Point::MoeIds`) are
/// the buffer the router top-k writes (its third argument), ranked over the global expert
/// count even on a rank that owns a share, on the site's stream.
#[test]
fn the_dumped_ids_are_the_buffer_the_top_k_writes() {
    use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
    use crate::glm5next_mlp::weights::{
        Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextMoePtrTables, Glm5NextMoeWeights,
    };
    use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
    use metrale_gpu_runtime::gpu::mock::MockArg;
    let null = DevicePtr::NULL;
    let table = Glm5NextExpertPtrTable {
        packed_ptrs: null,
        scale_ptrs: null,
        scale2_vals: null,
        scale_bases: null,
    };
    let w = Glm5NextMoeWeights {
        router: null,
        router_bias: DevicePtr(0x77),
        shared: Glm5NextDenseMlpWeights {
            gate_proj: null,
            up_proj: null,
            down_proj: null,
        },
        experts: Vec::new(),
        ptrs: Glm5NextMoePtrTables {
            gate: table,
            up: table,
            down: table,
        },
        precision: GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            0,
            true,
        )
        .unwrap(),
        act_scales: None,
    };
    let gpu = MockGpuBackend::new();
    // 2026-10-22: An EP=2 rank owning 144 of 288 experts still ranks all 288.
    let mut c = cfg(288);
    c.local_experts = 144;
    let ws = Glm5NextMlpWorkspace::new_forked(&gpu, &c, 2, false, 0).unwrap();
    super::router_topk(&gpu, &kernels(RANK), &c, &w, &ws, 2, 9).unwrap();
    let l = gpu.launches_snapshot().pop().expect("the top-k launched");
    assert_eq!((l.func, l.grid[0], l.stream), (BASE, 2, 9));
    assert_eq!(l.args[2], MockArg::Buffer(ws.ids()), "top-k ids output");
    assert_eq!(l.args[4], MockArg::Bytes(288u32.to_le_bytes().to_vec()));
}
