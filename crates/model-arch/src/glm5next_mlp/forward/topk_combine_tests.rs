// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The launch shapes `topk_combine` picks. The bytes of the rank kernel against the
//! base kernel, and of the split combine against the one-block combine, are the GPU microtest's
//! job (`examples/glm5next_c1_fuse_microtest.rs`).
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::{ROUTER_RANK_MAX_EXPERTS, combine_blocks_per_row, topk_launch};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

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
    let mut k = Glm5NextMlpKernels::resolve(&gpu).expect("mock resolves every kernel");
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
