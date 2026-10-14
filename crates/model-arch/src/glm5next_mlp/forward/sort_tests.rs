// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The launches `sort` issues. The bytes of the multi-block sort against the
//! one-block sort and a host stable sort are the GPU microtest's job
//! (`examples/glm5next_c1_fuse_microtest.rs`, `C1_FUSE_ONLY=sort`).
//!
//! Owner: model-arch (GLM-5.3 MLP).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::{SORT_MB_CHUNK, SORT_MB_MAX_EXPERTS, sort_by_expert_with, sort_hist_bytes};
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

const SORT: u64 = 0x71;
const MB: [u64; 3] = [0x72, 0x73, 0x74];

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

fn kernels(gpu: &MockGpuBackend, mb: [u64; 3]) -> Glm5NextMlpKernels {
    let mut k = Glm5NextMlpKernels::resolve(gpu).expect("mock resolves every kernel");
    k.moe_sort_by_expert = KernelHandle(SORT);
    k.sort_mb = mb.map(KernelHandle);
    k
}

fn u32a(v: usize) -> MockArg {
    MockArg::Bytes((v as u32).to_le_bytes().to_vec())
}

/// 2026-10-14: The launches one `sort_by_expert_with` call issued on a fresh mock.
fn run(
    c: &Glm5NextMlpConfig,
    rows: usize,
    slots: usize,
    multiblock: bool,
) -> (Glm5NextMlpWorkspace, Vec<MockLaunch>, MockGpuBackend) {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu, MB);
    let ws = Glm5NextMlpWorkspace::new(&gpu, c, rows).unwrap();
    let before = gpu.launches_snapshot().len();
    sort_by_expert_with(&gpu, &k, c, &ws, slots, multiblock, 9).unwrap();
    let l = gpu.launches_snapshot()[before..].to_vec();
    (ws, l, gpu)
}

/// 2026-10-14: Off, exactly the launch `w4a4_grouped` issued inline before the helper: one block
/// of 256 threads with the ids, the four routing tables, the slot, expert and top-k counts.
#[test]
fn off_is_the_one_block_sort_unchanged() {
    let c = cfg(288);
    let (ws, l, _) = run(&c, 4, 32, false);
    assert_eq!(l.len(), 1);
    assert_eq!(
        (l[0].func, l[0].grid, l[0].block, l[0].stream),
        (SORT, [1, 1, 1], [256, 1, 1], 9)
    );
    let want = vec![
        MockArg::Buffer(ws.ids),
        MockArg::Buffer(ws.sorted_token_ids),
        MockArg::Buffer(ws.sorted_expert_ids),
        MockArg::Buffer(ws.expert_offsets),
        MockArg::Buffer(ws.token_to_perm),
        u32a(32),
        u32a(288),
        u32a(8),
    ];
    assert_eq!(l[0].args, want);
}

/// 2026-10-14: On, hist and scatter run one block of 256 per 1024 slots (at least one, so an
/// empty sort still writes zero offsets), the scan one block of the experts in whole warps, all
/// on the caller's stream, and they share one `sort_hist` and `expert_offsets`.
#[test]
fn on_runs_hist_scan_scatter_over_chunks_of_1024_slots() {
    for (experts, scan_block) in [(288usize, 288u32), (289, 320), (1, 32), (1024, 1024)] {
        let c = cfg(experts);
        for (slots, nb) in [(0usize, 1u32), (1, 1), (1024, 1), (1025, 2), (8192, 8)] {
            let (ws, l, _) = run(&c, 1024, slots, true);
            let shape: Vec<_> = l
                .iter()
                .map(|x| (x.func, x.grid, x.block, x.stream))
                .collect();
            assert_eq!(
                shape,
                vec![
                    (MB[0], [nb, 1, 1], [256, 1, 1], 9),
                    (MB[1], [1, 1, 1], [scan_block, 1, 1], 9),
                    (MB[2], [nb, 1, 1], [256, 1, 1], 9),
                ],
                "{experts} experts, {slots} slots"
            );
            let (b, n, e) = (MockArg::Buffer, u32a(slots), u32a(experts));
            assert_eq!(
                l[0].args,
                vec![b(ws.ids), b(ws.sort_hist), n.clone(), e.clone()]
            );
            assert_eq!(
                l[1].args,
                vec![
                    b(ws.sort_hist),
                    b(ws.expert_offsets),
                    u32a(nb as usize),
                    e.clone()
                ]
            );
            assert_eq!(
                l[2].args,
                vec![
                    b(ws.ids),
                    b(ws.sort_hist),
                    b(ws.expert_offsets),
                    b(ws.sorted_token_ids),
                    b(ws.sorted_expert_ids),
                    b(ws.token_to_perm),
                    n,
                    e,
                    u32a(8),
                ]
            );
        }
    }
}

/// 2026-10-14: `sort_hist` holds a count per expert for every block the widest sort launches.
#[test]
fn sort_hist_holds_every_block_of_the_widest_sort() {
    for (rows, experts) in [
        (1usize, 288usize),
        (128, 288),
        (129, 288),
        (2048, 288),
        (3, 1024),
    ] {
        let c = cfg(experts);
        let (ws, _, gpu) = run(&c, rows, 0, false);
        let blocks = (rows * 8).div_ceil(SORT_MB_CHUNK);
        assert_eq!(sort_hist_bytes(&c, rows * 8), blocks * experts * 4);
        let got = gpu.read_alloc(ws.sort_hist).unwrap().len();
        assert_eq!(got, blocks * experts * 4, "{rows} rows, {experts} experts");
    }
}

/// 2026-10-14: The lever where the multi-block sort cannot run is refused, not served by the
/// one-block sort: a kernel missing, no experts or more than one scan block holds, or more slots
/// than the workspace was sized for. Nothing is launched.
#[test]
fn multiblock_that_cannot_run_is_an_error() {
    let err = |k: &Glm5NextMlpKernels, c: &Glm5NextMlpConfig, slots: usize| {
        let gpu = MockGpuBackend::new();
        let ws = Glm5NextMlpWorkspace::new(&gpu, c, 16).unwrap();
        let e = sort_by_expert_with(&gpu, k, c, &ws, slots, true, 0).unwrap_err();
        assert!(gpu.launches_snapshot().is_empty());
        e.to_string()
    };
    let gpu = MockGpuBackend::new();
    for missing in 0..3 {
        let mut mb = MB;
        mb[missing] = 0;
        assert!(err(&kernels(&gpu, mb), &cfg(288), 128).contains("not in this target"));
    }
    let k = kernels(&gpu, MB);
    assert!(err(&k, &cfg(SORT_MB_MAX_EXPERTS + 1), 128).contains("1..=1024"));
    assert!(err(&k, &cfg(0), 0).contains("1..=1024"));
    assert!(err(&k, &cfg(288), 129).contains("129 slots past the workspace's 128"));
}
