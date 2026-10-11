// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The folded post + pre-norm launch on the mock backend: argument order, the
//! partials' fields, and the refusals before any launch. The bits are the GPU gate's job
//! (`examples/glm5next_hc_post_fold_gate.rs`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_comm::PeerPartials;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::*;

const FOLD: u64 = 0x77;

fn kernels(gpu: &MockGpuBackend) -> Glm5NextMhcKernels {
    let mut k = Glm5NextMhcKernels::resolve(gpu).expect("mock resolves every kernel");
    k.hc_post_pre_norm = KernelHandle(FOLD);
    k
}

fn site(bf16: bool) -> Glm5NextMhcSiteWeights {
    Glm5NextMhcSiteWeights {
        hc_fn: DevicePtr(0x100),
        hc_fn_bf16: bf16,
        hc_scale: DevicePtr(0x200),
        hc_base: DevicePtr(0x300),
        mix: DevicePtr(0x400),
        done: DevicePtr(0x480),
    }
}

fn io() -> PostFoldIo {
    PostFoldIo {
        own: DevicePtr(0x1000),
        residual: DevicePtr(0x2000),
        streams_out: DevicePtr(0x3000),
        post: DevicePtr(0x4000),
        comb: DevicePtr(0x5000),
        norm_w: DevicePtr(0x6000),
        y_out: DevicePtr(0x7000),
        normed_out: DevicePtr(0x8000),
    }
}

fn ptr(l: &MockLaunch, i: usize) -> DevicePtr {
    match &l.args[i] {
        MockArg::Buffer(p) => *p,
        other => panic!("arg {i} is {other:?}, not a pointer"),
    }
}

fn bytes(l: &MockLaunch, i: usize) -> Vec<u8> {
    match &l.args[i] {
        MockArg::Bytes(b) => b.clone(),
        other => panic!("arg {i} is {other:?}, not bytes"),
    }
}

const PARTIALS: PeerPartials = PeerPartials {
    recv: 0x9000,
    slot_elems: 8192,
    world_size: 3,
    rank: 1,
};

/// 2026-10-10: One launch over the next site's 24 mix rows: own, the peers' slot and ranks (or
/// a null slot and 0 ranks without partials), residual, the previous post/comb, the streams
/// out, the site's weights and scratch, the norm, then post/comb again as the outputs.
#[test]
fn one_launch_in_the_kernels_argument_order() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    for partials in [Some(&PARTIALS), None] {
        let from = gpu.launch_count();
        glm_hc_post_pre_norm(
            &gpu,
            &k,
            &site(true),
            io(),
            partials,
            [4096, 4, 20],
            [1e-6; 3],
            5,
        )
        .unwrap();
        assert_eq!(gpu.launch_count(), from + 1);
        let l = gpu.launches_snapshot()[from].clone();
        assert_eq!(
            (l.func, l.grid, l.block, l.stream),
            (FOLD, [1, 24, 1], [256, 1, 1], 5)
        );
        let io = io();
        assert_eq!(ptr(&l, 0), io.own);
        let (peers, slot, n, r) = match partials {
            Some(p) => (DevicePtr(p.recv), p.slot_elems as u64, 3i32, 1i32),
            None => (DevicePtr::NULL, 0, 0, 0),
        };
        assert_eq!(ptr(&l, 1), peers);
        assert_eq!(bytes(&l, 2), slot.to_le_bytes());
        assert_eq!(bytes(&l, 3), n.to_le_bytes());
        assert_eq!(bytes(&l, 4), r.to_le_bytes());
        let w = site(true);
        let want = [
            io.residual,
            io.post,
            io.comb,
            io.streams_out,
            w.hc_fn,
            w.hc_scale,
            w.hc_base,
            w.mix,
            w.done,
            io.norm_w,
            io.y_out,
            io.post,
            io.comb,
            io.normed_out,
        ];
        for (i, p) in want.iter().enumerate() {
            assert_eq!(ptr(&l, 5 + i), *p, "pointer argument {}", 5 + i);
        }
        assert_eq!(bytes(&l, 19), 4096u32.to_le_bytes());
        assert_eq!(bytes(&l, 20), 4u32.to_le_bytes());
    }
}

/// 2026-10-10: A missing kernel, a width the kernel does not hold, in-place streams, an FP32
/// `hc_fn` or a peer slot narrower than the row is refused, and nothing is launched.
#[test]
fn refusals_come_before_any_launch() {
    let gpu = MockGpuBackend::new();
    let k = kernels(&gpu);
    let mut no_kernel = k;
    no_kernel.hc_post_pre_norm = KernelHandle(0);
    let mut in_place = io();
    in_place.streams_out = in_place.residual;
    let narrow = PeerPartials {
        slot_elems: 2048,
        ..PARTIALS
    };
    let run = |k: &Glm5NextMhcKernels, w, io, p: Option<&PeerPartials>, h: u32| {
        glm_hc_post_pre_norm(&gpu, k, &w, io, p, [h, 4, 20], [1e-6; 3], 0)
            .unwrap_err()
            .to_string()
    };
    for (e, want) in [
        (
            run(&no_kernel, site(true), io(), None, 4096),
            "not in this build",
        ),
        (run(&k, site(true), io(), None, 4100), "hidden 4100"),
        (run(&k, site(true), io(), None, 8192), "hidden 8192"),
        (run(&k, site(true), in_place, None, 4096), "own buffer"),
        (run(&k, site(false), io(), None, 4096), "FP32"),
        (
            run(&k, site(true), io(), Some(&narrow), 4096),
            "peer slot of 2048",
        ),
    ] {
        assert!(e.contains(want), "{e}");
    }
    assert_eq!(gpu.launch_count(), 0);
}
