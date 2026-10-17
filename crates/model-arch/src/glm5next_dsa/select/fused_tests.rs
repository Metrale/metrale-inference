// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The host side of `METRALE_GLM_DSA_SELECT_FUSED` (`fused.rs`, `fit.rs`): which
//! passes take it, what bounds their rows, what fails instead of falling back, and the launch's
//! argument order against `dsa_select_fused_pipe`'s signature. The kernel's selection is gated
//! on a GB10 by `examples/glm5next_dsa_select_fused_gate/`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::passes_tests::{cfg, inputs, kernels};
use super::*;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn geom(rows: usize) -> DsaSelectGeometry {
    DsaSelectGeometry::plan(&cfg(), 4096, rows).unwrap()
}

/// 2026-10-17: GLM-5.3's prefill shape is fused; one row, no pools, a width past the radix
/// bound (and zero), or another head shape is not. Both width bounds are probed.
#[test]
fn fused_shape_is_the_tensor_core_shape_at_a_radix_width() {
    let g = geom(8);
    assert_eq!(g.select_k, 512, "GLM-5.3: 2048 tokens at kpool 4");
    assert!(fused_shape(&g));
    let with = |f: &dyn Fn(&mut DsaSelectGeometry)| {
        let mut x = g;
        f(&mut x);
        fused_shape(&x)
    };
    assert!(with(&|x| x.select_k = 1));
    assert!(with(&|x| x.select_k = SELECT_FUSED_MAXK));
    assert!(!with(&|x| x.select_k = SELECT_FUSED_MAXK + 1));
    assert!(!with(&|x| x.select_k = 0));
    assert!(!with(&|x| x.q_rows = 1));
    assert!(with(&|x| x.q_rows = 2));
    assert!(!with(&|x| x.n_pools = 0));
    assert!(!with(&|x| x.index_heads = 16));
    assert!(!with(&|x| x.index_head_dim = 64));
}

/// 2026-10-17: The kernel's dynamic shared memory: the pipe's ring plus 8 warps x 704 entries x
/// 8 B, within the 99 KB per-block opt-in with the walk's 8 B of static shared memory.
#[test]
fn fused_shared_memory_is_the_ring_plus_eight_buffers() {
    assert_eq!(SELECT_FUSED_SMEM, SCORES_PIPE_SMEM + 8 * 704 * 8);
    assert_eq!(SELECT_FUSED_SMEM, 98_048);
    const { assert!(SELECT_FUSED_SMEM + 8 <= 99 * 1024) };
}

/// 2026-10-17: A fused scratch bounds a pass's rows by the selected region alone, and `fits_for`
/// skips the scores and candidacy regions only for a fused pass; the same scratch unfused keeps
/// the old bound (3 rows of scores) and refuses the 8-row pass. The selected region still
/// bounds a fused pass (7 rows of it refuse 8).
#[test]
fn a_fused_pass_is_bounded_by_its_selected_rows_only() {
    let c = cfg();
    let g = geom(8);
    let gpu = MockGpuBackend::new();
    let mut s =
        DsaSelectScratch::alloc_rows(&gpu, &c, &DsaSelectGeometry { q_rows: 3, ..g }, 8).unwrap();
    assert!(!s.fused, "the lever is off unless set");
    assert_eq!(s.rows_per_pass(&g), 3);
    assert!(s.fits(&c, &g).is_err());
    s.fused = true;
    assert!(s.fused_on(&g));
    assert_eq!(s.rows_per_pass(&g), 8);
    assert!(s.fits_for(&c, &g, true).is_ok());
    assert!(
        s.fits(&c, &g).is_err(),
        "an unfused pass is still checked in full"
    );
    let other = DsaSelectGeometry {
        index_heads: 16,
        ..g
    };
    assert!(!s.fused_on(&other));
    assert_eq!(
        s.rows_per_pass(&other),
        3,
        "a pass outside the shape keeps the old bound"
    );
    s.capacity[5] = 7 * g.select_k * 4;
    assert_eq!(s.rows_per_pass(&g), 7);
    assert!(s.fits_for(&c, &g, true).is_err());
}

/// 2026-10-17: A fused scratch on an exact multi-row pass takes the kernel, and fails (never
/// falls back) when a kernel it needs is missing or the pass carries a device geometry. A
/// ceiling launch, a one-row pass or an unfused scratch never takes it and never fails here,
/// whatever kernels resolved.
#[test]
fn takes_fused_needs_its_kernels_and_host_geometry() {
    let c = cfg();
    let g = geom(8);
    let gpu = MockGpuBackend::new();
    let inp = inputs(&gpu, &c, 4096, 8);
    let mut s = DsaSelectScratch::alloc_rows(&gpu, &c, &g, 8).unwrap();
    let mut k = kernels();
    let exact = DsaSelectLaunch::Exact;
    assert!(
        !fused::takes_fused(&k, &g, &inp, &s, exact).unwrap(),
        "unfused"
    );
    s.fused = true;
    assert!(
        fused::takes_fused(&k, &g, &inp, &s, exact).is_err(),
        "no fused kernel"
    );
    k.rows.select_fused = KernelHandle(30);
    assert!(
        fused::takes_fused(&k, &g, &inp, &s, exact).is_err(),
        "no prep kernel"
    );
    k.rows.pool_prep = KernelHandle(31);
    assert!(fused::takes_fused(&k, &g, &inp, &s, exact).unwrap());
    k.rows.expand_folded = KernelHandle(0);
    assert!(
        fused::takes_fused(&k, &g, &inp, &s, exact).is_err(),
        "no folded expansion"
    );
    k.rows.expand_folded = KernelHandle(22);
    let dev = DsaSelectInputs {
        geom_dev: DevicePtr(0x1000),
        ..inp
    };
    assert!(
        fused::takes_fused(&k, &g, &dev, &s, exact).is_err(),
        "device geometry"
    );
    let ceiling = DsaSelectLaunch::Ceiling { max_pools: 1024 };
    assert!(!fused::takes_fused(&k, &g, &dev, &s, ceiling).unwrap());
    let one = DsaSelectGeometry { q_rows: 1, ..g };
    assert!(!fused::takes_fused(&k, &one, &inp, &s, exact).unwrap());
}

/// 2026-10-17: `fused::launch` sends `dsa_select_fused_pipe` its arguments in signature order
/// (q, keys_bf16, weights, vis, q_pos, q_mask, selected, Q, P, H, D, KP, select_k, scale), one
/// CTA of 256 threads per 8 rows, and the shared memory above; it refuses a scratch without
/// prepared pools.
#[test]
fn fused_launch_matches_the_kernel_signature() {
    let c = cfg();
    let g = geom(13);
    let gpu = MockGpuBackend::new();
    let inp = inputs(&gpu, &c, 4096, 13);
    let mut s = DsaSelectScratch::alloc_rows(&gpu, &c, &g, 13).unwrap();
    let mut k = kernels();
    k.rows.select_fused = KernelHandle(30);
    assert!(
        fused::launch(&gpu, &k, &g, &inp, &s, 5).is_err(),
        "no prepared pools"
    );
    s.pipe = pipe::DsaPipeRegions::alloc(&gpu, &g, true).unwrap();
    fused::launch(&gpu, &k, &g, &inp, &s, 5).unwrap();
    let l = gpu.launches_snapshot().pop().unwrap();
    assert_eq!((l.func, l.grid, l.block), (30, [2, 1, 1], [256, 1, 1]));
    assert_eq!((l.shared_mem, l.stream), (SELECT_FUSED_SMEM, 5));
    let b = MockArg::Buffer;
    let u = |v: usize| MockArg::Bytes((v as u32).to_le_bytes().to_vec());
    let want = vec![
        b(inp.q),
        b(s.pipe.keys_bf16),
        b(inp.weights),
        b(s.pipe.vis),
        b(inp.q_pos),
        b(inp.q_mask),
        b(s.selected),
        u(13),
        u(g.n_pools),
        u(32),
        u(128),
        u(4),
        u(512),
        MockArg::Bytes((128f32).powf(-0.5).to_le_bytes().to_vec()),
    ];
    assert_eq!(l.args, want);
}
