// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: CPU dispatch/refusal controls; numerical CUDA evidence is separate.

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::Nvfp4SmallRowKernels;

fn gpu() -> MockGpuBackend {
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_handle(
        "moe_w4a16",
        "moe_w4a16_grouped_gemm_ptrtable_small16_n32",
        KernelHandle(1),
    );
    gpu.set_kernel_handle(
        "moe_w4a16",
        "moe_w4a16_grouped_gemm_ptrtable_large64",
        KernelHandle(2),
    );
    gpu.set_kernel_n_tile(KernelHandle(1), 32);
    gpu.set_kernel_n_tile(KernelHandle(2), 64);
    gpu
}

#[test]
fn disjoint_grids_keep_the_entire_large_expert_fallback_and_stream() {
    let gpu = gpu();
    let pair = Nvfp4SmallRowKernels::resolve(&gpu).unwrap();
    let p = DevicePtr(0x1000);
    pair.launch(
        &gpu,
        p,
        p,
        p,
        p,
        p,
        p,
        DevicePtr::NULL,
        256,
        129,
        2048,
        15,
        71,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 2);
    assert_eq!(launches[0].grid, [5, 1, 256]);
    assert_eq!(launches[0].block, [64, 1, 1]);
    assert_eq!(launches[1].block, [128, 1, 1]);
    assert_eq!(launches[1].grid, [3, 15, 256]);
    assert!(launches.iter().all(|x| x.stream == 71));
    assert_eq!(launches[0].args, launches[1].args);
    assert_eq!(gpu.kernel_lookups_snapshot().len(), 2);
}

#[test]
fn invalid_shape_or_pointer_never_partially_launches() {
    for (experts, n, k, tiles, ptr) in [
        (0, 128, 2048, 1, 0x1000),
        (256, 0, 2048, 1, 0x1000),
        (256, 128, 0, 1, 0x1000),
        (256, 128, 2049, 1, 0x1000),
        (256, 128, 2048, 0, 0x1000),
        (256, 128, 2048, 1, 0),
        (256, 128, 2048, 1, 0x1001),
    ] {
        let gpu = gpu();
        let pair = Nvfp4SmallRowKernels::resolve(&gpu).unwrap();
        let p = DevicePtr(ptr);
        assert!(
            pair.launch(
                &gpu,
                p,
                p,
                p,
                p,
                p,
                p,
                DevicePtr::NULL,
                experts,
                n,
                k,
                tiles,
                0
            )
            .is_err()
        );
        assert!(gpu.launches_snapshot().is_empty());
    }
}

#[test]
fn missing_or_wrong_precision_kernel_pair_is_refused() {
    let gpu = gpu();
    gpu.deny_kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_large64");
    assert!(Nvfp4SmallRowKernels::resolve(&gpu).is_err());
    let gpu = self::gpu();
    gpu.set_kernel_n_tile(KernelHandle(1), 64);
    assert!(Nvfp4SmallRowKernels::resolve(&gpu).is_err());
    let gpu = self::gpu();
    gpu.set_kernel_a_e4m3(KernelHandle(1));
    assert!(Nvfp4SmallRowKernels::resolve(&gpu).is_err());
}

#[test]
fn narrow_and_wide_tiles_cover_each_n_tail() {
    for n in [1u32, 31, 32, 33, 64, 129, 512, 2048] {
        let gpu = gpu();
        let pair = Nvfp4SmallRowKernels::resolve(&gpu).unwrap();
        let p = DevicePtr(0x1000);
        pair.launch(
            &gpu,
            p,
            p,
            p,
            p,
            p,
            p,
            DevicePtr::NULL,
            256,
            n,
            2048,
            15,
            71,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches[0].grid, [n.div_ceil(32), 1, 256]);
        assert_eq!(launches[0].block, [64, 1, 1]);
        assert_eq!(launches[1].grid, [n.div_ceil(64), 15, 256]);
        assert_eq!(launches[1].block, [128, 1, 1]);
        assert_eq!(launches[0].args, launches[1].args);
    }
}

#[test]
fn malformed_gather_and_large_tile_metadata_are_refused() {
    let gpu = gpu();
    let pair = Nvfp4SmallRowKernels::resolve(&gpu).unwrap();
    let p = DevicePtr(0x1000);
    assert!(
        pair.launch(&gpu, p, p, p, p, p, p, DevicePtr(3), 256, 128, 2048, 1, 0)
            .is_err()
    );
    assert!(gpu.launches_snapshot().is_empty());
    gpu.set_kernel_n_tile(KernelHandle(2), 32);
    assert!(Nvfp4SmallRowKernels::resolve(&gpu).is_err());
}
