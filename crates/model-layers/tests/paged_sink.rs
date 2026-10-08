// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Sink launcher geometry and appended-pointer ABI controls.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
use metrale_model_layers::layers::ops::{PagedSinkGeometry, paged_decode_attn_bf16_sink};

fn geometry() -> PagedSinkGeometry {
    PagedSinkGeometry {
        sequences: 2,
        max_blocks: 9,
        q_heads: 64,
        kv_heads: 8,
        head_dim: 64,
        block_size: 16,
        scale: 0.125,
        q_stride: 4096,
        window: 128,
    }
}

#[test]
fn sink_pointer_is_appended_without_moving_legacy_arguments() {
    let gpu = MockGpuBackend::new();
    let buffers = std::array::from_fn(|i| DevicePtr(0x1000 * (i as u64 + 1)));
    paged_decode_attn_bf16_sink(&gpu, KernelHandle(7), buffers, &geometry(), 3).unwrap();
    let launches = gpu.launches_snapshot();
    let l = &launches[0];
    assert_eq!(l.grid, [64, 2, 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(l.stream, 3);
    assert_eq!(l.args.len(), 15);
    assert_eq!(l.args[0], MockArg::Buffer(buffers[0]));
    assert_eq!(l.args[13], MockArg::Bytes(128u32.to_le_bytes().to_vec()));
    assert_eq!(l.args[14], MockArg::Buffer(buffers[6]));
}

#[test]
fn invalid_geometry_and_missing_buffers_do_not_launch() {
    let gpu = MockGpuBackend::new();
    let b = std::array::from_fn(|i| DevicePtr(0x1000 * (i as u64 + 1)));
    for case in 0..9 {
        let mut g = geometry();
        match case {
            0 => g.head_dim = 128,
            1 => g.kv_heads = 0,
            2 => g.kv_heads = 7,
            3 => g.q_stride = 1,
            4 => g.sequences = 0,
            5 => g.scale = f32::NAN,
            6 => g.max_blocks = u32::MAX,
            7 => g.block_size = 0,
            _ => g.sequences = 65536,
        }
        assert!(paged_decode_attn_bf16_sink(&gpu, KernelHandle(7), b, &g, 0).is_err());
    }
    for i in 0..7 {
        let mut bad = b;
        bad[i] = DevicePtr::NULL;
        assert!(paged_decode_attn_bf16_sink(&gpu, KernelHandle(7), bad, &geometry(), 0).is_err());
    }
    assert!(gpu.launches_snapshot().is_empty());
}
