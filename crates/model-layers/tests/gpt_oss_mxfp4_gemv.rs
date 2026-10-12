// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Packed-view launcher ABI and refusal tests, not numeric GPU results.
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};
pub use metrale_model_layers::weight_map;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;
#[path = "../src/layers/ops/gpt_oss_mxfp4_gemv.rs"]
mod launch;

fn store() -> WeightStore {
    WeightStore::from_map(HashMap::from([
        (
            "w".into(),
            WeightTensor {
                ptr: DevicePtr(0x1000),
                shape: vec![2, 35, 3, 16],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "s".into(),
            WeightTensor {
                ptr: DevicePtr(0x5000),
                shape: vec![2, 35, 3],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]))
}
#[test]
fn expert_pointer_offsets_and_tail_rows_reach_exact_abi() {
    let weights = store();
    let view = weight_map::PackedMxfp4Experts::bind(&weights, "w", "s", 2, 35, 96).unwrap();
    let gpu = MockGpuBackend::new();
    launch::gpt_oss_mxfp4_gemv_bf16(
        &gpu,
        KernelHandle(7),
        DevicePtr(0x8000),
        &view.expert(1).unwrap(),
        DevicePtr(0x9000),
        19,
    )
    .unwrap();
    let calls = gpu.launches_snapshot();
    let call = &calls[0];
    assert_eq!(call.grid, [9, 1, 1]);
    assert_eq!(call.block, [128, 1, 1]);
    assert_eq!(call.stream, 19);
    assert_eq!(
        call.args,
        vec![
            MockArg::Buffer(DevicePtr(0x1000 + 35 * 48)),
            MockArg::Buffer(DevicePtr(0x5000 + 35 * 3)),
            MockArg::Buffer(DevicePtr(0x8000)),
            MockArg::Buffer(DevicePtr(0x9000)),
            MockArg::Bytes(35u32.to_le_bytes().to_vec()),
            MockArg::Bytes(96u32.to_le_bytes().to_vec())
        ]
    );
}
#[test]
fn null_misaligned_wrapping_or_aliasing_buffers_never_launch() {
    let weights = store();
    let views = weight_map::PackedMxfp4Experts::bind(&weights, "w", "s", 2, 35, 96).unwrap();
    let view = views.expert(0).unwrap();
    let gpu = MockGpuBackend::new();
    for (kernel, input, output) in [
        (0, 0x8000, 0x9000),
        (7, 0, 0x9000),
        (7, 0x8001, 0x9000),
        (7, u64::MAX - 1, 0x9000),
        (7, 0x8000, 0),
        (7, 0x8000, 0x9001),
        (7, 0x8000, u64::MAX - 1),
        (7, 0x8000, 0x1000),
        (7, 0x8000, 0x5000),
        (7, 0x8000, 0x8000),
    ] {
        assert!(
            launch::gpt_oss_mxfp4_gemv_bf16(
                &gpu,
                KernelHandle(kernel),
                DevicePtr(input),
                &view,
                DevicePtr(output),
                0
            )
            .is_err()
        );
    }
    assert!(gpu.launches_snapshot().is_empty());
}
