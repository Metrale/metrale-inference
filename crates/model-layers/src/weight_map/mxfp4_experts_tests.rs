// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-06: Packed-view geometry tests. Addresses are inert metadata, not GPU results.

use std::collections::HashMap;

use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;

fn store(shape: Vec<usize>, scales: Vec<usize>) -> WeightStore {
    WeightStore::from_map(HashMap::from([
        (
            "blocks".into(),
            WeightTensor {
                ptr: DevicePtr(0x1000),
                shape,
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "scales".into(),
            WeightTensor {
                ptr: DevicePtr(0x10000000),
                shape: scales,
                dtype: WeightDtype::UInt8,
            },
        ),
    ]))
}

#[test]
fn checkpoint_shapes_preserve_first_middle_and_last_expert_offsets() {
    // 2026-10-06: Shapes from pinned GPT-OSS-20B revision 6cee5e81 metadata.
    for n in [5760, 2880] {
        let weights = store(vec![32, n, 90, 16], vec![32, n, 90]);
        let packed = PackedMxfp4Experts::bind(&weights, "blocks", "scales", 32, n, 2880).unwrap();
        for expert in [0, 17, 31] {
            let view = packed.expert(expert).unwrap();
            assert_eq!(
                view.weight(),
                DevicePtr(0x1000 + (expert * n * 1440) as u64)
            );
            assert_eq!(
                view.scales(),
                DevicePtr(0x10000000 + (expert * n * 90) as u64)
            );
            assert_eq!((view.rows(), view.cols()), (n, 2880));
            assert_eq!(view.format(), WeightQuantFormat::Mxfp4E8m0);
            assert_eq!(view.packed_bytes(), n * 1440);
            assert_eq!(view.scale_bytes(), n * 90);
        }
        assert!(packed.expert(32).is_err());
        assert!(packed.expert(usize::MAX).is_err());
    }
}

#[test]
fn refuses_wrong_rank_layout_and_same_element_count_scale_transpose() {
    for (blocks, scales) in [
        (vec![2, 4, 32], vec![2, 4, 2]),
        (vec![2, 4, 2, 16], vec![2, 2, 4]),
        (vec![2, 4, 4, 8], vec![2, 4, 2]),
        (vec![1, 8, 2, 16], vec![2, 4, 2]),
        (vec![2, 4, 2, 16], vec![2, 4, 4]),
    ] {
        let weights = store(blocks, scales);
        assert!(PackedMxfp4Experts::bind(&weights, "blocks", "scales", 2, 4, 64).is_err());
    }
}

#[test]
fn refuses_wrong_dtype_missing_tensors_and_null_addresses() {
    for key in ["blocks", "scales"] {
        for dtype in [
            WeightDtype::FP8E4M3,
            WeightDtype::FP8E8M0,
            WeightDtype::FP32,
            WeightDtype::BF16,
        ] {
            let mut map = HashMap::new();
            for (name, shape, ptr) in [
                ("blocks", vec![2, 4, 2, 16], 0x1000),
                ("scales", vec![2, 4, 2], 0x2000),
            ] {
                map.insert(
                    name.into(),
                    WeightTensor {
                        ptr: DevicePtr(ptr),
                        shape,
                        dtype: if name == key {
                            dtype
                        } else {
                            WeightDtype::UInt8
                        },
                    },
                );
            }
            assert!(
                PackedMxfp4Experts::bind(&WeightStore::from_map(map), "blocks", "scales", 2, 4, 64)
                    .is_err()
            );
        }
    }
    let empty = WeightStore::from_map(HashMap::new());
    assert!(PackedMxfp4Experts::bind(&empty, "blocks", "scales", 2, 4, 64).is_err());
    for null_key in ["blocks", "scales"] {
        let mut map = HashMap::new();
        for (name, shape) in [("blocks", vec![2, 4, 2, 16]), ("scales", vec![2, 4, 2])] {
            map.insert(
                name.into(),
                WeightTensor {
                    ptr: DevicePtr(if name == null_key { 0 } else { 4096 }),
                    shape,
                    dtype: WeightDtype::UInt8,
                },
            );
        }
        assert!(
            PackedMxfp4Experts::bind(&WeightStore::from_map(map), "blocks", "scales", 2, 4, 64)
                .is_err()
        );
    }
}

#[test]
fn refuses_zero_partial_groups_size_overflow_and_address_wrap() {
    let valid = store(vec![2, 4, 2, 16], vec![2, 4, 2]);
    for (e, n, k) in [(0, 4, 64), (2, 0, 64), (2, 4, 0), (2, 4, 63)] {
        assert!(PackedMxfp4Experts::bind(&valid, "blocks", "scales", e, n, k).is_err());
    }
    let huge = store(vec![usize::MAX, 4, 2, 16], vec![usize::MAX, 4, 2]);
    assert!(PackedMxfp4Experts::bind(&huge, "blocks", "scales", usize::MAX, 4, 64).is_err());
    for cols in [32, 64] {
        let huge_rows = store(
            vec![1, usize::MAX, cols / 32, 16],
            vec![1, usize::MAX, cols / 32],
        );
        assert!(
            PackedMxfp4Experts::bind(&huge_rows, "blocks", "scales", 1, usize::MAX, cols).is_err()
        );
    }
    for key in ["blocks", "scales"] {
        let mut map = HashMap::new();
        for (name, shape) in [("blocks", vec![2, 4, 2, 16]), ("scales", vec![2, 4, 2])] {
            map.insert(
                name.into(),
                WeightTensor {
                    ptr: DevicePtr(if name == key { u64::MAX - 2 } else { 4096 }),
                    shape,
                    dtype: WeightDtype::UInt8,
                },
            );
        }
        assert!(
            PackedMxfp4Experts::bind(&WeightStore::from_map(map), "blocks", "scales", 2, 4, 64)
                .is_err()
        );
    }
}

#[test]
fn views_read_exact_expert_bytes_without_dequantizing_or_reordering() {
    use metrale_gpu_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
    let gpu = MockGpuBackend::new();
    let blocks: Vec<u8> = (0..3 * 4 * 32).map(|i| (i % 256) as u8).collect();
    // 2026-10-06: Include extreme E8M0 encodings; binding must not reinterpret them.
    let scales: Vec<u8> = (0..3 * 4 * 2).map(|i| [0, 127, 254, 255][i % 4]).collect();
    let wptr = gpu.alloc(blocks.len()).unwrap();
    let sptr = gpu.alloc(scales.len()).unwrap();
    gpu.copy_h2d(&blocks, wptr).unwrap();
    gpu.copy_h2d(&scales, sptr).unwrap();
    let weights = WeightStore::from_map(HashMap::from([
        (
            "blocks".into(),
            WeightTensor {
                ptr: wptr,
                shape: vec![3, 4, 2, 16],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "scales".into(),
            WeightTensor {
                ptr: sptr,
                shape: vec![3, 4, 2],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]));
    let packed = PackedMxfp4Experts::bind(&weights, "blocks", "scales", 3, 4, 64).unwrap();
    assert_eq!(packed.expert_count(), 3);
    for index in 0..3 {
        let view = packed.expert(index).unwrap();
        let mut actual_blocks = vec![0; view.packed_bytes()];
        let mut actual_scales = vec![0; view.scale_bytes()];
        gpu.copy_d2h(view.weight(), &mut actual_blocks).unwrap();
        gpu.copy_d2h(view.scales(), &mut actual_scales).unwrap();
        assert_eq!(actual_blocks, blocks[index * 128..(index + 1) * 128]);
        assert_eq!(actual_scales, scales[index * 8..(index + 1) * 8]);
    }
}
