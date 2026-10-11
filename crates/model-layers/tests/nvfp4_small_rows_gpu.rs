// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-07: Constructed exact dyadic-dot oracle, legacy parity and wrong-grid control.
//! Run with a built GB10 Laguna target; no checkpoint or learned data is required.

#![cfg(feature = "cuda")]

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::{Nvfp4SmallRowKernels, moe_w4a16_grouped_gemm_ptrtable};

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> DevicePtr {
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, ptr).unwrap();
    ptr
}

fn download(gpu: &dyn GpuBackend, ptr: DevicePtr, elements: usize) -> Vec<u16> {
    let mut bytes = vec![0; elements * 2];
    gpu.copy_d2h(ptr, &mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
        .collect()
}

#[test]
#[ignore = "requires the built GB10 Laguna CUDA target"]
fn small_rows_preserve_legacy_and_exact_dot_through_934_rows() {
    let set = metrale_kernels::all_ptx_sets()
        .into_iter()
        .find(|x| x.target.model == "laguna-xs-2.1" && x.ptx_arch.starts_with("sm_121"))
        .expect("build gb10/laguna-xs-2.1/nvfp4 first");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules).unwrap();
    for k in [32, 256] {
        check(&gpu, k);
    }
}

fn check(gpu: &dyn GpuBackend, k: usize) {
    let pair = Nvfp4SmallRowKernels::resolve(gpu).unwrap();
    let legacy = gpu
        .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable")
        .unwrap();
    let small = gpu
        .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_small16")
        .unwrap();
    let large = gpu
        .kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_large64")
        .unwrap();
    let counts = [0usize, 1, 15, 16, 17, 63, 64, 65, 934];
    let n = 129usize;
    let total: usize = counts.iter().sum();
    let mut offsets = vec![0u32];
    for rows in counts {
        offsets.push(offsets.last().unwrap() + rows as u32);
    }
    let offsets_dev = upload(
        gpu,
        &offsets
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let sorted: Vec<u32> = (0..total as u32).rev().collect();
    let sorted_dev = upload(
        gpu,
        &sorted
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let a: Vec<f32> = (0..total * k)
        .map(|i| ((i * 37 % 13) as f32 - 6.0) / 8.0)
        .collect();
    let a_dev = upload(
        gpu,
        &a.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let mut packed = Vec::new();
    let mut scale_ptrs = Vec::new();
    let mut weight_ptrs = Vec::new();
    for expert in 0..counts.len() {
        let bytes: Vec<u8> = (0..n * k / 2)
            .map(|i| (i * 29 + expert * 71) as u8)
            .collect();
        weight_ptrs.push(upload(gpu, &bytes).0);
        scale_ptrs.push(upload(gpu, &vec![0x38; n * k / 16]).0);
        packed.push(bytes);
    }
    let weight_dev = upload(
        gpu,
        &weight_ptrs
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let scales_dev = upload(
        gpu,
        &scale_ptrs
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let scale2: Vec<f32> = (0..counts.len()).map(|e| [1.0, 0.5, 2.0][e % 3]).collect();
    let scale2_dev = upload(
        gpu,
        &scale2
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    );
    let lut = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ];
    let mut expected = vec![0u16; total * n];
    for e in 0..counts.len() {
        for r in offsets[e] as usize..offsets[e + 1] as usize {
            for col in 0..n {
                let mut sum = 0.0f64;
                for j in 0..k {
                    let byte = packed[e][col * k / 2 + j / 2];
                    let code = if j.is_multiple_of(2) {
                        byte & 15
                    } else {
                        byte >> 4
                    };
                    sum += f64::from(a[sorted[r] as usize * k + j])
                        * f64::from(lut[code as usize] * scale2[e]);
                }
                expected[r * n + col] = half::bf16::from_f32(sum as f32).to_bits();
            }
        }
    }
    // 2026-10-07: Exact dyadic sums fit FP32, independently fixing the final BF16 result.
    let sentinel = 0x7fc1u16;
    for mode in 0..4 {
        let bytes: Vec<u8> = (0..total * n + 64)
            .flat_map(|_| sentinel.to_le_bytes())
            .collect();
        let out = upload(gpu, &bytes);
        let args = (counts.len() as u32, n as u32, k as u32);
        match mode {
            0 => moe_w4a16_grouped_gemm_ptrtable(
                gpu,
                legacy,
                a_dev,
                weight_dev,
                scales_dev,
                scale2_dev,
                out,
                offsets_dev,
                sorted_dev,
                args.0,
                args.1,
                args.2,
                15,
                0,
            )
            .unwrap(),
            1 | 3 => pair
                .launch(
                    gpu,
                    a_dev,
                    weight_dev,
                    scales_dev,
                    scale2_dev,
                    out,
                    offsets_dev,
                    sorted_dev,
                    args.0,
                    args.1,
                    args.2,
                    if mode == 3 { 1 } else { 15 },
                    0,
                )
                .unwrap(),
            2 => {
                moe_w4a16_grouped_gemm_ptrtable(
                    gpu,
                    large,
                    a_dev,
                    weight_dev,
                    scales_dev,
                    scale2_dev,
                    out,
                    offsets_dev,
                    sorted_dev,
                    args.0,
                    args.1,
                    args.2,
                    15,
                    0,
                )
                .unwrap();
                moe_w4a16_grouped_gemm_ptrtable(
                    gpu,
                    small,
                    a_dev,
                    weight_dev,
                    scales_dev,
                    scale2_dev,
                    out,
                    offsets_dev,
                    sorted_dev,
                    args.0,
                    args.1,
                    args.2,
                    1,
                    0,
                )
                .unwrap();
            }
            _ => unreachable!(),
        }
        gpu.synchronize(0).unwrap();
        let result = download(gpu, out, total * n + 64);
        assert!(
            result[total * n..].iter().all(|x| *x == sentinel),
            "tail overwrite"
        );
        if mode == 3 {
            assert_ne!(&result[..total * n], expected, "wrong grid must fail");
        } else {
            assert_eq!(&result[..total * n], expected, "mode={mode}, k={k}");
        }
    }
}
