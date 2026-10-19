// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The comparison and transfer helpers of `real_file_tests`, moved out to keep that
//! file under the length bound.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

pub(super) fn rms(v: &[f32]) -> f32 {
    (v.iter().map(|x| x * x).sum::<f32>() / v.len().max(1) as f32).sqrt()
}

pub(super) fn report(what: &str, got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let max = want.iter().fold(0f32, |a, v| a.max(v.abs()));
    let worst = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    println!(
        "  {what:<28} rms got {:.5} want {:.5}  max|want| {:.4}  worst {:.5}  rel {:.2e}",
        rms(got),
        rms(want),
        max,
        worst,
        worst / max.max(1e-12)
    );
    worst / max.max(1e-12)
}

pub(super) fn dl_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(2)
        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
        .collect()
}

pub(super) fn dl_f32(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<f32> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b).unwrap();
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub(super) fn up_bf16(gpu: &dyn GpuBackend, v: &[f32]) -> DevicePtr {
    let b = crate::moe_v41::bf16_bytes(v);
    let p = gpu.alloc(b.len()).unwrap();
    gpu.copy_h2d(&b, p).unwrap();
    p
}
