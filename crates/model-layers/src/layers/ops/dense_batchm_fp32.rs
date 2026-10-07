// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Bounded scalar-order batch projection with FP32 output.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

fn validate(
    kernel: KernelHandle,
    pointers: [DevicePtr; 3],
    m: u32,
    n: u32,
    k: u32,
    stride: u32,
) -> Result<()> {
    ensure!(
        kernel.0 != 0
            && (1..=16).contains(&m)
            && n > 0
            && n.is_multiple_of(4)
            && k > 0
            && k.is_multiple_of(8)
            && stride >= n,
        "dense batch FP32 invalid kernel/geometry"
    );
    let sizes = [
        u64::from(m) * u64::from(k) * 2,
        u64::from(n)
            .checked_mul(u64::from(k))
            .and_then(|v| v.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("dense batch FP32 size overflow"))?,
        ((u64::from(m) - 1) * u64::from(stride) + u64::from(n)) * 4,
    ];
    let alignment = [16, 16, 4];
    let mut ranges = [(0, 0); 3];
    for i in 0..3 {
        let start = pointers[i].0;
        ensure!(
            start != 0 && start.is_multiple_of(alignment[i]),
            "dense batch FP32 invalid pointer alignment"
        );
        let end = start
            .checked_add(sizes[i])
            .ok_or_else(|| anyhow::anyhow!("dense batch FP32 address overflow"))?;
        ranges[i] = (start, end);
    }
    for i in 0..2 {
        ensure!(
            ranges[i].1 <= ranges[2].0 || ranges[2].1 <= ranges[i].0,
            "dense batch FP32 output aliases input"
        );
    }
    Ok(())
}
/// 2026-10-07: `A[M,K]` and `B[N,K]` BF16 to `C[M,stride]` FP32. No bias or cast.
/// Requires M in1..=16, N divisible by4, K divisible by8, and stride>=N.
/// N%4 refusal avoids the existing partial-CTA barrier path; BF16 admission is unchanged.
/// Exact scalar reduction policy; callers supply buffers of the validated sizes.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemv_batchm_fp32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stride: u32,
    stream: u64,
) -> Result<()> {
    validate(kernel, [input, weight, output], m, n, k, stride)?;
    KernelLaunch::new(gpu, kernel)
        .grid([n / 4, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(stride)
        .launch(stream)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refuses_truncation_partial_barrier_and_alignment() {
        let p = [DevicePtr(0x10000), DevicePtr(0x20000), DevicePtr(0x30000)];
        for m in [1, 2, 8, 16] {
            assert!(validate(KernelHandle(1), p, m, 32, 64, 36).is_ok());
        }
        for (m, n, k, s) in [
            (0, 32, 64, 32),
            (17, 32, 64, 32),
            (2, 31, 64, 32),
            (2, 32, 63, 32),
            (2, 32, 64, 31),
        ] {
            assert!(validate(KernelHandle(1), p, m, n, k, s).is_err());
        }
        assert!(validate(KernelHandle(0), p, 1, 32, 64, 32).is_err());
        assert!(validate(KernelHandle(1), [DevicePtr(3), p[1], p[2]], 1, 32, 64, 32).is_err());
    }
    #[test]
    fn refuses_alias_and_address_wrap() {
        assert!(
            validate(
                KernelHandle(1),
                [DevicePtr(0x10000), DevicePtr(0x20000), DevicePtr(0x30000)],
                1,
                u32::MAX - 3,
                u32::MAX - 7,
                u32::MAX - 3
            )
            .is_err()
        );
        assert!(
            validate(
                KernelHandle(1),
                [DevicePtr(0x10000), DevicePtr(0x20000), DevicePtr(0x10004)],
                2,
                32,
                64,
                32
            )
            .is_err()
        );
        assert!(
            validate(
                KernelHandle(1),
                [
                    DevicePtr(u64::MAX - 15),
                    DevicePtr(0x20000),
                    DevicePtr(0x30000)
                ],
                2,
                32,
                64,
                32
            )
            .is_err()
        );
    }
}
