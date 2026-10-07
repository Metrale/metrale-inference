// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Optional diagnostic counters; no model computation or default switch.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{gpu::GpuBackend, kernel_args::KernelLaunch};

pub(super) fn read(gpu: &dyn GpuBackend, manifest: &serde_json::Value) -> Result<Option<[u64; 3]>> {
    if manifest["policy_override"]["midpoint_counts"] != true {
        return Ok(None);
    }
    let buffer = gpu.alloc(24)?;
    let result = (|| {
        KernelLaunch::new(
            gpu,
            gpu.kernel("gpt_oss_mxfp4_gemv", "gpt_oss_midpoint_get_counts")?,
        )
        .grid([1, 1, 1])
        .block([1, 1, 1])
        .arg_ptr(buffer)
        .launch(gpu.default_stream())?;
        let mut bytes = [0u8; 24];
        gpu.copy_d2h_on_stream(buffer, &mut bytes, gpu.default_stream())?;
        Ok(Some(std::array::from_fn(|i| {
            u64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap())
        })))
    })();
    gpu.free(buffer)?;
    result
}

pub(super) fn report(
    gpu: &dyn GpuBackend,
    manifest: &serde_json::Value,
    steps: usize,
) -> Result<Option<serde_json::Value>> {
    let Some(counts) = read(gpu, manifest)? else {
        return Ok(None);
    };
    let gate = steps as u64 * 24 * 4 * 5760;
    let down = steps as u64 * 24 * 4 * 2880;
    ensure!(
        counts[2] == 0 && counts[0] <= gate && counts[1] <= down,
        "invalid midpoint counter geometry"
    );
    Ok(Some(
        serde_json::json!({"gate_retries":counts[0],"down_retries":counts[1],"gate_rows":gate,"down_rows":down,"total_fraction":(counts[0]+counts[1]) as f64/(gate+down) as f64,"scope":"diagnostic exact-FP32-midpoint retries; no performance qualification"}),
    ))
}
