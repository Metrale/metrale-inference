// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Decode-policy LM-head replay from saved post-layer traces; never a timed path.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::{layers::ops, weight_map::DenseWeight};
pub fn logits(
    gpu: &dyn GpuBackend,
    trace: &[u8],
    norm: DevicePtr,
    head: DevicePtr,
    vocab: usize,
    eps: f32,
    stream: u64,
) -> Result<Vec<u8>> {
    ensure!(
        trace.len().is_multiple_of(24 * 5760) && vocab <= 201088 && vocab > 0,
        "trace/head geometry"
    );
    let norm_kernel = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
    let head_kernel = gpu.kernel("gemv", "dense_gemv_bf16")?;
    let size = 11520 + vocab * 2;
    ensure!(
        gpu.device_free_memory()? >= size + gpu.total_memory()?.div_ceil(100) * 15,
        "diagnostic head memory limit"
    );
    let allocation = gpu.alloc(size)?;
    let result = (|| {
        let input = allocation;
        let normalized = allocation.offset(5760);
        let output = allocation.offset(11520);
        let mut all = Vec::with_capacity(trace.len() / (24 * 5760) * vocab * 2);
        for token in trace.chunks_exact(24 * 5760) {
            gpu.copy_h2d_async(&token[23 * 5760..], input, stream)?;
            ops::rms_norm(
                gpu,
                norm_kernel,
                input,
                &DenseWeight { weight: norm },
                normalized,
                1,
                2880,
                eps,
                stream,
            )?;
            ops::dense_gemv(
                gpu,
                head_kernel,
                normalized,
                &DenseWeight { weight: head },
                output,
                vocab as u32,
                2880,
                stream,
            )?;
            let mut row = vec![0; vocab * 2];
            gpu.copy_d2h_on_stream(output, &mut row, stream)?;
            ensure!(
                row.chunks_exact(2)
                    .all(|x| u16::from_le_bytes([x[0], x[1]]) & 0x7f80 != 0x7f80),
                "nonfinite diagnostic logits"
            );
            all.extend(row);
        }
        Ok(all)
    })();
    gpu.synchronize(stream)?;
    gpu.free(allocation)?;
    result
}
