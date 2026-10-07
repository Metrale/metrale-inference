// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Scale-only image modulation and staged BF16 tanh residual.
//! Norm/projection execution is separate. These operators do not register a model.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

/// Validated row-selection and parameter-stride contract. Upload `selected_rows`
/// unchanged to a U32 device buffer before executing either operator.
pub struct ImageModulationLayout {
    selected_rows: Vec<u32>,
    width: u32,
    stride: u32,
    offset: u32,
}
impl ImageModulationLayout {
    /// Qwen's mask is shared across samples. Target tokens use their sample's
    /// timestep row; other tokens use the trailing t=0 row. None uses each sample.
    pub fn new(
        samples: u32,
        tokens: u32,
        width: u32,
        components: u32,
        component: u32,
        target_mask: Option<&[bool]>,
    ) -> Result<Self> {
        ensure!(
            samples > 0 && tokens > 0 && width > 0,
            "empty image modulation shape"
        );
        ensure!(
            component < components,
            "modulation component outside parameter row"
        );
        ensure!(
            target_mask.is_none_or(|m| m.len() == tokens as usize),
            "modulation mask token count"
        );
        let rows = samples
            .checked_mul(tokens)
            .ok_or_else(|| anyhow::anyhow!("row count overflow"))?;
        let stride = width
            .checked_mul(components)
            .ok_or_else(|| anyhow::anyhow!("parameter stride overflow"))?;
        let elements = u64::from(rows) * u64::from(width);
        ensure!(
            elements.div_ceil(256) <= i32::MAX as u64,
            "launch grid overflow"
        );
        let mut selected_rows = Vec::new();
        selected_rows.try_reserve_exact(rows as usize)?;
        for sample in 0..samples {
            for token in 0..tokens {
                selected_rows.push(if target_mask.is_some_and(|m| !m[token as usize]) {
                    samples
                } else {
                    sample
                });
            }
        }
        Ok(Self {
            selected_rows,
            width,
            stride,
            offset: width * component,
        })
    }
    /// Host row map in sample-major, token-major order; no device I/O here.
    pub fn selected_rows(&self) -> &[u32] {
        &self.selected_rows
    }
    fn validate(&self, pointers: &[DevicePtr], selected: DevicePtr) -> Result<()> {
        ensure!(
            pointers
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(2)),
            "null/misaligned BF16 buffer"
        );
        ensure!(
            !selected.is_null() && selected.0.is_multiple_of(4),
            "null/misaligned row-selection buffer"
        );
        Ok(())
    }
    fn grid(&self) -> u32 {
        (self.selected_rows.len() as u64 * u64::from(self.width)).div_ceil(256) as u32
    }
}

/// `buffers` = normalized, parameters, uploaded U32 selection, output. All BF16
/// buffers must cover the declared shape. Output may alias normalized only.
pub fn image_modulation_scale_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    layout: &ImageModulationLayout,
    buffers: [DevicePtr; 4],
    stream: u64,
) -> Result<()> {
    let [normalized, params, selected, out] = buffers;
    layout.validate(&[normalized, params, out], selected)?;
    KernelLaunch::new(gpu, kernel)
        .grid([layout.grid(), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(normalized)
        .arg_ptr(params)
        .arg_ptr(selected)
        .arg_ptr(out)
        .arg_u32(layout.selected_rows.len() as u32)
        .arg_u32(layout.width)
        .arg_u32(layout.stride)
        .arg_u32(layout.offset)
        .launch(stream)
}
/// `buffers` = hidden, branch, parameters, uploaded U32 selection, output.
/// Output may alias hidden or branch; parameter/selection storage stays immutable.
pub fn image_modulation_residual_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    layout: &ImageModulationLayout,
    buffers: [DevicePtr; 5],
    stream: u64,
) -> Result<()> {
    let [hidden, branch, params, selected, out] = buffers;
    layout.validate(&[hidden, branch, params, out], selected)?;
    KernelLaunch::new(gpu, kernel)
        .grid([layout.grid(), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(branch)
        .arg_ptr(params)
        .arg_ptr(selected)
        .arg_ptr(out)
        .arg_u32(layout.selected_rows.len() as u32)
        .arg_u32(layout.width)
        .arg_u32(layout.stride)
        .arg_u32(layout.offset)
        .launch(stream)
}

/// 2026-10-07: Preserve the BF16 intermediate before applying a shared per-head
/// norm weight. `buffers` = normalized rows, BF16 weight vector, output. Output
/// may alias normalized. Width 128 is the pinned Qwen image head dimension.
pub fn image_head_weight_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: [DevicePtr; 3],
    rows: u32,
    stream: u64,
) -> Result<()> {
    let [normalized, weight, out] = buffers;
    ensure!(
        rows > 0 && rows <= i32::MAX as u32,
        "invalid image head row count"
    );
    ensure!(
        buffers
            .iter()
            .all(|p| !p.is_null() && p.0.is_multiple_of(2)),
        "invalid image head buffer"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([(u64::from(rows) * 128).div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(normalized)
        .arg_ptr(weight)
        .arg_ptr(out)
        .arg_u32(rows)
        .arg_u32(128)
        .launch(stream)
}

/// 2026-10-07: Complex-pair rotation for 32 heads of width128; cis is FP32
/// `[sequence,64,2]`, shared by samples. `buffers` = input, cis, output.
pub fn image_rope_complex_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: [DevicePtr; 3],
    samples: u32,
    sequence: u32,
    stream: u64,
) -> Result<()> {
    let [input, cis, out] = buffers;
    ensure!(samples > 0 && sequence > 0, "empty image rotary shape");
    let pairs = u64::from(samples)
        .checked_mul(u64::from(sequence))
        .and_then(|n| n.checked_mul(32 * 64))
        .ok_or_else(|| anyhow::anyhow!("image rotary size overflow"))?;
    ensure!(
        pairs.div_ceil(256) <= i32::MAX as u64,
        "image rotary grid overflow"
    );
    ensure!(
        !input.is_null()
            && input.0.is_multiple_of(2)
            && !out.is_null()
            && out.0.is_multiple_of(2)
            && !cis.is_null()
            && cis.0.is_multiple_of(4),
        "invalid image rotary pointers"
    );
    KernelLaunch::new(gpu, kernel)
        .grid([pairs.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(cis)
        .arg_ptr(out)
        .arg_u32(samples)
        .arg_u32(sequence)
        .arg_u32(32)
        .launch(stream)
}
