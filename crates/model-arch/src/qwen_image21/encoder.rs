// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Dense BF16 Qwen3-VL text block; one unpadded text-only sequence.
//! No vision/deepstack, tokenizer, cache, final norm or LM head substitution.
use anyhow::{Result, ensure};
use metrale_gpu_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};
use metrale_model_layers::{layers::ops, weight_map::DenseWeight};
use metrale_model_weights::weights::{WeightDtype, WeightTensor};
/// Ordered tensors: input norm,Q,K,V,O,Q norm,K norm,post norm,gate,up,down.
pub struct TextBlockWeights<'a>(pub [&'a WeightTensor; 11]);
pub struct DiagnosticTextBlock<'a> {
    gpu: &'a dyn GpuBackend,
    weights: TextBlockWeights<'a>,
    tokens: u32,
    allocation: DevicePtr,
    residual: DevicePtr,
    normed: DevicePtr,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    attended: DevicePtr,
    branch: DevicePtr,
    temp: DevicePtr,
    gate: DevicePtr,
    up: DevicePtr,
    activated: DevicePtr,
    ones: DevicePtr,
    rms: KernelHandle,
    weight: KernelHandle,
    gemm: KernelHandle,
    rope: KernelHandle,
    attention: KernelHandle,
    activation: KernelHandle,
    add: KernelHandle,
}
impl<'a> DiagnosticTextBlock<'a> {
    pub fn new(
        gpu: &'a dyn GpuBackend,
        tokens: u32,
        weights: TextBlockWeights<'a>,
    ) -> Result<Self> {
        ensure!(
            (1..=4096).contains(&tokens),
            "unsupported encoder sequence length"
        );
        let shapes: [&[usize]; 11] = [
            &[4096],
            &[4096, 4096],
            &[1024, 4096],
            &[1024, 4096],
            &[4096, 4096],
            &[128],
            &[128],
            &[4096],
            &[12288, 4096],
            &[12288, 4096],
            &[4096, 12288],
        ];
        for (w, shape) in weights.0.iter().zip(shapes) {
            ensure!(
                w.dtype == WeightDtype::BF16
                    && w.shape == shape
                    && !w.ptr.is_null()
                    && w.ptr.0.is_multiple_of(16),
                "invalid dense text encoder weight"
            );
        }
        let rms = gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?;
        let weight = gpu.kernel("image_modulation", "image_head_weight_bf16")?;
        let gemm = gpu.kernel("dense_gemm_bf16", "dense_gemm_bf16_pipelined")?;
        let rope = gpu.kernel("image_modulation", "image_text_rope_bf16")?;
        let attention = gpu.kernel("attn_prefill_h128", "attn_prefill_h128")?;
        let activation = gpu.kernel("image_modulation", "image_silu_staged_mul_bf16")?;
        let add = gpu.kernel("residual_add", "bf16_residual_add")?;
        let widths = [
            4096, 4096, 4096, 1024, 1024, 4096, 4096, 4096, 12288, 12288, 12288,
        ];
        let mut offsets = Vec::new();
        let mut size = 0usize;
        for width in widths {
            offsets.push(size);
            size += tokens as usize * width * 2;
        }
        let allocation = gpu.alloc(size + 8192)?;
        let value = Self {
            gpu,
            weights,
            tokens,
            allocation,
            residual: allocation.offset(offsets[0]),
            normed: allocation.offset(offsets[1]),
            q: allocation.offset(offsets[2]),
            k: allocation.offset(offsets[3]),
            v: allocation.offset(offsets[4]),
            attended: allocation.offset(offsets[5]),
            branch: allocation.offset(offsets[6]),
            temp: allocation.offset(offsets[7]),
            gate: allocation.offset(offsets[8]),
            up: allocation.offset(offsets[9]),
            activated: allocation.offset(offsets[10]),
            ones: allocation.offset(size),
            rms,
            weight,
            gemm,
            rope,
            attention,
            activation,
            add,
        };
        gpu.copy_h2d(
            &(0..4096)
                .flat_map(|_| 0x3f80u16.to_le_bytes())
                .collect::<Vec<_>>(),
            value.ones,
        )?;
        Ok(value)
    }
    /// Consumes BF16 `[tokens,4096]` and supplied native BF16 cosine/sine
    /// `[tokens,64]`; returns owned pre-final-norm output. No padding/vision path.
    pub fn forward(
        &mut self,
        input: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            [input, cos, sin]
                .iter()
                .all(|p| !p.is_null() && p.0.is_multiple_of(4)),
            "invalid encoder input pointers"
        );
        if input != self.residual {
            self.gpu
                .copy_d2d_async(input, self.residual, self.tokens as usize * 8192, stream)?;
        }
        self.normalize(self.residual, self.normed, 0, self.tokens, 4096, stream)?;
        self.linear(self.normed, self.q, 1, 4096, 4096, stream)?;
        self.linear(self.normed, self.k, 2, 1024, 4096, stream)?;
        self.linear(self.normed, self.v, 3, 1024, 4096, stream)?;
        self.normalize(self.q, self.q, 5, self.tokens * 32, 128, stream)?;
        self.normalize(self.k, self.k, 6, self.tokens * 8, 128, stream)?;
        for (ptr, heads) in [(self.q, 32), (self.k, 8)] {
            KernelLaunch::new(self.gpu, self.rope)
                .grid([(self.tokens * heads * 64).div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(ptr)
                .arg_ptr(cos)
                .arg_ptr(sin)
                .arg_u32(self.tokens)
                .arg_u32(heads)
                .launch(stream)?;
        }
        KernelLaunch::new(self.gpu, self.attention)
            .grid([32, self.tokens.div_ceil(32), 1])
            .block([128, 1, 1])
            .arg_ptr(self.q)
            .arg_ptr(self.k)
            .arg_ptr(self.v)
            .arg_ptr(self.attended)
            .arg_u32(self.tokens)
            .arg_u32(32)
            .arg_u32(8)
            .arg_u32(128)
            .arg_f32(1.0 / 128.0f32.sqrt())
            .arg_u32(1)
            .arg_u32(0)
            .launch(stream)?;
        self.linear(self.attended, self.branch, 4, 4096, 4096, stream)?;
        ops::residual_add(
            self.gpu,
            self.add,
            self.residual,
            self.branch,
            self.tokens * 4096,
            stream,
        )?;
        self.normalize(self.residual, self.normed, 7, self.tokens, 4096, stream)?;
        self.linear(self.normed, self.gate, 8, 12288, 4096, stream)?;
        self.linear(self.normed, self.up, 9, 12288, 4096, stream)?;
        ops::silu_mul(
            self.gpu,
            self.activation,
            self.gate,
            self.up,
            self.activated,
            self.tokens * 12288,
            stream,
        )?;
        self.linear(self.activated, self.branch, 10, 4096, 12288, stream)?;
        ops::residual_add(
            self.gpu,
            self.add,
            self.residual,
            self.branch,
            self.tokens * 4096,
            stream,
        )?;
        Ok(self.residual)
    }
    #[allow(clippy::too_many_arguments)]
    fn normalize(
        &self,
        input: DevicePtr,
        out: DevicePtr,
        index: usize,
        rows: u32,
        cols: u32,
        stream: u64,
    ) -> Result<()> {
        ops::rms_norm(
            self.gpu,
            self.rms,
            input,
            &DenseWeight { weight: self.ones },
            self.temp,
            rows,
            cols,
            1e-6,
            stream,
        )?;
        KernelLaunch::new(self.gpu, self.weight)
            .grid([(rows * cols).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.temp)
            .arg_ptr(self.weights.0[index].ptr)
            .arg_ptr(out)
            .arg_u32(rows)
            .arg_u32(cols)
            .launch(stream)
    }
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        input: DevicePtr,
        out: DevicePtr,
        index: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::dense_gemm_bf16_pipelined(
            self.gpu,
            self.gemm,
            input,
            &DenseWeight {
                weight: self.weights.0[index].ptr,
            },
            out,
            self.tokens,
            n,
            k,
            stream,
        )
    }
}
impl Drop for DiagnosticTextBlock<'_> {
    fn drop(&mut self) {
        let _ = self.gpu.free(self.allocation);
    }
}

/// Text-only positions share all three MRoPE axes. Native FP32 host arithmetic
/// remains independently testable; it is not assumed bit-identical to Torch pow.
pub fn text_rope_coefficients(tokens: u32) -> Result<(Vec<half::bf16>, Vec<half::bf16>)> {
    ensure!(
        (1..=4096).contains(&tokens),
        "unsupported encoder positions"
    );
    let frequencies: Vec<f32> = (0..64)
        .map(|i| 1.0 / 5_000_000.0f32.powf((2 * i) as f32 / 128.0))
        .collect();
    let mut cosines = Vec::with_capacity(tokens as usize * 64);
    let mut sines = Vec::with_capacity(tokens as usize * 64);
    for position in 0..tokens {
        for frequency in &frequencies {
            let angle = position as f32 * frequency;
            cosines.push(half::bf16::from_f32(angle.cos()));
            sines.push(half::bf16::from_f32(angle.sin()));
        }
    }
    Ok((cosines, sines))
}
