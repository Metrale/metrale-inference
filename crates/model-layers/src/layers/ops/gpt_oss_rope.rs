// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Nontruncated YaRN parameters and staged BF16 half-split rotation.
//! Named residual: `yarn_half_split_bf16_staged`; existing RoPE policies unchanged.
use anyhow::{Result, ensure};
use metrale_config::{GptOssRope, ModelConfig};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-07: Explicit scalar inputs to the host frequency table builder.
#[derive(Debug, Clone, Copy)]
pub struct GptOssYarn {
    pub head_dim: u32,
    pub base: f32,
    pub factor: f32,
    pub low: f32,
    pub span: f32,
    pub attention_factor: f32,
}
impl GptOssYarn {
    /// 2026-10-07: Transformers v4.55.0 continuous correction bounds use f64
    /// scalar logarithms, then f32 tensor operations. No floor/ceil substitution.
    pub fn from_config(c: &ModelConfig) -> Result<Self> {
        ensure!(
            c.model_type == "gpt_oss"
                && c.gpt_oss.is_some_and(
                    |p| p.rope == GptOssRope::HalfSplitYarnWithoutTruncatedCorrectionRange
                ),
            "GPT YaRN requires explicit family policy"
        );
        ensure!(c.head_dim == 64, "GPT YaRN currently requires head_dim64");
        ensure!(
            c.rope_theta.is_finite() && c.rope_theta > 1.0 && (c.rope_theta as f32).is_finite(),
            "GPT YaRN invalid base"
        );
        ensure!(
            c.yarn_beta_fast.is_finite()
                && c.yarn_beta_slow.is_finite()
                && c.yarn_beta_fast > 0.0
                && c.yarn_beta_slow > 0.0,
            "GPT YaRN invalid beta"
        );
        ensure!(
            c.yarn_original_max_position_embeddings > 0
                && c.max_position_embeddings > 0
                && c.yarn_factor.is_finite()
                && c.yarn_factor > 0.0,
            "GPT YaRN invalid context/factor"
        );
        let factor =
            c.max_position_embeddings as f64 / c.yarn_original_max_position_embeddings as f64;
        ensure!(
            factor as f32 == c.yarn_factor,
            "GPT YaRN context ratio differs from factor"
        );
        let attention_factor = if factor <= 1.0 {
            1.0
        } else {
            1.0 + 0.1 * factor.ln()
        };
        ensure!(
            attention_factor as f32 == c.yarn_attention_factor,
            "GPT YaRN attention factor differs from pinned policy"
        );
        let correction = |rot: f32| {
            64.0 * (c.yarn_original_max_position_embeddings as f64
                / (f64::from(rot) * 2.0 * std::f64::consts::PI))
                .ln()
                / (2.0 * c.rope_theta.ln())
        };
        let low = correction(c.yarn_beta_fast).max(0.0);
        let mut high = correction(c.yarn_beta_slow).min(63.0);
        if high == low {
            high += 0.001;
        }
        ensure!(high > low, "GPT YaRN invalid correction range");
        Ok(Self {
            head_dim: 64,
            base: c.rope_theta as f32,
            factor: factor as f32,
            low: low as f32,
            span: (high - low) as f32,
            attention_factor: attention_factor as f32,
        })
    }
    fn validate(&self) -> Result<()> {
        ensure!(
            self.head_dim == 64
                && self.base.is_finite()
                && self.base > 1.0
                && self.factor.is_finite()
                && self.factor > 0.0
                && self.low.is_finite()
                && self.span.is_finite()
                && self.span > 0.0
                && self.attention_factor.is_finite()
                && self.attention_factor > 0.0,
            "GPT YaRN invalid parameters"
        );
        Ok(())
    }
}

/// 2026-10-07: Host-built table of the 32 FP32 YaRN inverse frequencies.
///
/// Every device uploads these exact bits; no device math library is involved.
/// The steps follow the Transformers v4.55 `_compute_yarn_parameters` FP32
/// tensor order, with the same FP32 `low`/`span` scalars the ramp sees there:
///
/// ```text
/// pos[i]   = base ^ (2i / 64)
/// extra[i] = 1 - clamp((i - low) / span, 0, 1)
/// out[i]   = 1/(factor*pos[i]) * (1 - extra[i]) + (1/pos[i]) * extra[i]
/// ```
///
/// The power is the only transcendental step. It is evaluated in f64 and
/// rounded once to f32, which yields the correctly rounded FP32 power unless
/// the exact value lies within about 2^-52 (relative) of an FP32 rounding
/// midpoint; the GPT-OSS-20B entries were checked against a 200-bit
/// reference and are pinned bit for bit in `tests/gpt_oss_rope.rs`. All other
/// steps are IEEE FP32 add/sub/mul/div, correctly rounded by definition, and
/// Rust never contracts them into FMAs. A device `powf` is only accurate to a
/// few ulps: on gfx1151 HIP it differed from this table at 6/32 entries, and
/// the GB10 CUDA kernel (equal to the Torch CUDA reference) at 2/32.
///
/// Usage: `let table = gpt_oss_yarn_frequency_table(&GptOssYarn::from_config(&cfg)?)?;`
/// then copy `table` (little-endian, 128 bytes) to the frequency buffer.
pub fn gpt_oss_yarn_frequency_table(p: &GptOssYarn) -> Result<[f32; 32]> {
    p.validate()?;
    let mut out = [0.0f32; 32];
    for (i, slot) in out.iter_mut().enumerate() {
        // 2026-10-07: Exponent 2i/64 is exact in FP32 and f64 alike.
        let pos = f64::from(p.base).powf(f64::from(2 * i as u32) / 64.0) as f32;
        let extrapolation = 1.0 / pos;
        let interpolation = 1.0 / (p.factor * pos);
        let ramp = ((i as f32 - p.low) / p.span).clamp(0.0, 1.0);
        let extra = 1.0 - ramp;
        *slot = interpolation * (1.0 - extra) + extrapolation * extra;
    }
    ensure!(
        out.iter().all(|f| f.is_finite() && *f > 0.0),
        "GPT YaRN nonfinite or nonpositive frequency"
    );
    Ok(out)
}
/// 2026-10-07: In-place packed BF16 Q/K, token positions U32 and frequency table F32.
/// Callers own sufficient nonoverlapping buffers; positions must be within context.
pub fn gpt_oss_rope_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    buffers: [DevicePtr; 4],
    rows: u32,
    q_heads: u32,
    kv_heads: u32,
    p: &GptOssYarn,
    stream: u64,
) -> Result<()> {
    p.validate()?;
    ensure!(
        kernel.0 != 0
            && rows > 0
            && q_heads > 0
            && kv_heads > 0
            && q_heads.is_multiple_of(kv_heads),
        "GPT RoPE invalid launch"
    );
    let heads = q_heads
        .checked_add(kv_heads)
        .ok_or_else(|| anyhow::anyhow!("GPT RoPE head overflow"))?;
    let pairs = rows
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(32))
        .ok_or_else(|| anyhow::anyhow!("GPT RoPE size overflow"))?;
    let extents = [
        u64::from(rows) * u64::from(q_heads) * 128,
        u64::from(rows) * u64::from(kv_heads) * 128,
        u64::from(rows) * 4,
        128,
    ];
    for (i, pointer) in buffers.iter().enumerate() {
        let align = if i < 2 { 2 } else { 4 };
        ensure!(
            pointer.0 != 0
                && pointer.0.is_multiple_of(align)
                && pointer.0.checked_add(extents[i]).is_some(),
            "GPT RoPE null/misaligned buffer {i}"
        );
    }
    KernelLaunch::new(gpu, kernel)
        .grid([pairs.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(buffers[0])
        .arg_ptr(buffers[1])
        .arg_ptr(buffers[2])
        .arg_ptr(buffers[3])
        .arg_u32(rows)
        .arg_u32(q_heads)
        .arg_u32(kv_heads)
        .arg_f32(p.attention_factor)
        .launch(stream)
}
