// SPDX-License-Identifier: MIT OR Apache-2.0
//! 2026-10-07: Explicit GPT-OSS reference policies, not generic MoE defaults.

/// 2026-10-07: Biased linear router logits are selected before softmax.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssRouting {
    TopKLogitsThenSoftmax,
}

/// 2026-10-07: A learned per-head logit contributes to the denominator with zero V.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssAttention {
    BiasedGqaWithDenominatorSink,
}

/// 2026-10-07: Distinct from symmetric-clamp SiLU used by other engine families.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GptOssActivation {
    /// 2026-10-07: gate=min(gate, limit); up=clamp(up,-limit,limit);
    /// output=gate*sigmoid(alpha*gate)*(up+1). Gate/up channels are interleaved.
    InterleavedAsymmetricSwiGlu { alpha: f32 },
}

/// 2026-10-07: Expert down bias is inside each expert's routing-weighted sum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssExpertBias {
    GateUpAndDownBeforeRoutingReduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssNorm {
    Fp32NormalizationAndScaleBeforeCast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssRope {
    HalfSplitYarnWithoutTruncatedCorrectionRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GptOssExpertFormat {
    /// 2026-10-07: Two E2M1 nibbles per byte; one E8M0 scale per 32 values, no global scale.
    Mxfp4E2m1Group32E8m0,
}

/// 2026-10-07: Set only by the validated family parser; input JSON cannot inject it.
/// A loader must implement these semantics before accepting this configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GptOssPolicy {
    pub routing: GptOssRouting,
    pub attention: GptOssAttention,
    pub activation: GptOssActivation,
    pub expert_bias: GptOssExpertBias,
    pub norm: GptOssNorm,
    pub rope: GptOssRope,
    pub expert_format: GptOssExpertFormat,
}
