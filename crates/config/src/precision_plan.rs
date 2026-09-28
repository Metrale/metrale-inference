// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The declared precision plan: what a checkpoint's `quantization_config`
//! declares, per linear layer, for the weight AND the input activations.
//!
//! This is the one place that reads the precision half of `quantization_config`
//! (compressed-tensors `config_groups`, ModelOpt `quantized_layers` / `quant_algo`, and the
//! HF `fp8` block scheme). Loaders and kernel dispatch ask [`DeclaredPrecisionPlan::resolve`]
//! for a module and run what it says by default. A lever may only go BELOW what the plan
//! declares; running above it is not a silent default either.
//!
//! Owner: config (quantization).
//! Invariants:
//! - Pure: no environment, no I/O. A malformed block is an error, never a guess.
//! - A module the plan does not cover (ignored, unmatched, or no quantization at all)
//!   resolves to [`LayerPrecision::UNQUANTIZED`]: 16-bit weights and activations.

use anyhow::Result;

#[path = "precision_plan/parse.rs"]
mod parse;
#[path = "precision_plan/target.rs"]
mod target;

pub use target::Target;

/// 2026-09-28: Element type of a quantized operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumKind {
    /// 2026-09-28: A float format (E4M3 at 8 bits, E2M1 at 4 bits).
    Float,
    /// 2026-09-28: An integer format.
    Int,
}

/// 2026-09-28: How many values share one scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Granularity {
    /// 2026-09-28: One scale for the whole tensor.
    Tensor,
    /// 2026-09-28: One scale per output channel (weights).
    Channel,
    /// 2026-09-28: One scale per token (activations).
    Token,
    /// 2026-09-28: One scale per `n` consecutive values along K.
    Group(u32),
    /// 2026-09-28: One scale per `n` values plus a per-tensor global scale
    /// (compressed-tensors `tensor_group`, ModelOpt NVFP4).
    TensorGroup(u32),
    /// 2026-09-28: One scale per `[rows, cols]` block (weights).
    Block(u32, u32),
    /// 2026-09-28: Not stated by the checkpoint.
    Unstated,
}

/// 2026-09-28: When an operand's scales are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleTiming {
    /// 2026-09-28: Calibrated and stored in the checkpoint.
    Static,
    /// 2026-09-28: Computed at run time.
    Dynamic,
    /// 2026-09-28: Local scales at run time under a stored global scale
    /// (compressed-tensors `dynamic: "local"`).
    Local,
}

/// 2026-09-28: One quantized operand as declared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    /// 2026-09-28: Float or int.
    pub kind: NumKind,
    /// 2026-09-28: Bits per element.
    pub bits: u8,
    /// 2026-09-28: Scale sharing.
    pub granularity: Granularity,
    /// 2026-09-28: Scale timing.
    pub timing: ScaleTiming,
}

impl Operand {
    /// 2026-09-28: NVFP4 as ModelOpt and compressed-tensors declare it: E2M1, group 16
    /// under a per-tensor scale.
    pub const NVFP4: Operand = Operand {
        kind: NumKind::Float,
        bits: 4,
        granularity: Granularity::TensorGroup(16),
        timing: ScaleTiming::Static,
    };

    /// 2026-09-28: A 4-bit float (NVFP4 or MXFP4).
    pub fn is_fp4(&self) -> bool {
        self.kind == NumKind::Float && self.bits == 4
    }

    /// 2026-09-28: An 8-bit float (E4M3).
    pub fn is_fp8(&self) -> bool {
        self.kind == NumKind::Float && self.bits == 8
    }
}

/// 2026-09-28: What one linear layer declares. `None` means that operand is not quantized
/// (16-bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPrecision {
    /// 2026-09-28: The weight's declared format.
    pub weight: Option<Operand>,
    /// 2026-09-28: The input activations' declared format.
    pub activation: Option<Operand>,
}

impl LayerPrecision {
    /// 2026-09-28: 16-bit weights and activations.
    pub const UNQUANTIZED: LayerPrecision = LayerPrecision {
        weight: None,
        activation: None,
    };

    /// 2026-09-28: The activations are declared 4-bit float (W4A4 on an FP4 weight).
    pub fn activation_is_fp4(&self) -> bool {
        self.activation.is_some_and(|a| a.is_fp4())
    }

    /// 2026-09-28: The activations are declared 8-bit float.
    pub fn activation_is_fp8(&self) -> bool {
        self.activation.is_some_and(|a| a.is_fp8())
    }

    /// 2026-09-28: A short label, e.g. `W4A4`, `W8A8`, `W4A16`, `W16A16`.
    pub fn label(&self) -> String {
        let bits = |o: Option<Operand>| o.map_or(16, |o| o.bits);
        format!("W{}A{}", bits(self.weight), bits(self.activation))
    }
}

/// 2026-09-28: Which `quantization_config` dialect the plan came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlanSource {
    /// 2026-09-28: No precision declared (no block, or a block with no scheme).
    #[default]
    Undeclared,
    /// 2026-09-28: compressed-tensors `config_groups`.
    CompressedTensors,
    /// 2026-09-28: ModelOpt (`quantized_layers`, `config_groups` or a global `quant_algo`).
    ModelOpt,
    /// 2026-09-28: The HF `fp8` method (`weight_block_size`, `activation_scheme`).
    Fp8,
}

/// 2026-09-28: One scheme and the modules it targets.
#[derive(Debug, Clone)]
pub struct Rule {
    /// 2026-09-28: Targets, in the order the checkpoint lists them.
    pub targets: Vec<Target>,
    /// 2026-09-28: What the targeted modules declare.
    pub precision: LayerPrecision,
}

/// 2026-09-28: The per-layer precision a checkpoint declares.
#[derive(Debug, Clone, Default)]
pub struct DeclaredPrecisionPlan {
    /// 2026-09-28: Dialect.
    pub source: PlanSource,
    /// 2026-09-28: Schemes.
    pub rules: Vec<Rule>,
    /// 2026-09-28: Modules left unquantized.
    pub ignore: Vec<Target>,
}

impl DeclaredPrecisionPlan {
    /// 2026-09-28: A plan that declares nothing (no `quantization_config`).
    pub const UNDECLARED: DeclaredPrecisionPlan = DeclaredPrecisionPlan {
        source: PlanSource::Undeclared,
        rules: Vec::new(),
        ignore: Vec::new(),
    };

    /// 2026-09-28: Parse the `quantization_config` object (the value, not the whole
    /// config.json). A ModelOpt `hf_quant_config.json` nested under `"quantization"` is
    /// accepted as is.
    pub fn from_quantization_config(qc: &serde_json::Value) -> Result<Self> {
        parse::plan(qc)
    }

    /// 2026-09-28: What `module` (a checkpoint module path such as
    /// `model.language_model.layers.3.mlp.gate_proj`, no `.weight` suffix) declares.
    ///
    /// Ignored modules are unquantized. Otherwise the compressed-tensors precedence
    /// applies: exact names first, then patterns ordered by their text, then class
    /// targets (`Linear`), and the first match wins.
    pub fn resolve(&self, module: &str) -> LayerPrecision {
        if self.ignore.iter().any(|t| t.matches_name(module)) {
            return LayerPrecision::UNQUANTIZED;
        }
        let mut best: Option<(u8, &str, LayerPrecision)> = None;
        for rule in &self.rules {
            for t in &rule.targets {
                let Some(tier) = t.tier_for(module) else {
                    continue;
                };
                let key = (tier, t.text());
                if best.is_none_or(|(bt, bs, _)| key < (bt, bs)) {
                    best = Some((tier, t.text(), rule.precision));
                }
            }
        }
        best.map_or(LayerPrecision::UNQUANTIZED, |(_, _, p)| p)
    }

    /// 2026-09-28: True when no module declares anything.
    pub fn is_undeclared(&self) -> bool {
        self.source == PlanSource::Undeclared
    }
}

#[cfg(test)]
#[path = "precision_plan_tests.rs"]
mod precision_plan_tests;
