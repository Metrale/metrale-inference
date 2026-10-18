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

pub use target::{IgnoreDialect, Target};

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

    /// 2026-10-18: MXFP4 as the OCP Microscaling Formats (MX) Specification v1.0 defines
    /// it: E2M1, one power-of-two (E8M0) scale per 32 values along K, no per-tensor scale.
    pub const MXFP4: Operand = Operand {
        kind: NumKind::Float,
        bits: 4,
        granularity: Granularity::Group(32),
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
    /// 2026-10-18: How the HF `fp8` block scales are stored (`scale_fmt`); `None` when the
    /// block does not say, which for that dialect means F32 (`weight_scale_inv`).
    pub scale_fmt: Option<ScaleFmt>,
}

/// 2026-10-18: The HF `fp8` `scale_fmt` values the plan reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFmt {
    /// 2026-10-18: `ue8m0`: each block scale is one unsigned E8M0 byte (a power of two).
    Ue8m0,
}

/// 2026-10-18: The stored element type of a quantized weight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredElement {
    /// 2026-10-18: One E4M3 byte per value.
    E4m3,
    /// 2026-10-18: Two E2M1 values per byte along K (safetensors writes I8 or U8).
    E2m1Pair,
}

/// 2026-10-18: The stored element type of a weight's scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoredScale {
    /// 2026-10-18: One F32 per block.
    F32,
    /// 2026-10-18: One E8M0 byte per block or group.
    E8m0,
}

/// 2026-10-18: The on-disk tensors a declared weight format implies for a `[rows, cols]`
/// (`[out, in]`) linear weight: the weight tensor and its scale tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightStorage {
    /// 2026-10-18: Weight element type.
    pub element: StoredElement,
    /// 2026-10-18: Weight tensor shape.
    pub shape: [u64; 2],
    /// 2026-10-18: Scale element type.
    pub scale: StoredScale,
    /// 2026-10-18: Scale tensor shape.
    pub scale_shape: [u64; 2],
}

impl DeclaredPrecisionPlan {
    /// 2026-09-28: A plan that declares nothing (no `quantization_config`).
    pub const UNDECLARED: DeclaredPrecisionPlan = DeclaredPrecisionPlan {
        source: PlanSource::Undeclared,
        rules: Vec::new(),
        ignore: Vec::new(),
        scale_fmt: None,
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
    ///
    /// 2026-10-17: An expert projection no rule names is declared by its fused-experts module
    /// ([`expert_container`]), as a ModelOpt `quantized_layers` entry naming `...mlp.experts`
    /// covers every projection under it. A projection that matches any rule keeps that answer.
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
        match (best, expert_container(module)) {
            (Some((_, _, p)), _) => p,
            (None, Some(container)) => self.resolve(container),
            (None, None) => LayerPrecision::UNQUANTIZED,
        }
    }

    /// 2026-10-18: The tensors `module`'s declared weight format implies for a `[rows, cols]`
    /// weight; `None` when the module is declared unquantized. An E4M3 block-scaled weight
    /// has one scale per `[r, c]` block (ceil-divided); an MXFP4 weight packs two values per
    /// byte along K and has one E8M0 scale per 32 values of each row. Any other declared
    /// format is an error: its storage is not modelled here.
    pub fn weight_storage(
        &self,
        module: &str,
        rows: u64,
        cols: u64,
    ) -> Result<Option<WeightStorage>> {
        let Some(w) = self.resolve(module).weight else {
            return Ok(None);
        };
        let storage = match w.granularity {
            Granularity::Block(r, c) if w.is_fp8() => WeightStorage {
                element: StoredElement::E4m3,
                shape: [rows, cols],
                scale: match self.scale_fmt {
                    Some(ScaleFmt::Ue8m0) => StoredScale::E8m0,
                    None => StoredScale::F32,
                },
                scale_shape: [rows.div_ceil(u64::from(r)), cols.div_ceil(u64::from(c))],
            },
            Granularity::Group(g) if w == Operand::MXFP4 => {
                anyhow::ensure!(
                    cols.is_multiple_of(2),
                    "{module}: an MXFP4 weight needs an even K, got {cols}"
                );
                WeightStorage {
                    element: StoredElement::E2m1Pair,
                    shape: [rows, cols / 2],
                    scale: StoredScale::E8m0,
                    scale_shape: [rows, cols.div_ceil(u64::from(g))],
                }
            }
            _ => anyhow::bail!("{module}: the storage of a declared {w:?} weight is not modelled"),
        };
        Ok(Some(storage))
    }

    /// 2026-09-28: True when no module declares anything.
    pub fn is_undeclared(&self) -> bool {
        self.source == PlanSource::Undeclared
    }
}

/// 2026-10-02: The fused-experts module an expert projection belongs to: `<p>.experts` for
/// `<p>.experts.<i>.<proj>`. A checkpoint that quantizes the routed experts as one module (a
/// ModelOpt `quantized_layers` entry naming `...mlp.experts`) declares every expert projection
/// through it.
///
/// 2026-10-17: Moved here from the circuit crate so the plan's own resolve uses it.
pub fn expert_container(module: &str) -> Option<&str> {
    let at = module.rfind(".experts.")?;
    let rest = &module[at + ".experts.".len()..];
    let (index, proj) = rest.split_once('.')?;
    (!index.is_empty() && index.bytes().all(|b| b.is_ascii_digit()) && !proj.is_empty())
        .then(|| &module[..at + ".experts".len()])
}

#[cfg(test)]
#[path = "precision_plan_tests.rs"]
mod precision_plan_tests;

#[cfg(test)]
#[path = "precision_plan_storage_tests.rs"]
mod precision_plan_storage_tests;
