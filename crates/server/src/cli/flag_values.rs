// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The closed value sets of the enumerated `met serve` string
//! flags, and the parsers for `--kv-high-precision-layers` and the tristate
//! flags.
//!
//! `validate_serve_args` refuses a value outside a set (`check_enum`), and
//! [`options_for_flag`] offers the sets to the TUI option picker and the CLI
//! manifest, so what is offered and what is accepted come from one list.
//!
//! `--kv-cache-dtype` has no const here: its list is
//! `metrale_cache::kv_cache::KvCacheDtype::ALL`.
//!
//! The sets are not wired into clap as `PossibleValuesParser`s:
//! `validate_serve_args` reports every violation at once, where clap exits on
//! the first.
//!
//! Owner: server CLI (`met serve`).
//! Invariants: none beyond the types.

use metrale_config::WeightQuantization;
use metrale_model_layers::layers::{DenseQuantization, ExpertQuantization};

/// 2026-09-26: What `--kv-high-precision-layers auto` resolves to. The flag's
/// help text states the same number as "recommended".
pub(crate) const AUTO_KV_HIGH_PRECISION_LAYERS: usize = 2;

/// 2026-09-26: The accepted forms of `--kv-high-precision-layers`: a count or
/// a keyword, so not a `check_enum` list. `validate_serve_args` and
/// `serve_phases::kv_cache` both parse through this type, so they accept the
/// same forms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvHighPrecisionLayers {
    /// 2026-09-26: `max` / `all`: every attention layer stays BF16.
    All,
    /// 2026-09-26: `auto`: [`AUTO_KV_HIGH_PRECISION_LAYERS`].
    Auto,
    /// 2026-09-26: An explicit first-N-and-last-N count. `0`, the flag's
    /// default, defers to `auto_high_precision_layers` for the KV dtype.
    Count(usize),
}

impl std::str::FromStr for KvHighPrecisionLayers {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, String> {
        match raw.trim().to_lowercase().as_str() {
            "max" | "all" => Ok(Self::All),
            "auto" => Ok(Self::Auto),
            s => s.parse::<usize>().map(Self::Count).map_err(|_| {
                "expected a whole number of layers, or one of: auto, max, all".to_string()
            }),
        }
    }
}

impl KvHighPrecisionLayers {
    pub(crate) fn resolve(self, num_attn_layers: usize) -> usize {
        match self {
            Self::All => num_attn_layers,
            Self::Auto => AUTO_KV_HIGH_PRECISION_LAYERS,
            Self::Count(n) => n,
        }
    }
}

/// 2026-09-26: The value set of a lever whose default is decided off the
/// command line and which the command line may pin either way. `auto` pins
/// nothing. A presence flag could move it in one direction only.
pub(crate) const TRISTATES: &[&str] = &["auto", "on", "off"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tristate {
    Auto,
    On,
    Off,
}

impl std::str::FromStr for Tristate {
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, String> {
        match raw {
            "auto" => Ok(Self::Auto),
            "on" => Ok(Self::On),
            "off" => Ok(Self::Off),
            other => Err(format!("expected one of auto, on, off; got {other:?}")),
        }
    }
}

impl Tristate {
    /// 2026-09-26: Parse a value `validate_serve_args` has already accepted;
    /// panics on any other.
    pub(crate) fn validated(raw: &str) -> Self {
        raw.parse().expect("validated by validate_serve_args")
    }

    /// 2026-09-26: What the command line pins: `None` for `auto`, which leaves
    /// the lever's own default in charge.
    pub(crate) fn pinned(self) -> Option<bool> {
        match self {
            Self::Auto => None,
            Self::On => Some(true),
            Self::Off => Some(false),
        }
    }
}

/// 2026-10-08: `--kv-cache-dtype` parser: the canonical name for a value that has a
/// second spelling, so every consumer of the flag sees one spelling. `fp8_e4m3` is the
/// format `fp8` stores (E4M3, `metrale_cache::kv_cache::KvCacheDtype::Fp8`). Any other
/// value passes through unchanged and is checked by `validate_serve_args`.
pub(crate) fn canonical_kv_cache_dtype(raw: &str) -> Result<String, String> {
    let canonical = match raw {
        "fp8_e4m3" => "fp8",
        other => other,
    };
    Ok(canonical.to_string())
}

pub(crate) const LM_HEAD_DTYPES: &[&str] = &["default", "bf16", "nvfp4", "fp8"];
pub(crate) const MTP_QUANTS: &[&str] = &["bf16", "fp8", "nvfp4"];
pub(crate) const SCHEDULERS: &[&str] = &["fifo", "slai"];
/// 2026-09-26: The device routers: `sync` settles every device step before
/// the next host decision; `async` lets one plain decode step run ahead of the
/// host, and falls back to `sync` with a warning at startup when the router
/// cannot be built, as on a model without a device token feed.
pub(crate) const SCHEDULER_CONFIGS: &[&str] = &["sync", "async"];
/// 2026-09-26: `--telemetry`: the telemetry crate's own level names.
pub(crate) const TELEMETRY_LEVELS: &[&str] = &metrale_telemetry::Level::NAMES;
pub(crate) const SSM_H_DTYPES: &[&str] = &["f32", "f16", "f16-pool"];
pub(crate) const MTP_GATES: &[&str] = &["auto", "force"];
pub(crate) const TOOL_CALL_PARSERS: &[&str] = &[
    "hermes",
    "qwen3_coder",
    "qwen3_xml",
    "gemma4",
    "mistral",
    "minimax_xml",
    "bare_json",
    "poolside_v1",
    "glm47",
];

/// 2026-09-27: `--expert-quantization`: a clap value enum over the model layer's tiers
/// (`metrale_model_layers::layers::ExpertQuantization`), which own the names. Unlike the string
/// sets above, clap parses and refuses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertQuantizationArg(pub ExpertQuantization);

impl clap::ValueEnum for ExpertQuantizationArg {
    fn value_variants<'a>() -> &'a [Self] {
        const VARIANTS: [ExpertQuantizationArg; 3] = [
            ExpertQuantizationArg(ExpertQuantization::ALL[0]),
            ExpertQuantizationArg(ExpertQuantization::ALL[1]),
            ExpertQuantizationArg(ExpertQuantization::ALL[2]),
        ];
        &VARIANTS
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        // 2026-09-27: GB10, Qwen3.6-35B-A3B-FP8, canonical tiers, `--mtp-gate force`; the
        // same numbers as at `ExpertQuantization`.
        let help = match self.0 {
            ExpertQuantization::Fp8 => {
                "the checkpoint's FP8 experts (most stable, still fast). BFCL echolp N=1004 \
                 84.96/86.03 overall/normalized; agentic-webserver pass (535-546 s); C16 \
                 315.5 tok/s, 0.227 J/tok"
            }
            ExpertQuantization::Nvfp4GateUp => {
                "lowers the routed experts' gate and up projections to NVFP4 in decode; down, \
                 the shared expert and prefill stay FP8 (the stable speed lever). BFCL \
                 84.86/85.33; agentic-webserver pass 3/3 (504-533 s); C16 359.6 tok/s, \
                 0.211 J/tok"
            }
            ExpertQuantization::Nvfp4 => {
                "lowers every routed-expert projection to NVFP4 in decode; the shared expert and \
                 prefill stay FP8 (dangerous but fast). BFCL 85.46/86.57; agentic-webserver \
                 FAILS its 700 s ceiling (168 turns, 774 s); C16 383.1 tok/s, 0.201 J/tok"
            }
        };
        Some(clap::builder::PossibleValue::new(self.0.name()).help(help))
    }
}

/// 2026-10-09: `--dense-quantization`: a clap value enum over the model layer's tiers
/// (`metrale_model_layers::layers::DenseQuantization`), which own the names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenseQuantizationArg(pub DenseQuantization);

impl clap::ValueEnum for DenseQuantizationArg {
    fn value_variants<'a>() -> &'a [Self] {
        const VARIANTS: [DenseQuantizationArg; 3] = [
            DenseQuantizationArg(DenseQuantization::ALL[0]),
            DenseQuantizationArg(DenseQuantization::ALL[1]),
            DenseQuantizationArg(DenseQuantization::ALL[2]),
        ];
        &VARIANTS
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        let help = match self.0 {
            DenseQuantization::Declared => "16-bit dense projections at the checkpoint's width",
            DenseQuantization::Fp8 => {
                "16-bit dense projections quantized at load to FP8 per-channel and decoded \
                 W8A8 with per-token FP8 activations: below the checkpoint's declared \
                 precision (glm5_next only; unmeasured)"
            }
            DenseQuantization::W4a16 => {
                "FURTHER BELOW the checkpoint's declared precision than fp8: the KDA q/k/v, \
                 f_a, b, g_a and o projections and the shared expert quantized at load to \
                 NVFP4 (4-bit weights) and decoded W4A16 with 16-bit activations; the rest of \
                 fp8's set stays fp8 (glm5_next only; unmeasured)"
            }
        };
        Some(clap::builder::PossibleValue::new(self.0.name()).help(help))
    }
}

/// 2026-09-28: `--weight-quantization`: a clap value enum over the config crate's tiers
/// (`metrale_config::WeightQuantization`), which own the names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightQuantizationArg(pub WeightQuantization);

impl clap::ValueEnum for WeightQuantizationArg {
    fn value_variants<'a>() -> &'a [Self] {
        const VARIANTS: [WeightQuantizationArg; 2] = [
            WeightQuantizationArg(WeightQuantization::ALL[0]),
            WeightQuantizationArg(WeightQuantization::ALL[1]),
        ];
        &VARIANTS
    }

    fn to_possible_value(&self) -> Option<clap::builder::PossibleValue> {
        let help = match self.0 {
            WeightQuantization::Declared => WEIGHT_QUANT_DECLARED_HELP,
            WeightQuantization::Nvfp4 => WEIGHT_QUANT_NVFP4_HELP,
        };
        Some(clap::builder::PossibleValue::new(self.0.name()).help(help))
    }
}

/// 2026-09-28: The `declared` tier's one-line help (`met serve --help`).
const WEIGHT_QUANT_DECLARED_HELP: &str = "each layer at the precision its checkpoint's \
     quantization_config declares, weights and activations (W4A4 where it declares FP4 \
     activations, W4A16 where weight-only). Until the W8A8 decode kernels land, FP8-declared \
     layers run 16-bit activations (above declared), and the dense loader still requantizes \
     per-channel FP8 attention/GDN/MLP to NVFP4";

/// 2026-09-28: The `nvfp4` tier's one-line help.
const WEIGHT_QUANT_NVFP4_HELP: &str = "the engine before the tiers: FP8-declared layers \
     requantized to NVFP4 at load (below declared), 16-bit decode activations unless \
     --w4a4-downcast, FP4 MMQ prefill on every NVFP4 FFN; the certified recipes' tier";

/// 2026-09-26: The closed value set for a `met serve` flag, by its long name,
/// or `None` for a free-form flag. For `--kv-cache-dtype` it lists each
/// dtype's canonical name only; parse aliases such as `fp8k2v` for
/// `fp8k_turbo2v` are not offered.
pub(crate) fn options_for_flag(flag: &str) -> Option<Vec<String>> {
    let owned = |list: &[&str]| list.iter().map(|s| s.to_string()).collect();
    match flag {
        "lm-head-dtype" => Some(owned(LM_HEAD_DTYPES)),
        "mtp-quantization" => Some(owned(MTP_QUANTS)),
        "scheduler" => Some(owned(SCHEDULERS)),
        "scheduler-config" => Some(owned(SCHEDULER_CONFIGS)),
        "ssm-h-dtype" => Some(owned(SSM_H_DTYPES)),
        "telemetry" => Some(owned(TELEMETRY_LEVELS)),
        "mtp-gate" => Some(owned(MTP_GATES)),
        "tool-call-parser" => Some(owned(TOOL_CALL_PARSERS)),
        "ssm-batched-recurrent" | "content-loop-watchdog" | "tool-grammar" => {
            Some(owned(TRISTATES))
        }
        "weight-quantization" => Some(
            WeightQuantization::ALL
                .iter()
                .map(|q| q.name().to_string())
                .collect(),
        ),
        "dense-quantization" => Some(
            DenseQuantization::ALL
                .iter()
                .map(|q| q.name().to_string())
                .collect(),
        ),
        "expert-quantization" => Some(
            ExpertQuantization::ALL
                .iter()
                .map(|q| q.name().to_string())
                .collect(),
        ),
        "kv-cache-dtype" => Some(
            metrale_cache::kv_cache::KvCacheDtype::ALL
                .iter()
                .map(|d| d.name().to_string())
                .collect(),
        ),
        _ => None,
    }
}

#[cfg(test)]
#[path = "flag_values_tests.rs"]
mod tests;

/// 2026-09-28: `--forward`: which forward decode runs. `legacy` is the hand-written layer
/// loops; `circuit` runs the program compiled from the model's circuit plan
/// (`metrale_model_layers::circuit_exec`), and boot refuses a model the circuit does not model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum ForwardArg {
    #[value(help = "the hand-written layer loops")]
    Legacy,
    #[value(help = "the program compiled from the model's circuit plan")]
    Circuit,
    #[value(
        help = "the circuit plan with reference rules only (no bit-identical fusions), for A/B"
    )]
    CircuitReference,
}

impl ForwardArg {
    /// 2026-09-28: The spelling on the command line and in `GET /forward`.
    pub fn name(self) -> &'static str {
        match self {
            ForwardArg::Legacy => "legacy",
            ForwardArg::Circuit => "circuit",
            ForwardArg::CircuitReference => "circuit-reference",
        }
    }
}
