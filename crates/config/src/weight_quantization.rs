// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: `--weight-quantization`: the precision each linear layer runs at, and the one
//! policy that loaders and kernel dispatch ask for it.
//!
//! - `declared` (the default) follows the checkpoint's `quantization_config`
//!   ([`DeclaredPrecisionPlan`]) per layer, for the weights and for the input activations.
//! - `nvfp4` is the engine as it ran before the plan existed. FP8-declared projections are
//!   requantized to NVFP4 at load. Decode runs 16-bit activations, and FP4 activations only
//!   under `--w4a4-downcast` ([`W4a4Downcast`]). The dense-FFN MMQ prefill runs FP4
//!   activations on every NVFP4 FFN.
//!
//! The serve publishes a [`WeightQuantTier`] once. Each loader builds a [`WeightQuantPolicy`]
//! from it and the model's plan, and stamps what the policy answers on the weights it builds
//! ([`Nvfp4Act`]). Kernel dispatch reads those stamps and the published tier, and nothing
//! else.
//!
//! Owner: config (quantization).
//! Invariants:
//! - Pure: no environment, no I/O, no process state.
//! - Under `nvfp4` no answer depends on the plan.
//! - A checkpoint that declares nothing ([`DeclaredPrecisionPlan::is_undeclared`]) gets the
//!   `nvfp4` answers under both tiers. Load-time quantization of a 16-bit checkpoint is
//!   governed by its own flags, not this one.

use anyhow::{Result, bail};

use crate::precision_plan::{DeclaredPrecisionPlan, LayerPrecision};

/// 2026-09-28: The `--weight-quantization` tiers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum WeightQuantization {
    /// 2026-09-28: Each layer at the precision its checkpoint declares.
    #[default]
    Declared,
    /// 2026-09-28: FP8-declared projections requantized to NVFP4 at load, as before the plan
    /// existed.
    Nvfp4,
}

impl WeightQuantization {
    /// 2026-09-28: Every tier, in the order the flag lists them.
    pub const ALL: [Self; 2] = [Self::Declared, Self::Nvfp4];

    /// 2026-09-28: The flag value, recipe value and gate-record value.
    pub fn name(self) -> &'static str {
        match self {
            Self::Declared => "declared",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

/// 2026-09-28: `--w4a4-downcast` and `--w4a4-downcast-wide`, which exist only under the
/// `nvfp4` tier. They run every NVFP4 small-M projection's activations as NVFP4 (W4A4),
/// whatever the checkpoint declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum W4a4Downcast {
    /// 2026-09-28: 16-bit activations at decode.
    #[default]
    Off,
    /// 2026-09-28: `--w4a4-downcast`: up to 32 rows.
    Narrow,
    /// 2026-09-28: `--w4a4-downcast --w4a4-downcast-wide`: up to 64 rows.
    Wide,
}

impl W4a4Downcast {
    /// 2026-09-28: From the two presence flags; `-wide` alone does nothing, as it always has.
    pub fn from_flags(downcast: bool, wide: bool) -> Self {
        match (downcast, wide) {
            (false, _) => Self::Off,
            (true, false) => Self::Narrow,
            (true, true) => Self::Wide,
        }
    }
}

/// 2026-09-28: What the serve publishes: the tier and, under `nvfp4`, its W4A4 lever.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct WeightQuantTier {
    tier: WeightQuantization,
    downcast: W4a4Downcast,
}

impl WeightQuantTier {
    /// 2026-09-28: The `nvfp4` tier with its W4A4 lever. A `declared` tier refuses a lever:
    /// the checkpoint already decides every layer's activations there.
    pub fn new(tier: WeightQuantization, downcast: W4a4Downcast) -> Result<Self> {
        if tier == WeightQuantization::Declared && downcast != W4a4Downcast::Off {
            bail!(
                "--w4a4-downcast applies only with --weight-quantization nvfp4; under \
                 `declared` each layer's activations follow the checkpoint's \
                 quantization_config"
            );
        }
        Ok(Self { tier, downcast })
    }

    /// 2026-09-28: The tier.
    pub fn tier(self) -> WeightQuantization {
        self.tier
    }

    /// 2026-09-28: The `nvfp4` tier's W4A4 lever; always `Off` under `declared`.
    pub fn downcast(self) -> W4a4Downcast {
        self.downcast
    }

    /// 2026-09-28: Whether decode may take the FP4 block-scale projection path at all, so
    /// its kernels and activation scratch are prepared.
    pub fn uses_w4a4_decode(self) -> bool {
        match self.tier {
            WeightQuantization::Declared => true,
            WeightQuantization::Nvfp4 => self.downcast != W4a4Downcast::Off,
        }
    }

    /// 2026-09-28: The widest row count the W4A4 path serves for a weight stamped `act`,
    /// given `kernel_rows`, the widest the prepared kernels serve (0 when absent). 0 means
    /// the weight's decode activations stay 16-bit.
    ///
    /// Under `nvfp4` the lever decides for every weight alike: 32 rows, or 64 with `-wide`.
    /// Under `declared` a weight stamped [`Nvfp4Act::A4`] runs W4A4 at every row count the
    /// kernels serve, and any other weight runs 16-bit activations.
    pub fn w4a4_rows(
        self,
        act: Nvfp4Act,
        narrow_rows: u32,
        wide_rows: u32,
        kernel_rows: u32,
    ) -> u32 {
        match self.tier {
            WeightQuantization::Nvfp4 => match self.downcast {
                W4a4Downcast::Off => 0,
                W4a4Downcast::Narrow => narrow_rows,
                W4a4Downcast::Wide => wide_rows,
            },
            WeightQuantization::Declared => {
                if act == Nvfp4Act::A4 {
                    kernel_rows
                } else {
                    0
                }
            }
        }
    }

    /// 2026-09-28: Whether a dense FFN whose gate, up or down is stamped `act` sends its 2-
    /// and 3-row decode steps through the W4A4 projections. Under `declared` the stamps
    /// decide; under `nvfp4` the lever does, as `--w4a4-downcast` always has.
    pub fn ffn_small_batch_w4a4(self, any_a4: bool) -> bool {
        match self.tier {
            WeightQuantization::Nvfp4 => self.downcast != W4a4Downcast::Off,
            WeightQuantization::Declared => any_a4,
        }
    }

    /// 2026-09-28: Whether a dense FFN runs its single decode row W4A4. Only `declared` does,
    /// and only for an FFN that declares FP4 activations; `--w4a4-downcast` never reached
    /// the single row.
    pub fn ffn_single_row_w4a4(self, any_a4: bool) -> bool {
        self.tier == WeightQuantization::Declared && any_a4
    }
}

/// 2026-09-28: The activation format the policy set for one NVFP4 weight, which the loader
/// stamps on it (`QuantizedWeight::act`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Nvfp4Act {
    /// 2026-09-28: No stamp: the loader does not consult the policy, or the tier is `nvfp4`.
    /// Decode follows the published tier's lever, and prefill may run FP4 activations (the
    /// dense-FFN MMQ arm), as before the plan existed.
    #[default]
    Unstamped,
    /// 2026-09-28: The checkpoint declares FP4 input activations (W4A4): decode and prefill
    /// run them FP4.
    A4,
    /// 2026-09-28: The checkpoint declares wider activations (A8 or A16): no FP4 activations
    /// in decode or prefill.
    Wide,
}

impl Nvfp4Act {
    /// 2026-09-28: The stamp of a weight built from `parts` (a fused concatenation): FP4 only
    /// when every part is, `Wide` when any part is, else unstamped.
    pub fn combine(parts: impl IntoIterator<Item = Nvfp4Act>) -> Nvfp4Act {
        let mut all_a4 = true;
        let mut any = false;
        for p in parts {
            any = true;
            match p {
                Nvfp4Act::Wide => return Nvfp4Act::Wide,
                Nvfp4Act::Unstamped => all_a4 = false,
                Nvfp4Act::A4 => {}
            }
        }
        if any && all_a4 {
            Nvfp4Act::A4
        } else {
            Nvfp4Act::Unstamped
        }
    }

    /// 2026-09-28: Whether prefill may quantize this weight's activations to FP4.
    pub fn allows_fp4_prefill(self) -> bool {
        self != Nvfp4Act::Wide
    }
}

/// 2026-09-28: Which kernels the policy may rely on. The W8A8 decode family reports its
/// presence here (`metrale_model_layers::layers::kernel_caps`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct KernelCaps {
    /// 2026-09-28: A per-token E4M3 activation quantizer plus E4M3xE4M3 small-M GEMMs for the
    /// dense and non-expert projections. Until they land, an FP8-declared layer served at its
    /// FP8 weights runs the W8A16 kernels: its activations are above the declared precision.
    pub w8a8_decode: bool,
    /// 2026-09-28: The same for MoE experts (routed and shared), whose grouped kernels land
    /// separately ([`is_expert_module`]).
    pub w8a8_moe_decode: bool,
    /// 2026-09-28: The W8A8 decode of non-expert projections whose FP8 weights are 128x128
    /// block-scaled (HF `fp8`, e.g. the attention and GDN of Qwen3.6-35B-A3B-FP8), validated
    /// separately from the per-channel dense family
    /// ([`WeightQuantPolicy::fp8_block_scaled_decode_act`]).
    pub w8a8_block_scaled_decode: bool,
    /// 2026-09-28: An FP8 lm_head that serves every decode row count in one pass. The FP8
    /// head today launches once per row, which collapses throughput at width (C16 88.9 tok/s
    /// against 200.0 on the NVFP4 head, C128 not finishing), so `declared` takes a declared
    /// FP8 head only once this is set ([`LmHeadChoice::PendingFp8Kernel`]).
    pub fp8_lm_head_batched: bool,
}

/// 2026-09-28: Whether `module` is a MoE expert projection (routed or shared), which the
/// [`KernelCaps::w8a8_moe_decode`] kernels serve.
pub fn is_expert_module(module: &str) -> bool {
    module.contains(".experts.")
        || module.ends_with(".experts")
        || module.contains(".shared_expert.")
}

/// 2026-09-28: The activation format a layer runs at in decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActFormat {
    /// 2026-09-28: BF16 activations.
    Bf16,
    /// 2026-09-28: E4M3 activations with a per-token (or per-group) scale.
    Fp8,
    /// 2026-09-28: NVFP4 activations.
    Fp4,
}

/// 2026-09-28: The lm_head format `--lm-head-dtype default` resolves to under `declared`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadFormat {
    /// 2026-09-28: The head stays 16-bit (the checkpoint does not quantize it).
    Bf16,
    /// 2026-09-28: The checkpoint's FP8 head, decoded W8A16 (W8A8 once the kernels land).
    Fp8,
    /// 2026-09-28: NVFP4.
    Nvfp4,
}

/// 2026-09-28: What `--lm-head-dtype default` resolves to ([`WeightQuantPolicy::lm_head`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LmHeadChoice {
    /// 2026-09-28: The engine's per-model default head: under `nvfp4`, or with nothing
    /// declared.
    EngineDefault,
    /// 2026-09-28: The head format the checkpoint declares.
    Declared(LmHeadFormat),
    /// 2026-09-28: The checkpoint declares an FP8 head but the batched FP8 head kernel
    /// ([`KernelCaps::fp8_lm_head_batched`]) is absent: the engine's default head, which is
    /// above or below the declared precision depending on the model, until it lands.
    PendingFp8Kernel,
}

/// 2026-09-28: The per-layer answers for one model: its declared plan under the published
/// tier.
#[derive(Clone, Copy, Debug)]
pub struct WeightQuantPolicy<'a> {
    tier: WeightQuantTier,
    plan: &'a DeclaredPrecisionPlan,
    caps: KernelCaps,
}

impl<'a> WeightQuantPolicy<'a> {
    /// 2026-09-28: The policy for a model whose `quantization_config` parsed to `plan`.
    pub fn new(tier: WeightQuantTier, plan: &'a DeclaredPrecisionPlan, caps: KernelCaps) -> Self {
        Self { tier, plan, caps }
    }

    /// 2026-09-28: The policy for a checkpoint whose parsed `quantization_config` is `qc`;
    /// `None` (no block) declares nothing.
    pub fn for_checkpoint(
        tier: WeightQuantTier,
        qc: Option<&'a crate::QuantizationConfig>,
        caps: KernelCaps,
    ) -> Self {
        static UNDECLARED: DeclaredPrecisionPlan = DeclaredPrecisionPlan::UNDECLARED;
        Self::new(tier, qc.map_or(&UNDECLARED, |q| &q.precision), caps)
    }

    /// 2026-09-28: The tier in force.
    pub fn tier(&self) -> WeightQuantTier {
        self.tier
    }

    /// 2026-09-28: Whether answers follow the checkpoint: the `declared` tier on a checkpoint
    /// that declares something.
    pub fn follows_plan(&self) -> bool {
        self.tier.tier == WeightQuantization::Declared && !self.plan.is_undeclared()
    }

    /// 2026-09-28: What `module` declares.
    pub fn declared(&self, module: &str) -> LayerPrecision {
        self.plan.resolve(module)
    }

    /// 2026-09-28: The stamp for `module`'s NVFP4 weight (`QuantizedWeight::act`): `A4` where
    /// the checkpoint declares FP4 activations, `Wide` where it declares any other (so no
    /// FP4 activations run below it), unstamped under `nvfp4` or with no declaration.
    pub fn nvfp4_act(&self, module: &str) -> Nvfp4Act {
        if !self.follows_plan() {
            return Nvfp4Act::Unstamped;
        }
        if self.plan.resolve(module).activation_is_fp4() {
            Nvfp4Act::A4
        } else {
            Nvfp4Act::Wide
        }
    }

    /// 2026-09-28: Whether the policy asks for `module`'s FP8 weights to be served as FP8,
    /// not requantized to NVFP4 at load: under `declared`, every weight the checkpoint
    /// declares FP8. A load site without an FP8 decode arm for the declared scale layout
    /// still requantizes, logs it, and stamps the NVFP4 copy `Wide`.
    pub fn wants_fp8_weights(&self, module: &str) -> bool {
        self.follows_plan() && self.plan.resolve(module).weight.is_some_and(|w| w.is_fp8())
    }

    /// 2026-09-28: The decode activations of `module` when it is served at its FP8 weights
    /// (`None` when the policy does not ask for them): FP8 where the checkpoint declares FP8
    /// activations and the W8A8 decode kernels are present, BF16 otherwise (above declared,
    /// until those kernels land). This is WHEN; the W8A8 family says whether it CAN serve a
    /// given weight and row count.
    pub fn fp8_decode_act(&self, module: &str) -> Option<ActFormat> {
        if !self.wants_fp8_weights(module) {
            return None;
        }
        let a8 = self
            .plan
            .resolve(module)
            .activation
            .is_some_and(|a| a.is_fp8());
        let kernels = if is_expert_module(module) {
            self.caps.w8a8_moe_decode
        } else {
            self.caps.w8a8_decode
        };
        Some(if kernels && a8 {
            ActFormat::Fp8
        } else {
            ActFormat::Bf16
        })
    }

    /// 2026-09-28: [`Self::fp8_decode_act`] for a non-expert `module` whose FP8 weight is
    /// 128x128 block-scaled: FP8 only when [`KernelCaps::w8a8_block_scaled_decode`] is set as
    /// well, BF16 (W8A16) otherwise. An expert module answers as `fp8_decode_act` does.
    pub fn fp8_block_scaled_decode_act(&self, module: &str) -> Option<ActFormat> {
        let act = self.fp8_decode_act(module)?;
        Some(
            if act == ActFormat::Fp8
                && !is_expert_module(module)
                && !self.caps.w8a8_block_scaled_decode
            {
                ActFormat::Bf16
            } else {
                act
            },
        )
    }

    /// 2026-09-28: Whether the policy serves `module` at its FP8 weights and the checkpoint
    /// declares FP8 activations for it, whatever the kernels. Where the W8A8 answer is BF16,
    /// the layer runs above its declared activations until its W8A8 path is validated, which
    /// the load log states.
    pub fn declares_fp8_activations(&self, module: &str) -> bool {
        self.wants_fp8_weights(module)
            && self
                .plan
                .resolve(module)
                .activation
                .is_some_and(|a| a.is_fp8())
    }

    /// 2026-10-03: Whether the block-scaled attention/GDN W8A8 may run although
    /// [`KernelCaps::w8a8_block_scaled_decode`] is off. That cap is held because, stacked with
    /// the expert W8A8 on Qwen/Qwen3.6-35B-A3B-FP8, it flipped a greedy tie. Only a MoE
    /// checkpoint that declares FP8 activations on none of its `experts` (the NVFP4 35B,
    /// whose experts are NVFP4) has no such stack. A dense checkpoint, or the FP8 35B with its
    /// experts held at BF16 activations, keeps W8A16, as the combination is unvalidated there.
    pub fn lifts_block_scaled_w8a8(&self, experts: &[String], experts_decode_fp8: bool) -> bool {
        self.follows_plan()
            && !experts_decode_fp8
            && !experts.is_empty()
            && experts.iter().all(|m| !self.declares_fp8_activations(m))
    }

    /// 2026-09-28: The head `--lm-head-dtype default` takes: under `declared`, the declared
    /// format, except a declared FP8 head while [`KernelCaps::fp8_lm_head_batched`] is false;
    /// the engine's per-model default otherwise, or for a format with no head kernel.
    pub fn lm_head(&self) -> LmHeadChoice {
        if !self.follows_plan() {
            return LmHeadChoice::EngineDefault;
        }
        match self.plan.resolve("lm_head").weight {
            None => LmHeadChoice::Declared(LmHeadFormat::Bf16),
            Some(w) if w.is_fp8() && self.caps.fp8_lm_head_batched => {
                LmHeadChoice::Declared(LmHeadFormat::Fp8)
            }
            Some(w) if w.is_fp8() => LmHeadChoice::PendingFp8Kernel,
            Some(w) if w.is_fp4() => LmHeadChoice::Declared(LmHeadFormat::Nvfp4),
            Some(_) => LmHeadChoice::EngineDefault,
        }
    }
}

#[cfg(test)]
#[path = "weight_quantization_tests.rs"]
mod weight_quantization_tests;
