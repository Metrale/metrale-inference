// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: The model side of a hardware plan: a circuit at its checkpoint's formats, the
//! serving policy its rules read, and the kernel target (`<model>/<quant>`) whose sources a
//! class compiles for it. [`CircuitSource`] is the one seam between the hardware axis and how
//! a circuit is instantiated (today `kernels/circuits/INSTANCES.toml`; the checkpoint
//! instantiation plugs in behind the same trait).
//!
//! Owner: metrale-circuit (hardware).
//! Invariants:
//! - Settings a class's `HARDWARE.toml [defaults]` decides are re-read from the device's class
//!   ([`policy_on_class`]), except on the class a recipe was written for
//!   ([`ModelUnderPlan::settings_class`]), where the recipe's value stands: it may pin one away
//!   from the class default.
//! - `recipe` precision is the instance's own (the golden plans' formats); `declared` is the
//!   checkpoint's declared formats. Neither is chosen silently: the report prints which.

use std::collections::BTreeMap;

use super::HwError;
use super::class::ClassInfo;
use crate::fuser::Policy;
use crate::instances::{Instance, PrecisionSpec, parse_instances};
use crate::ir::Circuit;
use crate::render::Header;
use crate::venn::repo::{Repo, load_instance, resolve};

/// 2026-09-30: Which formats a plan serves.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionChoice {
    /// 2026-09-30: The recipe's pinned formats (a golden instance's tier and engine choices).
    Recipe,
    /// 2026-09-30: The checkpoint's declared formats.
    Declared,
}

impl PrecisionChoice {
    /// 2026-09-30: The CLI and report spelling.
    pub fn name(self) -> &'static str {
        match self {
            PrecisionChoice::Recipe => "recipe",
            PrecisionChoice::Declared => "declared",
        }
    }
}

/// 2026-09-30: What names the model.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec<'a> {
    /// 2026-09-30: Checkpoint id (`org/name`) or recipe id.
    pub checkpoint: &'a str,
    /// 2026-09-30: The checkpoint's `config.json`, when read.
    pub config_json: Option<&'a str>,
    /// 2026-09-30: Its `hf_quant_config.json`, when it ships one.
    pub hf_quant: Option<&'a str>,
    /// 2026-09-30: Formats served.
    pub precision: PrecisionChoice,
}

/// 2026-09-30: A model ready to plan.
#[derive(Debug, Clone)]
pub struct ModelUnderPlan {
    /// 2026-09-30: Recipe id, or the checkpoint id when no recipe serves it.
    pub label: String,
    /// 2026-09-30: Checkpoint id.
    pub checkpoint: String,
    /// 2026-09-30: Kernel model directory (`qwen3.8-27b`).
    pub kernel_model: String,
    /// 2026-09-30: Kernel quant directory (`nvfp4`).
    pub kernel_quant: String,
    /// 2026-09-30: The circuit.
    pub circuit: Circuit,
    /// 2026-09-30: Policy as the model's source states it.
    pub policy: Policy,
    /// 2026-09-30: Plan header lines (`target` is re-spelled per class).
    pub header: Header,
    /// 2026-09-30: Where the formats come from, one line.
    pub precision: String,
    /// 2026-09-30: Which formats were asked for.
    pub precision_choice: PrecisionChoice,
    /// 2026-09-30: Where each policy setting comes from, when derived rather than stated by a
    /// recipe (empty: the recipe states them in INSTANCES.toml).
    pub policy_sources: Vec<(String, String)>,
    /// 2026-09-30: The class a recipe's settings were stated for (its INSTANCES.toml target's
    /// hardware); `None` for a model planned from its checkpoint, which takes every class's
    /// defaults.
    pub settings_class: Option<String>,
}

/// 2026-09-30: Instantiates the model a spec names.
pub trait CircuitSource {
    /// 2026-09-30: The model.
    fn model(&self, spec: &ModelSpec<'_>) -> Result<ModelUnderPlan, HwError>;
}

/// 2026-09-30: The source over `kernels/circuits/INSTANCES.toml`.
pub struct InstancesSource<'a> {
    /// 2026-09-30: The repository.
    pub repo: &'a dyn Repo,
}

impl CircuitSource for InstancesSource<'_> {
    fn model(&self, spec: &ModelSpec<'_>) -> Result<ModelUnderPlan, HwError> {
        let text = self
            .repo
            .read("kernels/circuits/INSTANCES.toml")
            .map_err(HwError::Model)?;
        let all = parse_instances(&text).map_err(|e| HwError::Model(e.to_string()))?;
        let inst = resolve(&all, spec.checkpoint).map_err(|e| HwError::Model(e.to_string()))?;
        let (inst, precision) = match (spec.precision, &inst.precision) {
            (PrecisionChoice::Recipe, p) => {
                let line = format!("recipe {} ({})", inst.recipe, describe(p));
                (inst, line)
            }
            (PrecisionChoice::Declared, PrecisionSpec::Policy { .. }) => {
                let mut i = inst.clone();
                if let PrecisionSpec::Policy { tier, .. } = &mut i.precision {
                    *tier = "declared".into();
                }
                let line = format!("declared ({})", describe(&i.precision));
                (i, line)
            }
            (PrecisionChoice::Declared, PrecisionSpec::Table(t)) => {
                let line = format!("declared, as stated by the precision table `{t}`");
                (inst, line)
            }
        };
        model_of(self.repo, &inst, precision, spec.precision)
    }
}

fn describe(p: &PrecisionSpec) -> String {
    match p {
        PrecisionSpec::Table(t) => format!("precision table `{t}`"),
        PrecisionSpec::Policy {
            checkpoint_plan,
            tier,
            ..
        } => format!("checkpoint plan `{checkpoint_plan}` at tier `{tier}`"),
    }
}

/// 2026-09-30: `inst` loaded as a model to plan.
pub fn model_of(
    repo: &dyn Repo,
    inst: &Instance,
    precision: String,
    precision_choice: PrecisionChoice,
) -> Result<ModelUnderPlan, HwError> {
    let loaded = load_instance(repo, inst).map_err(|e| HwError::Model(e.to_string()))?;
    let mut parts = inst.target.split('/');
    let (hw, model, quant) = (parts.next(), parts.next(), parts.next());
    let (Some(hw), Some(model), Some(quant)) = (hw, model, quant) else {
        return Err(HwError::Model(format!(
            "{}: target `{}` is not hw/model/quant",
            inst.recipe, inst.target
        )));
    };
    Ok(ModelUnderPlan {
        label: inst.recipe.clone(),
        checkpoint: inst.checkpoint.clone(),
        kernel_model: model.to_string(),
        kernel_quant: quant.to_string(),
        circuit: loaded.circuit,
        policy: inst.policy.clone(),
        header: crate::header(inst),
        precision,
        precision_choice,
        policy_sources: Vec::new(),
        settings_class: Some(hw.to_string()),
    })
}

/// 2026-09-30: Policy settings a class's `HARDWARE.toml [defaults]` decides: setting to the
/// defaults key (booleans map to `on` / `off`). INSTANCES.toml cites these rows for the values
/// its recipes state.
pub const CLASS_DEFAULT_SETTINGS: [(&str, &str); 3] = [
    ("ssm_batched_recurrent", "ssm_batched_recurrent"),
    ("ssm_ba_gates_hopper", "ssm_ba_gates_hopper"),
    ("decode_split_silu", "decode_split_silu"),
];

/// 2026-09-30: `policy` with the class-decided settings re-read from `class`, unless `class` is
/// the `settings_class` the policy was stated for; the changed keys are returned for the report.
/// A class that does not state one is an error either way.
pub fn policy_on_class(
    policy: &Policy,
    settings_class: Option<&str>,
    class: &ClassInfo,
) -> Result<(Policy, BTreeMap<String, String>), HwError> {
    let own = settings_class == Some(class.name.as_str());
    let mut out = policy.clone();
    let mut changed = BTreeMap::new();
    for (setting, key) in CLASS_DEFAULT_SETTINGS {
        if !policy.settings.contains_key(setting) {
            continue;
        }
        let raw = class.defaults.get(key).ok_or_else(|| {
            HwError::Class(format!(
                "kernels/{}/HARDWARE.toml [defaults] does not state `{key}`",
                class.name
            ))
        })?;
        let v = match raw.as_str() {
            "true" => "on",
            "false" => "off",
            other => other,
        }
        .to_string();
        if own {
            continue;
        }
        if policy.settings.get(setting) != Some(&v) {
            changed.insert(setting.to_string(), v.clone());
        }
        out.settings.insert(setting.to_string(), v);
    }
    Ok((out, changed))
}

/// 2026-09-30: `header` with the target re-spelled for `class`.
pub fn header_on(model: &ModelUnderPlan, class: &str) -> Header {
    model
        .header
        .iter()
        .map(|(k, v)| {
            if k == "target" {
                (
                    k.clone(),
                    format!("{class}/{}/{}", model.kernel_model, model.kernel_quant),
                )
            } else {
                (k.clone(), v.clone())
            }
        })
        .collect()
}
