// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The contract file (`kernels/<hw>/common/ACCURACY.toml`): one `[[contract]]` per
//! kernel family entry point set, naming the reference, the class, what the family's pipeline
//! does not state (reduction depth, flush-to-zero, approximate functions, where block scales
//! fold), the input classes and the mutations, plus the calibration rows a calibration run
//! writes and a reviewer reads in the diff.
//!
//! Owner: metrale-accuracy.
//! Invariants:
//! - No field has a default: every contract states every field its class needs (PCND);
//!   unknown keys are refused.
//! - The contract references the family manifest, it never restates it: formats come from the
//!   family's `pipeline.<op>` ([`crate::plan`]).

use std::collections::BTreeMap;

use serde::Deserialize;

use crate::inputs::InputClass;
use crate::mutation::Mutation;

/// 2026-10-09: How the output is judged.
#[derive(Debug, Clone, PartialEq)]
pub enum Class {
    /// 2026-10-09: Within the bound derived from the declared pipeline.
    Derived,
    /// 2026-10-09: Byte for byte equal to another kernel's output on the same operands.
    BitIdentical {
        /// 2026-10-09: The sibling entry point (`module::function`).
        against: String,
    },
}

/// 2026-10-09: One level of a reduction tree, innermost first.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Level {
    /// 2026-10-09: Where the level runs (`thread`, `warp`, `cta`, `mma`, `split`), for reports.
    pub level: String,
    /// 2026-10-09: Terms combined at this level: a literal, or `k/<lit>` (the reduced length
    /// divided by a literal, rounded up).
    pub width: String,
    /// 2026-10-09: `sequential` (depth = width) or `tree` (depth = ceil(log2 width)).
    pub order: String,
}

/// 2026-10-09: Where a block-scaled weight's scale is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleFold {
    /// 2026-10-09: Each element is decoded with its scale into the operand precision, then
    /// multiplied (the pipeline's `weight` step rounds the product of value and scale).
    Element,
    /// 2026-10-09: Each group's partial sum is accumulated unscaled and multiplied by the
    /// group's scale (the pipeline's `scale` step rounds that product).
    Group,
    /// 2026-10-09: The op has no block-scaled weight.
    None,
}

/// 2026-10-09: The split of an output dimension across launches (tensor-parallel shards).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Split {
    /// 2026-10-09: Shards.
    pub world: u32,
    /// 2026-10-09: Shard boundaries are multiples of this (1: none).
    pub align: u32,
}

/// 2026-10-09: A calibration row: what the arms measured at one point and input class.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Calibration {
    /// 2026-10-09: The point key ([`crate::points::AccuracyPoint::key`] or a case key).
    pub point: String,
    /// 2026-10-09: Input class.
    pub input: String,
    /// 2026-10-09: The good arm's max err/bound.
    pub ratio: f64,
    /// 2026-10-09: The good arm's share of misrounded outputs.
    pub misrounded: f64,
    /// 2026-10-09: The noise floor's (worst legitimate bracketing's) share of misrounded outputs.
    pub floor_misrounded: f64,
    /// 2026-10-09: The smallest share of misrounded outputs over the mutation arms.
    pub mutation_min_misrounded: f64,
    /// 2026-10-09: The target closure hash the arms ran on.
    pub closure: String,
}

/// 2026-10-09: One contract.
#[derive(Debug, Clone, PartialEq)]
pub struct Contract {
    /// 2026-10-09: Family id in KERNEL_FAMILIES.toml.
    pub family: String,
    /// 2026-10-09: The entry points launched (`module::function`), each in the family.
    pub kernels: Vec<String>,
    /// 2026-10-09: The family op whose pipeline the reference reads.
    pub op: String,
    /// 2026-10-09: Reference name ([`crate::refs::Reference`]).
    pub reference: String,
    /// 2026-10-09: Judgement.
    pub class: Class,
    /// 2026-10-09: The reduction tree of each reduced dimension, by dimension name.
    pub reduction: BTreeMap<String, Vec<Level>>,
    /// 2026-10-09: Results below the normal range are flushed.
    pub ftz: bool,
    /// 2026-10-09: Relative error of each approximate function the reference uses.
    pub approx: BTreeMap<String, f64>,
    /// 2026-10-09: Kernel constants a reference needs (a quantizer's scale floor), by name; a
    /// reference asks for the ones it uses and errors on a missing one.
    pub constants: BTreeMap<String, f64>,
    /// 2026-10-09: Where block scales fold.
    pub scale_fold: ScaleFold,
    /// 2026-10-09: Output split across launches, when the contract tests one.
    pub split: Option<Split>,
    /// 2026-10-09: Input classes run.
    pub inputs: Vec<InputClass>,
    /// 2026-10-09: Mutations that must fail.
    pub mutations: Vec<Mutation>,
    /// 2026-10-09: Calibration rows.
    pub calibration: Vec<Calibration>,
}

/// 2026-10-09: The whole file.
#[derive(Debug, Clone, PartialEq)]
pub struct Contracts {
    /// 2026-10-09: Hardware class.
    pub hardware: String,
    /// 2026-10-09: Corpus seed.
    pub seed: u64,
    /// 2026-10-09: Contracts, in file order.
    pub contracts: Vec<Contract>,
}

/// 2026-10-09: A contract file that cannot be used.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ContractError {
    /// 2026-10-09: TOML or schema error.
    #[error("ACCURACY.toml: {0}")]
    Parse(String),
    /// 2026-10-09: A contract field is wrong for its class or op.
    #[error("contract `{family}`: {problem}")]
    Invalid {
        /// 2026-10-09: Family.
        family: String,
        /// 2026-10-09: What is wrong.
        problem: String,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    schema: u32,
    hardware: String,
    seed: u64,
    contract: Vec<ContractFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ContractFile {
    family: String,
    kernels: Vec<String>,
    op: String,
    reference: String,
    class: String,
    against: Option<String>,
    reduction: Option<BTreeMap<String, Vec<Level>>>,
    ftz: Option<bool>,
    approx: Option<BTreeMap<String, String>>,
    constants: Option<BTreeMap<String, f64>>,
    scale_fold: Option<String>,
    split: Option<Split>,
    inputs: Vec<String>,
    mutations: Vec<String>,
    #[serde(default)]
    calibration: Vec<Calibration>,
}

/// 2026-10-09: Parse a relative error spelled `2^-22` or as a decimal.
pub fn parse_rel(s: &str) -> Option<f64> {
    if let Some(e) = s.strip_prefix("2^") {
        let e: f64 = e.parse().ok()?;
        return Some(2f64.powf(e));
    }
    s.parse().ok().filter(|v: &f64| v.is_finite() && *v >= 0.0)
}

/// 2026-10-09: Parse ACCURACY.toml. Validation against the families is [`crate::validate`].
pub fn parse_contracts(text: &str) -> Result<Contracts, ContractError> {
    let f: File = toml::from_str(text).map_err(|e| ContractError::Parse(e.to_string()))?;
    if f.schema != 1 {
        return Err(ContractError::Parse(format!(
            "schema {} (this build reads 1)",
            f.schema
        )));
    }
    let contracts = f
        .contract
        .into_iter()
        .map(contract)
        .collect::<Result<_, _>>()?;
    Ok(Contracts {
        hardware: f.hardware,
        seed: f.seed,
        contracts,
    })
}

fn contract(c: ContractFile) -> Result<Contract, ContractError> {
    let bad = |p: String| ContractError::Invalid {
        family: c.family.clone(),
        problem: p,
    };
    let class = match (c.class.as_str(), &c.against) {
        ("derived", None) => Class::Derived,
        ("bit_identical", Some(a)) => Class::BitIdentical { against: a.clone() },
        ("derived", Some(_)) => return Err(bad("a derived contract names no `against`".into())),
        ("bit_identical", None) => {
            return Err(bad("a bit_identical contract needs `against`".into()));
        }
        (other, _) => return Err(bad(format!("class `{other}` (derived | bit_identical)"))),
    };
    let derived = class == Class::Derived;
    let need = |name: &str, present: bool| -> Result<(), ContractError> {
        match (derived, present) {
            (true, false) => Err(bad(format!("a derived contract states `{name}`"))),
            (false, true) => Err(bad(format!("`{name}` is for derived contracts only"))),
            _ => Ok(()),
        }
    };
    need("reduction", c.reduction.is_some())?;
    need("ftz", c.ftz.is_some())?;
    need("approx", c.approx.is_some())?;
    need("scale_fold", c.scale_fold.is_some())?;
    let approx = c
        .approx
        .clone()
        .unwrap_or_default()
        .into_iter()
        .map(|(k, v)| {
            parse_rel(&v)
                .map(|r| (k.clone(), r))
                .ok_or_else(|| bad(format!("approx `{k}` = `{v}`")))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let scale_fold = match c.scale_fold.as_deref() {
        None | Some("none") => ScaleFold::None,
        Some("element") => ScaleFold::Element,
        Some("group") => ScaleFold::Group,
        Some(o) => return Err(bad(format!("scale_fold `{o}` (element | group | none)"))),
    };
    for levels in c.reduction.iter().flat_map(|r| r.values()) {
        for l in levels {
            if l.order != "sequential" && l.order != "tree" {
                return Err(bad(format!(
                    "reduction order `{}` (sequential | tree)",
                    l.order
                )));
            }
            if crate::plan::level_width(&l.width, 1).is_none() {
                return Err(bad(format!(
                    "reduction width `{}` (a literal or k/<lit>)",
                    l.width
                )));
            }
        }
    }
    let inputs = c
        .inputs
        .iter()
        .map(|s| InputClass::parse(s).ok_or_else(|| bad(format!("input class `{s}`"))))
        .collect::<Result<Vec<_>, _>>()?;
    if !inputs.contains(&InputClass::Gaussian) {
        return Err(bad("every contract runs the `gaussian` class".into()));
    }
    let mutations = c
        .mutations
        .iter()
        .map(|s| Mutation::parse(s).ok_or_else(|| bad(format!("mutation `{s}`"))))
        .collect::<Result<Vec<_>, _>>()?;
    if mutations.is_empty() {
        return Err(bad(
            "a contract without a mutation cannot prove it detects anything".into(),
        ));
    }
    if c.kernels.is_empty() {
        return Err(bad("no kernels".into()));
    }
    Ok(Contract {
        family: c.family.clone(),
        kernels: c.kernels.clone(),
        op: c.op.clone(),
        reference: c.reference.clone(),
        class,
        reduction: c.reduction.clone().unwrap_or_default(),
        ftz: c.ftz.unwrap_or(false),
        approx,
        constants: c.constants.clone().unwrap_or_default(),
        scale_fold,
        split: c.split,
        inputs,
        mutations,
        calibration: c.calibration.clone(),
    })
}

#[cfg(test)]
#[path = "contract_tests.rs"]
mod tests;
