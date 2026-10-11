// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The mock spec: every choice that shapes a mock (rehearsal) checkpoint, stated
//! explicitly. There is no default for any key; a missing key, an unknown key or a value this
//! build does not implement is refused.
//!
//! ```toml
//! schema = 1
//! seed = 20261003
//! [layers]
//! per_signature = 1          # or one count per layer signature, in `inspect` order: [2, 1]
//! [experts]
//! keep = "all"
//! [vocab]
//! keep = "all"
//! [mtp]
//! keep = true
//! [vision]
//! keep = true
//! [capacity]
//! kv = "free"
//! [routing]
//! mode = "uniform"           # or "histogram", with `histogram = "<profile.json>"` and
//!                            # `calibration = "none"` or "<calibrate-routing file>"
//! [values]
//! mode = "init"              # or "stats", with `stats = "<met ml-utils value-stats file>"`
//! [speculative]
//! accept = "natural"
//! ```
//!
//! Owner: metrale-ml-utils.
//! Invariants:
//! - [`MockSpec::canonical`] is a pure function of the parsed values, so two files that say
//!   the same thing have the same canonical text and digest.
//! - Values reserved for later milestones (expert, vocab, MTP or vision shrinking, a pinned KV
//!   capacity, forced acceptance) are refused with the reason.

use serde::Deserialize;

use crate::error::{MlError, Result};

/// 2026-10-03: How many units of each layer signature a mock keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerSignature {
    /// 2026-10-03: The same count for every signature.
    All(u32),
    /// 2026-10-03: One count per signature, in the order `inspect` lists them.
    Each(Vec<u32>),
}

/// 2026-10-03: How the MoE router weights are synthesized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingMode {
    /// 2026-10-03: Random routers: near-uniform expert load.
    Uniform,
    /// 2026-10-03: Router weights that reproduce an imported expert-load profile (the bias
    /// channel, `routing.rs`). The paths are read by the caller, never by this crate.
    Histogram {
        /// 2026-10-03: The profile file, as the spec states it.
        path: String,
        /// 2026-10-04: A per-layer gain file from `met ml-utils calibrate-routing` (the bias's
        /// scale against the noise the mock's hidden states actually produce); `None` for the
        /// first, uncalibrated mock (`calibration = "none"`).
        calibration: Option<String>,
    },
}

/// 2026-10-04: Where weight values come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValuesMode {
    /// 2026-10-03: Analytic initialisation (`values.rs`).
    Init,
    /// 2026-10-04: Sampled from a checkpoint's per-class bit-pattern statistics (`stats.rs`).
    /// The path is read by the caller.
    Stats {
        /// 2026-10-04: The statistics file, as the spec states it.
        path: String,
    },
}

/// 2026-10-03: A parsed, validated mock spec.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockSpec {
    /// 2026-10-03: The synthesis seed.
    pub seed: u64,
    /// 2026-10-03: Units kept per layer signature.
    pub per_signature: PerSignature,
    /// 2026-10-03: Router synthesis.
    pub routing: RoutingMode,
    /// 2026-10-04: Weight values.
    pub values: ValuesMode,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecFile {
    schema: u32,
    seed: u64,
    layers: LayersFile,
    experts: KeepStr,
    vocab: KeepStr,
    mtp: KeepBool,
    vision: KeepBool,
    capacity: CapacityFile,
    routing: RoutingFile,
    values: ValuesFile,
    speculative: SpeculativeFile,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LayersFile {
    per_signature: toml::Value,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepStr {
    keep: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct KeepBool {
    keep: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CapacityFile {
    kv: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RoutingFile {
    mode: String,
    histogram: Option<String>,
    calibration: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ValuesFile {
    mode: String,
    stats: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeculativeFile {
    accept: String,
}

fn later(key: &str, value: &str, only: &str) -> MlError {
    MlError::Spec(format!(
        "{key} = {value}: this build implements only {only}; the other values are later \
         milestones (ml-utils DESIGN.md, section 8)"
    ))
}

fn count(v: &toml::Value) -> Result<u32> {
    v.as_integer()
        .and_then(|i| u32::try_from(i).ok())
        .filter(|&n| n >= 1)
        .ok_or_else(|| MlError::Spec(format!("layers.per_signature: {v} is not a count >= 1")))
}

impl MockSpec {
    /// 2026-10-03: Parse and validate a spec file's text.
    pub fn parse(text: &str) -> Result<Self> {
        let f: SpecFile = toml::from_str(text).map_err(|e| MlError::Spec(e.to_string()))?;
        if f.schema != 1 {
            return Err(MlError::Spec(format!(
                "schema {} (this build reads 1)",
                f.schema
            )));
        }
        let per_signature = match &f.layers.per_signature {
            toml::Value::Array(a) if !a.is_empty() => {
                PerSignature::Each(a.iter().map(count).collect::<Result<_>>()?)
            }
            v @ toml::Value::Integer(_) => PerSignature::All(count(v)?),
            other => {
                return Err(MlError::Spec(format!(
                    "layers.per_signature: {other} is neither a count nor a non-empty list"
                )));
            }
        };
        if f.experts.keep != "all" {
            return Err(later("experts.keep", &f.experts.keep, "\"all\""));
        }
        if f.vocab.keep != "all" {
            return Err(later("vocab.keep", &f.vocab.keep, "\"all\""));
        }
        if !f.mtp.keep {
            return Err(later("mtp.keep", "false", "true"));
        }
        if !f.vision.keep {
            return Err(later("vision.keep", "false", "true"));
        }
        if f.capacity.kv != "free" {
            return Err(later("capacity.kv", &f.capacity.kv, "\"free\""));
        }
        if f.speculative.accept != "natural" {
            return Err(later(
                "speculative.accept",
                &f.speculative.accept,
                "\"natural\"",
            ));
        }
        let values = match (f.values.mode.as_str(), f.values.stats) {
            ("init", None) => ValuesMode::Init,
            ("stats", Some(path)) if !path.is_empty() => ValuesMode::Stats { path },
            ("init", Some(_)) => {
                return Err(MlError::Spec(
                    "values.stats is set but values.mode is \"init\"".into(),
                ));
            }
            ("stats", _) => {
                return Err(MlError::Spec(
                    "values.mode = \"stats\" needs values.stats = \"<file>\"".into(),
                ));
            }
            (other, _) => {
                return Err(MlError::Spec(format!(
                    "values.mode = {other:?}: expected \"init\" or \"stats\""
                )));
            }
        };
        let calibration = match (f.routing.mode.as_str(), f.routing.calibration) {
            ("histogram", Some(c)) if c == "none" => None,
            ("histogram", Some(c)) if !c.is_empty() => Some(c),
            ("histogram", _) => {
                return Err(MlError::Spec(
                    "routing.mode = \"histogram\" needs routing.calibration = \"none\" or a \
                     calibration file"
                        .into(),
                ));
            }
            (_, Some(_)) => {
                return Err(MlError::Spec(
                    "routing.calibration is set but routing.mode is not \"histogram\"".into(),
                ));
            }
            (_, None) => None,
        };
        let routing = match (f.routing.mode.as_str(), f.routing.histogram) {
            ("uniform", None) => RoutingMode::Uniform,
            ("histogram", Some(path)) if !path.is_empty() => {
                RoutingMode::Histogram { path, calibration }
            }
            ("uniform", Some(_)) => {
                return Err(MlError::Spec(
                    "routing.histogram is set but routing.mode is \"uniform\"".into(),
                ));
            }
            ("histogram", _) => {
                return Err(MlError::Spec(
                    "routing.mode = \"histogram\" needs routing.histogram = \"<profile>\"".into(),
                ));
            }
            (other, _) => {
                return Err(MlError::Spec(format!(
                    "routing.mode = {other:?}: expected \"uniform\" or \"histogram\""
                )));
            }
        };
        Ok(MockSpec {
            seed: f.seed,
            per_signature,
            routing,
            values,
        })
    }

    /// 2026-10-03: The units kept for each of `n` signatures.
    pub fn counts(&self, n: usize) -> Result<Vec<u32>> {
        match &self.per_signature {
            PerSignature::All(c) => Ok(vec![*c; n]),
            PerSignature::Each(v) if v.len() == n => Ok(v.clone()),
            PerSignature::Each(v) => Err(MlError::Spec(format!(
                "layers.per_signature lists {} counts; this checkpoint has {n} layer signatures",
                v.len()
            ))),
        }
    }

    /// 2026-10-03: The spec as canonical TOML: fixed key order, every key written.
    pub fn canonical(&self) -> String {
        let per = match &self.per_signature {
            PerSignature::All(c) => c.to_string(),
            PerSignature::Each(v) => format!(
                "[{}]",
                v.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
            ),
        };
        let routing = match &self.routing {
            RoutingMode::Uniform => "mode = \"uniform\"\n".to_string(),
            RoutingMode::Histogram { path, calibration } => format!(
                "mode = \"histogram\"\nhistogram = {}\ncalibration = {}\n",
                toml_str(path),
                toml_str(calibration.as_deref().unwrap_or("none"))
            ),
        };
        let values = match &self.values {
            ValuesMode::Init => "mode = \"init\"\n".to_string(),
            ValuesMode::Stats { path } => format!("mode = \"stats\"\nstats = {}\n", toml_str(path)),
        };
        format!(
            "schema = 1\nseed = {}\n\n[layers]\nper_signature = {per}\n\n[experts]\nkeep = \"all\"\n\n\
             [vocab]\nkeep = \"all\"\n\n[mtp]\nkeep = true\n\n[vision]\nkeep = true\n\n\
             [capacity]\nkv = \"free\"\n\n[routing]\n{routing}\n[values]\n{values}\n\
             [speculative]\naccept = \"natural\"\n",
            self.seed
        )
    }
}

/// 2026-10-03: `s` as a TOML basic string.
pub(crate) fn toml_str(s: &str) -> String {
    toml::Value::String(s.to_string()).to_string()
}

#[cfg(test)]
#[path = "spec_tests.rs"]
mod spec_tests;
