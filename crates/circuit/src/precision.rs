// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The precision a linear node runs at: the [`EdgePrecision`] resolver, and
//! [`PrecisionTable`], a table-driven implementation read from
//! `kernels/circuits/precision/<checkpoint>.toml`.
//!
//! The trait is shaped after `DeclaredPrecisionPlan::resolve(module) -> LayerPrecision`
//! (branch perf/weight-quantization, `crates/config/src/precision_plan.rs`): a module path
//! in, the weight and input-activation formats out. Once that plan and `WeightQuantPolicy`
//! are on main, an adapter implements this trait from them and the tables go.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - A table ends with a catch-all `match = "*"` entry, so every module resolves to a
//!   stated format; there is no implicit default.
//! - The first matching entry wins, in file order.

use serde::Deserialize;

use crate::format::{Format, FormatError};

/// 2026-09-28: The formats one linear module runs at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinearFormats {
    /// 2026-09-28: Weight format as loaded.
    pub weight: Format,
    /// 2026-09-28: Input activation format the kernel reads.
    pub activation: Format,
}

/// 2026-09-28: Answers the formats of a linear module, e.g.
/// `layers.3.mlp.down_proj`. Module paths are relative to the circuit's `module_prefix`.
pub trait EdgePrecision {
    /// 2026-09-28: The formats `module` runs at.
    fn linear(&self, module: &str) -> LinearFormats;
}

/// 2026-09-28: Why a precision table did not load.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PrecisionError {
    /// 2026-09-28: Not valid TOML, or not the table shape.
    #[error("precision table: {0}")]
    Parse(String),
    /// 2026-09-28: A format string no spelling matches.
    #[error("precision table entry `{pattern}`: {source}")]
    Format {
        /// 2026-09-28: The entry's `match`.
        pattern: String,
        /// 2026-09-28: The format error.
        source: FormatError,
    },
    /// 2026-09-28: The last entry is not `match = "*"`.
    #[error("precision table must end with a catch-all `match = \"*\"` entry")]
    NoCatchAll,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TableFile {
    schema: u32,
    checkpoint: String,
    tier: String,
    linear: Vec<EntryFile>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct EntryFile {
    #[serde(rename = "match")]
    pattern: String,
    weight: String,
    activation: String,
}

/// 2026-09-28: A glob-matched table of linear formats.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrecisionTable {
    /// 2026-09-28: The checkpoint the table describes.
    pub checkpoint: String,
    /// 2026-09-28: The `--weight-quantization` tier it describes.
    pub tier: String,
    entries: Vec<(String, LinearFormats)>,
    catch_all: LinearFormats,
}

impl PrecisionTable {
    /// 2026-09-28: Parse a table from its TOML text.
    pub fn parse(text: &str) -> Result<Self, PrecisionError> {
        let file: TableFile =
            toml::from_str(text).map_err(|e| PrecisionError::Parse(e.to_string()))?;
        if file.schema != 1 {
            return Err(PrecisionError::Parse(format!(
                "schema {} (this build reads 1)",
                file.schema
            )));
        }
        let mut entries = Vec::with_capacity(file.linear.len());
        for e in file.linear {
            let fmt = |s: &str| {
                Format::parse(s).map_err(|source| PrecisionError::Format {
                    pattern: e.pattern.clone(),
                    source,
                })
            };
            let formats = LinearFormats {
                weight: fmt(&e.weight)?,
                activation: fmt(&e.activation)?,
            };
            entries.push((e.pattern, formats));
        }
        let catch_all = match entries.pop() {
            Some((p, f)) if p == "*" => f,
            _ => return Err(PrecisionError::NoCatchAll),
        };
        Ok(PrecisionTable {
            checkpoint: file.checkpoint,
            tier: file.tier,
            entries,
            catch_all,
        })
    }
}

impl EdgePrecision for PrecisionTable {
    fn linear(&self, module: &str) -> LinearFormats {
        self.entries
            .iter()
            .find(|(p, _)| glob(p, module))
            .map_or(self.catch_all, |(_, f)| *f)
    }
}

/// 2026-09-28: `*` matches any run of characters, dots included; everything else is literal.
pub(crate) fn glob(pattern: &str, text: &str) -> bool {
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.len() == 1 {
        return pattern == text;
    }
    let (first, last) = (parts[0], parts[parts.len() - 1]);
    if !text.starts_with(first) || text.len() < first.len() + last.len() || !text.ends_with(last) {
        return false;
    }
    let mut rest = &text[first.len()..text.len() - last.len()];
    for mid in &parts[1..parts.len() - 1] {
        match rest.find(mid) {
            Some(at) => rest = &rest[at + mid.len()..],
            None => return false,
        }
    }
    true
}

#[cfg(test)]
#[path = "precision_tests.rs"]
mod precision_tests;
