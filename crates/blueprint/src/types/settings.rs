// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Typed serve settings: levers, flags, the chat template, precision provenance and
//! the host resources (unsigned).
//!
//! Owner: blueprint.
//! Invariants:
//! - No floats: a decimal is [`Fixed`] (`0.82` is `Fixed { mantissa: 82, exp: -2 }`), and a
//!   value is typed only when it renders back to exactly the text it was parsed from, so
//!   projecting a recipe and rendering it back is lossless.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};

use serde::{Deserialize, Serialize};

use crate::ids::Sha256;

/// 2026-10-22: A decimal `mantissa × 10^exp`, written with exactly `-exp` fraction digits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Fixed {
    /// 2026-10-22: The digits as an integer.
    pub mantissa: i64,
    /// 2026-10-22: Zero or minus the fraction digit count.
    pub exp: i8,
}

impl Fixed {
    /// 2026-10-22: Parse `[-]d+.d+`; `None` for anything else (NaN, inf, exponents) or text
    /// that would not render back identically.
    pub fn parse(s: &str) -> Option<Self> {
        let (int, frac) = s.split_once('.')?;
        let digits = int.strip_prefix('-').unwrap_or(int);
        if digits.is_empty()
            || frac.is_empty()
            || !digits
                .bytes()
                .chain(frac.bytes())
                .all(|b| b.is_ascii_digit())
        {
            return None;
        }
        let exp = -i8::try_from(frac.len()).ok()?;
        let mut joined = String::from(int);
        joined.push_str(frac);
        let v = Fixed {
            mantissa: joined.parse().ok()?,
            exp,
        };
        (v.render() == s).then_some(v)
    }

    /// 2026-10-22: The decimal text.
    pub fn render(self) -> String {
        let frac = usize::from(self.exp.unsigned_abs());
        let mut digits = self.mantissa.unsigned_abs().to_string();
        while digits.len() <= frac {
            digits.insert(0, '0');
        }
        let point = digits.len() - frac;
        let mut out = String::new();
        if self.mantissa < 0 {
            out.push('-');
        }
        out.push_str(&digits[..point]);
        if frac > 0 {
            out.push('.');
            out.push_str(&digits[point..]);
        }
        out
    }
}

fn int_of(s: &str) -> Option<i64> {
    let v: i64 = s.parse().ok()?;
    (v.to_string() == s).then_some(v)
}

/// 2026-10-22: A `METRALE_*` lever value, typed by the lever table's rule.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeverValue {
    /// 2026-10-22: On (`1`) or off (`0`).
    Switch(bool),
    /// 2026-10-22: Set (`1`).
    Presence,
    /// 2026-10-22: An integer.
    Int(i64),
    /// 2026-10-22: A decimal.
    Fixed(Fixed),
    /// 2026-10-22: Free text.
    Text(String),
    /// 2026-10-22: A path.
    Path(String),
}

impl LeverValue {
    /// 2026-10-22: The env text.
    pub fn render(&self) -> String {
        match self {
            LeverValue::Switch(true) | LeverValue::Presence => String::from("1"),
            LeverValue::Switch(false) => String::from("0"),
            LeverValue::Int(v) => v.to_string(),
            LeverValue::Fixed(v) => v.render(),
            LeverValue::Text(v) | LeverValue::Path(v) => v.clone(),
        }
    }

    /// 2026-10-22: Parse an integer lever value; `None` unless canonical.
    pub fn int(s: &str) -> Option<Self> {
        int_of(s).map(LeverValue::Int)
    }
}

/// 2026-10-22: A `met serve` flag value as the recipe wrote it, typed where the type renders
/// back exactly.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FlagValue {
    /// 2026-10-22: `true` / `false`.
    Bool(bool),
    /// 2026-10-22: An integer.
    Int(i64),
    /// 2026-10-22: A decimal.
    Fixed(Fixed),
    /// 2026-10-22: Anything else.
    Text(String),
}

impl FlagValue {
    /// 2026-10-22: Type `s`: total, and `render(parse(s)) == s` for every `s`.
    pub fn parse(s: &str) -> Self {
        match s {
            "true" => FlagValue::Bool(true),
            "false" => FlagValue::Bool(false),
            _ => int_of(s)
                .map(FlagValue::Int)
                .or_else(|| Fixed::parse(s).map(FlagValue::Fixed))
                .unwrap_or_else(|| FlagValue::Text(String::from(s))),
        }
    }

    /// 2026-10-22: The text.
    pub fn render(&self) -> String {
        match self {
            FlagValue::Bool(b) => b.to_string(),
            FlagValue::Int(v) => v.to_string(),
            FlagValue::Fixed(v) => v.render(),
            FlagValue::Text(v) => v.clone(),
        }
    }
}

/// 2026-10-22: The levers a serve runs under.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeSettings {
    /// 2026-10-22: Digest of the lever table these values were typed against; an unset lever
    /// is the table's default, so the table digest plus `levers` is the full resolution.
    pub registry_sha256: Sha256,
    /// 2026-10-22: Every lever the recipe sets, by env name.
    pub levers: BTreeMap<String, LeverValue>,
    /// 2026-10-22: Non-`METRALE_` variables the recipe sets (e.g. `NCCL_*`), verbatim.
    pub passthrough: BTreeMap<String, String>,
}

/// 2026-10-22: The chat template's content, pinned (its path is a host resource).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemplatePin {
    /// 2026-10-22: SHA-256 of `text`.
    pub sha256: Sha256,
    /// 2026-10-22: The template.
    pub text: String,
}

/// 2026-10-22: The serve flags (the recipe's `defaults:` minus host resources).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServePolicy {
    /// 2026-10-22: By recipe key.
    pub flags: BTreeMap<String, FlagValue>,
    /// 2026-10-22: The chat template, when the recipe names one.
    pub chat_template: Option<TemplatePin>,
}

/// 2026-10-22: Where the per-node formats came from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrecisionSpec {
    /// 2026-10-22: The precision table or checkpoint plan name (INSTANCES.toml `precision`).
    pub tier: String,
    /// 2026-10-22: SHA-256 of that file's text.
    pub table_sha256: Sha256,
}

/// 2026-10-22: Host resources: launch facts that are never signed (the unsigned section).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Resources {
    /// 2026-10-22: The recipe id (`family/stem`) it was projected from.
    pub recipe_id: String,
    /// 2026-10-22: `recipe_version`.
    pub recipe_version: String,
    /// 2026-10-22: `runtime`.
    pub runtime: Option<String>,
    /// 2026-10-22: The container image.
    pub container: String,
    /// 2026-10-22: Host flags by recipe key: `port`, `host`, the chat template path.
    pub flags: BTreeMap<String, String>,
}
