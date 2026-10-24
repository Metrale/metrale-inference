// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Dimension expressions: a sum of products of named dims and integer literals,
//! e.g. `q_heads*head_dim*2` or `lin_qk*2+lin_v`. 2026-10-02: A factor may be a dim divided by a
//! literal, rounded up (`moe_inter/128`: the scale count of a 128-wide group). 2026-10-08: Or by
//! another dim, rounded up (`index_head_dim/index_kpool`: a pooled key's share of one token).
//! 2026-10-18: Divisions chain left to right, each step rounded up (`head_dim/2/csa_ratio`: the
//! FP4 bytes of a latent row pooled over `csa_ratio` tokens, per token).
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - An expression keeps its source text, so a plan renders the shape the template wrote.
//! - Evaluation names every unknown dim and refuses overflow; it never substitutes a value.

use crate::prelude::*;

use alloc::collections::BTreeMap;

/// 2026-09-28: A parsed dimension expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimExpr {
    text: String,
    terms: Vec<Vec<Factor>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Factor {
    Name(String),
    Lit(u64),
    /// 2026-10-02: `name/by/by…`, each step rounded up (2026-10-18: chained).
    CeilDiv(String, Vec<Divisor>),
}

/// 2026-10-08: One divisor of a [`Factor::CeilDiv`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum Divisor {
    /// 2026-10-02: An integer literal, never zero.
    Lit(u64),
    /// 2026-10-08: A dim; zero at evaluation is an error.
    Dim(String),
}

/// 2026-09-28: Why a dimension expression did not parse or evaluate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DimError {
    /// 2026-09-28: Empty term, stray character or a literal of zero. 2026-10-08: Also a division
    /// by a dim whose value is zero.
    #[error("bad dimension expression `{0}`")]
    Syntax(String),
    /// 2026-09-28: A name the arch shape does not define.
    #[error("dimension `{name}` in `{expr}` is not defined by the arch shape")]
    Unknown {
        /// 2026-09-28: The missing name.
        name: String,
        /// 2026-09-28: The expression that named it.
        expr: String,
    },
    /// 2026-09-28: The value does not fit in u64.
    #[error("dimension expression `{0}` overflows")]
    Overflow(String),
}

/// 2026-10-22: Serialized as its text (whitespace removed), the one canonical form; a decoded
/// text is parsed again, so the terms always agree with the text.
impl serde::Serialize for DimExpr {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.text)
    }
}

impl<'de> serde::Deserialize<'de> for DimExpr {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let text = String::deserialize(d)?;
        DimExpr::parse(&text).map_err(serde::de::Error::custom)
    }
}

impl DimExpr {
    /// 2026-09-28: Parse `text`. Whitespace is ignored.
    pub fn parse(text: &str) -> Result<Self, DimError> {
        let syntax = || DimError::Syntax(text.to_string());
        let compact: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.is_empty() {
            return Err(syntax());
        }
        let mut terms = Vec::new();
        for term in compact.split('+') {
            let mut factors = Vec::new();
            for f in term.split('*') {
                factors.push(parse_factor(f).ok_or_else(syntax)?);
            }
            terms.push(factors);
        }
        Ok(DimExpr {
            text: compact,
            terms,
        })
    }

    /// 2026-09-28: The expression as written, whitespace removed.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// 2026-09-28: Every name the expression reads.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.terms.iter().flatten().flat_map(|f| match f {
            Factor::Name(n) => vec![n.as_str()],
            Factor::CeilDiv(n, by) => core::iter::once(n.as_str())
                .chain(by.iter().filter_map(|d| match d {
                    Divisor::Dim(d) => Some(d.as_str()),
                    Divisor::Lit(_) => None,
                }))
                .collect(),
            Factor::Lit(_) => Vec::new(),
        })
    }

    /// 2026-09-28: The value under `dims`.
    pub fn eval(&self, dims: &BTreeMap<String, u64>) -> Result<u64, DimError> {
        let overflow = || DimError::Overflow(self.text.clone());
        let mut sum: u64 = 0;
        for term in &self.terms {
            let mut prod: u64 = 1;
            for f in term {
                let dim = |n: &String| {
                    dims.get(n).copied().ok_or_else(|| DimError::Unknown {
                        name: n.clone(),
                        expr: self.text.clone(),
                    })
                };
                let v = match f {
                    Factor::Lit(v) => *v,
                    Factor::Name(n) => dim(n)?,
                    Factor::CeilDiv(n, by) => {
                        let mut v = dim(n)?;
                        for d in by {
                            v = match d {
                                Divisor::Lit(l) => v.div_ceil(*l),
                                Divisor::Dim(d) => match dim(d)? {
                                    0 => return Err(DimError::Syntax(self.text.clone())),
                                    l => v.div_ceil(l),
                                },
                            };
                        }
                        v
                    }
                };
                prod = prod.checked_mul(v).ok_or_else(overflow)?;
            }
            sum = sum.checked_add(prod).ok_or_else(overflow)?;
        }
        Ok(sum)
    }
}

fn parse_factor(f: &str) -> Option<Factor> {
    if let Some((name, rest)) = f.split_once('/') {
        let Factor::Name(n) = parse_factor(name)? else {
            return None;
        };
        let by = rest
            .split('/')
            .map(|d| match parse_factor(d)? {
                Factor::Lit(l) => Some(Divisor::Lit(l)),
                Factor::Name(d) => Some(Divisor::Dim(d)),
                Factor::CeilDiv(..) => None,
            })
            .collect::<Option<Vec<_>>>()?;
        return Some(Factor::CeilDiv(n, by));
    }
    let first = f.chars().next()?;
    if first.is_ascii_digit() {
        return f.parse::<u64>().ok().filter(|&v| v > 0).map(Factor::Lit);
    }
    let ident = (first.is_ascii_lowercase() || first == '_')
        && f.chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    ident.then(|| Factor::Name(f.to_string()))
}

#[cfg(test)]
#[path = "dims_tests.rs"]
mod dims_tests;
