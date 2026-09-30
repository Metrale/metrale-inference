// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Dimension expressions: a sum of products of named dims and integer literals,
//! e.g. `q_heads*head_dim*2` or `lin_qk*2+lin_v`.
//!
//! Owner: metrale-circuit.
//! Invariants:
//! - An expression keeps its source text, so a plan renders the shape the template wrote.
//! - Evaluation names every unknown dim and refuses overflow; it never substitutes a value.

use std::collections::BTreeMap;

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
}

/// 2026-09-28: Why a dimension expression did not parse or evaluate.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DimError {
    /// 2026-09-28: Empty term, stray character or a literal of zero.
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
        self.terms.iter().flatten().filter_map(|f| match f {
            Factor::Name(n) => Some(n.as_str()),
            Factor::Lit(_) => None,
        })
    }

    /// 2026-09-28: The value under `dims`.
    pub fn eval(&self, dims: &BTreeMap<String, u64>) -> Result<u64, DimError> {
        let overflow = || DimError::Overflow(self.text.clone());
        let mut sum: u64 = 0;
        for term in &self.terms {
            let mut prod: u64 = 1;
            for f in term {
                let v = match f {
                    Factor::Lit(v) => *v,
                    Factor::Name(n) => *dims.get(n).ok_or_else(|| DimError::Unknown {
                        name: n.clone(),
                        expr: self.text.clone(),
                    })?,
                };
                prod = prod.checked_mul(v).ok_or_else(overflow)?;
            }
            sum = sum.checked_add(prod).ok_or_else(overflow)?;
        }
        Ok(sum)
    }
}

fn parse_factor(f: &str) -> Option<Factor> {
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
