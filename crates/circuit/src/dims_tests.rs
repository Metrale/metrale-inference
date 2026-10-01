// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: Dimension expressions.
//!
//! Owner: metrale-circuit.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;

fn dims() -> BTreeMap<String, u64> {
    [("q_heads", 24), ("head_dim", 256), ("n", 3), ("top_k", 8)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect()
}

#[test]
fn products_and_sums_evaluate_left_to_right_by_precedence() {
    let e = DimExpr::parse("q_heads*head_dim*2 + n*top_k").unwrap();
    assert_eq!(e.eval(&dims()), Ok(24 * 256 * 2 + 3 * 8));
    assert_eq!(e.text(), "q_heads*head_dim*2+n*top_k");
    let names: Vec<&str> = e.names().collect();
    assert_eq!(names, ["q_heads", "head_dim", "n", "top_k"]);
}

#[test]
fn an_unknown_name_is_reported_not_zeroed() {
    let e = DimExpr::parse("hidden*2").unwrap();
    assert_eq!(
        e.eval(&dims()),
        Err(DimError::Unknown {
            name: "hidden".into(),
            expr: "hidden*2".into()
        })
    );
}

#[test]
fn syntax_errors_are_refused() {
    for s in ["", "a+", "*a", "0", "a*0", "Hidden", "a-b", "a**b", "2x"] {
        assert!(
            matches!(DimExpr::parse(s), Err(DimError::Syntax(_))),
            "{s:?} parsed"
        );
    }
}

#[test]
fn overflow_is_an_error() {
    let mut d = BTreeMap::new();
    d.insert("big".to_string(), u64::MAX);
    assert_eq!(
        DimExpr::parse("big*2").unwrap().eval(&d),
        Err(DimError::Overflow("big*2".into()))
    );
    assert_eq!(
        DimExpr::parse("big+1").unwrap().eval(&d),
        Err(DimError::Overflow("big+1".into()))
    );
}
