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
    for s in [
        "", "a+", "*a", "0", "a*0", "Hidden", "a-b", "a**b", "2x", "a/0", "4/2", "a/2/2",
    ] {
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

/// 2026-10-02: `name/lit` divides the dim by the literal, rounding up, and names the dim.
#[test]
fn a_ceil_division_rounds_up_and_names_its_dim() {
    let e = DimExpr::parse("n*head_dim/100*4").unwrap();
    assert_eq!(e.eval(&dims()), Ok(3 * 3 * 4));
    assert_eq!(e.names().collect::<Vec<_>>(), ["n", "head_dim"]);
    assert_eq!(DimExpr::parse("q_heads/24").unwrap().eval(&dims()), Ok(1));
}

/// 2026-10-08: `name/name` divides by the other dim, rounding up, and names both; a divisor dim
/// of zero, an unknown divisor and a nested division are refused.
#[test]
fn a_division_by_a_dim_rounds_up_and_names_both() {
    let e = DimExpr::parse("head_dim/top_k").unwrap();
    assert_eq!(e.eval(&dims()), Ok(32));
    assert_eq!(e.names().collect::<Vec<_>>(), ["head_dim", "top_k"]);
    assert_eq!(DimExpr::parse("q_heads/n*2").unwrap().eval(&dims()), Ok(16));
    assert_eq!(DimExpr::parse("head_dim/n").unwrap().eval(&dims()), Ok(86));
    let mut zero = dims();
    zero.insert("z".into(), 0);
    assert_eq!(
        DimExpr::parse("head_dim/z").unwrap().eval(&zero),
        Err(DimError::Syntax("head_dim/z".into()))
    );
    assert!(matches!(
        DimExpr::parse("head_dim/kpool").unwrap().eval(&dims()),
        Err(DimError::Unknown { name, .. }) if name == "kpool"
    ));
    assert!(DimExpr::parse("4/head_dim").is_err());
    assert!(DimExpr::parse("head_dim/top_k/2").is_err());
}
