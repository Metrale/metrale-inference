// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Dim predicates: each test, their conjunction, and every malformed form.

use super::*;

fn file(eq: Option<u64>, min: Option<u64>, max: Option<u64>, m: Option<u64>, r: Option<u64>) -> DimPredFile {
    DimPredFile {
        eq,
        min,
        max,
        modulo: m,
        rem: r,
    }
}

fn one(f: DimPredFile) -> Result<DimPred, String> {
    parse([("moe_inter".to_string(), f)].into_iter().collect()).map(|m| m["moe_inter"])
}

/// 2026-10-23: The k64 split: 704 % 128 == 64 takes the twins, 640 and 2048 do not.
#[test]
fn modulo_tells_the_k64_slice_from_the_plain_one() {
    let k64 = one(file(None, None, None, Some(128), Some(64))).unwrap();
    let plain = one(file(None, None, None, Some(128), Some(0))).unwrap();
    for (v, twin) in [(704, true), (640, false), (2048, false), (64, true)] {
        assert_eq!(k64.holds(v), twin, "{v}");
        assert_eq!(plain.holds(v), !twin, "{v}");
    }
    assert_eq!(k64.describe(), "%128=64");
}

#[test]
fn every_stated_test_must_hold() {
    let p = one(file(None, Some(2), Some(16), Some(2), Some(0))).unwrap();
    assert!(p.holds(2) && p.holds(16) && p.holds(8));
    assert!(!p.holds(1), "below min");
    assert!(!p.holds(18), "above max");
    assert!(!p.holds(7), "odd");
    let e = one(file(Some(704), None, None, None, None)).unwrap();
    assert!(e.holds(704) && !e.holds(705));
    assert_eq!(p.describe(), ">=2,<=16,%2=0");
}

#[test]
fn malformed_predicates_are_refused() {
    assert!(one(file(None, None, None, None, None)).is_err(), "no test");
    assert!(one(file(None, None, None, Some(128), None)).is_err(), "mod without rem");
    assert!(one(file(None, None, None, None, Some(0))).is_err(), "rem without mod");
    assert!(one(file(None, None, None, Some(0), Some(0))).is_err(), "mod 0");
    assert!(one(file(None, None, None, Some(128), Some(128))).is_err(), "rem >= mod");
}

#[test]
fn an_undefined_dim_is_named_not_read_as_false() {
    let preds = parse([("moe_inter".to_string(), file(Some(1), None, None, None, None))].into_iter().collect())
        .unwrap();
    let dims: BTreeMap<String, u64> = [("hidden".to_string(), 1)].into_iter().collect();
    assert_eq!(all_hold(&preds, &dims), Err("moe_inter".to_string()));
    let dims: BTreeMap<String, u64> = [("moe_inter".to_string(), 1)].into_iter().collect();
    assert_eq!(all_hold(&preds, &dims), Ok(true));
    let dims: BTreeMap<String, u64> = [("moe_inter".to_string(), 2)].into_iter().collect();
    assert_eq!(all_hold(&preds, &dims), Ok(false));
}
