// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: kg_status degrades per section (an unavailable section is a declared mode, the
//! others still answer), and id/campaign refusals name what the caller probably meant.

mod common;

use metrale_kg_derive::derive;
use metrale_kg_derive::queries::resolve_id;
use metrale_kg_derive::status::{Freshness, status};

fn fresh() -> Freshness {
    // 2026-10-23: test value: no failed re-derive, epoch time (the clock is not under test).
    Freshness {
        stale_reason: None,
        derived_at_ms: 0,
        source_commit: Some("381f9cb6ec3b".into()),
        newest_change: Some(("c/LEDGER.tsv".into(), 1_792_000_000_000)),
    }
}

#[test]
fn one_unavailable_section_does_not_take_down_the_others() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let v = status(&b, &common::config(), "glm/mk", &[], &fresh(), None).unwrap();
    assert_eq!(v["owed_lkb"]["unavailable"], true, "{}", v["owed_lkb"]);
    assert_eq!(v["owed_lkb"]["code"], "KgSource");
    assert!(
        v["progress"].is_object() && v["blocked"]["first"].is_array(),
        "{v}"
    );
    assert!(v["progress"]["P0"].is_object(), "{}", v["progress"]);
}

#[test]
fn an_unknown_section_or_campaign_is_refused_with_the_known_names() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let e = status(
        &b,
        &common::config(),
        "glm/mk",
        &["progres".into()],
        &fresh(),
        None,
    )
    .unwrap_err()
    .to_string();
    assert!(
        e.contains("progres") && e.contains("progress, blocked"),
        "{e}"
    );
    let e = status(&b, &common::config(), "glm/megakernel", &[], &fresh(), None)
        .unwrap_err()
        .to_string();
    assert!(e.contains("known: glm/mk"), "{e}");
}

#[test]
fn a_bare_or_misspelled_id_is_refused_with_the_closest_ids() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    assert_eq!(
        resolve_id(&b, "ledger:seg").unwrap().to_string(),
        "ledger:seg"
    );
    let bare = resolve_id(&b, "seg").unwrap_err().to_string();
    assert!(
        bare.contains("did you mean ledger:seg")
            && bare.contains("prefixes: ")
            && bare.contains(", ledger"),
        "{bare}"
    );
    let wrong = resolve_id(&b, "ledgr:seg").unwrap_err().to_string();
    assert!(
        wrong.contains("unknown label prefix") && wrong.contains("ledger:seg"),
        "{wrong}"
    );
    let typo = resolve_id(&b, "ledger:sgg").unwrap_err().to_string();
    assert!(
        typo.contains("no node ledger:sgg") && typo.contains("ledger:seg"),
        "{typo}"
    );
    let far = resolve_id(&b, "ledger:zzzzzzzzzzzz")
        .unwrap_err()
        .to_string();
    assert!(!far.contains("did you mean"), "{far}");
}

#[test]
fn freshness_names_the_source_commit_and_null_without_an_lkb() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let s = ["freshness".to_string()];
    let v = status(&b, &common::config(), "glm/mk", &s, &fresh(), None).unwrap();
    assert_eq!(v["freshness"]["source_commit"], "381f9cb6ec3b");
    let none = Freshness {
        source_commit: None,
        ..fresh()
    };
    let v = status(&b, &common::config(), "glm/mk", &s, &none, None).unwrap();
    assert!(v["freshness"]["source_commit"].is_null(), "{v}");
}

#[test]
fn a_default_status_caps_every_list_and_says_how_many_there_were() {
    use metrale_kg_derive::status::compact;
    use metrale_kg_derive::tools::LIST_CAP;
    let v = compact(
        serde_json::json!({"rows": [1, 2, 3, 4, 5], "n": {"refs": ["a", "b", "c", "d"]}, "few": [1]}),
    );
    assert_eq!(v["rows"].as_array().unwrap().len(), LIST_CAP);
    assert_eq!(v["rows_total"], 5);
    assert_eq!(v["n"]["refs_total"], 4);
    assert_eq!(v["few"], serde_json::json!([1]));
    let top = compact(serde_json::json!([1, 2, 3, 4]));
    assert_eq!(top["total"], 4);
    fn longest(v: &serde_json::Value) -> usize {
        match v {
            serde_json::Value::Array(a) => a.len().max(a.iter().map(longest).max().unwrap_or(0)),
            serde_json::Value::Object(o) => o.values().map(longest).max().unwrap_or(0),
            _ => 0,
        }
    }
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let d = status(&b, &common::config(), "glm/mk", &[], &fresh(), None).unwrap();
    assert!(longest(&d) <= LIST_CAP + 1, "{d}");
    assert!(d.get("schema").is_none() && d.get("files").is_none());
    let named = status(
        &b,
        &common::config(),
        "glm/mk",
        &["freshness".to_string()],
        &fresh(),
        None,
    )
    .unwrap();
    assert!(named["freshness"].get("files").is_some_and(|f| f.is_u64()));
    assert_eq!(named["freshness"]["newest_change"]["path"], "c/LEDGER.tsv");
}
