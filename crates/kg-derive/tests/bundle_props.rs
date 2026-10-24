// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The bundle invariants (queries part §5 `bundle_fixpoint`, design §4.1) and the
//! secrets denylist (owner decision 7), over the in-memory fixture tree.

mod common;

use metrale_kg_derive::jsonl::{digest, render};
use metrale_kg_derive::{KgId, Label, derive};

#[test]
fn derive_twice_and_in_any_read_order_gives_one_digest() {
    let cfg = common::config();
    let a = derive(&cfg, &common::sources(), None).unwrap();
    let b = derive(&cfg, &common::sources(), None).unwrap();
    let rev: Vec<usize> = (0..common::FILE_COUNT).rev().collect();
    let c = derive(&cfg, &common::sources_in(&rev), None).unwrap();
    assert_eq!(render(&a), render(&b));
    assert_eq!(digest(&a), digest(&c));
    assert_eq!(render(&a), render(&c));
}

#[test]
fn one_byte_of_one_hypothesis_changes_one_node_sha_one_text_sha_and_the_digest() {
    let cfg = common::config();
    let a = derive(&cfg, &common::sources(), None).unwrap();
    let mut src = common::sources();
    let ledger = src.files.get_mut("c/LEDGER.tsv").unwrap();
    *ledger = ledger.replace("dense segment", "dense segmenT");
    let b = derive(&cfg, &src, None).unwrap();
    let changed_nodes: Vec<String> = a
        .nodes
        .iter()
        .zip(&b.nodes)
        .filter(|(x, y)| x.sha != y.sha)
        .map(|(x, _)| x.id.to_string())
        .collect();
    let changed_texts: Vec<String> = a
        .texts
        .iter()
        .zip(&b.texts)
        .filter(|(x, y)| x.sha != y.sha)
        .map(|(x, _)| format!("{}.{}", x.node, x.field))
        .collect();
    assert_eq!(a.nodes.len(), b.nodes.len());
    assert_eq!(changed_nodes, ["ledger:seg"]);
    assert_eq!(changed_texts, ["ledger:seg.hypothesis"]);
    assert_ne!(digest(&a), digest(&b));
}

#[test]
fn a_seeded_secret_never_reaches_the_bundle_and_is_reported_by_pattern_name() {
    let cfg = common::config();
    let mut src = common::sources();
    let fake = "sk-or-v1-FAKEFAKEFAKE0000";
    for (path, needle) in [
        ("mem/project_dense_seg.md", "dense segment LOSS"),
        ("c/TOOLING.md", "pkill -f matched my shell"),
        ("c/LEDGER.tsv", "the read is at 89% of floor"),
    ] {
        let t = src.files.get_mut(path).unwrap();
        let seeded = t.replace(needle, &format!("{needle} {fake}"));
        assert_ne!(*t, seeded, "seed must apply to {path}");
        *t = seeded;
    }
    src.add("mem/github.pat.txt", format!("ghp_{fake}"));
    let b = derive(&cfg, &src, None).unwrap();
    let out = render(&b);
    assert!(!out.contains("FAKEFAKEFAKE"), "a secret reached the bundle");
    assert!(
        !out.contains("cfat_SECRETVALUE"),
        "a credential file's contents reached the bundle"
    );
    assert!(!out.contains("ghp_"));
    let redacted: Vec<&str> = b
        .errors
        .iter()
        .filter(|f| f.code == "KgSecretRedacted")
        .map(|f| f.msg.as_str())
        .collect();
    assert!(redacted.len() >= 3, "{redacted:?}");
    assert!(
        redacted.iter().all(|m| m.contains("openrouter-key")),
        "{redacted:?}"
    );
    let cf = KgId::new(Label::Memory, "reference_cloudflare_tokens_metrale").unwrap();
    assert_eq!(b.node(&cf).unwrap().p["redacted"], true);
    assert!(b.texts.iter().all(|t| !t.text.contains("sk-or-")));
}

#[test]
fn dangling_references_become_unresolved_nodes_not_dropped_edges() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    let u = KgId::parse("unresolved:memory:missing_one").unwrap();
    assert!(b.node(&u).is_some());
    let from = KgId::parse("feedback:feedback_tests_prove_the_bug").unwrap();
    assert!(b.edges.iter().any(|e| e.from == from && e.to == u));
    assert!(
        b.errors
            .iter()
            .any(|f| f.code == "KgForwardLink" && f.msg.contains("missing_one"))
    );
    assert!(
        !b.errors
            .iter()
            .any(|f| f.code == "KgDanglingRef" && f.msg.contains("missing_one"))
    );
    let names = |v: Vec<(KgId, Vec<KgId>)>| -> Vec<String> {
        v.into_iter().map(|(u, _)| u.to_string()).collect()
    };
    // A non-memory dangling ref (the unnamed blocker of p0-x) must land in `dangling`, not in
    // `forward_links`, so the split is tested in both directions.
    let mut src = common::sources();
    let l = src.files.get_mut("c/LEDGER.tsv").unwrap();
    *l = l.replace("blocked on decision:2026-10-22/p0-split, ", "");
    let b = derive(&common::config(), &src, None).unwrap();
    let fwd = names(metrale_kg_derive::queries::forward_links(&b));
    let dang = names(metrale_kg_derive::queries::dangling(&b));
    assert_eq!(fwd, ["unresolved:memory:missing_one"]);
    assert!(
        dang.iter().any(|d| d.starts_with("unresolved:gate:")),
        "{dang:?}"
    );
    assert!(
        !dang.iter().any(|d| d.starts_with("unresolved:memory:")),
        "{dang:?}"
    );
    assert!(b.fatal().is_empty(), "{:?}", b.fatal());
}

#[test]
fn every_node_has_a_source_line_and_every_edge_endpoint_is_a_node() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    for n in &b.nodes {
        let (path, line) = n.src.rsplit_once(':').unwrap();
        assert!(!path.is_empty() && line.parse::<u32>().is_ok(), "{}", n.src);
    }
    for e in &b.edges {
        assert!(
            b.node(&e.from).is_some() && b.node(&e.to).is_some(),
            "{e:?}"
        );
    }
}

#[test]
fn a_non_table_tooling_line_is_a_bad_row_finding() {
    let b = derive(&common::config(), &common::sources(), None).unwrap();
    assert!(
        b.errors
            .iter()
            .any(|f| f.code == "KgBadRow" && f.at == "c/TOOLING.md:7"),
        "{:?}",
        b.errors
    );
    assert_eq!(b.of(Label::Friction).count(), 2);
}
