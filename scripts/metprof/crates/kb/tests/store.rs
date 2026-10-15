// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the durable LatticeDB store behind the sink (PLAN W1.2):
//! upsert and history, deterministic export, replay of the spool, and the
//! refusal to mix embedding spaces. `DenseFake` stands behind the `Embedder`
//! port because this crate ships no dense embedder (its client lives outside,
//! ADDENDUM-EMBEDDINGS E1); it only has to be of another space.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::run;
use lattice_storage::MemStorage;
use metprof_kb::derive::{derive, run_key};
use metprof_kb::embed::{Embedder, Embedding, Hashed, Space};
use metprof_kb::lattice::Store;
use metprof_kb::lease::SystemLeaseEnv;
use metprof_kb::port::RunRecord;
use metprof_kb::schema::{Kind, MAX_SAMPLE_DEPTH, NodeRef, SUPERSEDES_KEPT};
use metprof_kb::sink::{KbPaths, LatticeSink, Materialized, block_on, open_durable};
use metprof_kb::{Result, canon, export};
use serde_json::{Value, json};

fn families() -> RunRecord {
    run(include_str!("fixtures/families_tree.json"))
}

/// The same capture and command (so the same ProfileRun) with another result.
fn with_result(n: u64) -> RunRecord {
    let mut r = families();
    r.result.insert("n".into(), json!(n));
    r
}

/// A different ProfileRun: another argument word.
fn other_run(n: u64) -> RunRecord {
    let mut r = families();
    r.args.push(metprof_kb::port::Arg::Word(format!("--x={n}")));
    r
}

/// Claims one space, gives vectors of another (equal unless it lies).
struct DenseFake {
    claims: Space,
    gives: Space,
}

impl Embedder for DenseFake {
    fn space(&self) -> &Space {
        &self.claims
    }
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        Ok(texts
            .iter()
            .map(|_| Embedding {
                space: self.gives.clone(),
                vector: vec![1.0; self.gives.dims()],
            })
            .collect())
    }
}

fn dense_space() -> Space {
    Space::dense("embed.example", "m", 256).expect("space")
}

fn dense() -> DenseFake {
    DenseFake {
        claims: dense_space(),
        gives: dense_space(),
    }
}

fn header(export: &[u8]) -> Value {
    let first = export.split(|b| *b == b'\n').next().expect("header line");
    serde_json::from_slice(first).expect("header JSON")
}

fn profile_runs(export: &[u8]) -> usize {
    export
        .split(|b| *b == b'\n')
        .filter(|l| {
            serde_json::from_slice::<Value>(l)
                .is_ok_and(|v| v.pointer("/payload/kind") == Some(&json!("ProfileRun")))
        })
        .count()
}

fn paths(root: &std::path::Path) -> KbPaths {
    KbPaths {
        spool_dir: root.join("spool"),
        durable_dir: root.join("db"),
        export_dir: root.join("export"),
    }
}

#[test]
fn a_record_upserted_twice_is_one_point_and_supersedes_on_change() {
    block_on(async {
        let h = Hashed::new();
        let mut store = Store::open(&Space::hashed(), MemStorage::new(), MemStorage::new())
            .await
            .expect("opens");
        let a = derive(&with_result(1), MAX_SAMPLE_DEPTH).expect("derives");
        let first = store.apply(&a, &h).await.expect("applies");
        assert_eq!(first.inserted, a.records.len());
        assert_eq!(first.edges_added, a.edges.len());
        let points = header(&export::render(&store).expect("exports"))["points"].clone();

        let again = store.apply(&a, &h).await.expect("applies");
        assert!(
            again.is_noop(),
            "same graph again writes nothing: {again:?}"
        );
        assert_eq!(again.unchanged, a.records.len());

        let mut later = with_result(1);
        later.stamps.recorded_at = "2026-02-02T00:00:00Z".into();
        let later = derive(&later, MAX_SAMPLE_DEPTH).expect("derives");
        assert!(
            store.apply(&later, &h).await.expect("applies").is_noop(),
            "recorded_at alone is no change"
        );

        let node = NodeRef::new(Kind::ProfileRun, run_key(&with_result(1)).expect("key"));
        let sha = |g: &metprof_kb::derive::Graph| {
            let r = g
                .records
                .iter()
                .find(|r| r.node() == node)
                .expect("run record");
            canon::content_sha256(r).expect("sha")
        };
        let mut want = Vec::new();
        let mut prev = sha(&a);
        for n in 2..=(SUPERSEDES_KEPT as u64 + 3) {
            let g = derive(&with_result(n), MAX_SAMPLE_DEPTH).expect("derives");
            let d = store.apply(&g, &h).await.expect("applies");
            assert_eq!((d.inserted, d.updated), (0, 1), "only the run changed");
            want.push(prev.clone());
            prev = sha(&g);
            let got = store.get(&node).expect("reads").expect("present");
            assert_eq!(got.content_sha256, prev);
            assert_eq!(got.record.recorded_at, "2026-01-01T00:00:00Z");
            let tail = want.len().saturating_sub(SUPERSEDES_KEPT);
            assert_eq!(got.supersedes, want[tail..].to_vec());
        }
        let out = export::render(&store).expect("exports");
        assert_eq!(header(&out)["points"], points, "one point per entity");
        assert_eq!(profile_runs(&out), 1);
    })
    .expect("runtime");
}

#[test]
fn export_is_byte_identical_on_repeat_and_across_reopen() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let h = Hashed::new();
    let env = SystemLeaseEnv;
    let sink = LatticeSink::new(paths(tmp.path()), &h, &env);
    let Materialized::Done { export, .. } = sink.materialize(&[families()]).expect("materialises")
    else {
        panic!("the lease is free")
    };
    let one = std::fs::read(&export).expect("export");
    assert_eq!(profile_runs(&one), 1);
    assert!(one.ends_with(b"\n"));
    let Materialized::Done { applied, .. } = sink.materialize(&[]).expect("exports") else {
        panic!("the lease is free")
    };
    assert!(applied.is_noop());
    assert_eq!(std::fs::read(&export).expect("export"), one, "repeat");

    let reopened = block_on(async {
        let store = open_durable(&tmp.path().join("db"), &h)
            .await
            .expect("reopens");
        export::render(&store).expect("exports")
    })
    .expect("runtime");
    assert_eq!(reopened, one, "after a WAL replay");
}

#[test]
fn replay_materialises_every_spool_file_and_is_idempotent() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let h = Hashed::new();
    let env = SystemLeaseEnv;
    let sink = LatticeSink::new(paths(tmp.path()), &h, &env);
    for n in 0..3 {
        sink.spool().write(&other_run(n)).expect("spools");
    }
    assert!(!tmp.path().join("db").exists(), "spooled only");

    let Materialized::Done { applied, export } = sink.replay().expect("replays") else {
        panic!("the lease is free")
    };
    assert!(applied.inserted > 3);
    let first = std::fs::read(&export).expect("export");
    assert_eq!(profile_runs(&first), 3);

    let Materialized::Done { applied, .. } = sink.replay().expect("replays") else {
        panic!("the lease is free")
    };
    assert!(
        applied.is_noop(),
        "second replay writes nothing: {applied:?}"
    );
    assert_eq!(std::fs::read(&export).expect("export"), first);
}

#[test]
fn a_store_refuses_an_embedder_or_points_of_another_space() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let h = Hashed::new();
    let env = SystemLeaseEnv;
    let sink = LatticeSink::new(paths(tmp.path()), &h, &env);
    sink.materialize(&[families()]).expect("materialises");

    let other = dense();
    block_on(async {
        let err = open_durable(&tmp.path().join("db"), &other)
            .await
            .err()
            .expect("refused");
        assert!(err.0.contains("never mix"), "{err}");

        let mut store = Store::open(&Space::hashed(), MemStorage::new(), MemStorage::new())
            .await
            .expect("opens");
        let g = derive(&families(), MAX_SAMPLE_DEPTH).expect("derives");
        let liar = DenseFake {
            claims: Space::hashed(),
            gives: dense_space(),
        };
        let err = store.apply(&g, &liar).await.expect_err("refused");
        assert!(err.0.contains("never mix"), "{err}");
        assert_eq!(
            header(&export::render(&store).expect("exports"))["points"],
            json!(0)
        );

        // Even with nothing to embed (every record unchanged) the space is checked.
        store.apply(&g, &h).await.expect("applies");
        let err = store.apply(&g, &other).await.expect_err("refused");
        assert!(err.0.contains("never mix"), "{err}");
    })
    .expect("runtime");
}
