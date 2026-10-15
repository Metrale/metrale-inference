// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the spool and the lease (PLAN W1.2), on temporary directories.
//! The lease runs against `FakeEnv`, a fake behind the `LeaseEnv` port: its
//! clock only moves when the code under test sleeps, so a 30 s wait costs no
//! wall time and its length is exact; liveness is a fixed pid set.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::cell::Cell;
use std::collections::BTreeSet;
use std::fs;
use std::time::Duration;

use common::run;
use metprof_kb::embed::Hashed;
use metprof_kb::lease::{
    Acquired, HOLDER_FILE, Holder, LEASE_DIR, LeaseEnv, STALE_AFTER_S, WAIT_S, acquire,
};
use metprof_kb::port::{KbSink, RunRecord};
use metprof_kb::sink::{KbPaths, LatticeSink, Sealing};
use metprof_kb::spool::{Spool, decode, file_name};

const T0_MS: u64 = 1_800_000_000_000;

struct FakeEnv {
    pid: u32,
    live: BTreeSet<u32>,
    now_ms: Cell<u64>,
    slept_ms: Cell<u64>,
}

impl FakeEnv {
    fn new(pid: u32, live: &[u32]) -> FakeEnv {
        FakeEnv {
            pid,
            live: live.iter().copied().collect(),
            now_ms: Cell::new(T0_MS),
            slept_ms: Cell::new(0),
        }
    }
}

impl LeaseEnv for FakeEnv {
    fn pid(&self) -> u32 {
        self.pid
    }
    fn alive(&self, pid: u32) -> bool {
        self.live.contains(&pid)
    }
    fn now_s(&self) -> u64 {
        self.now_ms.get() / 1000
    }
    fn sleep(&self, d: Duration) {
        let ms = d.as_millis() as u64;
        self.now_ms.set(self.now_ms.get() + ms);
        self.slept_ms.set(self.slept_ms.get() + ms);
    }
}

fn families() -> RunRecord {
    run(include_str!("fixtures/families_tree.json"))
}

fn plant(dir: &std::path::Path, h: Holder) {
    let lease = dir.join(LEASE_DIR);
    fs::create_dir_all(&lease).expect("lease dir");
    fs::write(lease.join(HOLDER_FILE), h.encode()).expect("holder");
}

fn holder_of(a: &Acquired) -> Holder {
    match a {
        Acquired::Held(l) => l.holder(),
        Acquired::Busy(h) => panic!("expected the lease, got Busy({h:?})"),
    }
}

#[test]
fn a_crash_between_write_and_rename_leaves_no_spool_file() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let spool = Spool::new(tmp.path().join("spool"));
    let r = families();

    let staged = spool.stage(&r).expect("stages");
    drop(staged); // the process dies before the rename
    assert!(
        spool.list().expect("lists").is_empty(),
        "a staged file is not in the spool"
    );

    let path = spool.write(&r).expect("spools");
    assert_eq!(spool.list().expect("lists"), vec![path.clone()]);
    let name = path
        .file_name()
        .expect("name")
        .to_string_lossy()
        .to_string();
    assert_eq!(name, file_name(&r).expect("name"));
    assert!(name.starts_with("2026-01-01T00:00:00Z-") && name.ends_with(".kkb.jsonl"));
    assert_eq!(Spool::read(&path).expect("reads"), r);

    let bytes = fs::read(&path).expect("bytes");
    assert!(
        decode(&bytes[..bytes.len() - 1]).is_err(),
        "no newline: refused"
    );
    assert!(
        decode(&bytes[..bytes.len() / 2]).is_err(),
        "truncated: refused"
    );
    let mut spaced = b" ".to_vec();
    spaced.extend_from_slice(&bytes);
    assert!(decode(&spaced).is_err(), "not canonical: refused");

    let mut bad = r.clone();
    bad.stamps.recorded_at = "../../x".into();
    assert!(file_name(&bad).is_err(), "recorded_at never becomes a path");
}

#[test]
fn a_second_writer_waits_the_full_window_then_is_busy() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let durable = tmp.path().join("db");
    let first = FakeEnv::new(100, &[100, 200]);
    let second = FakeEnv::new(200, &[100, 200]);

    let held = acquire(&durable, &first).expect("acquires");
    let mine = holder_of(&held);
    assert_eq!(
        mine,
        Holder {
            pid: 100,
            since: T0_MS / 1000
        }
    );

    match acquire(&durable, &second).expect("acquire") {
        Acquired::Busy(h) => assert_eq!(h, Some(mine)),
        Acquired::Held(_) => panic!("a live lease must not be taken"),
    }
    let waited = second.slept_ms.get();
    assert!(
        (WAIT_S * 1000..WAIT_S * 1000 + 1000).contains(&waited),
        "waited {waited} ms, not the {WAIT_S} s window"
    );

    // The sink records the run anyway and says why it is not materialised.
    let embedder = Hashed::new();
    let paths = KbPaths {
        spool_dir: tmp.path().join("spool"),
        durable_dir: durable.clone(),
        export_dir: tmp.path().join("export"),
    };
    let sink = LatticeSink::new(paths, &embedder, &second, &Sealing::Off);
    let rec = sink.record(&families()).expect("spooled");
    assert!(!rec.materialized);
    assert!(
        rec.reason
            .as_deref()
            .is_some_and(|r| r.contains("pid 100") && r.contains("replay"))
    );
    assert_eq!(sink.spool().list().expect("lists").len(), 1);
    assert!(
        !tmp.path().join("export").exists(),
        "nothing exported without the lease"
    );

    let Acquired::Held(lease) = held else {
        unreachable!()
    };
    lease.release().expect("releases");
    assert!(!durable.join(LEASE_DIR).exists());
    let again = acquire(&durable, &second).expect("acquires");
    assert_eq!(holder_of(&again).pid, 200);
}

#[test]
fn a_stale_lease_is_taken_and_a_fresh_one_is_not() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let durable = tmp.path().join("db");
    let env = FakeEnv::new(200, &[100, 200]);
    let now = T0_MS / 1000;

    plant(
        &durable,
        Holder {
            pid: 300,
            since: now,
        },
    );
    let got = acquire(&durable, &env).expect("acquires");
    assert_eq!(holder_of(&got).pid, 200, "a dead holder's lease is taken");
    assert_eq!(env.slept_ms.get(), 0, "without waiting");
    drop(got);
    assert!(!durable.join(LEASE_DIR).exists(), "dropping releases");

    plant(
        &durable,
        Holder {
            pid: 100,
            since: now - STALE_AFTER_S - 1,
        },
    );
    let got = acquire(&durable, &env).expect("acquires");
    assert_eq!(
        holder_of(&got).pid,
        200,
        "a lease older than the limit is taken"
    );
    drop(got);

    // Aged exactly the limit when the wait ends: still live, so Busy.
    plant(
        &durable,
        Holder {
            pid: 100,
            since: now - STALE_AFTER_S + WAIT_S,
        },
    );
    match acquire(&durable, &env).expect("acquire") {
        Acquired::Busy(h) => assert_eq!(h.map(|h| h.pid), Some(100)),
        Acquired::Held(_) => panic!("a live lease at the limit is not stale"),
    }
    let leftovers: Vec<_> = fs::read_dir(&durable)
        .expect("lists")
        .map(|e| e.expect("entry").file_name())
        .collect();
    assert_eq!(
        leftovers,
        vec![std::ffi::OsString::from(LEASE_DIR)],
        "stale leases are deleted"
    );
}
