// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the spool and the lease (PLAN W1.2), on temporary directories.
//! The lease is a real file lock; the waiting runs against `FakeEnv`, a fake
//! behind the `LeaseEnv` port whose clock only moves when the code under test
//! sleeps, so a 30 s wait costs no wall time and its length is exact. A second
//! writer in the same process opens its own descriptor (file locks conflict
//! between descriptors); the crash test runs the holder as a real child
//! process (this test binary re-run on `lease_child`) and kills it.
#![cfg(not(target_arch = "wasm32"))]

mod common;

use std::cell::Cell;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use common::run;
use metprof_kb::atomic;
use metprof_kb::embed::Hashed;
use metprof_kb::lease::{
    Acquired, Holder, LOCK_FILE, LeaseEnv, SystemLeaseEnv, WAIT_S, acquire, read_holder,
};
use metprof_kb::port::{KbSink, RunRecord};
use metprof_kb::sink::{KbPaths, LatticeSink, Sealing};
use metprof_kb::spool::{Spool, decode, file_name};
use metprof_kb::{Error, Result};

const T0_MS: u64 = 1_800_000_000_000;
const CHILD_DIR: &str = "METPROF_KB_LEASE_CHILD_DIR";
const HELD: &str = "lease-child: held";

struct FakeEnv {
    pid: u32,
    now_ms: Cell<u64>,
    slept_ms: Cell<u64>,
    clock_broken: bool,
}

impl FakeEnv {
    fn new(pid: u32) -> FakeEnv {
        FakeEnv {
            pid,
            now_ms: Cell::new(T0_MS),
            slept_ms: Cell::new(0),
            clock_broken: false,
        }
    }
}

impl LeaseEnv for FakeEnv {
    fn pid(&self) -> u32 {
        self.pid
    }
    fn now_s(&self) -> Result<u64> {
        if self.clock_broken {
            return Err(Error("fake clock: broken".into()));
        }
        Ok(self.now_ms.get() / 1000)
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

fn holder_of(a: &Acquired) -> Holder {
    match a {
        Acquired::Held(l) => l.holder(),
        Acquired::Busy(h) => panic!("expected the lease, got Busy({h:?})"),
    }
}

fn busy_holder(a: Acquired) -> Option<Holder> {
    match a {
        Acquired::Busy(h) => h,
        Acquired::Held(l) => panic!("a held lease must not be taken; got {:?}", l.holder()),
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
fn staging_never_remakes_a_missing_directory_and_create_dir_makes_ancestors() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let gone = tmp.path().join("gone");
    assert!(atomic::stage(&gone, "x.json", b"{}").is_err());
    assert!(!gone.exists(), "stage does not create directories");

    let deep = tmp.path().join("a").join("b").join("c");
    atomic::create_dir(&deep).expect("creates");
    atomic::create_dir(&deep).expect("an existing directory is fine");
    let path = atomic::write(&deep, "x.json", b"{}").expect("writes");
    assert_eq!(fs::read(path).expect("reads"), b"{}");

    let file = tmp.path().join("plain");
    fs::write(&file, b"").expect("file");
    assert!(atomic::create_dir(&file.join("sub")).is_err());
    assert!(atomic::create_dir(&file).is_err(), "a file is not a dir");
}

#[test]
fn a_second_writer_waits_the_full_window_then_is_busy() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let durable = tmp.path().join("db");
    let first = FakeEnv::new(100);
    let second = FakeEnv::new(200);

    let held = acquire(&durable, &first).expect("acquires");
    let mine = holder_of(&held);
    assert_eq!(
        mine,
        Holder {
            pid: 100,
            since: T0_MS / 1000
        }
    );
    let lock = durable.join(LOCK_FILE);
    assert_eq!(read_holder(&lock), Some(mine), "the holder states itself");

    assert_eq!(
        busy_holder(acquire(&durable, &second).expect("acquire")),
        Some(mine)
    );
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
    assert_eq!(fs::read(&lock).expect("lock file stays"), b"", "unstated");
    let slept = second.slept_ms.get();
    let again = acquire(&durable, &second).expect("acquires");
    assert_eq!(holder_of(&again).pid, 200);
    assert_eq!(second.slept_ms.get(), slept, "a released lease: no wait");
    drop(again);
    assert_eq!(
        fs::read(&lock).expect("lock file"),
        b"",
        "dropping unstates"
    );
    assert_eq!(
        holder_of(&acquire(&durable, &first).expect("acquires")).pid,
        100
    );
}

#[test]
fn neither_a_statement_nor_age_decides_who_holds_the_lease() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let durable = tmp.path().join("db");
    fs::create_dir_all(&durable).expect("dir");
    let lock = durable.join(LOCK_FILE);

    // A crashed holder's leftover statement (dead pid, ancient) is no lock.
    fs::write(&lock, Holder { pid: 300, since: 1 }.encode()).expect("plant");
    let env = FakeEnv::new(200);
    let got = acquire(&durable, &env).expect("acquires");
    assert_eq!(holder_of(&got).pid, 200);
    assert_eq!(env.slept_ms.get(), 0, "without waiting");
    assert_eq!(read_holder(&lock).map(|h| h.pid), Some(200), "overwritten");

    // A holder older than any limit keeps the lease: a writer a day later
    // waits its window and is busy.
    let late = FakeEnv::new(400);
    late.now_ms.set(T0_MS + 86_400_000);
    assert_eq!(
        busy_holder(acquire(&durable, &late).expect("acquire")).map(|h| h.pid),
        Some(200)
    );
    drop(got);

    // A garbled statement names nobody and still decides nothing.
    fs::write(&lock, b"pid=\x00garbage").expect("plant");
    assert_eq!(read_holder(&lock), None);
    assert_eq!(
        holder_of(&acquire(&durable, &late).expect("acquires")).pid,
        400
    );
}

#[test]
fn a_clock_that_fails_fails_the_acquire() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let env = FakeEnv {
        clock_broken: true,
        ..FakeEnv::new(100)
    };
    let err = acquire(&tmp.path().join("db"), &env).expect_err("no time, no lease");
    assert!(err.0.contains("fake clock"), "{err}");
}

/// Kills the child when the test ends, however it ends.
struct Reap(Child);

impl Drop for Reap {
    fn drop(&mut self) {
        let _kill_if_alive = self.0.kill();
        let _status = self.0.wait();
    }
}

fn spawn_holder(durable: &Path) -> Reap {
    let exe = std::env::current_exe().expect("test binary");
    let mut child = Command::new(exe)
        .args(["--exact", "lease_child", "--ignored", "--nocapture"])
        .env(CHILD_DIR, durable)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("spawns the holder");
    let out = child.stdout.take().expect("stdout");
    let mut reap = Reap(child);
    let held = BufReader::new(out)
        .lines()
        .map(|l| l.expect("reads"))
        .any(|l| l == HELD);
    if !held {
        let status = reap.0.wait().expect("waits");
        panic!("the holder exited ({status}) without taking the lease");
    }
    reap
}

#[test]
fn a_crashed_holder_releases_the_lease() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let durable = tmp.path().join("db");
    let mut child = spawn_holder(&durable);
    let pid = child.0.id();

    let env = FakeEnv::new(200);
    assert_eq!(
        busy_holder(acquire(&durable, &env).expect("acquire")).map(|h| h.pid),
        Some(pid),
        "another process holds the lease"
    );

    child.0.kill().expect("SIGKILL: no unwinding, no release");
    child.0.wait().expect("reaped");
    let slept = env.slept_ms.get();
    let got = acquire(&durable, &env).expect("acquires");
    assert_eq!(holder_of(&got).pid, 200);
    assert_eq!(env.slept_ms.get(), slept, "a dead holder's lock is free");
}

/// The child half of `a_crashed_holder_releases_the_lease`: take the lease
/// on `$METPROF_KB_LEASE_CHILD_DIR`, say so, and hold it until killed.
#[test]
#[ignore = "run by a_crashed_holder_releases_the_lease as its child process"]
fn lease_child() {
    let Some(dir) = std::env::var_os(CHILD_DIR) else {
        return;
    };
    let lease = acquire(Path::new(&dir), &SystemLeaseEnv).expect("acquires");
    assert!(matches!(lease, Acquired::Held(_)), "the lease is free");
    println!("{HELD}");
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
