// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the `metprof kb` subcommands against the real local KB
//! (spool, lease, LatticeDB store, export and seal) in a temporary
//! directory, and their exit statuses: 0, 3 when the KB could not be
//! brought to the state asked for (a busy lease, an unwritable lock), 4
//! when `kb verify` finds a sealed export that fails.
#![cfg(unix)]

mod common;

use std::cell::Cell;
use std::fs;
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::time::Duration;

use common::kb::{SUMMARIZE, capture};
use common::{FAMILIES_TOML, HASHED, Mem, kb_toml};
use metprof::adapters::files::FsFiles;
use metprof::cli::kb::VERIFY_FAILED;
use metprof::cli::{Io, Outcome, run};
use metprof::ports::Files;
use metprof_kb::lease::{Acquired, LOCK_FILE, LeaseEnv, SystemLeaseEnv, acquire};
use metprof_kb::local::LocalKbs;
use metprof_kb::time::SystemClock;
use serde_json::{Value, json};

struct Dir {
    _tmp: tempfile::TempDir,
    root: PathBuf,
    kb: String,
    capture: String,
    families: String,
}

fn write(path: &Path, text: &str) {
    fs::write(path, text).unwrap();
}

fn dir(sealed: bool) -> Dir {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().to_path_buf();
    let p = |n: &str| root.join(n).display().to_string();
    let key = root.join("export.key");
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&key)
        .and_then(|mut f| std::io::Write::write_all(&mut f, "ab".repeat(32).as_bytes()))
        .unwrap();
    write(&root.join("embedding.toml"), HASHED);
    let seal = if sealed {
        format!("seal_key_file = \"{}\"", key.display())
    } else {
        "seal = \"off\"".to_string()
    };
    let text = kb_toml(&seal)
        .replace("/kb/spool", &p("spool"))
        .replace("/kb/durable", &p("durable"))
        .replace("/kb/export", &p("export"))
        .replace(common::EMBEDDING_TOML, &p("embedding.toml"));
    write(&root.join("kb.toml"), &text);
    write(&root.join("f.toml"), FAMILIES_TOML);
    write(&root.join("c.sqlite"), "synthetic capture");
    Dir {
        kb: p("kb.toml"),
        capture: p("c.sqlite"),
        families: p("f.toml"),
        _tmp: tmp,
        root,
    }
}

impl Dir {
    fn run(&self, argv: &[&str]) -> Outcome {
        self.run_with(SystemLeaseEnv, argv)
    }

    /// Run with `env` as the machine the lease waits on.
    fn run_with(&self, env: impl LeaseEnv, argv: &[&str]) -> Outcome {
        let mut full = vec!["--json", "--kb", self.kb.as_str()];
        full.extend_from_slice(argv);
        let argv: Vec<String> = full.iter().map(|s| s.to_string()).collect();
        let mem = Mem::new().capture(&self.capture, capture());
        run(
            &argv,
            &Io {
                traces: &mem,
                files: &FsFiles,
                chat: &mem,
                salt: &mem,
                kb: &LocalKbs::new("metprof test", env),
                clock: &SystemClock,
            },
        )
    }

    fn ok(&self, argv: &[&str]) -> Value {
        let out = self.run(argv);
        assert_eq!((out.code, out.stderr.as_str()), (0, ""), "{argv:?}");
        serde_json::from_str(&out.stdout).unwrap()
    }

    fn summarize(&self) -> Outcome {
        let mut argv: Vec<&str> = SUMMARIZE.to_vec();
        argv[1] = &self.capture;
        argv[5] = &self.families;
        self.run(&argv)
    }

    fn export(&self) -> PathBuf {
        self.root.join("export").join("kkb.jsonl")
    }
}

#[test]
fn selfcheck_runs_and_seal_round_trip_through_the_local_kb() {
    let d = dir(true);
    let v = d.ok(&["kb", "selfcheck"]);
    assert_eq!(v["read_back"], json!("ok"));
    assert!(Path::new(v["sealed"].as_str().unwrap()).is_file());
    d.ok(&["kb", "verify"]);
    let out = d.summarize();
    assert_eq!((out.code, out.stderr.as_str()), (0, ""));
    let exported = fs::read_to_string(d.export()).unwrap();
    let cap = FsFiles.sha256(&d.capture).unwrap();
    assert!(exported.contains(&cap), "the export holds the run");
    assert!(!exported.contains(&d.capture), "the export holds no path");
    d.ok(&["kb", "verify"]);
    write(&d.export(), &format!("{exported}\n"));
    let out = d.run(&["kb", "verify"]);
    assert_eq!((out.code, out.stdout.as_str()), (VERIFY_FAILED, ""));
    assert!(
        out.stderr.contains("does not seal the current kkb.jsonl"),
        "{}",
        out.stderr
    );
    d.ok(&["kb", "seal"]);
    d.ok(&["kb", "verify"]);
    d.ok(&["kb", "export"]);
    assert_eq!(fs::read_to_string(d.export()).unwrap(), exported);
    d.ok(&["kb", "verify"]);
    let v = d.ok(&["kb", "replay"]);
    assert_eq!(
        (v["spool_files"].clone(), v["inserted"].clone()),
        (json!(2), json!(0))
    );
    // A flipped byte, or no sealed export at all, fails verification.
    let sealed = d.root.join("export").join("kkb.jsonl.kkb");
    let mut bytes = fs::read(&sealed).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&sealed, &bytes).unwrap();
    let out = d.run(&["kb", "verify"]);
    assert_eq!((out.code, out.stdout.as_str()), (VERIFY_FAILED, ""));
    assert!(
        out.stderr.contains("FAILED: seal: opening"),
        "{}",
        out.stderr
    );
    fs::remove_file(&sealed).unwrap();
    assert_eq!(d.run(&["kb", "verify"]).code, VERIFY_FAILED);
    d.ok(&["kb", "seal"]);
    d.ok(&["kb", "verify"]);
    // Another key cannot open the sealed export.
    write(&d.root.join("export.key"), &"cd".repeat(32));
    let out = d.run(&["kb", "verify"]);
    assert_eq!((out.code, out.stdout.as_str()), (VERIFY_FAILED, ""));
    assert!(out.stderr.contains("seal: opening"), "{}", out.stderr);
}

#[test]
fn an_unsafe_key_or_a_dense_embedder_is_refused_before_the_analysis() {
    let d = dir(true);
    let key = d.root.join("export.key");
    fs::set_permissions(&key, fs::Permissions::from_mode(0o644)).unwrap();
    for out in [d.summarize(), d.run(&["kb", "selfcheck"])] {
        assert_eq!((out.code, out.stdout.as_str()), (2, ""));
        assert!(out.stderr.contains("(0600)"), "{}", out.stderr);
    }
    write(&key, &"ab".repeat(31));
    fs::set_permissions(&key, fs::Permissions::from_mode(0o600)).unwrap();
    let out = d.summarize();
    assert_eq!((out.code, out.stdout.as_str()), (2, ""));
    assert!(
        out.stderr.contains("exactly 64 hex digits"),
        "{}",
        out.stderr
    );
    write(
        &d.root.join("embedding.toml"),
        "[embedder]\nkind = \"openai_compatible\"\nendpoint = \"http://embed.invalid/v1\"\n\
         model = \"m\"\ndims = 8\nprecision = 6\n[reranker]\nkind = \"none\"\n",
    );
    let out = d.summarize();
    assert_eq!((out.code, out.stdout.as_str()), (2, ""));
    assert!(
        out.stderr.contains("kind = \"hashed\" only"),
        "{}",
        out.stderr
    );
    assert!(!d.root.join("spool").exists(), "nothing written");
}

#[test]
fn seal_off_exports_without_sealing() {
    let d = dir(false);
    let v = d.ok(&["kb", "selfcheck"]);
    assert_eq!(
        (v["read_back"].clone(), v["sealed"].clone()),
        (json!("ok"), json!("off"))
    );
    assert!(!d.root.join("export").join("kkb.jsonl.kkb").exists());
    for op in ["seal", "verify"] {
        let out = d.run(&["kb", op]);
        assert_eq!(out.code, 2);
        assert!(
            out.stderr.contains("kb.toml has seal = \"off\""),
            "{}",
            out.stderr
        );
    }
}

/// The lease's machine with a clock that moves only when the lease sleeps:
/// waiting out the window costs no wall time.
struct FastEnv(Cell<u64>);

impl LeaseEnv for FastEnv {
    fn pid(&self) -> u32 {
        std::process::id()
    }
    fn now_s(&self) -> metprof_kb::Result<u64> {
        Ok(self.0.get() / 1000)
    }
    fn sleep(&self, d: Duration) {
        self.0.set(self.0.get() + d.as_millis() as u64);
    }
}

fn fast() -> FastEnv {
    FastEnv(Cell::new(1_800_000_000_000))
}

#[test]
fn kb_verify_behind_a_busy_lease_exits_3_without_looking() {
    let d = dir(true);
    d.ok(&["kb", "selfcheck"]);
    let sealed = d.root.join("export").join("kkb.jsonl.kkb");
    fs::write(&sealed, b"not a sealed export").unwrap();
    let held = acquire(&d.root.join("durable"), &SystemLeaseEnv).unwrap();
    assert!(matches!(held, Acquired::Held(_)));

    let out = d.run_with(fast(), &["kb", "verify"]);
    assert_eq!(out.code, 3, "{}", out.stderr);
    let me = format!("leased by pid {}", std::process::id());
    assert!(out.stderr.contains(&me), "{}", out.stderr);
    assert!(!out.stderr.contains("FAILED"), "{}", out.stderr);

    drop(held);
    assert_eq!(d.run(&["kb", "verify"]).code, VERIFY_FAILED, "it looks now");
}

#[test]
fn kb_verify_of_a_kb_it_cannot_lock_exits_3_not_4() {
    let d = dir(true);
    d.ok(&["kb", "selfcheck"]);
    let durable = d.root.join("durable");
    let lock = durable.join(LOCK_FILE);
    let mode = |p: &Path, m: u32| fs::set_permissions(p, fs::Permissions::from_mode(m)).unwrap();
    mode(&lock, 0o444);
    mode(&durable, 0o555);
    assert!(
        fs::OpenOptions::new().write(true).open(&lock).is_err(),
        "the lock file is read-only (this test needs a non-root user)"
    );

    let out = d.run(&["kb", "verify"]);
    mode(&durable, 0o755);
    mode(&lock, 0o644);
    assert_eq!((out.code, out.stdout.as_str()), (3, ""), "{}", out.stderr);
    assert!(out.stderr.contains("lease: opening"), "{}", out.stderr);
    assert!(!out.stderr.contains("FAILED"), "{}", out.stderr);
    d.ok(&["kb", "verify"]);
}
