// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KB side of the command line (DESIGN §3.1): the required
//! `--kb`, `--kb off`, the run record of a command (digests and stamps, never
//! paths), the exit status when the sink fails, `kb.toml` refusals, and the
//! `metprof kb` subcommands against the real local KB in a temporary
//! directory. Every `Mem::run` elsewhere also checks one record per run.

mod common;

use common::kb::NOW;
use common::tree::{SHA_ABC, argv as tree_argv, mem as tree_mem, written};
use common::{Capture, FAMILIES_TOML, HASHED, Mem, Sink, kb_toml};
use metprof::cli::kb::OFF_NOTICE;
use metprof::cli::{Io, Outcome, run};
use metprof::core::provenance::sha256_hex;
use metprof::ports::Files;
use metprof_kb::port::{Arg, FileRef, Identity};
use serde_json::{Value, json};

const SUMMARIZE: [&str; 10] = [
    "summarize",
    "c.sqlite",
    "--device",
    "0",
    "--families",
    "f.toml",
    "--gap-ms",
    "1",
    "--top",
    "2",
];

fn capture() -> Capture {
    Capture::new()
        .kernel("gemm_a", 0.0, 100.0)
        .kernel("norm_x", 3000.0, 3010.0)
}

fn mem() -> Mem {
    Mem::new()
        .file("f.toml", FAMILIES_TOML)
        .capture("c.sqlite", capture())
}

fn with_kb<'a>(kb: &'a str, argv: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["--kb", kb];
    v.extend_from_slice(argv);
    v
}

#[test]
fn a_run_without_kb_is_refused_and_records_nothing() {
    let m = mem();
    let out = m.run_raw(&SUMMARIZE, &m);
    assert_eq!(
        out,
        Outcome {
            code: 2,
            stdout: String::new(),
            stderr: "metprof: error: --kb is required (kb.toml path, or 'off')\n".into(),
            files: Vec::new(),
        }
    );
    assert!(m.records().is_empty());
    for help in [&["--help"][..], &["kb", "selfcheck", "--help"]] {
        let out = m.run_raw(help, &m);
        assert_eq!(out.code, 0, "{}", out.stderr);
        assert!(out.stdout.contains("--kb KB.TOML|off"), "{}", out.stdout);
    }
}

#[test]
fn kb_off_is_logged_and_records_nothing() {
    let m = mem();
    let recorded = m.run(&SUMMARIZE);
    assert_eq!((recorded.code, recorded.stderr.as_str()), (0, ""));
    let mut after = SUMMARIZE.to_vec();
    after.push("--kb=off");
    for argv in [with_kb("off", &SUMMARIZE), after] {
        let out = m.run(&argv);
        assert_eq!(out.code, 0, "{}", out.stderr);
        assert_eq!(out.stdout, recorded.stdout);
        assert_eq!(out.stderr, OFF_NOTICE);
    }
    assert_eq!(m.records().len(), 1, "only the run with a kb.toml");
    let e = m.refused(&["--kb", "off", "kb", "selfcheck"]);
    assert!(e.contains("needs a kb.toml"), "{e}");
}

#[test]
fn kb_is_given_once_with_a_value() {
    let m = mem();
    let mut twice = with_kb("off", &SUMMARIZE);
    twice.extend(["--kb", common::KB_TOML]);
    let mut bare = SUMMARIZE.to_vec();
    bare.push("--kb");
    for (argv, msg) in [
        (twice, "argument --kb: given more than once"),
        (bare, "argument --kb: expected one argument"),
        (
            with_kb("--json", &SUMMARIZE),
            "argument --kb: expected one argument",
        ),
    ] {
        let e = m.refused(&argv);
        assert!(e.contains(msg), "{argv:?}: {e}");
    }
}

#[test]
fn a_run_records_digests_and_stamps_never_paths() {
    let m = mem();
    let mut argv = vec!["--json"];
    argv.extend(
        SUMMARIZE
            .iter()
            .filter(|a| !["--families", "f.toml"].contains(a)),
    );
    argv.push("--families=f.toml");
    let out = m.run(&argv);
    assert_eq!(out.code, 0, "{}", out.stderr);
    let [r] = <[_; 1]>::try_from(m.records()).expect("one record");
    let fam = sha256_hex(FAMILIES_TOML.as_bytes());
    let cap = m.sha256("c.sqlite").unwrap();
    let w = |s: &str| Arg::Word(s.into());
    let file = |sha: &str, name: &str| Arg::File {
        sha256: sha.into(),
        basename: name.into(),
    };
    assert_eq!(r.command, "summarize");
    assert_eq!(
        r.args,
        vec![
            file(&cap, "c.sqlite"),
            w("--device"),
            w("0"),
            w("--gap-ms"),
            w("1"),
            w("--top"),
            w("2"),
            w("--families"),
            file(&fam, "f.toml"),
        ]
    );
    let stdout: Value = serde_json::from_str(&out.stdout).unwrap();
    assert_eq!(Value::Object(r.result.clone()), stdout);
    assert!(r.files.is_empty());
    let s = &r.stamps;
    assert_eq!(
        (
            s.recorded_at.as_str(),
            s.metprof_version.as_str(),
            s.config_sha256.as_str(),
            s.capture_sha256.as_deref()
        ),
        (
            NOW,
            env!("CARGO_PKG_VERSION"),
            fam.as_str(),
            Some(cap.as_str())
        )
    );
    let sha = &s.metprof_git_sha;
    assert!(
        sha == "unknown" || (sha.len() == 40 && sha.chars().all(|c| c.is_ascii_hexdigit())),
        "{sha}"
    );
    assert_eq!(
        s.identity,
        Identity {
            box_alias: "box-a".into(),
            campaign: "campaign-a".into(),
            hardware_class: "class-a".into(),
        }
    );
}

#[test]
fn a_tree_run_records_the_tree_it_writes() {
    let m = tree_mem();
    let out = m.run(&tree_argv(&["--model", "m"]));
    let tree = written(&out);
    let [r] = <[_; 1]>::try_from(m.records()).expect("one record");
    let sha = sha256_hex(&out.files[0].1);
    assert_eq!(
        r.files,
        vec![FileRef {
            name: "t.json".into(),
            sha256: sha.clone(),
            json: Some(tree),
        }]
    );
    let at = r.args.iter().position(|a| *a == Arg::Word("--out".into()));
    assert_eq!(
        at.map(|i| &r.args[i + 1]),
        Some(&Arg::File {
            sha256: sha,
            basename: "t.json".into()
        })
    );
    assert_eq!(r.result["out"], json!("t.json"));
    assert_eq!(r.stamps.config_sha256, m.sha256("/cfg.toml").unwrap());
    assert_eq!(r.stamps.capture_sha256.as_deref(), Some(SHA_ABC));
}

#[test]
fn a_failing_sink_exits_3_with_the_analysis_on_stdout() {
    let ok = mem().run(&SUMMARIZE);
    let m = mem().sink(Sink::Fails);
    let out = m.run(&SUMMARIZE);
    assert_eq!((out.code, &out.stdout), (3, &ok.stdout));
    assert_eq!(
        out.stderr,
        "metprof: kb: this run is not recorded: disk full (synthetic)\n"
    );
    assert!(m.records().is_empty());
    let m = mem().sink(Sink::SpoolsOnly);
    let out = m.run(&SUMMARIZE);
    assert_eq!((out.code, &out.stdout), (3, &ok.stdout));
    assert_eq!(
        out.stderr,
        "metprof: kb: recorded in /kb/spool/run.json but not materialised: another writer \
         holds the lease (synthetic)\n"
    );
    let t = tree_mem().sink(Sink::Fails);
    let out = t.run(&tree_argv(&["--model", "m"]));
    assert_eq!(
        (out.code, out.files.len()),
        (3, 1),
        "the tree is still written"
    );
}

#[test]
fn a_bad_kb_toml_is_a_configuration_error_with_nothing_recorded() {
    let off = "seal = \"off\"";
    let cases = [
        (
            kb_toml(off).replace("\"/kb/spool\"", "\"kb/spool\""),
            "spool_dir \"kb/spool\" is not an absolute path",
        ),
        (
            kb_toml("seal = \"off\"\nseal_key_file = \"/kb/k\""),
            "are exclusive",
        ),
        (kb_toml(""), "seal_key_file, or seal = \"off\", is required"),
        (kb_toml("seal = \"on\""), "seal \"on\" is not \"off\""),
        (
            format!("colour = 1\n{}", kb_toml(off)),
            "unknown field `colour`",
        ),
        (
            kb_toml(off).replace("schema = 1", "schema = 2"),
            "schema 2 is not 1",
        ),
        (
            kb_toml(off).replace("\"campaign-a\"", "\" \""),
            "[identity] campaign is empty",
        ),
        (
            kb_toml(off).replace(common::EMBEDDING_TOML, "/kb/gone.toml"),
            "cannot read embedding.toml /kb/gone.toml",
        ),
        (
            kb_toml(off).replace(common::EMBEDDING_TOML, "/kb/bad.toml"),
            "embedding.toml: [embedder] kind \"dense\"",
        ),
    ];
    for (text, msg) in cases {
        let m = mem()
            .file("/kb/other.toml", &text)
            .file("/kb/bad.toml", &HASHED.replace("hashed", "dense"));
        let e = m.refused(&with_kb("/kb/other.toml", &SUMMARIZE));
        assert!(e.contains(msg), "{msg}: {e}");
        assert!(m.records().is_empty());
    }
    let e = mem().refused(&with_kb("/kb/none.toml", &SUMMARIZE));
    assert!(e.contains("cannot read kb.toml /kb/none.toml"), "{e}");
}

#[cfg(unix)]
mod local {
    //! The real local KB: spool, lease, LatticeDB store, export and seal.

    use std::fs;
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    use std::path::{Path, PathBuf};

    use metprof::adapters::files::FsFiles;
    use metprof_kb::local::LocalKbs;
    use metprof_kb::time::SystemClock;

    use super::*;

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
                    kb: &LocalKbs::new("metprof test"),
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
        assert_eq!(out.code, 3);
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
        // Another key cannot open the sealed export.
        write(&d.root.join("export.key"), &"cd".repeat(32));
        let out = d.run(&["kb", "verify"]);
        assert_eq!(out.code, 3);
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
}
