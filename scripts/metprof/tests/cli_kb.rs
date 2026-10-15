// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KB side of the command line (DESIGN §3.1): the required
//! `--kb`, `--kb off`, the run record of a command (digests and stamps, never
//! paths), the exit status when the sink fails, `kb.toml` refusals (the `metprof kb`
//! subcommands against the real local KB are in `cli_kb_local.rs`). Every `Mem::run` elsewhere also checks one record per run.

mod common;

use common::kb::{NOW, SUMMARIZE, capture};
use common::tree::{argv as tree_argv, mem as tree_mem};
use common::{FAMILIES_TOML, HASHED, Mem, Sink, kb_toml};
use metprof::cli::Outcome;
use metprof::cli::kb::OFF_NOTICE;
use metprof::core::provenance::sha256_hex;
use metprof::ports::Files;
use metprof_kb::port::{Arg, Identity};
use serde_json::Value;

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
