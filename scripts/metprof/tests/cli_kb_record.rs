// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: what the run record of a file-writing or path-taking command
//! holds: the tree a `tree` run writes (and an input at the output path read
//! as the old file), and `ttft` runs keeping digests and base names, never a
//! path.

mod common;

use common::Mem;
use common::tree::{SHA_ABC, argv as tree_argv, mem as tree_mem, written};
use common::ttft::{CONFIG, config_with_client, join_lines, ok_reply, req};
use metprof::core::provenance::sha256_hex;
use metprof::ports::Files;
use metprof_kb::port::{Arg, FileRef};
use serde_json::json;

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
fn a_baseline_at_the_output_path_is_recorded_as_the_old_file() {
    let old = tree_mem()
        .run(&tree_argv(&["--model", "m"]))
        .files
        .remove(0)
        .1;
    let m = tree_mem().bytes("/t.json", &old);
    let out = m.run(&tree_argv(&["--baseline", "/t.json"]));
    let new = serde_json::to_vec(&written(&out)).unwrap();
    assert_ne!(sha256_hex(&new), sha256_hex(&old), "a different tree");
    let [r] = <[_; 1]>::try_from(m.records()).expect("one record");
    let digest_after = |opt: &str| {
        let at = r.args.iter().position(|a| *a == Arg::Word(opt.into()));
        match at.map(|i| &r.args[i + 1]) {
            Some(Arg::File { sha256, basename }) if basename == "t.json" => sha256.clone(),
            other => panic!("{opt}: {other:?}"),
        }
    };
    let written_sha = sha256_hex(&out.files[0].1);
    assert_eq!(digest_after("--baseline"), sha256_hex(&old), "the old tree");
    assert_eq!(digest_after("--out"), written_sha, "the new tree");
    assert_eq!(r.files[0].sha256, written_sha);
}

#[test]
fn ttft_runs_record_digests_and_base_names_never_paths() {
    let dir = "/runs/cell-a";
    let p = |n: &str| format!("{dir}/{n}");
    let (cfg, prompt, out, log, b) = (
        p("t.toml"),
        p("p.txt"),
        p("c.jsonl"),
        p("a.log"),
        p("b.log"),
    );
    let serve_log = join_lines(&[
        req(100, 32, false, 7, 900.0),
        req(100, 8, false, 7, 450.0),
        req(100, 8, false, 9, 470.0),
        req(100, 8, true, 9, 90.0),
    ]);
    let client = Mem::new()
        .file(&cfg, &config_with_client())
        .file(&prompt, "id {salt} then text")
        .reply(ok_reply(100, 900.0, 1_000_000))
        .reply(ok_reply(100, 450.0, 3_000_000))
        .reply(ok_reply(100, 470.0, 5_000_000))
        .reply(ok_reply(100, 90.0, 1_000_000));
    let ran = client.run(&[
        "ttft",
        "client",
        "--url",
        "http://h:9",
        "--config",
        &cfg,
        "--cell",
        "a",
        "--prompt",
        &prompt,
        "--cold",
        "2",
        "--warm",
        "1",
        "--out",
        &out,
    ]);
    assert_eq!(ran.code, 0, "{}", ran.stderr);
    let logs = Mem::new()
        .file(&log, &serve_log)
        .file(&b, &serve_log)
        .file(&cfg, CONFIG)
        .bytes(&out, &ran.files[0].1);
    let joined = logs.run(&[
        "ttft", "log", "--log", &log, "--config", &cfg, "--client", &out,
    ]);
    assert_eq!(joined.code, 0, "{}", joined.stderr);
    let compared = logs.run(&[
        "ttft",
        "compare",
        "--a",
        &log,
        "--b",
        &b,
        "--config",
        &cfg,
        "--min-reps",
        "1",
    ]);
    assert_eq!(compared.code, 0, "{}", compared.stderr);

    let records: Vec<_> = client.records().into_iter().chain(logs.records()).collect();
    let file_args: Vec<Vec<&str>> = records
        .iter()
        .map(|r| {
            r.args
                .iter()
                .filter_map(|a| match a {
                    Arg::File { basename, .. } => Some(basename.as_str()),
                    _ => None,
                })
                .collect()
        })
        .collect();
    assert_eq!(
        file_args,
        vec![
            vec!["t.toml", "p.txt", "c.jsonl"],
            vec!["a.log", "t.toml", "c.jsonl"],
            vec!["a.log", "b.log", "t.toml"],
        ]
    );
    for r in &records {
        let text = serde_json::to_string(r).unwrap();
        assert!(!text.contains(dir), "a path in the record: {text}");
    }
    assert_eq!(
        records[0].stamps.config_sha256,
        sha256_hex(config_with_client().as_bytes())
    );
}
