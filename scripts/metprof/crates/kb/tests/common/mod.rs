// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the synthetic run record the tests share: a `tree` command
//! over a made-up capture, carrying `tree` as its kernel tree file.

use metprof_kb::port::{Arg, FileRef, Identity, RUN_SCHEMA, RunRecord, Stamps};
use serde_json::{Map, Value};

pub fn run(tree: &str) -> RunRecord {
    let json: Value = serde_json::from_str(tree).expect("fixture parses");
    RunRecord {
        schema: RUN_SCHEMA.into(),
        command: "tree".into(),
        args: vec![
            Arg::Word("tree".into()),
            Arg::File {
                sha256: "aa".repeat(32),
                basename: "capture.sqlite".into(),
            },
        ],
        result: Map::new(),
        files: vec![FileRef {
            name: "tree.json".into(),
            sha256: "bb".repeat(32),
            json: Some(json),
        }],
        stamps: Stamps {
            recorded_at: "2026-01-01T00:00:00Z".into(),
            metprof_version: "0.0.0".into(),
            metprof_git_sha: "unknown".into(),
            config_sha256: "cc".repeat(32),
            capture_sha256: Some("aa".repeat(32)),
            identity: Identity {
                box_alias: "box-a".into(),
                campaign: "campaign-a".into(),
                hardware_class: "class-a".into(),
            },
        },
    }
}
