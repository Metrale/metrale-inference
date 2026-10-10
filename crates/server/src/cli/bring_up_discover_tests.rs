// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: Tests for local-serve discovery: which processes are rank-0 `met serve`s, the
//! port each one's argv names, and the identity check against `/serve-config`.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants: none beyond the types.

use super::*;

fn argv(s: &str) -> Vec<String> {
    s.split(' ').map(str::to_string).collect()
}

#[test]
fn the_port_comes_from_the_argv_or_the_serve_default() {
    let d = default_port().unwrap();
    assert_eq!(d, 8888, "met serve's clap default");
    let c = candidate(
        7,
        argv("/home/claude/glm-bench/met serve --model-from-path /srv/m --port 8890 --rank 0"),
        d,
    )
    .unwrap();
    assert_eq!((c.pid, c.port), (7, 8890));
    assert_eq!(
        c.model_from_path.as_deref(),
        Some(std::path::Path::new("/srv/m"))
    );
    assert_eq!(
        candidate(8, argv("met serve Qwen/X --port=8891"), d)
            .unwrap()
            .port,
        8891
    );
    let plain = candidate(9, argv("met serve Qwen/X"), d).unwrap();
    assert_eq!((plain.port, plain.model_from_path), (8888, None));
}

#[test]
fn other_processes_and_ranks_are_not_candidates() {
    let d = 8888;
    for s in [
        "met serve --port 8891 --rank 1",
        "/usr/bin/python3 -m vllm serve --port 8888",
        "vllm serve /models/x",
        "met bench run ttft-cold-gate",
        "met",
        "met serve --port notaport",
    ] {
        assert_eq!(candidate(1, argv(s), d), None, "{s}");
    }
}

#[test]
fn only_the_processes_own_identity_confirms_it() {
    let c = candidate(42, argv("met serve --port 8890"), 8888).unwrap();
    let own = ServeIdentity {
        argv_sha256: argv_fingerprint(&argv("serve --port 8890")),
        binary_sha256: "b".into(),
        env_sha256: String::new(),
        pid: 42,
    };
    assert!(is_own(&c, &own));
    assert!(
        !is_own(
            &c,
            &ServeIdentity {
                pid: 43,
                ..own.clone()
            }
        ),
        "another pid"
    );
    let other_args = ServeIdentity {
        argv_sha256: argv_fingerprint(&argv("serve --port 8891")),
        ..own
    };
    assert!(!is_own(&c, &other_args), "another argv");
}
