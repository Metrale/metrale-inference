// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The embedding pipeline against the Embedder port (the only fake: a provider that
//! refuses any request holding a marked input and rate-limits on demand; the clock and timer are
//! the ports' trivial forms, since waiting is not under test, only that it happens).

use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};
use metrale_abstract_io::{Clock, MonoNs, Timer, UnixMs};
use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::embed::missing;
use metrale_kg_derive::embed::{Embedder, Mode, Plan, Reply, Role, embed_all, request_body};
use metrale_kg_derive::hash::{Sha256, sha256};
use metrale_kg_derive::settings::{
    EmbedLayer, Embedder as SettingsEmbedder, merge, parse_global, resolve,
};

mod common;

fn run<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("the fakes never pend"),
    }
}

struct Now;
impl Clock for Now {
    fn now_unix_ms(&self) -> UnixMs {
        UnixMs(0)
    }
    fn monotonic_ns(&self) -> MonoNs {
        MonoNs(0)
    }
}

#[derive(Default)]
struct Waits(Cell<u32>);
impl Timer for Waits {
    async fn sleep_until(&self, _at: MonoNs) {
        self.0.set(self.0.get() + 1);
    }
}

#[derive(Default)]
struct Provider {
    calls: RefCell<Vec<usize>>,
    roles: RefCell<Vec<Role>>,
    limited: Cell<u32>,
}

impl Embedder for Provider {
    fn mode(&self) -> Mode {
        Mode::Available {
            model: "m".into(),
            dim: 2,
        }
    }
    async fn embed(&self, role: Role, inputs: &[String]) -> Result<Reply> {
        self.calls.borrow_mut().push(inputs.len());
        self.roles.borrow_mut().push(role);
        if self.limited.get() > 0 {
            self.limited.set(self.limited.get() - 1);
            return Ok(Reply::Retryable {
                status: 429,
                retry_after_ms: None,
            });
        }
        if inputs.iter().any(|t| t.contains("BAD")) {
            return Err(error!(ErrorCode::KgEmbedProtocol, "HTTP 400"));
        }
        Ok(Reply::Vectors(
            inputs.iter().map(|t| vec![t.len() as f32, 1.0]).collect(),
        ))
    }
}

fn todo(n: usize, bad: &[usize]) -> Vec<(Sha256, String)> {
    (0..n)
        .map(|i| {
            let t = if bad.contains(&i) {
                format!("BAD {i}")
            } else {
                format!("text {i}")
            };
            (sha256(t.as_bytes()), t)
        })
        .collect()
}

const PLAN: Plan = Plan {
    batch: 8,
    max_attempts: 3,
    backoff_ms: 10,
    max_input_chars: 12,
};

#[test]
fn a_refused_input_is_isolated_and_every_other_input_is_embedded() {
    let p = Provider::default();
    let t = todo(20, &[5, 13]);
    let got = run(embed_all(
        &p,
        &Now,
        &Waits::default(),
        &PLAN,
        Role::Document,
        &t,
    ));
    assert!(got.stopped.is_none(), "{:?}", got.stopped);
    assert_eq!(got.done.len(), 18);
    let refused: Vec<Sha256> = got.refused.iter().map(|(s, _)| *s).collect();
    assert_eq!(refused, vec![t[5].0, t[13].0]);
    // 3 batches (8, 8, 4); each refused one bisects 8 -> 4 -> 2 -> 1: 3 extra levels x 2.
    assert!(
        p.calls.borrow().len() <= 3 + 2 * 6,
        "{:?}",
        p.calls.borrow()
    );
    let done: Vec<Sha256> = got.done.iter().map(|(s, _)| *s).collect();
    for (i, (s, _)) in t.iter().enumerate() {
        assert_eq!(done.contains(s), ![5, 13].contains(&i), "input {i}");
    }
}

#[test]
fn a_rate_limit_waits_and_retries_then_stops_after_max_attempts() {
    let p = Provider::default();
    p.limited.set(2);
    let w = Waits::default();
    let got = run(embed_all(
        &p,
        &Now,
        &w,
        &PLAN,
        Role::Document,
        &todo(4, &[]),
    ));
    assert!(got.stopped.is_none());
    assert_eq!((got.done.len(), w.0.get()), (4, 2));

    let p = Provider::default();
    p.limited.set(10);
    let got = run(embed_all(
        &p,
        &Now,
        &Waits::default(),
        &PLAN,
        Role::Document,
        &todo(4, &[]),
    ));
    assert_eq!(
        got.stopped.map(|e| e.code()),
        Some(ErrorCode::KgEmbedRateLimited)
    );
    assert!(got.done.is_empty());
}

#[test]
fn a_seeded_secret_never_reaches_the_embedder_so_never_the_vector_store() {
    // 2026-10-23: the cache stores only (sha, vector); what it could leak is what the embedder
    // was sent. Seed the same fakes as bundle_props and record every input the provider sees.
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
    let b = metrale_kg_derive::derive(&common::config(), &src, None).unwrap();
    let todo = missing(
        b.texts.iter().map(|t| (t.sha, t.text.as_str())),
        &Default::default(),
    );
    assert!(!todo.is_empty());
    #[derive(Default)]
    struct Recorder(RefCell<Vec<String>>);
    impl Embedder for Recorder {
        fn mode(&self) -> Mode {
            Mode::Available {
                model: "m".into(),
                dim: 1,
            }
        }
        async fn embed(&self, _role: Role, inputs: &[String]) -> Result<Reply> {
            self.0.borrow_mut().extend(inputs.iter().cloned());
            Ok(Reply::Vectors(inputs.iter().map(|_| vec![1.0]).collect()))
        }
    }
    let r = Recorder::default();
    let plan = Plan {
        max_input_chars: 1_000_000,
        ..PLAN
    };
    let got = run(embed_all(
        &r,
        &Now,
        &Waits::default(),
        &plan,
        Role::Document,
        &todo,
    ));
    assert_eq!(got.done.len(), todo.len());
    let sent = r.0.borrow();
    assert_eq!(sent.len(), todo.len());
    for t in sent.iter() {
        assert!(
            !t.contains("FAKEFAKEFAKE") && !t.contains("cfat_") && !t.contains("ghp_"),
            "{t}"
        );
    }
}

#[test]
fn an_input_over_max_input_chars_is_refused_unsent() {
    let p = Provider::default();
    let mut t = todo(3, &[]);
    t.push((sha256(b"long"), "x".repeat(13)));
    let got = run(embed_all(
        &p,
        &Now,
        &Waits::default(),
        &PLAN,
        Role::Document,
        &t,
    ));
    assert_eq!(got.done.len(), 3);
    assert_eq!(got.refused.len(), 1);
    assert!(got.refused[0].1.contains("not sent"), "{:?}", got.refused);
    assert_eq!(
        p.calls.borrow().iter().sum::<usize>(),
        3,
        "the long input never reached the provider"
    );
}

#[test]
fn every_request_carries_the_callers_role_through_retries_and_bisection() {
    for role in [Role::Query, Role::Document] {
        let p = Provider::default();
        p.limited.set(1);
        let got = run(embed_all(
            &p,
            &Now,
            &Waits::default(),
            &PLAN,
            role,
            &todo(8, &[3]),
        ));
        assert_eq!(got.done.len(), 7);
        let roles = p.roles.borrow();
        assert!(
            roles.len() > 2,
            "retry and bisection must both have run: {roles:?}"
        );
        assert!(roles.iter().all(|r| *r == role), "{role:?}: {roles:?}");
    }
}

const GLOBAL: &str = r#"schema = "latlas-config/v1"
[embedder]
mode = "http"
endpoint = "https://openrouter.ai/api/v1/embeddings"
model = "m"
dimensions = 4
headers = {}
key_env = "K"
timeout_ms = 1
batch = 1
max_attempts = 1
backoff_ms = 1
min_interval_ms = 0
max_input_chars = 1
input_type = { query = "query", document = "passage" }
"#;

fn resolved(global: &str) -> metrale_error::Result<SettingsEmbedder> {
    let (m, _) = merge(&parse_global(global)?, &EmbedLayer::default());
    resolve(&m)
}

#[test]
fn the_input_type_is_required_non_empty_sent_per_role_and_part_of_the_vector_space() {
    let SettingsEmbedder::Available(s) = resolved(GLOBAL).unwrap() else {
        panic!("must be available")
    };
    let inputs = ["t".to_string()];
    let q: serde_json::Value = serde_json::from_str(&request_body(
        &s.model,
        s.input_type.of(Role::Query),
        &inputs,
    ))
    .unwrap();
    let d: serde_json::Value = serde_json::from_str(&request_body(
        &s.model,
        s.input_type.of(Role::Document),
        &inputs,
    ))
    .unwrap();
    assert_eq!(
        (q["input_type"].as_str(), d["input_type"].as_str()),
        (Some("query"), Some("passage"))
    );
    assert_eq!(s.space(), "m/4/passage");
    let other = GLOBAL.replace("document = \"passage\"", "document = \"document\"");
    let SettingsEmbedder::Available(o) = resolved(&other).unwrap() else {
        panic!("must be available")
    };
    assert_ne!(
        s.space(),
        o.space(),
        "a cache under another document input_type must not be reused"
    );

    let without: String = GLOBAL
        .lines()
        .filter(|l| !l.starts_with("input_type"))
        .collect::<Vec<_>>()
        .join("\n");
    let e = resolved(&without).unwrap_err().to_string();
    assert!(e.contains("embedder.input_type"), "{e}");
    let empty = GLOBAL.replace("query = \"query\"", "query = \"\"");
    let e = resolved(&empty).unwrap_err().to_string();
    assert!(e.contains("embedder.input_type"), "{e}");
}
