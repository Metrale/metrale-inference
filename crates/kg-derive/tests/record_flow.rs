// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `record_fails_closed` and the taint preview (queries doc §5; DESIGN-latlas.md §4),
//! over the two ports `kg_record` writes through. The fakes are the ports themselves: an
//! in-memory file map that counts replaces, and a validator that refuses on demand.

use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::pin;
use core::task::{Context, Poll, Waker};
use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::Config;
use metrale_kg_derive::record::Kind;
use metrale_kg_derive::record_flow::{RecordFs, Request, Validator, record};
use std::collections::BTreeMap;

mod common;

fn run<F: Future>(f: F) -> F::Output {
    let mut f = pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(v) => v,
        Poll::Pending => panic!("the fakes never pend"),
    }
}

#[derive(Default)]
struct Files {
    m: RefCell<BTreeMap<String, String>>,
    replaces: Cell<u32>,
}

impl RecordFs for Files {
    async fn read(&self, path: &str) -> Result<Option<String>> {
        Ok(self.m.borrow().get(path).cloned())
    }
    async fn replace(&self, path: &str, expected: &str, new: &str) -> Result<()> {
        let mut m = self.m.borrow_mut();
        if m.get(path).map_or("", String::as_str) != expected {
            return Err(error!(
                ErrorCode::KgRecordWrite,
                path.to_string(),
                "changed"
            ));
        }
        m.insert(path.to_string(), new.to_string());
        self.replaces.set(self.replaces.get() + 1);
        Ok(())
    }
}

struct Says(bool);
impl Validator for Says {
    async fn validate(&self, _kind: Kind, path: &str, _new: &str) -> Result<String> {
        if self.0 {
            Ok("ok".into())
        } else {
            Err(error!(
                ErrorCode::KgValidatorRejected,
                path.to_string(),
                "ledger.py check: no"
            ))
        }
    }
}

fn setup() -> (Config, Files) {
    let cfg = common::config();
    let f = Files::default();
    f.m.borrow_mut()
        .insert("c/LEDGER.tsv".into(), common::LEDGER.into());
    (cfg, f)
}

fn row(id: &str, deps: &str, retracts: &str) -> BTreeMap<String, String> {
    let header = common::LEDGER.lines().next().unwrap();
    header
        .split('\t')
        .map(|k| {
            let v = match k {
                "id" => id,
                "verdict" => "NULL (no win)",
                "depends_on" => deps,
                "retracts" => retracts,
                "gate" => "tost(1%,n=3)",
                "resolves" => "-",
                _ => "x",
            };
            (k.to_string(), v.to_string())
        })
        .collect()
}

fn req(fields: BTreeMap<String, String>, dry_run: bool, ack: Option<u32>) -> Request {
    Request {
        kind: Kind::Ledger,
        fields,
        dry_run,
        acknowledge_taint: ack,
    }
}

#[test]
fn a_rejected_record_leaves_the_file_byte_identical_and_never_writes() {
    let (cfg, f) = setup();
    let c = cfg.campaign("glm/mk").unwrap();
    let mut tab = row("new-a", "-", "-");
    tab.insert("hypothesis".into(), "a\tb".into());
    let e = run(record(&f, &Says(true), c, &req(tab, false, None))).unwrap_err();
    assert_eq!(e.code(), ErrorCode::KgToolArgument);
    let e = run(record(
        &f,
        &Says(true),
        c,
        &req(row("new-a", "ghost", "-"), false, None),
    ))
    .unwrap_err();
    assert_eq!(
        e.code(),
        ErrorCode::KgValidatorRejected,
        "the Rust ledger check refuses an unknown id"
    );
    let e = run(record(
        &f,
        &Says(false),
        c,
        &req(row("new-a", "-", "-"), false, None),
    ))
    .unwrap_err();
    assert_eq!(
        e.code(),
        ErrorCode::KgValidatorRejected,
        "the validator of record refuses"
    );
    assert_eq!(f.replaces.get(), 0);
    assert_eq!(f.m.borrow()["c/LEDGER.tsv"], common::LEDGER);
}

#[test]
fn a_retraction_previews_its_taint_and_writes_only_when_acknowledged() {
    let (cfg, f) = setup();
    let c = cfg.campaign("glm/mk").unwrap();
    // p0-x <- l2-pf <- seg: retracting p0-x taints all three.
    let dry = run(record(
        &f,
        &Says(true),
        c,
        &req(row("undo", "-", "p0-x"), true, None),
    ))
    .unwrap();
    assert_eq!(dry.would_taint, ["l2-pf", "p0-x", "seg"]);
    assert!(!dry.written);
    let e = run(record(
        &f,
        &Says(true),
        c,
        &req(row("undo", "-", "p0-x"), false, Some(2)),
    ))
    .unwrap_err();
    assert_eq!(e.code(), ErrorCode::KgToolArgument);
    assert_eq!(f.replaces.get(), 0);
    let ok = run(record(
        &f,
        &Says(true),
        c,
        &req(row("undo", "-", "p0-x"), false, Some(3)),
    ))
    .unwrap();
    assert!(ok.written && f.replaces.get() == 1);
    assert!(f.m.borrow()["c/LEDGER.tsv"].ends_with("\tp0-x\ttost(1%,n=3)\t-\n"));
    let plain = run(record(
        &f,
        &Says(true),
        c,
        &req(row("plain", "-", "-"), false, None),
    ))
    .unwrap();
    assert!(plain.would_taint.is_empty() && plain.written, "{plain:?}");
}
