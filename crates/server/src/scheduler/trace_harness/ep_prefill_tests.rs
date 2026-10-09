// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched prefill (`METRALE_EP_PREFILL_BATCH`) from the scheduler's
//! trace: requests that arrive together on a multi-rank model are prefilled in one
//! `prefill_batch_chunk` with nothing sent to the workers per request, with or without decodes
//! active and without `--prefill-codispatch`, within the model's batched-step rows; only when
//! the lever is on and the model supports it; and every client receives what the
//! one-request-at-a-time prefill gives it.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;

fn script(n: usize, base: u32) -> Vec<u32> {
    let mut g: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
    g.push(EOS);
    g
}

/// 2026-10-09: Three requests on a multi-rank model whose batched step carries `rows` rows
/// (`None`: unsupported), with `METRALE_EP_PREFILL_BATCH` as `lever`. With `late`, the first
/// request starts alone and the other two arrive at tick 2, while it decodes; otherwise all
/// three are queued before the loop starts.
fn scenario(rows: Option<usize>, lever: bool, late: bool) -> Scenario {
    let mut reqs = vec![
        ReqSpec::new(1, 12, script(6, 10)),
        ReqSpec::new(2, 9, script(3, 20)),
        ReqSpec::new(3, 15, script(5, 30)),
    ];
    if late {
        for r in &mut reqs[1..] {
            r.arrive_at_tick = Some(2);
        }
    }
    Scenario {
        name: "ep_prefill_burst",
        cfg: ModelCfg {
            ep: true,
            ep_prefill_batch: rows,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: 64,
            ep_prefill_batch: lever,
            ..RunOptions::default()
        },
        reqs,
    }
}

fn trace(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

fn outputs(t: &[String]) -> Vec<&String> {
    t.iter().filter(|l| l.starts_with("out s")).collect()
}

fn lines<'a>(t: &'a [String], prefix: &str) -> Vec<(usize, &'a String)> {
    t.iter()
        .enumerate()
        .filter(|(_, l)| l.starts_with(prefix))
        .collect()
}

fn announced(t: &[String]) -> usize {
    t.iter().filter(|l| l.contains("cmd=0xfffffff0")).count()
}

/// 2026-10-09: With the lever on, a burst is one batched prefill of all three prompts, without
/// `--prefill-codispatch`; no request's chunk 0 is announced to the workers on its own
/// (`0xfffffff0`) and none runs through `prefill_chunk`; each client gets the bytes of the
/// per-request prefill, which the lever-off control runs.
#[test]
fn a_burst_is_one_batched_prefill_with_no_per_request_announcement() {
    let on = trace(&scenario(Some(512), true, false));
    let off = trace(&scenario(Some(512), false, false));
    let batched = lines(&on, "prefill_batch_chunk(");
    assert_eq!(batched.len(), 1, "one batched prefill expected:\n{on:#?}");
    for len in ["len=12", "len=9", "len=15"] {
        assert!(
            batched[0].1.contains(len),
            "{len} missing from {}",
            batched[0].1
        );
    }
    assert_eq!(
        announced(&on),
        0,
        "a deferred chunk 0 was announced:\n{on:#?}"
    );
    assert!(lines(&on, "prefill_chunk(").is_empty());
    assert_eq!(announced(&off), 3);
    assert_eq!(lines(&off, "prefill_chunk(").len(), 3);
    assert!(lines(&off, "prefill_batch_chunk(").is_empty());
    assert_eq!(outputs(&on).len(), 3);
    assert_eq!(outputs(&on), outputs(&off), "a client's output changed");
}

/// 2026-10-09: Requests that arrive while a decode runs are deferred too, and batched in a
/// prefill-only step before that tick's decode: the next decode carries all three sequences.
/// The first request, alone with nothing in flight, keeps its inline prefill.
#[test]
fn arrivals_behind_an_active_decode_are_batched_before_the_decode() {
    let on = trace(&scenario(Some(512), true, true));
    let off = trace(&scenario(Some(512), false, true));
    let batched = lines(&on, "prefill_batch_chunk(");
    assert_eq!(batched.len(), 1, "one batched prefill expected:\n{on:#?}");
    assert!(batched[0].1.contains("len=9") && batched[0].1.contains("len=15"));
    assert!(!batched[0].1.contains("len=12"));
    assert_eq!(
        announced(&on),
        1,
        "only the lone first request is announced"
    );
    let next_decode = on[batched[0].0..]
        .iter()
        .find(|l| l.starts_with("decode_batch("))
        .expect("a decode after the batched prefill");
    let seqs = next_decode
        .split("seqs=[")
        .nth(1)
        .and_then(|r| r.split(']').next())
        .expect("seqs in the decode line");
    assert_eq!(seqs.split(", ").count(), 3, "{next_decode}");
    assert_eq!(announced(&off), 3);
    assert_eq!(outputs(&on), outputs(&off), "a client's output changed");
}

/// 2026-10-09: A step takes only the leading streams that fit the model's batched-step rows:
/// 12 + 9 fit 22 rows, 15 waits for the next tick, where it is the only prefill left and runs
/// through the single-stream path, announced on its own.
#[test]
fn a_step_takes_the_streams_that_fit_its_rows() {
    let on = trace(&scenario(Some(22), true, false));
    let off = trace(&scenario(Some(22), false, false));
    let batched = lines(&on, "prefill_batch_chunk(");
    assert_eq!(batched.len(), 1, "{on:#?}");
    assert!(batched[0].1.contains("len=12") && batched[0].1.contains("len=9"));
    assert!(!batched[0].1.contains("len=15"));
    let single = lines(&on, "prefill_chunk(");
    assert_eq!(single.len(), 1);
    assert!(single[0].1.contains("len=15") && single[0].0 > batched[0].0);
    assert_eq!(announced(&on), 1);
    assert_eq!(outputs(&on), outputs(&off), "a client's output changed");
}

/// 2026-10-09: The lever does nothing on a model that cannot announce the pass: the trace is
/// the lever-off trace line for line, for a burst and for arrivals behind a decode.
#[test]
fn the_lever_is_inert_on_a_model_without_support() {
    for late in [false, true] {
        assert_eq!(
            trace(&scenario(None, true, late)),
            trace(&scenario(Some(512), false, late))
        );
    }
}
