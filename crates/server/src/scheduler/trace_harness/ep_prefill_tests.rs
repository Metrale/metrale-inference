// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The multi-rank batched prefill (`METRALE_EP_PREFILL_BATCH`) from the scheduler's
//! trace: a burst of requests on a multi-rank model is prefilled in one `prefill_batch_chunk`
//! with nothing sent to the workers per request, only when the lever is on and the model
//! supports it, and every client receives what the one-request-at-a-time prefill gives it.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;

/// 2026-10-09: Three requests queued before the loop starts (one burst), on a multi-rank model
/// that does (`supported`) or does not support the batched prefill, with
/// `--prefill-codispatch` on and `METRALE_EP_PREFILL_BATCH` as `lever`.
fn burst(supported: bool, lever: bool) -> Scenario {
    let script = |n: usize, base: u32| {
        let mut g: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
        g.push(EOS);
        g
    };
    Scenario {
        name: "ep_prefill_burst",
        cfg: ModelCfg {
            ep: true,
            ep_prefill_batch: supported,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: 64,
            prefill_codispatch: true,
            ep_prefill_batch: lever,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 12, script(4, 10)),
            ReqSpec::new(2, 9, script(3, 20)),
            ReqSpec::new(3, 15, script(5, 30)),
        ],
    }
}

fn trace(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

fn outputs(t: &[String]) -> Vec<&String> {
    t.iter().filter(|l| l.starts_with("out s")).collect()
}

/// 2026-10-09: With the lever on, the burst is one batched prefill of all three prompts, no
/// request's chunk 0 is announced to the workers on its own (`0xfffffff0`) and none runs
/// through `prefill_chunk`; each client gets the bytes of the per-request prefill.
#[test]
fn a_burst_is_one_batched_prefill_with_no_per_request_announcement() {
    let on = trace(&burst(true, true));
    let off = trace(&burst(true, false));
    let batched: Vec<&String> = on
        .iter()
        .filter(|l| l.starts_with("prefill_batch_chunk("))
        .collect();
    assert_eq!(batched.len(), 1, "one batched prefill expected:\n{on:#?}");
    for len in ["len=12", "len=9", "len=15"] {
        assert!(
            batched[0].contains(len),
            "{len} missing from {}",
            batched[0]
        );
    }
    assert!(
        !on.iter().any(|l| l.contains("cmd=0xfffffff0")),
        "a deferred chunk 0 was announced on its own:\n{on:#?}"
    );
    assert!(!on.iter().any(|l| l.starts_with("prefill_chunk(")));
    // 2026-10-09: The control: without the lever each request is announced and prefilled.
    assert_eq!(
        off.iter().filter(|l| l.contains("cmd=0xfffffff0")).count(),
        3
    );
    assert_eq!(
        off.iter()
            .filter(|l| l.starts_with("prefill_chunk("))
            .count(),
        3
    );
    assert!(!off.iter().any(|l| l.starts_with("prefill_batch_chunk(")));
    assert_eq!(outputs(&on).len(), 3);
    assert_eq!(outputs(&on), outputs(&off), "a client's output changed");
}

/// 2026-10-09: The lever does nothing on a model that cannot announce the pass: the trace is
/// the lever-off trace line for line.
#[test]
fn the_lever_is_inert_on_a_model_without_support() {
    assert_eq!(trace(&burst(false, true)), trace(&burst(true, false)));
}
