// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-19: The two-phase prefill on a multi-rank serve, from the scheduler's
//! trace. A prompt longer than one chunk never takes `prefill_twophase` on a
//! multi-rank model: its fallback loop calls `prefill_chunk` without the
//! `0xFFFFFFF0` head/worker handshake, which desynchronizes the wire (a 62k
//! prompt parked the workers and left every later request unserved). And a
//! failed two-phase prefill fails the request instead of falling back to the
//! chunked path: the failure is a poisoned communicator, and falling back
//! would accept requests the engine can never answer.
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

/// 2026-10-19: Two requests whose prompts (128 tokens) exceed
/// `max_prefill_tokens` (64), so the single-stream path chooses between
/// the two-phase prefill and the chunked loop at `chunk_offset == 0` —
/// the point the 2026-10-19 incident turned wrong. On the single-GPU
/// model the varlen lever defers chunk 0 and `bisect_q12_disable` keeps
/// the batched paths off, so both requests reach the single-stream path;
/// on the multi-rank one the batched prefill carries one stream per tick
/// and the leftover reaches it. `twophase_ok` makes `prefill_twophase`
/// succeed or fail.
fn scenario(ep: bool, twophase_ok: bool, healthy: bool) -> Scenario {
    Scenario {
        name: "twophase_fail",
        cfg: ModelCfg {
            ep,
            ep_prefill_batch: ep.then_some(1),
            twophase_ok,
            healthy,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: 64,
            // 2026-10-19: The single-GPU deferral: the varlen lever defers
            // chunk 0, and the q12 bisect keeps the batched paths off so
            // the deferred prefills reach the single-stream path.
            prefill_varlen: !ep,
            bisect_q12_disable: !ep,
            ep_prefill_batch: ep,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 128, script(6, 10)),
            ReqSpec::new(2, 128, script(6, 20)),
        ],
    }
}

fn traced(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

fn lines<'a>(t: &'a [String], needle: &str) -> Vec<&'a String> {
    t.iter().filter(|l| l.contains(needle)).collect()
}

fn client_output(t: &[String]) -> Vec<&String> {
    t.iter().filter(|l| l.starts_with("out s")).collect()
}

/// 2026-10-19: The single-GPU control: a long prompt goes through
/// `prefill_twophase`, which succeeds and serves the client.
#[test]
fn single_gpu_long_prompt_still_takes_the_twophase_prefill() {
    let t = traced(&scenario(false, true, true));
    assert_eq!(lines(&t, "prefill_twophase(").len(), 2, "{t:#?}");
    assert_eq!(client_output(&t).len(), 2, "{t:#?}");
    assert!(client_output(&t).iter().all(|l| l.contains("Done")));
}

/// 2026-10-19: On a multi-rank model the long prompt must NOT take the
/// two-phase path: it advances through the chunked loop, whose per-chunk
/// `0xFFFFFFF0` broadcast is the handshake the parked workers wait on.
/// The single-GPU control above runs `prefill_twophase` for the same
/// scenario, so the assertion fails without the `is_ep` gate.
#[test]
fn a_multi_rank_serve_never_takes_the_twophase_path_for_a_long_prompt() {
    let ep = traced(&scenario(true, true, true));
    assert!(
        lines(&ep, "prefill_twophase(").is_empty(),
        "a multi-rank serve ran the two-phase prefill:\n{ep:#?}"
    );
    assert!(
        !lines(&ep, "cmd=0xfffffff0").is_empty(),
        "the chunked loop never announced a chunk:\n{ep:#?}"
    );
    assert!(lines(&ep, "prefill_chunk(").len() >= 2, "{ep:#?}");
    assert!(client_output(&ep).iter().any(|l| l.contains("Done")));
    // 2026-10-19: The path differs, the client's output does not.
    let solo = traced(&scenario(false, true, true));
    assert_eq!(client_output(&ep), client_output(&solo));
}

/// 2026-10-19: A failed two-phase prefill fails the request with an error
/// frame; it does not fall back to the chunked path (which on the real
/// engine desynchronizes the wire) and does not hang the client. The
/// failure is injected on the single-GPU scenario, where the two-phase
/// prefill is attempted; the other request still completes.
#[test]
fn a_failed_twophase_prefill_errors_the_client_instead_of_falling_back() {
    let t = traced(&scenario(false, false, true));
    let errs = lines(&t, "out s1: Error(");
    assert_eq!(errs.len(), 1, "the client got no error frame:\n{t:#?}");
    assert!(
        errs[0].contains("two-phase prefill failed"),
        "wrong error: {}",
        errs[0]
    );
    // 2026-10-19: No fallback: s1's prompt was never prefilled in chunks.
    assert!(
        !lines(&t, "prefill_chunk(").iter().any(|l| l.contains("s1@")),
        "the chunked fallback ran for s1:\n{t:#?}"
    );
    // 2026-10-19: Every request that reaches the two-phase gate gets its
    // own error frame; none hangs.
    assert_eq!(
        lines(&t, "out s2: Error(").len(),
        1,
        "s2 got no error frame:\n{t:#?}"
    );
}

/// 2026-10-19: A two-phase failure on a model whose communicator reports
/// unhealthy requests shutdown, so the process stops serving instead of
/// accepting requests forever. The shutdown latch is process-global and
/// one-way (the TUI quit tests in this same binary set it, on parallel
/// threads), so both assertions need a fresh process: the test re-runs
/// itself with a marker env var, and the child does the actual work.
#[test]
fn an_unhealthy_communicator_after_a_twophase_failure_requests_shutdown() {
    use metrale_abstract_io::Env as _;
    let child = metrale_io_host_std::ProcessEnv
        .var("METRALE_TWOPHASE_SHUTDOWN_CHILD")
        .expect("unicode marker")
        .is_some();
    if child {
        // 2026-10-19: The healthy control: the same failure must NOT request
        // shutdown.
        let healthy = traced(&scenario(false, false, true));
        assert_eq!(lines(&healthy, "out s1: Error(").len(), 1, "{healthy:#?}");
        assert!(
            !crate::tui::shutdown::requested(),
            "a healthy model after a two-phase failure requested shutdown:\n{healthy:#?}"
        );

        let poisoned = traced(&scenario(false, false, false));
        assert_eq!(lines(&poisoned, "out s1: Error(").len(), 1, "{poisoned:#?}");
        assert!(
            crate::tui::shutdown::requested(),
            "an unhealthy model after a two-phase failure did not request shutdown"
        );
        return;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args(["--exact", "an_unhealthy_communicator_after_a_twophase_failure_requests_shutdown"])
        .env("METRALE_TWOPHASE_SHUTDOWN_CHILD", "1")
        .output()
        .expect("child test run");
    assert!(
        out.status.success(),
        "child run failed:\n{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
