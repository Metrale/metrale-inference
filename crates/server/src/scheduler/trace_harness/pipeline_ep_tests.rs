// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The pipelined decode lane for a multi-rank model with recurrent state
//! (`FedAheadPolicy::RepickOnly`, the GLM-5.3 serve under `METRALE_EP_DEVICE_FEED`), driven
//! through the real core with the asynchronous router over the recording model.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The fake keeps recurrent state (`model_recurrent.rs`): a step run twice, or a continuing
//! row fed at a position it already consumed, corrupts the state, and the client stream
//! shows it. What is proved here, on the CPU:
//! * the lane runs ahead under `RepickOnly` with one row, never with more, and every client
//!   stream equals the synchronous router's, with no recurrent state consumed twice;
//! * the gate is the policy: the same model under `Redo` never runs ahead;
//! * the oracle can fail: a forward at a consumed position turns the output to garbage.

use metrale_model_engine::traits::SequenceState;

use super::model::{GARBAGE, ModelCfg, RecordingModel};
use super::pipeline_tests::{async_run, client_lines, count, sync_run};
use super::runner::{EOS, ReqSpec, RunOptions, Scenario};
use super::scenarios;

/// 2026-10-10: The GLM-5.3 multi-rank serve as the scheduler sees it.
fn ep_recurrent(rows: Option<usize>) -> ModelCfg {
    ModelCfg {
        ep: true,
        has_ssm_layers: true,
        recurrent_fed_rows: rows,
        ..ModelCfg::default()
    }
}

/// 2026-10-10: C1: one request at a time (batch size 1), one stopping by EOS and one by its
/// token budget.
fn c1(name: &'static str, cfg: ModelCfg) -> Scenario {
    let mut eos = ReqSpec::new(1, 5, (10..22).chain([EOS]).collect());
    eos.max_tokens = 30;
    let mut budget = ReqSpec::new(2, 7, (30..50).collect());
    budget.max_tokens = 9;
    Scenario {
        name,
        cfg,
        opts: RunOptions {
            max_batch_size: 1,
            ..RunOptions::default()
        },
        reqs: vec![eos, budget],
    }
}

/// 2026-10-10: Every fed launch in a trace carries exactly one sequence.
fn fed_rows_per_launch(lines: &[String]) -> Vec<usize> {
    lines
        .iter()
        .filter_map(|l| l.split_once("decode_batch_fed(").map(|(_, rest)| rest))
        .map(|rest| rest.matches("len=").count())
        .collect()
}

/// 2026-10-10: The trace lines before the first line containing `needle`, for a failure message.
fn around(lines: &[String], needle: &str) -> String {
    let at = lines.iter().position(|l| l.contains(needle)).unwrap_or(0);
    lines[at.saturating_sub(12)..=at.min(lines.len().saturating_sub(1))].join("\n")
}

#[test]
fn a_recurrent_multi_rank_model_runs_one_row_ahead_with_the_synchronous_streams() {
    let sc = c1("ep_recurrent_c1", ep_recurrent(Some(1)));
    let expected = client_lines(&sync_run(&sc));
    let live = async_run(&sc);
    assert_eq!(
        client_lines(&live),
        expected,
        "the async router changed a stream"
    );
    assert!(count(&live, "fx launch DecodeFed") > 0, "never ran ahead");
    assert_eq!(
        count(&live, "consumed again"),
        0,
        "recurrent state was reused"
    );
    assert!(fed_rows_per_launch(&live).iter().all(|&n| n == 1));
}

#[test]
fn the_plain_lanes_keep_their_streams_and_never_feed_two_rows() {
    for name in [
        "plain_decode",
        "chunked_prefill",
        "mixed_prefill",
        "batched_prefill_waves",
        "batched_mixed",
        "preempt_requeue",
        "spill_swap_out_in",
        "request_timeout",
        "slai_policy",
        "host_logits_paths",
    ] {
        let mut sc = scenarios::all()
            .into_iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("no scenario {name}"));
        sc.cfg = ModelCfg {
            ep: true,
            has_ssm_layers: true,
            recurrent_fed_rows: Some(1),
            ..sc.cfg
        };
        let expected = sync_run(&sc);
        let live = async_run(&sc);
        assert_eq!(client_lines(&live), client_lines(&expected), "{name}");
        assert_eq!(
            count(&expected, "consumed again"),
            0,
            "{name} (sync):\n{}",
            around(&expected, "consumed again")
        );
        assert_eq!(
            count(&live, "consumed again"),
            0,
            "{name}:\n{}",
            around(&live, "consumed again")
        );
        assert!(
            fed_rows_per_launch(&live).iter().all(|&n| n == 1),
            "{name} fed more than one row"
        );
    }
}

#[test]
fn the_same_model_under_redo_never_runs_ahead() {
    let live = async_run(&c1("ep_recurrent_redo", ep_recurrent(None)));
    assert_eq!(count(&live, "fx launch DecodeFed"), 0);
}

#[test]
fn the_recurrent_oracle_flags_a_position_consumed_twice() {
    let model = RecordingModel::new(ep_recurrent(Some(1)));
    let mut seq = SequenceState::host_only(0);
    seq.session_hash = 7;
    assert!(model.recurrent_consume(&seq, 0, 5));
    assert!(model.recurrent_consume(&seq, 5, 1));
    assert!(
        !model.recurrent_consume(&seq, 5, 1),
        "a re-run went unnoticed"
    );
    assert!(
        !model.recurrent_consume(&seq, 6, 1),
        "corruption must persist"
    );
    assert_eq!(model.recurrent_out(false, 42), GARBAGE);
    assert_eq!(model.take_trace().len(), 1);
}

/// 2026-10-10: The negative control for the C1 equivalence above: with the over-run discard
/// switched off, the step run past an EOS reaches a client, so that comparison can fail.
#[test]
fn the_c1_comparison_fails_when_the_over_run_discard_is_off() {
    let sc = c1("ep_recurrent_c1_control", ep_recurrent(Some(1)));
    let expected = client_lines(&sync_run(&sc));
    let mut control = sc.clone();
    control.opts.pipeline_faults = crate::scheduler::PipelineFaults { keep_overrun: true };
    let live = async_run(&control);
    assert!(count(&live, "fx launch DecodeFed") > 0);
    assert_ne!(client_lines(&live), expected);
}
