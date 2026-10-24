// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The batched MTP bootstrap's stash, from the scheduler's trace: every decode row is
//! stashed (slot j = decode row j), and the proposes name their own decode row's slot. A
//! multi-rank worker stashes every row after the same batched decode (`ep_worker_decode_batch`)
//! without knowing which rows propose, so a slot that counts only the proposing rows would read
//! another sequence's hidden on the worker.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::pipeline_tests::sync_run;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario};

/// 2026-10-11: Three MTP requests prefilled together; the first one's second token is EOS, so
/// it finishes in the batched bootstrap and only decode rows 1 and 2 propose.
fn first_row_finishes() -> Scenario {
    let toks = |base: u32| (0..10).map(|i| base + i).collect::<Vec<u32>>();
    Scenario {
        name: "bootstrap_stash",
        cfg: ModelCfg {
            has_proposer: true,
            can_batch_verify: true,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            use_speculative: true,
            num_drafts: 1,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 5, vec![10, EOS]),
            ReqSpec::new(2, 5, toks(20)),
            ReqSpec::new(3, 5, toks(40)),
        ],
    }
}

/// 2026-10-11: After the first three-row batched decode, the stash holds rows 0, 1 and 2, and
/// the proposes read slots 1 and 2 (rows of the sequences still alive), whether batched or one
/// at a time.
#[test]
fn a_batched_bootstrap_stashes_every_row_and_proposes_from_its_own() {
    let lines = sync_run(&first_row_finishes());
    let d = lines
        .iter()
        .position(|l| l.starts_with("decode_batch(") && l.contains("\"s3"))
        .unwrap_or_else(|| panic!("no three-row batched decode: {lines:#?}"));
    let after = &lines[d + 1..];
    let stash = after
        .iter()
        .find(|l| l.starts_with("stash_verify_hidden_rows("))
        .expect("the batched bootstrap stashed nothing");
    assert!(
        stash.starts_with("stash_verify_hidden_rows([0, 1, 2]"),
        "{stash}"
    );
    let end = after
        .iter()
        .position(|l| l.starts_with("-- tick"))
        .unwrap_or(after.len());
    let mut slots: Vec<String> = Vec::new();
    for l in &after[..end] {
        if let Some(rest) = l.strip_prefix("save_hidden_for_mtp_from_stash(") {
            slots.push(rest.split(',').next().unwrap().to_string());
        }
        if l.starts_with("run_mtp_propose_batched(") {
            let s = l
                .split("stash=[")
                .nth(1)
                .unwrap()
                .split(']')
                .next()
                .unwrap();
            slots.extend(s.split(", ").map(str::to_string));
        }
    }
    assert_eq!(slots, ["1", "2"], "{:#?}", &after[..end]);
}
