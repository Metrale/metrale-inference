// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: On a multi-rank serve every prefill forward the head runs is one its workers were
//! told to run. Two-phase prefill (`Model::prefill_twophase`) has no worker command, so a head
//! that takes it leaves the workers parked in their command receive while its own prefix-lookup
//! agreement (`ep_min_u32`, one rooted broadcast per rank) waits on them: the collective rooted
//! at rank 1 never completes and the communicator is poisoned.
//!
//! The repro shape: three ranks, two requests decoding, a long prompt mid-prefill, and a second
//! long prompt arriving. `METRALE_EP_PREFILL_BATCH` defers the newcomer's chunk 0, the batched
//! step takes only the stream already in flight (both do not fit its rows), and on a later tick
//! the newcomer is the lone prefill with nothing started: the one state in which the scheduler
//! chose two-phase prefill.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;

/// 2026-10-17: The chunk size and the batched step's rows, scaled down from a serve's.
const CHUNK: usize = 64;
/// 2026-10-17: The long prompt already prefilling when the newcomer arrives.
const LONG_PREFIX: usize = 5 * CHUNK + 10;
/// 2026-10-17: The newcomer, longer than one chunk.
const NEWCOMER: usize = 3 * CHUNK + 7;

fn script(n: usize, base: u32) -> Vec<u32> {
    let mut g: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
    g.push(EOS);
    g
}

/// 2026-10-17: Two short requests decode together from the start; the long prefix arrives at
/// tick 2 and starts alone (inline chunk 0); the newcomer arrives at tick 3, while it prefills.
fn scenario(ep: bool) -> Scenario {
    let mut reqs = vec![
        ReqSpec::new(1, 8, script(40, 10)),
        ReqSpec::new(2, 8, script(40, 20)),
        ReqSpec::new(3, LONG_PREFIX, script(4, 30)),
        ReqSpec::new(4, NEWCOMER, script(4, 40)),
    ];
    reqs[2].arrive_at_tick = Some(2);
    reqs[3].arrive_at_tick = Some(3);
    Scenario {
        name: "ep_twophase_repro",
        cfg: ModelCfg {
            ep,
            ep_prefill_batch: Some(CHUNK),
            has_ssm_layers: true,
            twophase_ok: true,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: CHUNK,
            ep_prefill_batch: true,
            ..RunOptions::default()
        },
        reqs,
    }
}

fn trace(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

/// 2026-10-17: The `ep_broadcast_cmd` value lines after `i`, up to the next model call that is
/// not an announcement word.
fn announced_words(t: &[String], i: usize) -> Vec<usize> {
    t[i + 1..]
        .iter()
        .map_while(|l| {
            l.strip_prefix("ep_broadcast_cmd(0x")
                .and_then(|r| r.strip_suffix(')'))
                .map(|hex| usize::from_str_radix(hex, 16).expect("hex word"))
        })
        .collect()
}

fn field(line: &str, key: &str) -> usize {
    line.split(key)
        .nth(1)
        .and_then(|r| r.split([',', ')']).next())
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("{key} missing from {line}"))
}

/// 2026-10-17: Every single-stream prefill forward on the head is the one the workers were sent:
/// a `prefill_chunk` follows a `0xfffffff0` announcement whose words (chunk_len, chunk_start,
/// full_len) are that chunk's bounds, and no `prefill_twophase` runs at all.
fn assert_every_prefill_forward_is_announced(t: &[String]) {
    let mut pending: Option<(usize, usize)> = None;
    for (i, line) in t.iter().enumerate() {
        if line.contains("cmd=0xfffffff0") {
            let w = announced_words(t, i);
            assert!(
                w.len() >= 3,
                "announcement without its words at {i}:\n{t:#?}"
            );
            pending = Some((w[0], w[1]));
        } else if line.starts_with("prefill_twophase(") {
            panic!("the head ran a two-phase prefill its workers were never sent, at {i}:\n{t:#?}");
        } else if line.starts_with("prefill_chunk(") {
            let ran = (field(line, "len="), field(line, "start="));
            assert_eq!(
                pending.take(),
                Some(ran),
                "prefill_chunk at {i} is not the chunk announced to the workers:\n{t:#?}"
            );
        }
    }
}

/// 2026-10-17: The incident's shape on a multi-rank model: the newcomer's chunk 0 is deferred,
/// it reaches the single-stream path with nothing started while two requests decode, and the
/// head advances it in announced chunks. Fails on a scheduler that runs it through two-phase
/// prefill.
#[test]
fn a_deferred_long_prompt_on_a_multi_rank_serve_runs_announced_chunks() {
    let t = trace(&scenario(true));

    // 2026-10-17: The repro shape was reached: before the newcomer's first forward a two-row
    // decode is live and a batched step ran without it (its chunk 0 was deferred and did not fit
    // the step's rows), and that first forward starts at token 0.
    let newcomer = "s4@";
    let first = t
        .iter()
        .position(|l| {
            (l.starts_with("prefill_chunk(") || l.starts_with("prefill_twophase("))
                && l.contains(newcomer)
        })
        .unwrap_or_else(|| panic!("no single-stream prefill forward for {newcomer}:\n{t:#?}"));
    assert!(
        t[first].starts_with("prefill_twophase(") || t[first].contains("start=0,"),
        "the newcomer had started before the single-stream path:\n{t:#?}"
    );
    let two_row_decode = t[..first]
        .iter()
        .any(|l| l.starts_with("decode_batch(") && l.matches('@').count() >= 2);
    assert!(
        two_row_decode,
        "no batch-2 decode before the newcomer:\n{t:#?}"
    );
    let batched_without_it = t[..first]
        .iter()
        .any(|l| l.starts_with("prefill_batch_chunk(") && !l.contains(newcomer));
    assert!(
        batched_without_it,
        "no batched step ran while the newcomer waited:\n{t:#?}"
    );

    assert_every_prefill_forward_is_announced(&t);

    let done = t
        .iter()
        .filter(|l| l.starts_with("out s") && !l.contains("Err(") && !l.contains("dropped"))
        .count();
    assert_eq!(done, 4, "every client finishes:\n{t:#?}");
}
