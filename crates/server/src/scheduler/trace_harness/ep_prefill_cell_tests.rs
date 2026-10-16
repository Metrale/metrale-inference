// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: The multi-rank batched prefill (`METRALE_EP_PREFILL_BATCH`) under
//! `METRALE_GLM_PREFILL_CELL_GRID`, from the scheduler's trace: a prompt the cell walk would
//! touch (longer than the grid's first point) never rides a batched prefill or the two-phase
//! prefill, and runs through `prefill_chunk` with every non-last chunk ending on the grid;
//! whole prompts inside the first cell still batch. The grid-off control of each scenario
//! routes the long prompt through the batched or two-phase pass, so the routing is what the
//! grid changes. Every client receives the grid-off output.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;
use metrale_model_arch::glm5next_prefill_cell::CellGrid;

/// 2026-10-16: Grid points 32, 64, 96, ...: KV block boundaries of the fake's 16-token blocks.
const GRID: CellGrid = CellGrid {
    first: 32,
    step: 32,
};
/// 2026-10-16: The long prompt: three chunks under the 48-token budget on the grid.
const LONG: usize = 100;
/// 2026-10-16: The long prompt's sequence in a trace line (request 2).
const LONG_SEQ: &str = "s2@";

fn script(n: usize, base: u32) -> Vec<u32> {
    let mut g: Vec<u32> = (0..n as u32).map(|i| base + i).collect();
    g.push(EOS);
    g
}

/// 2026-10-16: A 12-token, a `second`-token and a 9-token request on a multi-rank model with the
/// lever on, a 48-token prefill budget and a working two-phase prefill. Without `late`, all three
/// are queued before the loop starts and a batched step carries 512 rows. With `late`, the
/// 9-token and then the second request arrive together at tick 2, behind the first one's
/// decode, so both are deferred; a batched step carries 22 rows, so the 9-token one batches
/// alone and the second one is left as the only prefill, at position 0.
fn scenario(grid: Option<CellGrid>, late: bool, second: usize) -> Scenario {
    let mut reqs = vec![
        ReqSpec::new(1, 12, script(6, 10)),
        ReqSpec::new(2, second, script(3, 20)),
        ReqSpec::new(3, 9, script(5, 30)),
    ];
    if late {
        reqs.swap(1, 2);
        for r in &mut reqs[1..] {
            r.arrive_at_tick = Some(2);
        }
    }
    Scenario {
        name: "ep_prefill_cell",
        cfg: ModelCfg {
            ep: true,
            ep_prefill_batch: Some(if late { 22 } else { 512 }),
            twophase_ok: true,
            cell_grid: grid,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: 48,
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

fn outputs(t: &[String]) -> Vec<&String> {
    t.iter().filter(|l| l.starts_with("out s")).collect()
}

fn with<'a>(t: &'a [String], prefix: &str, needle: &str) -> Vec<&'a String> {
    t.iter()
        .filter(|l| l.starts_with(prefix) && l.contains(needle))
        .collect()
}

/// 2026-10-16: The long prompt's `(start, len)` single-sequence chunks, in order.
fn long_chunks(t: &[String]) -> Vec<(usize, usize)> {
    let field = |l: &str, k: &str| -> usize {
        l.split(k)
            .nth(1)
            .and_then(|r| r.split(',').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{k} in {l}"))
    };
    t.iter()
        .filter(|l| l.starts_with("prefill_chunk(") && l.contains(LONG_SEQ))
        .map(|l| (field(l, "start="), field(l, "len=")))
        .collect()
}

/// 2026-10-16: The long prompt in the grid-on trace: no batched, mixed or two-phase pass carries
/// it, its single-sequence chunks cover it in order with every non-last chunk ending on a grid
/// point, and each client's output is the grid-off output.
fn assert_floored(on: &[String], off: &[String]) {
    for pass in [
        "prefill_batch_chunk(",
        "mixed_forward_batch(",
        "prefill_twophase(",
    ] {
        assert!(
            with(on, pass, LONG_SEQ).is_empty(),
            "{pass} carried it:\n{on:#?}"
        );
    }
    let chunks = long_chunks(on);
    assert_eq!(chunks, vec![(0, 32), (32, 32), (64, 36)], "{on:#?}");
    for &(s, l) in &chunks[..chunks.len() - 1] {
        assert_eq!((s + l - GRID.first) % GRID.step, 0, "{s}+{l} off the grid");
    }
    assert_eq!(outputs(on).len(), outputs(off).len());
    assert_eq!(outputs(on), outputs(off), "a client's output changed");
}

/// 2026-10-16: A burst: the leading 12-token prompt batches alone, the long one takes the
/// floored single-sequence chunks. Firing negative control: grid off, the burst is one batched
/// prefill carrying the long prompt's first chunk.
#[test]
fn under_the_grid_a_long_prompt_leaves_the_batched_prefill() {
    let on = trace(&scenario(Some(GRID), false, LONG));
    let off = trace(&scenario(None, false, LONG));
    assert!(
        !with(&off, "prefill_batch_chunk(", LONG_SEQ).is_empty(),
        "control: grid off batches the long prompt:\n{off:#?}"
    );
    let batched = with(&on, "prefill_batch_chunk(", "len=12");
    assert_eq!(batched.len(), 1, "the short head still batches:\n{on:#?}");
    assert_floored(&on, &off);
}

/// 2026-10-16: A long prompt deferred behind an active decode and left alone at position 0 by a
/// full batched step: grid on, it takes the floored single-sequence chunks. Firing negative
/// control: grid off, it runs as one two-phase prefill.
#[test]
fn under_the_grid_a_deferred_long_prompt_skips_the_two_phase_prefill() {
    let on = trace(&scenario(Some(GRID), true, LONG));
    let off = trace(&scenario(None, true, LONG));
    assert_eq!(
        with(&off, "prefill_twophase(", LONG_SEQ).len(),
        1,
        "control: grid off runs it two-phase:\n{off:#?}"
    );
    assert_floored(&on, &off);
}

/// 2026-10-16: Positive control, no batching lost: when every prompt is whole inside the first
/// cell (the second one exactly `GRID.first` tokens long), the grid-on trace is the grid-off
/// trace line for line, for a burst (one batched prefill of all three) and for a deferred
/// arrival.
#[test]
fn under_the_grid_whole_prompts_inside_the_first_cell_batch_as_before() {
    for late in [false, true] {
        let on = trace(&scenario(Some(GRID), late, GRID.first));
        let off = trace(&scenario(None, late, GRID.first));
        assert_eq!(on, off, "late={late}");
    }
    let burst = trace(&scenario(Some(GRID), false, GRID.first));
    let batched = with(&burst, "prefill_batch_chunk(", "");
    assert_eq!(batched.len(), 1, "{burst:#?}");
    for len in ["len=12 ", "len=32 ", "len=9 "] {
        assert!(
            batched[0].contains(len),
            "{len} missing from {}",
            batched[0]
        );
    }
}
