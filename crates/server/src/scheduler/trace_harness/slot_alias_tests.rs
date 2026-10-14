// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: DEF-1 repro. Decodes that retire while another sequence is still in
//! chunked prefill must never compact a survivor onto that sequence's slot.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The fake model hands out slots like the SSM pool (lowest free first) and, like
//! the engine before the fix, lets `compact_sequence` move onto any slot. The
//! check reads only the trace: every `s<id>@<slot>` a line names, which sequences
//! `free_sequence` has released, and each `compact_sequence(.., new_slot=N)`.

use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};

static SERIAL: Mutex<()> = Mutex::new(());

fn gen_eos(n: usize, base: u32) -> Vec<u32> {
    let mut g: Vec<u32> = (0..n as u32).map(|i| base + (i % 20)).collect();
    g.push(EOS);
    g
}

/// 2026-10-14: s1 decodes from tick 1. At tick 2 the 64-token s2 arrives first
/// (slot 1, eight 8-token chunks, so it stays mid-prefill for several ticks),
/// then two one-step shorts (slots 2 and 3) and two long decodes (slots 4 and 5).
/// When the shorts retire, three decodes survive; on d9b77748d the compaction
/// range is [0..3) and slot 1 looks free, so a survivor lands on s2's slot.
fn long_prefill_with_retiring_decodes() -> Scenario {
    let at_tick_2 = |id, prompt_len, toks| {
        let mut r = ReqSpec::new(id, prompt_len, toks);
        r.arrive_at_tick = Some(2);
        r
    };
    Scenario {
        name: "slot_alias_mid_prefill",
        cfg: ModelCfg::default(),
        opts: RunOptions {
            max_prefill_tokens: 8,
            ..RunOptions::default()
        },
        reqs: vec![
            ReqSpec::new(1, 4, gen_eos(30, 10)),
            at_tick_2(2, 64, gen_eos(3, 20)),
            at_tick_2(3, 4, gen_eos(1, 30)),
            at_tick_2(4, 4, gen_eos(1, 40)),
            at_tick_2(5, 4, gen_eos(30, 50)),
            at_tick_2(6, 4, gen_eos(30, 60)),
        ],
    }
}

/// 2026-10-14: Every `s<id>@<slot>` in `line`.
fn seq_slots(line: &str) -> Vec<(u64, usize)> {
    let mut out = Vec::new();
    let b = line.as_bytes();
    let digits = |from: usize| -> (usize, usize) {
        let end = (from..b.len())
            .find(|&j| !b[j].is_ascii_digit())
            .unwrap_or(b.len());
        (from, end)
    };
    for i in 0..b.len() {
        if b[i] != b's' || (i > 0 && b[i - 1].is_ascii_alphanumeric()) {
            continue;
        }
        let (s, e) = digits(i + 1);
        if e == s || e >= b.len() || b[e] != b'@' {
            continue;
        }
        let (s2, e2) = digits(e + 1);
        if e2 == s2 {
            continue;
        }
        if let (Ok(id), Ok(slot)) = (line[s..e].parse(), line[s2..e2].parse()) {
            out.push((id, slot));
        }
    }
    out
}

/// 2026-10-14: The violations in a trace: a compaction onto a slot another live
/// sequence holds, or one decode batch naming a slot twice.
fn slot_aliasing(lines: &[String]) -> Vec<String> {
    let mut slot_of: HashMap<u64, usize> = HashMap::new();
    let mut live: HashSet<u64> = HashSet::new();
    let mut bad = Vec::new();
    for (n, l) in lines.iter().enumerate() {
        let named = seq_slots(l);
        if l.starts_with("free_sequence(") {
            for (id, _) in &named {
                live.remove(id);
                slot_of.remove(id);
            }
            continue;
        }
        for &(id, slot) in &named {
            live.insert(id);
            slot_of.insert(id, slot);
        }
        if l.starts_with("compact_sequence(")
            && let Some((id, _)) = named.first().copied()
            && let Some(target) = l
                .split("new_slot=")
                .nth(1)
                .and_then(|t| t.trim_end_matches(')').parse::<usize>().ok())
        {
            for other in live.iter().filter(|&&o| o != id) {
                if slot_of.get(other) == Some(&target) {
                    bad.push(format!(
                        "line {n}: {l} moves s{id} onto live s{other}'s slot"
                    ));
                }
            }
            slot_of.insert(id, target);
        }
        if l.starts_with("decode_batch(") {
            let mut seen = HashSet::new();
            for (id, slot) in &named {
                if !seen.insert(*slot) {
                    bad.push(format!("line {n}: {l} names slot {slot} twice (s{id})"));
                }
            }
        }
    }
    bad
}

#[test]
fn a_retiring_decode_never_compacts_onto_a_mid_prefill_sequences_slot() {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let lines = run_scenario(&long_prefill_with_retiring_decodes());
    // 2026-10-14: The scenario must reach the trigger, or a pass proves nothing:
    // s2 is still being prefilled when the first short is released.
    let first_free = lines
        .iter()
        .position(|l| l.starts_with("free_sequence(s3@") || l.starts_with("free_sequence(s4@"))
        .expect("a short retires");
    let s2_last_chunk = lines
        .iter()
        .rposition(|l| l.contains("s2@") && l.contains("last=true"))
        .expect("s2's last prefill chunk");
    assert!(
        first_free < s2_last_chunk,
        "the shorts must retire while s2 is mid-prefill (free at {first_free}, s2 done at \
         {s2_last_chunk})"
    );
    assert!(
        lines.iter().any(|l| l.starts_with("compact_sequence(")),
        "the retirement must compact a survivor"
    );
    let bad = slot_aliasing(&lines);
    assert!(bad.is_empty(), "slot aliasing:\n{}", bad.join("\n"));
    for id in 1..=6 {
        assert!(
            lines
                .iter()
                .any(|l| l.starts_with(&format!("out s{id}: ")) && l.contains("Done(stop")),
            "s{id} did not finish"
        );
    }
}

/// 2026-10-14: The checker itself flags an aliasing compaction and a doubled decode
/// slot, so a pass above is not a blind checker.
#[test]
fn the_aliasing_checker_flags_both_violation_shapes() {
    let lines: Vec<String> = [
        "prefill_chunk(start=0, len=8, last=false, s2@1:len=8, stream=1)",
        "decode_batch(tokens=[1, 2], seqs=[\"s5@4:len=5\", \"s6@5:len=5\"], stream=0)",
        "compact_sequence(s6@5:len=5, new_slot=1)",
        "decode_batch(tokens=[1, 2], seqs=[\"s5@1:len=6\", \"s6@1:len=6\"], stream=0)",
    ]
    .map(String::from)
    .to_vec();
    let bad = slot_aliasing(&lines);
    assert_eq!(bad.len(), 2, "{bad:#?}");
    assert!(bad[0].contains("onto live s2's slot"), "{}", bad[0]);
    assert!(bad[1].contains("names slot 1 twice"), "{}", bad[1]);
    assert!(slot_aliasing(&lines[..2]).is_empty());
}
