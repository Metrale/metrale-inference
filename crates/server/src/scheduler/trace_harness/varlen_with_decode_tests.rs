// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: `--prefill-varlen-with-decode`: prompts that land while a sequence
//! decodes run as varlen waves beside the decode, not one inline chunk 0 each.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! Every assertion reads the live trace of the real scheduler loop over the
//! recording model. Each positive claim has a control with the lever off, so a
//! scenario that never reaches the serial path cannot pass vacuously.

use super::model::ModelCfg;
use super::runner::{EOS, ReqSpec, RunOptions, Scenario, run_scenario};
use super::tests::SERIAL;

fn gen_eos(n: usize, base: u32) -> Vec<u32> {
    let mut g: Vec<u32> = (0..n as u32).map(|i| base + (i % 20)).collect();
    g.push(EOS);
    g
}

/// 2026-10-09: s1 decodes from tick 1; s2..s5 land together at tick 2. With a
/// 16-token chunk, s2+s3 (6+5) share a wave, s4 (7) opens a second, and s5 (20,
/// not its last chunk) a third.
fn burst(varlen: bool, with_decode: bool, spec: bool) -> Scenario {
    let mut reqs = vec![ReqSpec::new(1, 4, gen_eos(12, 10))];
    for (id, prompt) in [(2u64, 6usize), (3, 5), (4, 7), (5, 20)] {
        let mut r = ReqSpec::new(id, prompt, gen_eos(4, 10 * id as u32));
        r.arrive_at_tick = Some(2);
        reqs.push(r);
    }
    let apply_levers: fn(&mut crate::scheduler::levers::SchedLevers) = match (varlen, with_decode) {
        (true, true) => |l| {
            l.prefill_varlen = true;
            l.prefill_varlen_with_decode = true;
        },
        (true, false) => |l| l.prefill_varlen = true,
        (false, _) => |_| {},
    };
    Scenario {
        name: "varlen_with_decode",
        cfg: ModelCfg {
            has_proposer: spec,
            ..ModelCfg::default()
        },
        opts: RunOptions {
            max_prefill_tokens: 16,
            use_speculative: spec,
            num_drafts: if spec { 3 } else { 1 },
            apply_levers,
            ..RunOptions::default()
        },
        reqs,
    }
}

fn run(sc: &Scenario) -> Vec<String> {
    let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    run_scenario(sc)
}

/// 2026-10-09: The trace lines of the tick that first mentions a prompt of
/// `s2` (the tick the burst is admitted on).
fn burst_tick(lines: &[String]) -> Vec<String> {
    let mut ticks: Vec<Vec<String>> = vec![Vec::new()];
    for l in lines {
        if l.starts_with("-- tick") {
            ticks.push(Vec::new());
        }
        ticks.last_mut().expect("one tick").push(l.clone());
    }
    ticks
        .into_iter()
        .find(|t| t.iter().any(|l| l.contains("s2@")))
        .expect("the burst is admitted on some tick")
}

/// 2026-10-09: What each client received, without `acc=` (accepted draft tokens): the
/// tokens are the request's, but how many came from drafts depends on the tick a
/// sequence first speculates on, which the schedule decides.
fn outputs(lines: &[String]) -> Vec<String> {
    let out: Vec<String> = lines
        .iter()
        .filter(|l| l.starts_with("out s"))
        .map(|l| {
            let (head, tail) = l.split_once(" acc=").expect("a Done line names acc=");
            let rest = tail.split_once(',').map_or("", |(_, r)| r);
            format!("{head}{rest}")
        })
        .collect();
    assert_eq!(out.len(), 5, "five clients answered: {out:?}");
    for o in &out {
        assert!(o.contains("Done(stop"), "every client stopped on EOS: {o}");
    }
    out
}

/// 2026-10-09: The inline chunk-0 prefills of the burst's requests.
fn inline_chunk0(lines: &[String]) -> usize {
    lines
        .iter()
        .filter(|l| l.starts_with("prefill_chunk(start=0") && !l.contains("s1@"))
        .count()
}

fn batch_calls(tick: &[String]) -> Vec<&String> {
    tick.iter()
        .filter(|l| l.starts_with("prefill_batch_chunk("))
        .collect()
}

#[test]
fn burst_beside_decode_runs_as_waves_not_serial_prefills() {
    for spec in [false, true] {
        let off = run(&burst(true, false, spec));
        let on = run(&burst(true, true, spec));
        // 2026-10-09: Control: without the lever every burst request prefills
        // its chunk 0 inline, one after another.
        assert_eq!(inline_chunk0(&off), 4, "spec={spec}: control is serial");
        assert!(batch_calls(&burst_tick(&off)).is_empty(), "spec={spec}");

        assert_eq!(inline_chunk0(&on), 0, "spec={spec}: no inline chunk 0");
        let tick = burst_tick(&on);
        let waves = batch_calls(&tick);
        assert_eq!(
            waves.len(),
            1,
            "spec={spec}: one wave on the burst tick: {waves:?}"
        );
        assert!(
            waves[0].contains("s2@") && waves[0].contains("s3@"),
            "spec={spec}: s2 and s3 share the first wave: {}",
            waves[0]
        );
        // 2026-10-09: The decode still runs on that tick, after the wave, and no mixed step
        // replaced it.
        let wave_at = tick.iter().position(|l| l == waves[0]).expect("wave");
        // 2026-10-09: `decode(` is a new sequence's MTP bootstrap; `decode_verify` the
        // speculative step of the sequence that was already decoding.
        let decode = tick.iter().rposition(|l| {
            l.starts_with("decode_batch(")
                || l.starts_with("decode_verify")
                || l.starts_with("decode(")
        });
        assert!(
            decode.is_some_and(|d| d > wave_at),
            "spec={spec}: a decode follows the wave: {tick:#?}"
        );
        assert!(!tick.iter().any(|l| l.starts_with("mixed_forward")));
        if spec {
            assert!(
                tick.iter().any(|l| l.starts_with("decode_verify")),
                "the decoding sequence kept its speculative step on the burst tick"
            );
        }
        // 2026-10-09: The next wave (s4) is a batched forward of a later tick, not an inline
        // chunk 0.
        let later = on
            .iter()
            .skip_while(|l| *l != waves[0])
            .skip(1)
            .find(|l| l.starts_with("prefill_batch_chunk(") || l.starts_with("prefill_chunk("))
            .expect("the remaining streams prefill later");
        assert!(
            later.starts_with("prefill_batch_chunk(") && later.contains("s4@"),
            "spec={spec}: the second wave: {later}"
        );
        // 2026-10-09: The fake answers from per-request scripts, so what each
        // client receives does not depend on the path.
        assert_eq!(outputs(&on), outputs(&off), "spec={spec}");
    }
}

#[test]
fn the_lever_needs_varlen_to_change_anything() {
    // 2026-10-09: Varlen off: the lever's scheduler branches stay closed even if
    // its field were set (serve refuses that combination at startup).
    let mut sc = burst(false, false, false);
    sc.opts.apply_levers = |l| l.prefill_varlen_with_decode = true;
    let lines = run(&sc);
    assert_eq!(inline_chunk0(&lines), 4);
    assert!(lines.iter().all(|l| !l.starts_with("prefill_batch_chunk(")));
}

#[test]
fn serve_refuses_the_lever_without_varlen() {
    use crate::scheduler::levers::SchedLevers;
    let err = SchedLevers::defaults()
        .with_prefill_varlen_with_decode(true)
        .err()
        .expect("refused without varlen");
    assert!(format!("{err:#}").contains("--prefill-varlen-batch"));
    let mut l = SchedLevers::defaults();
    l.prefill_varlen = true;
    assert!(
        l.with_prefill_varlen_with_decode(true)
            .expect("accepted with varlen")
            .prefill_varlen_with_decode
    );
    assert!(
        !SchedLevers::defaults()
            .with_prefill_varlen_with_decode(false)
            .expect("off is always accepted")
            .prefill_varlen_with_decode
    );
}
