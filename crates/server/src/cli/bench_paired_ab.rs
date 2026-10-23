// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `met benchmark paired-ab`: the verdict of a paired in-serve A/B run
//! (`METRALE_PAIRED_AB`) from the serve log's `PAIRED_AB` lines (`metrale_bench::paired_ab`).
//! GPU-free.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: the arm count is the instrument's own (`PAIRED_AB on: ... N arms`); a log with
//! no such line, or two different ones, is refused.

use anyhow::{Context, Result, bail};
use metrale_bench::paired_ab::{Plan, judge, parse};

use super::bench_args::PairedAbArgs;

/// 2026-10-23: Bootstrap replicates and seed (stated in the output).
const REPS: usize = 2000;
const SEED: u64 = 0x7061_6972;

fn arms_of(text: &str) -> Result<usize> {
    let mut arms: Vec<usize> = text
        .lines()
        .filter_map(|l| {
            let rest = &l[l.find("PAIRED_AB on:")?..];
            let at = rest.find(" arms")?;
            rest[..at].rsplit(' ').next()?.parse().ok()
        })
        .collect();
    arms.dedup();
    match arms.as_slice() {
        [n] => Ok(*n),
        [] => bail!("no `PAIRED_AB on:` line: the serve did not run the instrument"),
        _ => bail!("the log holds runs of different arm counts {arms:?}"),
    }
}

pub fn paired_ab_cmd(args: PairedAbArgs) -> Result<i32> {
    let text = std::fs::read_to_string(&args.log)
        .with_context(|| format!("reading {}", args.log.display()))?;
    let plan = Plan {
        arms: arms_of(&text)?,
        warm: args.warm_cycles,
        margin: args.margin_pct / 100.0,
        huber_k: args.huber_k,
        reps: REPS,
        seed: SEED,
    };
    let steps = parse(&text)?;
    let v = judge(&steps, plan)?;
    println!(
        "paired A/B: {} ({} steps, {} arms; warm {} cycles dropped per request; BCa stationary \
         block bootstrap {REPS} reps seed {SEED:#x} of the Huber location at k {}; TOST at +-{}% of \
         the A step; Holm over {} tests)",
        args.log.display(),
        steps.len(),
        plan.arms,
        plan.warm,
        plan.huber_k,
        args.margin_pct,
        v.len()
    );
    println!(
        "kv0 | arm | cycles | A step us | A-X us Huber (95% BCa) | % of step | mean A-X us | ESS | TOST p | equiv | X faster p | Holm"
    );
    for x in &v {
        println!(
            "{} | {} | {} | {:.1} | {:+.2} [{:+.2}, {:+.2}] | {:+.3}% [{:+.3}, {:+.3}] | {:+.2} | {:.0} | {:.4} | {} | {:.4} | {}",
            x.kv0,
            x.arm,
            x.cycles,
            x.a_mean_us,
            x.d_us,
            x.d_lo,
            x.d_hi,
            x.pct(x.d_us),
            x.pct(x.d_lo),
            x.pct(x.d_hi),
            x.d_mean_us,
            x.ess,
            x.tost_p,
            if x.equivalent { "yes" } else { "no" },
            x.superior_p,
            if x.superior_holm { "WIN" } else { "-" }
        );
    }
    // 2026-10-23: A replaying serve (`METRALE_PAIRED_AB_TRAJ`): each arm's own acceptance and
    // row-0 agreement with the record, over steps whose previous step ran the same arm.
    let traj = metrale_bench::forced_ab::parse_traj(&text)?;
    if !traj.is_empty() {
        let timed = metrale_bench::forced_ab::join(&steps, &traj);
        println!(
            "forced trajectory: {} replayed steps, {} timed",
            traj.len(),
            timed.len()
        );
        println!("arm | steps (prev same arm) | own acc | agree0 with record");
        for (arm, (n, acc, agree)) in metrale_bench::forced_ab::acceptance_by_arm(&timed, plan.arms)
            .iter()
            .enumerate()
        {
            let pc = |x: usize| 100.0 * x as f64 / (*n).max(1) as f64;
            println!(
                "{arm} | {n} | {acc} ({:.2}%) | {agree} ({:.2}%)",
                pc(*acc),
                pc(*agree)
            );
        }
    }
    if v.is_empty() {
        bail!("no request had 16 complete cycles after the warm-up");
    }
    Ok(0)
}
