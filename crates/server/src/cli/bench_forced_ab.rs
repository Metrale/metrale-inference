// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: `met benchmark forced-ab`: the verdict of serves that replayed one recorded token
//! trajectory (`METRALE_PAIRED_AB_TRAJ`, `metrale_bench::forced_ab`). GPU-free.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants: as many lever logs as control logs (pair `i` = control `i`, lever `i`); a log
//! without replay lines is refused.

use anyhow::{Context, Result, bail};
use metrale_bench::forced_ab::{ForcedPlan, Timed, join, judge_forced, parse_traj};
use metrale_bench::paired_ab::parse;

use super::bench_args::ForcedAbArgs;

const REPS: usize = 2000;
const SEED: u64 = 0x666f_7263;

fn load(p: &std::path::Path) -> Result<Vec<Timed>> {
    let text = std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
    let traj = parse_traj(&text)?;
    if traj.is_empty() {
        bail!(
            "{}: no `PAIRED_TRAJ replay` lines: not a replaying serve",
            p.display()
        );
    }
    Ok(join(&parse(&text)?, &traj))
}

pub fn forced_ab_cmd(args: ForcedAbArgs) -> Result<i32> {
    if args.control.len() != args.lever.len() || args.control.is_empty() {
        bail!(
            "{} control and {} lever logs: pairs need one of each",
            args.control.len(),
            args.lever.len()
        );
    }
    let pairs: Vec<(Vec<Timed>, Vec<Timed>)> = args
        .control
        .iter()
        .zip(&args.lever)
        .map(|(c, l)| Ok((load(c)?, load(l)?)))
        .collect::<Result<_>>()?;
    let plan = ForcedPlan {
        lever_arm: args.lever_arm,
        warm: args.warm_steps,
        margin: args.margin_pct / 100.0,
        huber_k: args.huber_k,
        reps: REPS,
        seed: SEED,
    };
    let v = judge_forced(&pairs, plan)?;
    println!(
        "forced A/B: {} pairs; lever arm {:?}; warm {} steps; per pair Huber k {} with BCa \
         stationary bootstrap {REPS} reps seed {SEED:#x}; across pairs t interval (pair = unit), \
         TOST +-{}% of the control step, Holm over {} requests",
        pairs.len(),
        args.lever_arm,
        args.warm_steps,
        args.huber_k,
        args.margin_pct,
        v.len()
    );
    println!(
        "kv0 | control us | C-L us (95% t, pairs) | % | per pair: n, d [BCa] | TOST p | L faster p | Holm | own acc C / L / steps | discordant C-only/L-only | McNemar p"
    );
    for r in &v {
        let pct = |x: f64| 100.0 * x / r.control_us;
        let ci = r.ci.map_or("n=1, no CI".into(), |(lo, hi)| {
            format!(
                "[{lo:+.1}, {hi:+.1}] = [{:+.3}%, {:+.3}%]",
                pct(lo),
                pct(hi)
            )
        });
        let pp: Vec<String> = r
            .pairs
            .iter()
            .map(|p| format!("{} {:+.1} [{:+.1},{:+.1}]", p.0, p.2, p.3, p.4))
            .collect();
        println!(
            "{} | {:.1} | {:+.1} {ci} | {:+.3}% | {} | {:.4} | {:.4} | {} | {} / {} / {} ({:.2}% / {:.2}%) | {} / {} | {:.4}",
            r.kv0,
            r.control_us,
            r.d_us,
            pct(r.d_us),
            pp.join("; "),
            r.tost_p,
            r.superior_p,
            if r.superior_holm { "WIN" } else { "-" },
            r.acc.0,
            r.acc.1,
            r.acc.2,
            100.0 * r.acc.0 as f64 / r.acc.2.max(1) as f64,
            100.0 * r.acc.1 as f64 / r.acc.2.max(1) as f64,
            r.discordant.0,
            r.discordant.1,
            r.mcnemar_p
        );
    }
    if v.is_empty() {
        bail!("no request had 32 matched lever steps in any pair");
    }
    Ok(0)
}
