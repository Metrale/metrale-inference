// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A serve-level A/B verdict from the command line: TOST equivalence and one-sided
//! superiority of arm B against arm A at a relative margin, on one metric (higher is better,
//! e.g. tok/s), with a serve as the unit (each line one serve's value).
//!
//!   cargo run -p metrale-stats --example abba_tost -- <margin_pct> < rows.tsv
//!
//! rows.tsv: `<arm A|B> <value>` per line, any order (the ABBA order is the caller's design).
//! Prints n, means, B-A with its 95% Welch interval (absolute and relative to A), the TOST p at
//! +-margin and the superiority p at 0. Owner: metrale-stats examples.
use std::io::Read;

use metrale_stats::Summary;
use metrale_stats::compare::{superiority, tost, welch};
use metrale_stats::interval::CiMethod;

fn summary(xs: &[f64]) -> Summary {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / (n - 1.0);
    Summary {
        mean,
        se: (var / n).sqrt(),
        df: n - 1.0,
    }
}

fn main() {
    let margin_pct: f64 = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .expect("usage: abba_tost <margin_pct> < rows.tsv");
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).expect("stdin");
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for l in text.lines().filter(|l| !l.trim().is_empty()) {
        let mut f = l.split_whitespace();
        let (arm, v) = (f.next().expect("arm"), f.next().expect("value"));
        let v: f64 = v.parse().expect("value");
        match arm {
            "A" => a.push(v),
            "B" => b.push(v),
            other => panic!("arm `{other}` is not A or B"),
        }
    }
    assert!(a.len() >= 2 && b.len() >= 2, "need >= 2 serves per arm");
    let (sa, sb) = (summary(&a), summary(&b));
    let d = welch(sb, sa);
    let ci = d.interval(0.95, CiMethod::T);
    let margin = sa.mean * margin_pct / 100.0;
    let t = tost(d, margin, 0.05);
    let s = superiority(d, 0.0, 0.05);
    let pct = |x: f64| 100.0 * x / sa.mean;
    println!(
        "nA={} nB={} A={:.3} B={:.3} B-A={:+.3} [{:+.3}, {:+.3}] = {:+.2}% [{:+.2}%, {:+.2}%] df={:.1} | TOST +-{margin_pct}%: p={:.4} {} | B>A: p={:.4} {}",
        a.len(),
        b.len(),
        sa.mean,
        sb.mean,
        d.mean,
        ci.lo,
        ci.hi,
        pct(d.mean),
        pct(ci.lo),
        pct(ci.hi),
        d.df,
        t.p,
        if t.equivalent {
            "EQUIVALENT"
        } else {
            "not shown"
        },
        s.p,
        if s.reject { "SUPERIOR" } else { "not shown" }
    );
}
