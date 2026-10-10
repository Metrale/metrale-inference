// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The text of `met ml-utils model-bring-up-bench`: one run's table, and two runs
//! side by side with the winner per metric and every asymmetry between them disclosed.
//!
//! Owner: server CLI (`met ml-utils`).
//! Invariants:
//! - Pure: records in, text out.
//! - Lower TTFT wins; a metric missing on either side has no winner, and equal values tie.
//! - `--compare` lists every difference in what the two runs measured (model, tokenizer,
//!   server prompt count, reps, cache verdict) before the table, so no winner is read without it.

use std::fmt::Write as _;

use anyhow::{Result, bail};

use super::bring_up_conc::{Rung, Section};
use super::bring_up_core::{Bench, CacheVerdict, Record, Row};

fn ms(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |v| format!("{v:.1}"))
}

fn tokens(v: Option<f64>) -> String {
    v.map_or_else(|| "-".to_string(), |v| format!("{v:.0}"))
}

fn cache(r: &Row) -> String {
    let verdict = match r.cache.verdict {
        CacheVerdict::Verified => "verified",
        CacheVerdict::Violated => "VIOLATED",
        CacheVerdict::NotExposed => "not exposed",
    };
    match r.cache.hit_tokens {
        Some(h) => format!("{verdict} ({h:.0} hit tok)"),
        None => verdict.to_string(),
    }
}

fn bench_name(r: &Row) -> String {
    format!("{} @{}", r.bench.label(), r.tokens)
}

/// 2026-10-10: One run's table: bench | p50 | p90 | min | reps | prompt tok | cache | engine |
/// model, then any row that did not complete.
pub(crate) fn table(rec: &Record) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:<22} {:>9} {:>9} {:>9} {:>5} {:>10}  {:<28} {:<10} model",
        "bench (target tok)",
        "p50 ms",
        "p90 ms",
        "min ms",
        "reps",
        "prompt tok",
        "prefix cache",
        "engine"
    );
    for r in &rec.ttft {
        let _ = writeln!(
            out,
            "{:<22} {:>9} {:>9} {:>9} {:>5} {:>10}  {:<28} {:<10} {}",
            bench_name(r),
            ms(r.p50_ms),
            ms(r.p90_ms),
            ms(r.min_ms),
            r.samples,
            tokens(r.prompt_tokens),
            cache(r),
            rec.engine.label,
            rec.model
        );
    }
    for r in rec.ttft.iter().filter(|r| r.status != "completed") {
        let _ = writeln!(out, "  {}: {}", bench_name(r), r.status);
    }
    if let Some(c) = &rec.concurrency {
        out.push_str(&conc_table(c, &rec.engine.label));
    }
    out
}

fn joules(r: &Rung, s: &Section) -> String {
    match r.j_per_tok {
        Some(j) => format!("{j:.3}"),
        None if s.energy_hosts.is_empty() => "not measured".to_string(),
        None => "no reading".to_string(),
    }
}

fn instrument(s: &Section) -> String {
    let i = &s.instrument;
    format!(
        "concurrency ladder: isl {} osl {} prompt_mode {} warmup {} rungs {:?}; energy {}",
        i.isl,
        i.osl,
        i.prompt_mode,
        i.warmup,
        i.concs,
        if s.energy_hosts.is_empty() {
            "not measured".to_string()
        } else {
            format!("GPU rail, NVML counters on {}", s.energy_hosts.join(", "))
        }
    )
}

/// 2026-10-10: The ladder: C | tok/s | TTFT p50 | TPOT p50 | J/tok.
fn conc_table(s: &Section, engine: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "{} [{engine}]", instrument(s));
    let _ = writeln!(
        out,
        "{:>5} {:>10} {:>12} {:>12} {:>13}",
        "C", "tok/s", "TTFT p50 ms", "TPOT p50 ms", "J/tok"
    );
    for r in &s.rungs {
        if !r.comparable {
            let _ = writeln!(
                out,
                "{:>5}  not comparable (vacuous, errored or cache-uncontrolled; see the run record)",
                r.conc
            );
            continue;
        }
        let _ = writeln!(
            out,
            "{:>5} {:>10} {:>12} {:>12} {:>13}",
            r.conc,
            r.tok_s
                .map_or_else(|| "-".to_string(), |v| format!("{v:.2}")),
            ms(r.ttft_p50_ms),
            ms(r.tpot_p50_ms),
            joules(r, s)
        );
    }
    if s.status != "completed" {
        let _ = writeln!(out, "  concurrency: {}", s.status);
    }
    out
}

/// 2026-10-10: Which side wins a metric; `higher` when larger is better (tok/s).
fn winner(a: Option<f64>, b: Option<f64>, la: &str, lb: &str, higher: bool) -> String {
    match (a, b) {
        (Some(x), Some(y)) if x == y => "tie".to_string(),
        (Some(x), Some(y)) => {
            let a_wins = if higher { x > y } else { x < y };
            let ratio = if x > y { x / y } else { y / x };
            format!("{} {ratio:.2}x", if a_wins { la } else { lb })
        }
        _ => "-".to_string(),
    }
}

fn key(r: &Row) -> (Bench, usize) {
    (r.bench, r.tokens)
}

/// 2026-10-10: Every way the two runs did not measure the same thing.
fn asymmetries(a: &Record, b: &Record) -> Vec<String> {
    let mut out = Vec::new();
    if a.model != b.model {
        out.push(format!("served model name: {} vs {}", a.model, b.model));
    }
    if a.tokenizer_sha256 != b.tokenizer_sha256 {
        let show = |r: &Record| {
            format!(
                "{} ({})",
                r.tokenizer.as_deref().unwrap_or("none"),
                r.tokenizer_sha256
                    .as_deref()
                    .map_or("-", |s| &s[..s.len().min(12)])
            )
        };
        out.push(format!("tokenizer differs: {} vs {}", show(a), show(b)));
    }
    for ra in &a.ttft {
        let Some(rb) = b.ttft.iter().find(|r| key(r) == key(ra)) else {
            out.push(format!("{} ran only on {}", bench_name(ra), a.engine.label));
            continue;
        };
        if ra.prompt_tokens.is_some()
            && rb.prompt_tokens.is_some()
            && ra.prompt_tokens != rb.prompt_tokens
        {
            out.push(format!(
                "{}: server prompt tokens {} vs {} (the chat templates render the same message \
                 differently)",
                bench_name(ra),
                tokens(ra.prompt_tokens),
                tokens(rb.prompt_tokens)
            ));
        }
        if ra.samples != rb.samples {
            out.push(format!(
                "{}: samples {} vs {}",
                bench_name(ra),
                ra.samples,
                rb.samples
            ));
        }
        for (rec, r) in [(a, ra), (b, rb)] {
            if r.cache.verdict != CacheVerdict::Verified {
                out.push(format!(
                    "{} on {}: prefix cache {}",
                    bench_name(r),
                    rec.engine.label,
                    cache(r)
                ));
            }
            if r.status != "completed" {
                out.push(format!(
                    "{} on {}: {}",
                    bench_name(r),
                    rec.engine.label,
                    r.status
                ));
            }
        }
    }
    for rb in &b.ttft {
        if !a.ttft.iter().any(|r| key(r) == key(rb)) {
            out.push(format!("{} ran only on {}", bench_name(rb), b.engine.label));
        }
    }
    match (&a.concurrency, &b.concurrency) {
        (Some(x), Some(y)) => {
            if x.instrument != y.instrument {
                out.push(format!(
                    "concurrency instrument: {} vs {}",
                    instrument(x),
                    instrument(y)
                ));
            } else if x.energy_hosts != y.energy_hosts {
                out.push(format!(
                    "energy hosts: {:?} vs {:?}",
                    x.energy_hosts, y.energy_hosts
                ));
            }
            for (rec, s) in [(a, x), (b, y)] {
                for r in s.rungs.iter().filter(|r| !r.comparable) {
                    out.push(format!(
                        "C={} on {}: not comparable",
                        r.conc, rec.engine.label
                    ));
                }
                if s.status != "completed" {
                    out.push(format!("concurrency on {}: {}", rec.engine.label, s.status));
                }
            }
        }
        (Some(_), None) => out.push(format!(
            "the concurrency ladder ran only on {}",
            a.engine.label
        )),
        (None, Some(_)) => out.push(format!(
            "the concurrency ladder ran only on {}",
            b.engine.label
        )),
        (None, None) => {}
    }
    out
}

/// 2026-10-10: Two records side by side: p50, p90 and min per bench with the winner of each.
pub(crate) fn compare(a: &Record, b: &Record) -> Result<String> {
    let (la, lb) = (a.engine.label.as_str(), b.engine.label.as_str());
    if la == lb {
        bail!(
            "both records are labelled {la:?}; re-run one with --label so the table can name them"
        );
    }
    let mut out = String::new();
    let _ = writeln!(
        out,
        "A = {la}: {} at {} ({})",
        a.model,
        a.url,
        a.engine.version.as_deref().unwrap_or("-")
    );
    let _ = writeln!(
        out,
        "B = {lb}: {} at {} ({})",
        b.model,
        b.url,
        b.engine.version.as_deref().unwrap_or("-")
    );
    let disclosed = asymmetries(a, b);
    let _ = writeln!(
        out,
        "asymmetries: {}",
        if disclosed.is_empty() { "none" } else { "" }
    );
    for d in &disclosed {
        let _ = writeln!(out, "  - {d}");
    }
    let _ = writeln!(
        out,
        "{:<22} {:>6} {:>9} {:>9} {:>14}",
        "bench (target tok)", "metric", la, lb, "winner"
    );
    for ra in &a.ttft {
        let Some(rb) = b.ttft.iter().find(|r| key(r) == key(ra)) else {
            continue;
        };
        for (metric, x, y) in [
            ("p50", ra.p50_ms, rb.p50_ms),
            ("p90", ra.p90_ms, rb.p90_ms),
            ("min", ra.min_ms, rb.min_ms),
        ] {
            let _ = writeln!(
                out,
                "{:<22} {:>6} {:>9} {:>9} {:>14}",
                bench_name(ra),
                metric,
                ms(x),
                ms(y),
                winner(x, y, la, lb, false)
            );
        }
    }
    if let (Some(x), Some(y)) = (&a.concurrency, &b.concurrency) {
        out.push_str(&conc_compare(x, y, la, lb));
    }
    Ok(out)
}

fn num(v: Option<f64>, digits: usize) -> String {
    v.map_or_else(|| "-".to_string(), |v| format!("{v:.digits$}"))
}

/// 2026-10-10: The two ladders rung by rung: tok/s (higher wins), TTFT p50, TPOT p50 and
/// J/tok (lower wins).
fn conc_compare(x: &Section, y: &Section, la: &str, lb: &str) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{:>5} {:>9} {:>11} {:>11} {:>14}",
        "C", "metric", la, lb, "winner"
    );
    for ra in &x.rungs {
        let Some(rb) = y.rungs.iter().find(|r| r.conc == ra.conc) else {
            continue;
        };
        for (metric, p, q, digits, higher) in [
            ("tok/s", ra.tok_s, rb.tok_s, 2, true),
            ("TTFT p50", ra.ttft_p50_ms, rb.ttft_p50_ms, 1, false),
            ("TPOT p50", ra.tpot_p50_ms, rb.tpot_p50_ms, 2, false),
            ("J/tok", ra.j_per_tok, rb.j_per_tok, 3, false),
        ] {
            let _ = writeln!(
                out,
                "{:>5} {:>9} {:>11} {:>11} {:>14}",
                ra.conc,
                metric,
                num(p, digits),
                num(q, digits),
                winner(p, q, la, lb, higher)
            );
        }
    }
    out
}

#[cfg(test)]
#[path = "bring_up_render_tests.rs"]
mod tests;
