// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The router ids the P-WD go/no-go (`pwd.rs`) feeds the union sweeps, host side:
//! `rows` rows of `TOP_K` distinct expert ids each, from one of four models.
//!
//! - `uniform`: every row picks 8 of the 288 experts uniformly (the random router).
//! - `zipf`: rows independent, expert popularity `1 / rank^s` over a seeded permutation; `s`
//!   fitted so the mean union of 16 rows is the serve's measured U(16).
//! - `paired`: an MTP verify batch holds two consecutive positions of each sequence. Row `2i`
//!   is uniform; row `2i + 1` keeps each of row `2i`'s experts with probability `rho` and fills
//!   the rest uniformly. `rho` fitted to the same U(16).
//! - `file`: real ids, one row per line, `TOP_K` comma-separated ids, taken in consecutive
//!   chunks of `rows` lines (wrapping at the end of the file).
//!
//! The two fitted models agree at 16 rows by construction and extrapolate differently to 24
//! and 32; the bench reports both.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: every row [`check_rows`] accepts: `TOP_K` distinct ids in `0..EXPERTS`.

use anyhow::{Context, Result, ensure};

use super::device::{EXPERTS, Rng, TOP_K};

/// 2026-10-15: Samples the bisection averages per probe, on one seed per probe (common random
/// numbers keep the mean monotone in the parameter).
const FIT_SAMPLES: usize = 400;
const FIT_STEPS: usize = 40;
const FIT_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

pub enum Router {
    Uniform,
    Zipf {
        s: f64,
        cdf: Vec<f64>,
    },
    Paired {
        rho: f64,
    },
    File {
        rows: Vec<[i32; TOP_K]>,
        next: usize,
    },
}

impl Router {
    pub fn name(&self) -> String {
        match self {
            Router::Uniform => "uniform".into(),
            Router::Zipf { s, .. } => format!("zipf(s={s:.3})"),
            Router::Paired { rho } => format!("paired(rho={rho:.3})"),
            Router::File { rows, .. } => format!("file({} rows)", rows.len()),
        }
    }

    /// 2026-10-15: One routing of `rows` rows.
    pub fn draw(&mut self, rng: &mut Rng, rows: usize) -> Result<Vec<i32>> {
        let ids = match self {
            Router::Uniform => (0..rows).flat_map(|_| uniform_row(rng, &[])).collect(),
            Router::Zipf { cdf, .. } => (0..rows).flat_map(|_| zipf_row(rng, cdf)).collect(),
            Router::Paired { rho } => paired(rng, rows, *rho),
            Router::File { rows: lines, next } => {
                let v: Vec<i32> = (0..rows)
                    .flat_map(|i| lines[(*next + i) % lines.len()])
                    .collect();
                *next = (*next + rows) % lines.len();
                v
            }
        };
        check_rows(&ids)?;
        Ok(ids)
    }
}

/// 2026-10-15: Fails on a row that is not `TOP_K` distinct ids in `0..EXPERTS`.
pub fn check_rows(ids: &[i32]) -> Result<()> {
    ensure!(
        ids.len().is_multiple_of(TOP_K),
        "{} ids are not whole rows",
        ids.len()
    );
    for (r, row) in ids.chunks(TOP_K).enumerate() {
        for (i, &e) in row.iter().enumerate() {
            ensure!(
                (0..EXPERTS as i32).contains(&e) && !row[..i].contains(&e),
                "row {r}: {row:?} is not {TOP_K} distinct ids in 0..{EXPERTS}"
            );
        }
    }
    Ok(())
}

/// 2026-10-15: Distinct experts of a routing: the union the sweep streams.
pub fn union_size(ids: &[i32]) -> usize {
    let mut v = ids.to_vec();
    v.sort_unstable();
    v.dedup();
    v.len()
}

/// 2026-10-15: The expected union of `rows` uniform rows, `E (1 - (1 - k / E)^rows)`.
/// 2026-10-16: `SUMMARY` prints it as the no-locality reference the fitted routes sit under.
pub fn uniform_union(rows: usize) -> f64 {
    let e = EXPERTS as f64;
    e * (1.0 - (1.0 - TOP_K as f64 / e).powi(rows as i32))
}

/// 2026-10-15: A uniform value in `[0, 1)`, 53 random bits.
fn unit(rng: &mut Rng) -> f64 {
    (rng.next() >> 11) as f64 / (1u64 << 53) as f64
}

/// 2026-10-15: One row: `keep`, then uniform distinct ids up to `TOP_K`.
fn uniform_row(rng: &mut Rng, keep: &[i32]) -> Vec<i32> {
    let mut row = keep.to_vec();
    while row.len() < TOP_K {
        let e = (rng.next() % EXPERTS as u64) as i32;
        if !row.contains(&e) {
            row.push(e);
        }
    }
    row
}

fn zipf_row(rng: &mut Rng, cdf: &[f64]) -> Vec<i32> {
    let mut row = Vec::with_capacity(TOP_K);
    while row.len() < TOP_K {
        let u = unit(rng) * cdf[EXPERTS - 1];
        let e = cdf.partition_point(|&c| c <= u).min(EXPERTS - 1) as i32;
        if !row.contains(&e) {
            row.push(e);
        }
    }
    row
}

fn paired(rng: &mut Rng, rows: usize, rho: f64) -> Vec<i32> {
    let mut ids = Vec::with_capacity(rows * TOP_K);
    for r in 0..rows {
        let row = if r % 2 == 0 {
            uniform_row(rng, &[])
        } else {
            let prev = &ids[(r - 1) * TOP_K..r * TOP_K];
            let keep: Vec<i32> = prev.iter().copied().filter(|_| unit(rng) < rho).collect();
            uniform_row(rng, &keep)
        };
        ids.extend(row);
    }
    ids
}

/// 2026-10-15: The Zipf cumulative weights at exponent `s` over a permutation seeded by `seed`.
fn zipf_cdf(s: f64, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed);
    let mut perm: Vec<usize> = (0..EXPERTS).collect();
    for i in (1..EXPERTS).rev() {
        perm.swap(i, (rng.next() % (i as u64 + 1)) as usize);
    }
    let mut w = vec![0.0; EXPERTS];
    for (rank, &e) in perm.iter().enumerate() {
        w[e] = 1.0 / ((rank + 1) as f64).powf(s);
    }
    w.iter()
        .scan(0.0, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect()
}

/// 2026-10-15: The mean union of `rows` rows of `router` over [`FIT_SAMPLES`] draws on one seed.
pub fn mean_union(router: &mut Router, rows: usize) -> Result<f64> {
    let mut rng = Rng(FIT_SEED);
    let mut sum = 0usize;
    for _ in 0..FIT_SAMPLES {
        sum += union_size(&router.draw(&mut rng, rows)?);
    }
    Ok(sum as f64 / FIT_SAMPLES as f64)
}

/// 2026-10-15: Bisects a parameter in `[lo, hi]` (the union shrinks as it grows) until the mean
/// union of 16 rows is `target`; fails when `target` lies outside the model's range.
fn fit(target: f64, (lo, hi): (f64, f64), make: &dyn Fn(f64) -> Router) -> Result<Router> {
    let at = |p: f64| mean_union(&mut make(p), 16);
    let (u_lo, u_hi) = (at(lo)?, at(hi)?);
    ensure!(
        u_hi <= target && target <= u_lo,
        "U(16) target {target} is outside [{u_hi:.1}, {u_lo:.1}] of this model"
    );
    let (mut a, mut b) = (lo, hi);
    for _ in 0..FIT_STEPS {
        let m = 0.5 * (a + b);
        if at(m)? > target { a = m } else { b = m }
    }
    Ok(make(0.5 * (a + b)))
}

pub fn fit_zipf(target_u16: f64, seed: u64) -> Result<Router> {
    fit(target_u16, (0.0, 3.0), &|s| Router::Zipf {
        s,
        cdf: zipf_cdf(s, seed),
    })
}

pub fn fit_paired(target_u16: f64) -> Result<Router> {
    fit(target_u16, (0.0, 1.0), &|rho| Router::Paired { rho })
}

/// 2026-10-15: `path`: one row per line, `TOP_K` comma-separated ids; blank lines skipped.
pub fn from_file(path: &str) -> Result<Router> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {path}"))?;
    let mut rows = Vec::new();
    for (n, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let v: Vec<i32> = line
            .split(',')
            .map(|t| t.trim().parse::<i32>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("{path}:{}: not comma-separated ids", n + 1))?;
        check_rows(&v).with_context(|| format!("{path}:{}", n + 1))?;
        ensure!(
            v.len() == TOP_K,
            "{path}:{}: {} ids, want {TOP_K}",
            n + 1,
            v.len()
        );
        rows.push(v.try_into().expect("length checked"));
    }
    ensure!(!rows.is_empty(), "{path}: no rows");
    Ok(Router::File { rows, next: 0 })
}
