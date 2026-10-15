// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: measured requests to per-cell statistics, and two runs to a
//! per-cell comparison (pure).
//!
//! A request is measured when its `max_tokens` is the config's and its prompt
//! token count is a cell's. A measured request that could not be attributed
//! (concurrent, unfinished) or has no first-token line is an error, never a
//! silent drop. The statistic is the true median with n; p90 is reported only
//! at n >= 10 (linear interpolation, labelled as such by its key).
//!
//! The comparison's verdict is range separation over the reps: `win` when
//! every B rep is below every A rep, `loss` when every B rep is above every A
//! rep, `overlap` otherwise, and `too-few` when either side has fewer reps
//! than the caller's minimum. It says nothing about whether a lever was armed
//! or engaged (that is `metprof verdict`).

use serde_json::{Value, json};

use crate::core::error::Result;
use crate::core::stats::{median_f64, quantile_sorted_f64};
use crate::core::ttft::TtftConfig;
use crate::core::ttft::log::{ServeLog, ServerRequest, Unattributed};
use crate::fail;

/// Below this many values a p90 is not reported (METHODOLOGY 3.3).
pub const P90_MIN_N: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Cold,
    Warm,
}

impl Mode {
    pub fn name(self) -> &'static str {
        match self {
            Mode::Cold => "cold",
            Mode::Warm => "warm",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Class {
    Step,
    Other,
}

impl Class {
    pub fn name(self) -> &'static str {
        match self {
            Class::Step => "step",
            Class::Other => "other",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Measured<'a> {
    pub cell: String,
    pub mode: Mode,
    pub class: Class,
    pub first_token: u64,
    pub req: &'a ServerRequest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Selection<'a> {
    pub measured: Vec<Measured<'a>>,
    /// Requests with the measured max_tokens but no cell's prompt count.
    pub unassigned: usize,
    /// Requests with another max_tokens.
    pub other: usize,
}

fn refuse_unattributed(what: &str, list: &[Unattributed], cfg: &TtftConfig) -> Result<()> {
    for u in list {
        let cell = cfg
            .cells
            .iter()
            .find(|c| c.prompt_tokens == u.prompt_tokens);
        if let (Some(c), true) = (cell, u.max_tokens == cfg.max_tokens) {
            fail!(
                "the request of line {} ({} prompt tokens, cell {}) is {what}: its Done line cannot be attributed, so the run is not measurable",
                u.line,
                u.prompt_tokens,
                c.name
            );
        }
    }
    Ok(())
}

pub fn select<'a>(log: &'a ServeLog, cfg: &TtftConfig) -> Result<Selection<'a>> {
    refuse_unattributed("concurrent with another request", &log.concurrent, cfg)?;
    refuse_unattributed("unfinished at the end of the log", &log.unfinished, cfg)?;
    let mut s = Selection {
        measured: Vec::new(),
        unassigned: 0,
        other: 0,
    };
    for r in &log.requests {
        if r.max_tokens != cfg.max_tokens {
            s.other += 1;
            continue;
        }
        let Some(cell) = cfg
            .cells
            .iter()
            .find(|c| c.prompt_tokens == r.prompt_tokens)
        else {
            s.unassigned += 1;
            continue;
        };
        let Some(first_token) = r.first_token else {
            fail!(
                "the request of line {} (cell {}) has no `Prefill first token` line",
                r.line,
                cell.name
            );
        };
        s.measured.push(Measured {
            cell: cell.name.clone(),
            mode: if r.hit.is_some() {
                Mode::Warm
            } else {
                Mode::Cold
            },
            class: if first_token == cfg.step_token_id {
                Class::Step
            } else {
                Class::Other
            },
            first_token,
            req: r,
        });
    }
    Ok(s)
}

/// n, true median, min, max and (n >= 10) p90 of `values`, as JSON.
pub fn summary(values: &[f64]) -> Value {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let p90 = (v.len() >= P90_MIN_N)
        .then(|| quantile_sorted_f64(&v, 0.9))
        .flatten();
    json!({
        "n": v.len(),
        "median": median_f64(&v),
        "min": v.first(),
        "max": v.last(),
        "p90_interpolated": p90,
    })
}

/// One row per (cell in config order, mode): the summary of the chosen
/// times, plus the same per first-token class.
pub fn cell_rows<F>(sel: &Selection, cfg: &TtftConfig, time: F) -> Vec<Value>
where
    F: Fn(&Measured) -> Option<f64>,
{
    let mut rows = Vec::new();
    for cell in &cfg.cells {
        for mode in [Mode::Cold, Mode::Warm] {
            let of = |class: Option<Class>| -> Vec<f64> {
                sel.measured
                    .iter()
                    .filter(|m| m.cell == cell.name && m.mode == mode)
                    .filter(|m| class.is_none_or(|c| m.class == c))
                    .filter_map(&time)
                    .collect()
            };
            let all = of(None);
            if all.is_empty() {
                continue;
            }
            let mut row = summary(&all);
            row["cell"] = json!(cell.name);
            row["mode"] = json!(mode.name());
            for class in [Class::Step, Class::Other] {
                let c = of(Some(class));
                row[class.name()] = json!({"n": c.len(), "median": median_f64(&c)});
            }
            rows.push(row);
        }
    }
    rows
}

fn get<'v>(rows: &'v [Value], cell: &str, mode: &str) -> Option<&'v Value> {
    rows.iter().find(|r| r["cell"] == cell && r["mode"] == mode)
}

fn f(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

/// Compare two runs' `cell_rows` (lower is better). `min_reps` gates the
/// verdict; every (cell, mode) present in both runs gets a row.
pub fn compare(a: &[Value], b: &[Value], min_reps: usize) -> Result<Vec<Value>> {
    let mut out = Vec::new();
    for ra in a {
        let (Some(cell), Some(mode)) = (ra["cell"].as_str(), ra["mode"].as_str()) else {
            fail!("a cell row without cell and mode: {ra}");
        };
        let Some(rb) = get(b, cell, mode) else {
            continue;
        };
        let n = |r: &Value| r["n"].as_u64().unwrap_or(0) as usize;
        let num = |r: &Value, k: &str| match f(r, k) {
            Some(x) => Ok(x),
            None => fail!("cell {cell} {mode}: no {k}"),
        };
        let (amin, amax, bmin, bmax) = (
            num(ra, "min")?,
            num(ra, "max")?,
            num(rb, "min")?,
            num(rb, "max")?,
        );
        let verdict = if n(ra) < min_reps || n(rb) < min_reps {
            "too-few"
        } else if bmax < amin {
            "win"
        } else if bmin > amax {
            "loss"
        } else {
            "overlap"
        };
        out.push(json!({
            "cell": cell,
            "mode": mode,
            "n_a": n(ra),
            "n_b": n(rb),
            "median_a": num(ra, "median")?,
            "median_b": num(rb, "median")?,
            "delta": num(rb, "median")? - num(ra, "median")?,
            "range_a": [amin, amax],
            "range_b": [bmin, bmax],
            "verdict": verdict,
        }));
    }
    if out.is_empty() {
        fail!("the two runs share no (cell, mode)");
    }
    Ok(out)
}
