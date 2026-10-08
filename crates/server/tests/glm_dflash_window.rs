// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-08: Window checks for DFlash2 on GLM-5.3 Flash, run against a live serve. Every
//! test is `#[ignore]`d: the model spans three GPUs, so the serve is started by hand in a GPU
//! window and these tests only talk HTTP to it.
//!
//! Owner: server tests (GLM DFlash).
//! Invariants: the prompts, the sampling and the token budget are fixed here, so two runs
//! differ only in the serve.
//!
//! Environment (all required; a missing one fails the test):
//! - `METRALE_GLM_URL`: the serve's base URL, e.g. `http://<host>:<port>`.
//! - `METRALE_GLM_MODEL`: the served model name.
//! - `METRALE_GLM_TRANSCRIPTS`: transcript file (transcript test only).
//! - `METRALE_GLM_TRANSCRIPT_MODE`: `record` (the serve without DFlash) or `compare` (the serve
//!   with it) (transcript test only).
//!
//!   cargo test -p metrale-server --release --test glm_dflash_window -- --ignored --nocapture

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

/// 2026-10-08: Prose, code, a tool-shaped answer, arithmetic, a list and a long-context recall,
/// the mix a drafter's acceptance varies over.
const PROMPTS: [(&str, &str); 8] = [
    (
        "prose",
        "Write a 200-word history of the printing press for a general audience.",
    ),
    (
        "code_heap",
        "Implement a binary min-heap in Python with push, pop and peek, with docstrings.",
    ),
    (
        "code_rust",
        "Write a Rust function that parses an ISO-8601 date (YYYY-MM-DD) without external crates, with unit tests.",
    ),
    (
        "json",
        "Return a JSON object describing three fictional employees with name, role, start_date and skills.",
    ),
    (
        "math",
        "Compute 17 * 243 - 1089 / 9 step by step and give the final number.",
    ),
    (
        "list",
        "List the planets of the solar system in order with one fact each.",
    ),
    (
        "explain",
        "Explain how a hash map handles collisions with open addressing and with chaining.",
    ),
    (
        "recall",
        "Repeat this sentence exactly three times, numbered: The quick brown fox jumps over the lazy dog near the riverbank at dawn.",
    ),
];

/// 2026-10-08: Output budget per prompt: long enough for several verify steps of a block-8
/// drafter on every prompt, short enough for one window.
const MAX_TOKENS: u64 = 256;

/// 2026-10-08: The acceptance floor, drafts accepted over drafts verified. A drafter fed the
/// wrong hidden state accepts close to nothing, and one fed the right state accepts a large
/// share on code and recall; 0.2 separates the two without claiming a target.
const MIN_ACCEPT_RATE: f64 = 0.2;

fn env(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("{name} is not set (see the module doc)"))
}

fn complete(url: &str, model: &str, prompt: &str) -> Result<(String, u64)> {
    let body = json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": MAX_TOKENS,
        "stream": false,
    });
    let mut resp = ureq::post(&format!("{url}/v1/chat/completions"))
        .send_json(&body)
        .with_context(|| format!("POST {url}/v1/chat/completions"))?;
    let v: Value = resp.body_mut().read_json()?;
    let msg = &v["choices"][0]["message"];
    let text = format!(
        "{}\n--reasoning--\n{}",
        msg["content"].as_str().unwrap_or_default(),
        msg["reasoning_content"].as_str().unwrap_or_default()
    );
    let n = v["usage"]["completion_tokens"]
        .as_u64()
        .context("response without usage.completion_tokens")?;
    Ok((text, n))
}

/// 2026-10-08: `(drafts verified, drafts accepted)` summed over
/// `metrale_spec_verify_steps_total{drafts, accepted}` on the serve's `/metrics` page.
fn spec_totals(url: &str) -> Result<(f64, f64)> {
    let page = ureq::get(&format!("{url}/metrics"))
        .call()?
        .body_mut()
        .read_to_string()?;
    let (mut drafts, mut accepted) = (0.0, 0.0);
    for line in page.lines() {
        let Some(rest) = line.strip_prefix("metrale_spec_verify_steps_total{") else {
            continue;
        };
        let (labels, value) = rest.split_once("} ").context("malformed series")?;
        let mut l = BTreeMap::new();
        for kv in labels.split(',') {
            let (k, v) = kv.split_once('=').context("malformed label")?;
            l.insert(k, v.trim_matches('"').parse::<f64>()?);
        }
        let steps: f64 = value.trim().parse()?;
        drafts += l.get("drafts").context("no drafts label")? * steps;
        accepted += l.get("accepted").context("no accepted label")? * steps;
    }
    Ok((drafts, accepted))
}

/// 2026-10-08: The first number to take in a window: drafter acceptance on the fixed prompts.
/// Needs a DFlash serve with `--telemetry basic`.
#[test]
#[ignore = "needs a live GLM-5.3 DFlash serve (GPU window)"]
fn glm_dflash_acceptance_on_fixed_prompts() -> Result<()> {
    let (url, model) = (env("METRALE_GLM_URL")?, env("METRALE_GLM_MODEL")?);
    let before = spec_totals(&url)?;
    for (id, prompt) in PROMPTS {
        let (_, n) = complete(&url, &model, prompt)?;
        println!("{id}: {n} tokens");
    }
    let after = spec_totals(&url)?;
    let (drafts, accepted) = (after.0 - before.0, after.1 - before.1);
    if drafts <= 0.0 {
        bail!(
            "no verify steps recorded: is the serve running DFlash with --telemetry basic? \
             ({drafts} drafts)"
        );
    }
    let rate = accepted / drafts;
    println!("drafts verified {drafts}, accepted {accepted}, rate {rate:.3}");
    assert!(
        rate >= MIN_ACCEPT_RATE,
        "acceptance {rate:.3} below {MIN_ACCEPT_RATE}: check the hidden-state tap"
    );
    Ok(())
}

/// 2026-10-08: DFlash is lossless at temperature 0: run once against the serve without DFlash
/// (`record`), then against the same serve with it (`compare`); every transcript and its token
/// count must match.
#[test]
#[ignore = "needs a live GLM-5.3 serve, without then with DFlash (GPU window)"]
fn glm_dflash_greedy_transcripts_match_plain_decode() -> Result<()> {
    let (url, model) = (env("METRALE_GLM_URL")?, env("METRALE_GLM_MODEL")?);
    let path = env("METRALE_GLM_TRANSCRIPTS")?;
    let mode = env("METRALE_GLM_TRANSCRIPT_MODE")?;
    let mut got = BTreeMap::new();
    for (id, prompt) in PROMPTS {
        let (text, n) = complete(&url, &model, prompt)?;
        got.insert(
            id.to_string(),
            json!({"text": text, "completion_tokens": n}),
        );
    }
    match mode.as_str() {
        "record" => std::fs::write(&path, serde_json::to_string_pretty(&got)?)?,
        "compare" => {
            let want: BTreeMap<String, Value> =
                serde_json::from_str(&std::fs::read_to_string(&path)?)?;
            let differing: Vec<&String> = want
                .keys()
                .filter(|k| got.get(*k) != want.get(*k))
                .collect();
            assert!(
                differing.is_empty() && want.len() == got.len(),
                "transcripts differ with DFlash on: {differing:?}"
            );
        }
        other => bail!("METRALE_GLM_TRANSCRIPT_MODE must be record or compare, not {other}"),
    }
    Ok(())
}
