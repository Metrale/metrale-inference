// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the one TTFT definition (pure), and its config file.
//!
//! - TTFT_server: the rank-0 `Done: ... TTFT=` line, nothing else.
//! - TTFT_client: request write to the first SSE frame carrying content,
//!   reasoning or a tool call (the bench gate's rule).
//! - cold / warm: no / a `Marconi SSM cache hit` line inside the request.
//!   A cache-hit line in neither known format is an error. No latency
//!   threshold and no "first request" rule decides cold.
//! - first-token class: `step` when the first token is the configured id
//!   (a token that streams no delta), `other` otherwise.
//! - statistics: the true median with n; p90 only at n >= 10.
//!
//! ```toml
//! schema = 1
//! [select]
//! max_tokens = 8            # the measured requests' max_tokens
//! [class]
//! step_token_id = 0         # the first token that streams no delta
//! [[cell]]
//! name = "c256"
//! prompt_tokens = 256       # usage.prompt_tokens of the cell's requests
//! [client]                  # read by `ttft client` only
//! model = "..."
//! temperature = 0.0
//! top_p = 1.0
//! seed = 0
//! salt_digits = 16          # decimal digits that replace `{salt}`
//! timeout_s = 900           # per request
//! warmup_requests = 1       # unmeasured requests before the first rep
//! warmup_max_tokens = 32    # their max_tokens; must differ from [select]
//! ```
//!
//! Every key is required (the `[client]` table only by `ttft client`), and an
//! unknown key is an error.

pub mod cells;
pub mod client;
pub mod log;
pub mod sse;

use crate::core::error::Result;
use crate::core::families::check_keys;
use crate::core::repr::{quote, toml_value};
use crate::fail;

/// The placeholder a client prompt file carries exactly once.
pub const SALT_PLACEHOLDER: &str = "{salt}";

#[derive(Debug, Clone, PartialEq)]
pub struct Cell {
    pub name: String,
    pub prompt_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClientConfig {
    pub model: String,
    pub temperature: f64,
    pub top_p: f64,
    pub seed: u64,
    pub salt_digits: usize,
    pub timeout_s: f64,
    pub warmup_requests: u64,
    pub warmup_max_tokens: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtftConfig {
    pub max_tokens: u64,
    pub step_token_id: u64,
    pub cells: Vec<Cell>,
    pub client: Option<ClientConfig>,
}

impl TtftConfig {
    pub fn cell(&self, name: &str) -> Result<&Cell> {
        match self.cells.iter().find(|c| c.name == name) {
            Some(c) => Ok(c),
            None => {
                let names: Vec<&str> = self.cells.iter().map(|c| c.name.as_str()).collect();
                fail!(
                    "no cell {} in the config (cells: {})",
                    quote(name),
                    names.join(", ")
                )
            }
        }
    }

    pub fn client(&self) -> Result<&ClientConfig> {
        match &self.client {
            Some(c) => Ok(c),
            None => fail!("the config has no [client] table, which `ttft client` requires"),
        }
    }
}

fn table<'t>(place: &str, v: Option<&'t toml::Value>) -> Result<&'t toml::Table> {
    match v.and_then(toml::Value::as_table) {
        Some(t) => Ok(t),
        None => fail!("{place}: must be a table"),
    }
}

fn uint(place: &str, t: &toml::Table, key: &str) -> Result<u64> {
    match t[key].as_integer().and_then(|i| u64::try_from(i).ok()) {
        Some(v) => Ok(v),
        None => fail!(
            "{place}.{key}: must be a non-negative integer, got {}",
            toml_value(&t[key])
        ),
    }
}

fn finite(place: &str, t: &toml::Table, key: &str) -> Result<f64> {
    let v = match &t[key] {
        toml::Value::Float(f) => Some(*f),
        toml::Value::Integer(i) => Some(*i as f64),
        _ => None,
    };
    match v.filter(|x| x.is_finite()) {
        Some(x) => Ok(x),
        None => fail!(
            "{place}.{key}: must be a finite number, got {}",
            toml_value(&t[key])
        ),
    }
}

fn client(t: &toml::Table) -> Result<ClientConfig> {
    let p = "[client]";
    check_keys(
        p,
        t,
        &[
            "model",
            "temperature",
            "top_p",
            "seed",
            "salt_digits",
            "timeout_s",
            "warmup_requests",
            "warmup_max_tokens",
        ],
        &[],
    )?;
    let model = match t["model"].as_str().filter(|s| !s.is_empty()) {
        Some(m) => m.to_string(),
        None => fail!("{p}.model: must be a non-empty string"),
    };
    let salt_digits = uint(p, t, "salt_digits")?;
    if !(1..=19).contains(&salt_digits) {
        fail!("{p}.salt_digits: must be 1..=19, got {salt_digits}");
    }
    let timeout_s = finite(p, t, "timeout_s")?;
    if timeout_s <= 0.0 {
        fail!("{p}.timeout_s: must be positive, got {timeout_s}");
    }
    Ok(ClientConfig {
        model,
        temperature: finite(p, t, "temperature")?,
        top_p: finite(p, t, "top_p")?,
        seed: uint(p, t, "seed")?,
        salt_digits: salt_digits as usize,
        timeout_s,
        warmup_requests: uint(p, t, "warmup_requests")?,
        warmup_max_tokens: uint(p, t, "warmup_max_tokens")?,
    })
}

/// Validate a parsed `ttft.toml`.
pub fn ttft_config(t: &toml::Table) -> Result<TtftConfig> {
    check_keys(
        "ttft config",
        t,
        &["schema", "select", "class", "cell"],
        &["client"],
    )?;
    if t["schema"].as_integer() != Some(1) {
        fail!(
            "ttft config: schema must be 1, got {}",
            toml_value(&t["schema"])
        );
    }
    let select = table("[select]", t.get("select"))?;
    check_keys("[select]", select, &["max_tokens"], &[])?;
    let class = table("[class]", t.get("class"))?;
    check_keys("[class]", class, &["step_token_id"], &[])?;
    let Some(list) = t["cell"].as_array().filter(|a| !a.is_empty()) else {
        fail!("[[cell]]: must be a non-empty array of tables");
    };
    let mut cells: Vec<Cell> = Vec::new();
    for (i, v) in list.iter().enumerate() {
        let place = format!("[[cell]] {i}");
        let c = table(&place, Some(v))?;
        check_keys(&place, c, &["name", "prompt_tokens"], &[])?;
        let Some(name) = c["name"].as_str().filter(|s| !s.is_empty()) else {
            fail!("{place}.name: must be a non-empty string");
        };
        let prompt_tokens = uint(&place, c, "prompt_tokens")?;
        if cells.iter().any(|x| x.name == name) {
            fail!("[[cell]]: name {} is declared twice", quote(name));
        }
        if let Some(x) = cells.iter().find(|x| x.prompt_tokens == prompt_tokens) {
            fail!(
                "[[cell]]: cells {} and {} both claim prompt_tokens {prompt_tokens}",
                quote(&x.name),
                quote(name)
            );
        }
        cells.push(Cell {
            name: name.to_string(),
            prompt_tokens,
        });
    }
    let max_tokens = uint("[select]", select, "max_tokens")?;
    let client = match t.get("client") {
        Some(v) => Some(client(table("[client]", Some(v))?)?),
        None => None,
    };
    if let Some(c) = client
        .as_ref()
        .filter(|c| c.warmup_max_tokens == max_tokens)
    {
        fail!(
            "[client].warmup_max_tokens: must differ from [select].max_tokens ({}), or the warm-ups would be measured",
            c.warmup_max_tokens
        );
    }
    Ok(TtftConfig {
        max_tokens,
        step_token_id: uint("[class]", class, "step_token_id")?,
        cells,
        client,
    })
}
