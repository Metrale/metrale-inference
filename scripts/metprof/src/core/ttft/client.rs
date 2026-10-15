// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the TTFT client's pure half: the endpoint, the salted prompt,
//! the request bytes, the per-rep record (JSONL), and the join of client
//! records to the server's measured requests.
//!
//! The join is how a client time and a server time are known to be the same
//! request without a request id in the log. Records (warm-ups excluded) are
//! matched in order, as a subsequence of the server's measured requests, on
//! equal prompt tokens and a server `Done` TTFT within `JOIN_TOLERANCE_MS` of
//! the response's `usage.time_to_first_token_ms`. The usage time is only a
//! join key; TTFT_server stays the `Done` line. A record that finds no
//! request, a record without a usage time, and a rep whose intent (cold or
//! warm) disagrees with the server's cache-hit marker are errors.

use serde_json::{Value, json};

use crate::core::error::Result;
use crate::core::repr::quote;
use crate::core::ttft::cells::{Mode, Selection};
use crate::core::ttft::sse::Streamed;
use crate::core::ttft::{ClientConfig, SALT_PLACEHOLDER};
use crate::fail;

/// The one path the client posts to.
pub const CHAT_PATH: &str = "/v1/chat/completions";

/// The `Done` line prints TTFT with one decimal, so the same time read from
/// usage differs from it by at most half of 0.1 ms (plus float noise).
pub const JOIN_TOLERANCE_MS: f64 = 0.05 + 1e-9;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Endpoint {
    pub host: String,
    pub port: u16,
}

/// `http://HOST:PORT` (a trailing `/` allowed). No TLS, no path, no default
/// port.
pub fn parse_url(url: &str) -> Result<Endpoint> {
    let Some(rest) = url.strip_prefix("http://") else {
        fail!("--url {}: must start with http://", quote(url));
    };
    let rest = rest.strip_suffix('/').unwrap_or(rest);
    let Some((host, port)) = rest.rsplit_once(':') else {
        fail!("--url {}: must name a port (http://HOST:PORT)", quote(url));
    };
    let ok_host = !host.is_empty()
        && host
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"-._[]:".contains(&b));
    let Some(port) = port.parse::<u16>().ok().filter(|p| *p > 0) else {
        fail!("--url {}: bad port {}", quote(url), quote(port));
    };
    if !ok_host {
        fail!("--url {}: bad host (no path, user or query)", quote(url));
    }
    Ok(Endpoint {
        host: host.to_string(),
        port,
    })
}

/// The salt as exactly `digits` decimal digits (each digit is one token).
pub fn salt_text(salt: u64, digits: usize) -> String {
    let m = 10u64.pow(digits as u32);
    format!("{:0digits$}", salt % m)
}

/// The template with its one `{salt}` replaced.
pub fn render_prompt(template: &str, salt: &str) -> Result<String> {
    match template.matches(SALT_PLACEHOLDER).count() {
        1 => Ok(template.replacen(SALT_PLACEHOLDER, salt, 1)),
        n => fail!("the prompt must contain {SALT_PLACEHOLDER} exactly once, found {n}"),
    }
}

/// One complete HTTP/1.1 request: a streamed chat completion with usage.
pub fn request(ep: &Endpoint, c: &ClientConfig, max_tokens: u64, prompt: &str) -> Vec<u8> {
    let body = json!({
        "model": c.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": true,
        "stream_options": {"include_usage": true},
        "max_tokens": max_tokens,
        "temperature": c.temperature,
        "top_p": c.top_p,
        "seed": c.seed,
    })
    .to_string();
    let head = format!(
        "POST {CHAT_PATH} HTTP/1.1\r\nHost: {}:{}\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        ep.host,
        ep.port,
        body.len()
    );
    [head.into_bytes(), body.into_bytes()].concat()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Intent {
    Warmup,
    Cold,
    Warm,
}

impl Intent {
    pub fn name(self) -> &'static str {
        match self {
            Intent::Warmup => "warmup",
            Intent::Cold => "cold",
            Intent::Warm => "warm",
        }
    }

    fn parse(s: &str) -> Option<Intent> {
        [Intent::Warmup, Intent::Cold, Intent::Warm]
            .into_iter()
            .find(|i| i.name() == s)
    }

    fn mode(self) -> Option<Mode> {
        match self {
            Intent::Warmup => None,
            Intent::Cold => Some(Mode::Cold),
            Intent::Warm => Some(Mode::Warm),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClientRecord {
    pub cell: String,
    pub rep: u64,
    pub intent: Intent,
    pub salt: String,
    pub prompt_tokens: u64,
    pub completion_tokens: Option<u64>,
    pub ttft_client_ms: f64,
    pub e2e_ms: f64,
    pub usage_ttft_ms: Option<f64>,
    pub cached_tokens: Option<u64>,
    pub finish_reason: String,
    pub first_delta: String,
}

/// A rep's record; refuses a response whose prompt is not the cell's.
pub fn record(
    cell: &crate::core::ttft::Cell,
    rep: u64,
    intent: Intent,
    salt: &str,
    s: &Streamed,
) -> Result<ClientRecord> {
    if s.usage.prompt_tokens != cell.prompt_tokens {
        fail!(
            "cell {} {} rep {rep}: the server counted {} prompt tokens, the cell is {}; the prompt file does not produce this cell",
            cell.name,
            intent.name(),
            s.usage.prompt_tokens,
            cell.prompt_tokens
        );
    }
    Ok(ClientRecord {
        cell: cell.name.clone(),
        rep,
        intent,
        salt: salt.to_string(),
        prompt_tokens: s.usage.prompt_tokens,
        completion_tokens: s.usage.completion_tokens,
        ttft_client_ms: s.ttft_client_ms,
        e2e_ms: s.e2e_ms,
        usage_ttft_ms: s.usage.ttft_ms,
        cached_tokens: s.usage.cached_tokens,
        finish_reason: s.finish_reason.clone(),
        first_delta: s.first_delta.to_string(),
    })
}

fn to_json(r: &ClientRecord) -> Value {
    json!({
        "cell": r.cell,
        "rep": r.rep,
        "mode": r.intent.name(),
        "salt": r.salt,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "ttft_client_ms": r.ttft_client_ms,
        "e2e_ms": r.e2e_ms,
        "usage_ttft_ms": r.usage_ttft_ms,
        "cached_tokens": r.cached_tokens,
        "finish_reason": r.finish_reason,
        "first_delta": r.first_delta,
    })
}

pub fn records_jsonl(rs: &[ClientRecord]) -> String {
    rs.iter().map(|r| format!("{}\n", to_json(r))).collect()
}

const KEYS: [&str; 12] = [
    "cell",
    "rep",
    "mode",
    "salt",
    "prompt_tokens",
    "completion_tokens",
    "ttft_client_ms",
    "e2e_ms",
    "usage_ttft_ms",
    "cached_tokens",
    "finish_reason",
    "first_delta",
];

fn from_json(v: &Value, n: usize) -> Result<ClientRecord> {
    let Some(o) = v.as_object() else {
        fail!("line {n}: not a JSON object");
    };
    for k in o.keys() {
        if !KEYS.contains(&k.as_str()) {
            fail!("line {n}: unknown key {}", quote(k));
        }
    }
    let get = |k: &str| match o.get(k) {
        Some(x) => Ok(x),
        None => fail!("line {n}: missing key {}", quote(k)),
    };
    let s = |k: &str| match get(k)?.as_str() {
        Some(x) => Ok(x.to_string()),
        None => fail!("line {n}: {k} must be a string"),
    };
    let u = |k: &str| match get(k)?.as_u64() {
        Some(x) => Ok(x),
        None => fail!("line {n}: {k} must be a non-negative integer"),
    };
    let f = |k: &str| match get(k)?.as_f64() {
        Some(x) => Ok(x),
        None => fail!("line {n}: {k} must be a number"),
    };
    let opt = |k: &str| -> Result<Option<&Value>> { Ok(Some(get(k)?).filter(|x| !x.is_null())) };
    let ou = |k: &str| -> Result<Option<u64>> {
        match opt(k)? {
            None => Ok(None),
            Some(x) => match x.as_u64() {
                Some(v) => Ok(Some(v)),
                None => fail!("line {n}: {k} must be an integer or null"),
            },
        }
    };
    let Some(intent) = Intent::parse(&s("mode")?) else {
        fail!("line {n}: mode must be warmup, cold or warm");
    };
    Ok(ClientRecord {
        cell: s("cell")?,
        rep: u("rep")?,
        intent,
        salt: s("salt")?,
        prompt_tokens: u("prompt_tokens")?,
        completion_tokens: ou("completion_tokens")?,
        ttft_client_ms: f("ttft_client_ms")?,
        e2e_ms: f("e2e_ms")?,
        usage_ttft_ms: match opt("usage_ttft_ms")? {
            None => None,
            Some(x) => match x.as_f64() {
                Some(v) => Some(v),
                None => fail!("line {n}: usage_ttft_ms must be a number or null"),
            },
        },
        cached_tokens: ou("cached_tokens")?,
        finish_reason: s("finish_reason")?,
        first_delta: s("first_delta")?,
    })
}

/// Parse a client JSONL file (blank lines skipped).
pub fn parse_records(text: &str) -> Result<Vec<ClientRecord>> {
    let mut out = Vec::new();
    for (i, l) in text.lines().enumerate() {
        if l.trim().is_empty() {
            continue;
        }
        let v: Value = match serde_json::from_str(l) {
            Ok(v) => v,
            Err(e) => fail!("line {}: not JSON: {e}", i + 1),
        };
        out.push(from_json(&v, i + 1)?);
    }
    Ok(out)
}

/// (index into `sel.measured`, index into `records`) for every non-warm-up
/// record, in order.
pub fn join(records: &[ClientRecord], sel: &Selection) -> Result<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for (ri, r) in records.iter().enumerate() {
        let Some(mode) = r.intent.mode() else {
            continue;
        };
        let what = format!(
            "client record {} ({} {} rep {})",
            ri + 1,
            r.cell,
            r.intent.name(),
            r.rep
        );
        let Some(key) = r.usage_ttft_ms else {
            fail!("{what}: no usage_ttft_ms, so it cannot be joined to a server request");
        };
        let found = sel.measured[cursor..].iter().position(|m| {
            m.req.prompt_tokens == r.prompt_tokens
                && (m.req.ttft_ms - key).abs() <= JOIN_TOLERANCE_MS
        });
        let Some(off) = found else {
            fail!(
                "{what}: no later measured server request with {} prompt tokens and TTFT {key:.3} ms",
                r.prompt_tokens
            );
        };
        let mi = cursor + off;
        let m = &sel.measured[mi];
        if m.cell != r.cell {
            fail!(
                "{what}: joined the server request of line {}, which is cell {}",
                m.req.line,
                m.cell
            );
        }
        if m.mode != mode {
            fail!(
                "{what}: the server request of line {} was {} (cache-hit marker), not {}",
                m.req.line,
                m.mode.name(),
                mode.name()
            );
        }
        out.push((mi, ri));
        cursor = mi + 1;
    }
    Ok(out)
}
