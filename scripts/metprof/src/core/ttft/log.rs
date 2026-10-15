// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the rank-0 serve log to one record per request (pure).
//!
//! A request runs from its `Chunked prefill start: N prompt tokens,
//! chunk_size=C, max_tokens=M` line to the next `Done: ... TTFT=X.Xms, ...`
//! line. Inside it, `Prefill first token: T` gives the first token and a
//! `Marconi SSM cache hit: ...` line (either engine format) marks it warm.
//!
//! The log carries no request id, so a line can only be attributed while one
//! request is in flight. When a request starts while another is still open,
//! every request of that overlap (until none is in flight) is reported as
//! `concurrent`, with no TTFT: their `Done` lines cannot be told apart.
//!
//! Errors: a marker line that does not match its exact format (including a
//! cache-hit line in neither format), a `Done` with no request open, and a
//! cache-hit or first-token line outside a request or repeated inside one.
//! ANSI colour escapes are removed before matching.

use std::sync::LazyLock;

use regex::Regex;

use crate::core::error::Result;
use crate::core::repr::quote;
use crate::fail;

#[derive(Debug, Clone, PartialEq)]
pub struct CacheHit {
    pub skipped: u64,
    pub blocks: u64,
    pub snapshot: u64,
    /// The prefix-lookup format's `replaying R SSM tokens to reach N`.
    pub replay: Option<(u64, u64)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ServerRequest {
    /// 1-based line of the request's `Chunked prefill start`.
    pub line: usize,
    /// The line's leading timestamp, when it has one.
    pub time: Option<String>,
    pub prompt_tokens: u64,
    pub max_tokens: u64,
    pub hit: Option<CacheHit>,
    pub first_token: Option<u64>,
    /// The `Done` line's TTFT (ms).
    pub ttft_ms: f64,
    pub done_tokens: u64,
    pub reason: String,
}

/// A request whose lines could not be attributed (concurrent or unfinished).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unattributed {
    pub line: usize,
    pub prompt_tokens: u64,
    pub max_tokens: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ServeLog {
    pub requests: Vec<ServerRequest>,
    pub concurrent: Vec<Unattributed>,
    /// Requests still open at the end of the log.
    pub unfinished: Vec<Unattributed>,
}

fn re(p: &str) -> Regex {
    Regex::new(p).expect("a log pattern is a valid regex")
}

static START: LazyLock<Regex> = LazyLock::new(|| {
    re(r"Chunked prefill start: (\d+) prompt tokens, chunk_size=(\d+), max_tokens=(\d+)$")
});
static HIT_PREFILL: LazyLock<Regex> = LazyLock::new(|| {
    re(r"Marconi SSM cache hit: (\d+) tokens skipped \((\d+) blocks\), snapshot (\d+)$")
});
static HIT_LOOKUP: LazyLock<Regex> = LazyLock::new(|| {
    re(
        r"Marconi SSM cache hit: (\d+) tokens skipped \((\d+) blocks\), snapshot (\d+), replaying (\d+) SSM tokens to reach (\d+)$",
    )
});
static FIRST: LazyLock<Regex> = LazyLock::new(|| re(r"Prefill first token: (\d+)$"));
static DONE_TRIGGER: LazyLock<Regex> = LazyLock::new(|| re(r"Done: \d+ tokens \("));
static DONE: LazyLock<Regex> =
    LazyLock::new(|| re(r"Done: (\d+) tokens \(([^)]*)\) \S+ tok/s, TTFT=(\d+(?:\.\d+)?)ms, "));

/// `text` without ANSI SGR escapes (`ESC [ digits/semicolons m`).
pub fn strip_ansi(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('\x1b') {
        out.push_str(&rest[..i]);
        let tail = &rest[i + 1..];
        let body = tail.strip_prefix('[').map(|b| {
            let n = b
                .bytes()
                .take_while(|c| c.is_ascii_digit() || *c == b';')
                .count();
            (n, b[n..].starts_with('m'))
        });
        match body {
            Some((n, true)) => rest = &tail[1 + n + 1..],
            _ => {
                out.push('\x1b');
                rest = tail;
            }
        }
    }
    out.push_str(rest);
    out
}

fn num(c: &regex::Captures, i: usize, line: usize) -> Result<u64> {
    match c[i].parse::<u64>() {
        Ok(v) => Ok(v),
        Err(_) => fail!("line {line}: number {} out of range", quote(&c[i])),
    }
}

#[derive(Debug, Clone)]
struct Open {
    line: usize,
    time: Option<String>,
    prompt_tokens: u64,
    max_tokens: u64,
    hit: Option<CacheHit>,
    first_token: Option<u64>,
}

impl Open {
    fn unattributed(&self) -> Unattributed {
        Unattributed {
            line: self.line,
            prompt_tokens: self.prompt_tokens,
            max_tokens: self.max_tokens,
        }
    }
}

fn cache_hit(l: &str, n: usize) -> Result<CacheHit> {
    if let Some(c) = HIT_LOOKUP.captures(l) {
        return Ok(CacheHit {
            skipped: num(&c, 1, n)?,
            blocks: num(&c, 2, n)?,
            snapshot: num(&c, 3, n)?,
            replay: Some((num(&c, 4, n)?, num(&c, 5, n)?)),
        });
    }
    if let Some(c) = HIT_PREFILL.captures(l) {
        return Ok(CacheHit {
            skipped: num(&c, 1, n)?,
            blocks: num(&c, 2, n)?,
            snapshot: num(&c, 3, n)?,
            replay: None,
        });
    }
    fail!(
        "line {n}: a cache-hit line in neither known format: {}",
        quote(l)
    )
}

fn timestamp(l: &str) -> Option<String> {
    let t = l.split_whitespace().next()?;
    (t.len() >= 19 && t.as_bytes()[0].is_ascii_digit() && t.contains('T')).then(|| t.to_string())
}

struct State {
    out: ServeLog,
    open: Vec<Open>,
    overlapping: bool,
    members: Vec<Unattributed>,
}

impl State {
    /// The one request a line can belong to: `Ok(None)` inside an overlap.
    fn only(&mut self, what: &str, n: usize) -> Result<Option<&mut Open>> {
        if self.open.is_empty() {
            fail!("line {n}: a {what} line outside any request");
        }
        Ok(if self.overlapping {
            None
        } else {
            self.open.first_mut()
        })
    }

    fn done(&mut self, l: &str, n: usize) -> Result<()> {
        let Some(c) = DONE.captures(l) else {
            fail!("line {n}: a Done line in an unknown format: {}", quote(l));
        };
        if self.open.is_empty() {
            fail!("line {n}: a Done line with no request open");
        }
        let finished = self.open.remove(0);
        if self.overlapping {
            self.members.push(finished.unattributed());
            if self.open.is_empty() {
                self.out.concurrent.append(&mut self.members);
                self.overlapping = false;
            }
            return Ok(());
        }
        let ttft_ms = match c[3].parse::<f64>() {
            Ok(v) => v,
            Err(_) => fail!("line {n}: TTFT {} is not a number", quote(&c[3])),
        };
        self.out.requests.push(ServerRequest {
            line: finished.line,
            time: finished.time,
            prompt_tokens: finished.prompt_tokens,
            max_tokens: finished.max_tokens,
            hit: finished.hit,
            first_token: finished.first_token,
            ttft_ms,
            done_tokens: num(&c, 1, n)?,
            reason: c[2].to_string(),
        });
        Ok(())
    }

    fn line(&mut self, l: &str, n: usize) -> Result<()> {
        if l.contains("Chunked prefill start:") {
            let Some(c) = START.captures(l) else {
                fail!(
                    "line {n}: a prefill-start line in an unknown format: {}",
                    quote(l)
                );
            };
            self.overlapping |= !self.open.is_empty();
            self.open.push(Open {
                line: n,
                time: timestamp(l),
                prompt_tokens: num(&c, 1, n)?,
                max_tokens: num(&c, 3, n)?,
                hit: None,
                first_token: None,
            });
        } else if l.contains("SSM cache hit") {
            let hit = cache_hit(l, n)?;
            if let Some(r) = self.only("cache-hit", n)? {
                if r.hit.is_some() {
                    fail!(
                        "line {n}: a second cache-hit line in the request of line {}",
                        r.line
                    );
                }
                r.hit = Some(hit);
            }
        } else if l.contains("Prefill first token:") {
            let Some(c) = FIRST.captures(l) else {
                fail!(
                    "line {n}: a first-token line in an unknown format: {}",
                    quote(l)
                );
            };
            let t = num(&c, 1, n)?;
            if let Some(r) = self.only("first-token", n)? {
                if r.first_token.is_some() {
                    fail!(
                        "line {n}: a second first-token line in the request of line {}",
                        r.line
                    );
                }
                r.first_token = Some(t);
            }
        } else if DONE_TRIGGER.is_match(l) {
            self.done(l, n)?;
        }
        Ok(())
    }
}

/// Parse a whole rank-0 log.
pub fn parse_serve_log(text: &str) -> Result<ServeLog> {
    let mut s = State {
        out: ServeLog::default(),
        open: Vec::new(),
        overlapping: false,
        members: Vec::new(),
    };
    for (i, raw) in text.lines().enumerate() {
        let l = strip_ansi(raw);
        s.line(l.trim_end(), i + 1)?;
    }
    s.out.unfinished = s
        .members
        .into_iter()
        .chain(s.open.iter().map(Open::unattributed))
        .collect();
    Ok(s.out)
}
