// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The OpenAI-compatible embedder host (owner decision 6: OpenRouter today; every
//! setting from the resolved two-layer config, nothing hard-coded): one blocking `ureq` POST per
//! batch on tokio's blocking pool, behind metrale-kg-derive's [`Embedder`] port.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - The key is resolved from its reference: a file (the first run of `[A-Za-z0-9_-]` starting
//!   with the configured prefix) or an environment variable (its whole value); it is
//!   held in [`Secret`] (whose `Debug` prints `[redacted]`), sent only as the `Authorization`
//!   header, and never formatted into an error, a log line, argv or the environment.
//! - A key reference that does not yield a key gives [`Mode::Unavailable`] with the reason (never
//!   the file's contents or the variable's value); the binary reports it as `graph_only`.
//! - Requests are spaced by at least `min_interval_ms` (the provider's rate limit).
//! - HTTP 429 and 5xx are [`Reply::Retryable`]; any other non-200 is `KgEmbedProtocol` with the
//!   status only, because a provider error body can echo the request.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use metrale_error::{ErrorCode, Result, error};
use metrale_kg_derive::embed::{Embedder, Mode, Reply, Role, parse_response, request_body};
use metrale_kg_derive::settings::{EmbedSettings, KeyRef};

/// 2026-10-23: The key; it never prints.
pub struct Secret(String);

impl core::fmt::Debug for Secret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("[redacted]")
    }
}

/// 2026-10-23: The first key in `text` that starts with `prefix`.
pub fn extract_key(text: &str, prefix: &str) -> Option<Secret> {
    if prefix.is_empty() {
        return None;
    }
    let at = text.find(prefix)?;
    let key: String = text[at..]
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
        .collect();
    (key.len() > prefix.len()).then_some(Secret(key))
}

/// 2026-10-23: Resolve a key reference, or the reason it yields no key (never the key material).
pub fn resolve_key(r: &KeyRef) -> core::result::Result<Secret, String> {
    match r {
        KeyRef::File { path, prefix } => match std::fs::read_to_string(path) {
            Ok(t) => {
                extract_key(&t, prefix).ok_or_else(|| format!("key file holds no `{prefix}` key"))
            }
            Err(e) => Err(format!("key file unreadable ({:?})", e.kind())),
        },
        KeyRef::Env { name } => match std::env::var(name) {
            Ok(v) if !v.trim().is_empty() => Ok(Secret(v.trim().to_string())),
            Ok(_) => Err(format!("${name} is empty")),
            Err(_) => Err(format!("${name} is unset")),
        },
    }
}

pub struct Inner {
    agent: ureq::Agent,
    s: EmbedSettings,
    key: Secret,
    last: Mutex<Option<Instant>>,
    /// 2026-10-23: Every call's request shape and outcome, one JSON line each (never the key or
    /// the input text).
    log: PathBuf,
}

/// 2026-10-23: The OpenRouter host, or the declared unavailable mode.
pub enum OpenRouter {
    Live(Arc<Inner>),
    Unavailable(String),
}

impl OpenRouter {
    /// 2026-10-23: The host for the resolved settings; a key reference that yields no key is the
    /// unavailable mode.
    pub fn new(s: &EmbedSettings, log: PathBuf) -> Self {
        let key = match resolve_key(&s.key) {
            Ok(k) => k,
            Err(reason) => return Self::Unavailable(reason),
        };
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_millis(s.timeout_ms)))
            .http_status_as_error(false)
            .build()
            .into();
        Self::Live(Arc::new(Inner {
            agent,
            s: s.clone(),
            key,
            last: Mutex::new(None),
            log,
        }))
    }

    /// 2026-10-23: The declared unavailable mode.
    pub fn unavailable(reason: &str) -> Self {
        Self::Unavailable(reason.into())
    }
}

fn pace(i: &Inner) {
    let gap = Duration::from_millis(i.s.min_interval_ms);
    let mut last = i
        .last
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(t) = *last {
        let since = t.elapsed();
        if since < gap {
            std::thread::sleep(gap - since);
        }
    }
    *last = Some(Instant::now());
}

/// 2026-10-23: Whether a refusal contradicts what was sent: an input-length refusal (HTTP 400
/// naming a token limit) for a request whose longest input is within `max_input_chars`.
fn is_fault(status: u16, detail: &str, max_chars: u64, max_input_chars: u32) -> bool {
    status == 400
        && detail.contains("exceeds")
        && detail.contains("token")
        && max_chars <= u64::from(max_input_chars)
}

/// 2026-10-23: The request shape of one call, as logged and as a fault names it.
fn shape(i: &Inner, role: Role, inputs: &[String]) -> serde_json::Value {
    let lens: Vec<usize> = inputs.iter().map(|t| t.chars().count()).collect();
    serde_json::json!({
        "model": i.s.model,
        "input_type": i.s.input_type.of(role),
        "inputs": inputs.len(),
        "total_chars": lens.iter().sum::<usize>(),
        "max_chars": lens.iter().max().copied().unwrap_or(0),
    })
}

fn log_call(i: &Inner, mut shape: serde_json::Value, status: u16, ms: u128, detail: &str) {
    shape["status"] = status.into();
    shape["ms"] = u64::try_from(ms).unwrap_or(u64::MAX).into();
    shape["ts_ms"] = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
        .into();
    if !detail.is_empty() {
        shape["detail"] = detail.into();
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&i.log)
    {
        let _ = writeln!(f, "{shape}");
    }
}

/// 2026-10-23: One request, logged with its shape and the HTTP status it got (0: none, a
/// timeout or a transport error).
fn post(i: &Inner, role: Role, inputs: &[String]) -> Result<Reply> {
    let t0 = Instant::now();
    let sh = shape(i, role, inputs);
    let mut http = 0;
    let r = post_inner(i, role, inputs, &sh, &mut http);
    let detail = match &r {
        Ok(Reply::Vectors(_)) => String::new(),
        Ok(Reply::Retryable { status: 0, .. }) => "timeout".to_string(),
        Ok(Reply::Retryable { .. }) => String::new(),
        Err(e) => e.to_string(),
    };
    log_call(i, sh, http, t0.elapsed().as_millis(), &detail);
    r
}

fn post_inner(
    i: &Inner,
    role: Role,
    inputs: &[String],
    sh: &serde_json::Value,
    http: &mut u16,
) -> Result<Reply> {
    pace(i);
    let body = request_body(&i.s.model, i.s.input_type.of(role), inputs);
    let auth = format!("Bearer {}", i.key.0);
    let mut req = i
        .agent
        .post(&i.s.endpoint)
        .header("Authorization", &auth)
        .header("Content-Type", "application/json");
    for (k, v) in &i.s.headers {
        req = req.header(k.as_str(), v.as_str());
    }
    let sent = req.send(body.as_str());
    let mut resp = match sent {
        Ok(r) => r,
        Err(ureq::Error::Timeout(_)) => {
            return Ok(Reply::Retryable {
                status: 0,
                retry_after_ms: None,
            });
        }
        Err(e) => return Err(error!(ErrorCode::KgEmbedderUnavailable, transport_kind(&e))),
    };
    let status = resp.status().as_u16();
    *http = status;
    if status == 429 || (500..600).contains(&status) {
        let retry_after_ms = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(|s| s.saturating_mul(1000));
        return Ok(Reply::Retryable {
            status,
            retry_after_ms,
        });
    }
    if status != 200 {
        let detail = resp
            .body_mut()
            .with_config()
            .limit(64 * 1024)
            .read_to_string()
            .ok()
            .and_then(|t| serde_json::from_str::<serde_json::Value>(&t).ok())
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .map(|m| shown_detail(&m))
            .unwrap_or_default();
        let max = sh["max_chars"].as_u64().unwrap_or(0);
        if is_fault(status, &detail, max, i.s.max_input_chars) {
            return Err(error!(
                ErrorCode::KgEmbedderFault,
                format!(
                    "HTTP 400{detail} for a request of {sh}, every input within max_input_chars {}",
                    i.s.max_input_chars
                )
            ));
        }
        return Err(error!(
            ErrorCode::KgEmbedProtocol,
            format!("HTTP {status}{detail}")
        ));
    }
    let text = resp
        .body_mut()
        .with_config()
        .limit(256 * 1024 * 1024)
        .read_to_string()
        .map_err(|e| error!(ErrorCode::KgEmbedProtocol, transport_kind(&e)))?;
    let count = inputs.len();
    Ok(Reply::Vectors(parse_response(
        &text,
        count,
        i.s.dimensions,
    )?))
}

/// 2026-10-23: The provider's error message, at most 200 chars, dropped whole if it matches the
/// secrets denylist (a provider can echo a header back).
fn shown_detail(m: &str) -> String {
    if metrale_kg_derive::secrets::secret_pattern(m).is_some() {
        return " (provider message withheld: it matched the secrets denylist)".into();
    }
    let short: String = m.chars().take(200).collect();
    format!(" ({short})")
}

/// 2026-10-23: A transport error's kind without its text (which can carry the URL or headers).
fn transport_kind(e: &ureq::Error) -> String {
    let s = format!("{e:?}");
    s.split(['(', ' ', '{'])
        .next()
        .unwrap_or("transport")
        .to_string()
}

impl Embedder for OpenRouter {
    fn mode(&self) -> Mode {
        match self {
            Self::Live(i) => Mode::Available {
                model: i.s.model.clone(),
                dim: i.s.dimensions,
            },
            Self::Unavailable(r) => Mode::Unavailable { reason: r.clone() },
        }
    }

    async fn embed(&self, role: Role, inputs: &[String]) -> Result<Reply> {
        let i = match self {
            Self::Live(i) => Arc::clone(i),
            Self::Unavailable(r) => {
                return Err(error!(ErrorCode::KgEmbedderUnavailable, r.clone()));
            }
        };
        let inputs = inputs.to_vec();
        tokio::task::spawn_blocking(move || post(&i, role, &inputs))
            .await
            .map_err(|_| {
                error!(
                    ErrorCode::KgEmbedderUnavailable,
                    "the embed task was cancelled"
                )
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(key: KeyRef) -> EmbedSettings {
        EmbedSettings {
            endpoint: "http://127.0.0.1:9/".into(),
            model: "m".into(),
            dimensions: 4,
            headers: Default::default(),
            key,
            timeout_ms: 10,
            batch: 1,
            max_attempts: 1,
            backoff_ms: 1,
            min_interval_ms: 0,
            max_input_chars: 100,
            input_type: metrale_kg_derive::settings::InputTypes {
                query: "query".into(),
                document: "passage".into(),
            },
        }
    }

    #[test]
    fn an_input_length_refusal_within_the_cap_is_an_embedder_fault() {
        let d = " (HTTP 400: Input length 46419 exceeds maximum allowed token size 8192)";
        assert!(is_fault(400, d, 70, 8000));
        assert!(
            !is_fault(400, d, 9000, 8000),
            "an input over the cap is ours, not a fault"
        );
        assert!(!is_fault(400, " (bad model)", 70, 8000));
        assert!(!is_fault(429, d, 70, 8000));
    }

    #[test]
    fn the_key_is_found_by_its_prefix_and_never_prints() {
        let k = extract_key(
            "OpenRouter embedder\nkey: sk-or-v1-abc_DEF-123\nnotes\n",
            "sk-or-",
        )
        .unwrap();
        assert_eq!(k.0, "sk-or-v1-abc_DEF-123");
        assert_eq!(format!("{k:?}"), "[redacted]");
        assert!(extract_key("no key here, sk-or- alone", "sk-or-").is_none());
        assert!(extract_key("sk-or-v1-abc", "").is_none());
    }

    #[test]
    fn a_reference_without_a_key_is_the_declared_unavailable_mode_without_contents() {
        let f = OpenRouter::new(
            &settings(KeyRef::File {
                path: "/nonexistent/kg-key-file".into(),
                prefix: "sk-or-".into(),
            }),
            PathBuf::from("/dev/null"),
        );
        assert_eq!(
            f.mode(),
            Mode::Unavailable {
                reason: "key file unreadable (NotFound)".into()
            }
        );
        let e = OpenRouter::new(
            &settings(KeyRef::Env {
                name: "METRALE_KG_TEST_UNSET_KEY_VAR".into(),
            }),
            PathBuf::from("/dev/null"),
        );
        assert_eq!(
            e.mode(),
            Mode::Unavailable {
                reason: "$METRALE_KG_TEST_UNSET_KEY_VAR is unset".into()
            }
        );
    }
}
