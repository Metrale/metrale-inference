// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: `embedding.toml`, the one configuration of the embedder and
//! the reranker (ADDENDUM-EMBEDDINGS E2), as pure types and a parser.
//!
//! There is no default: the file, both tables and every field a kind needs
//! are required. `openrouter` lowers to `openai_compatible` at a fixed base
//! URL, so every dense consumer shares one shape. A key is only ever a
//! reference (`{ env = "..." }` or `{ file = "..." }`); a literal key is
//! rejected without being echoed. This module reads no file, no secret and
//! no network: the caller reads the file, and a `SecretResolver` (a port)
//! resolves references.

use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::embed::Space;
use crate::{Error, Result, bail};

/// The base URL an `openrouter` embedder lowers to.
pub const OPENROUTER_BASE_URL: &str = "https://openrouter.ai/api/v1";
/// The host label of the `openrouter` kind in a space id.
pub const OPENROUTER_HOST: &str = "openrouter";

/// Where a key lives; never the key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecretRef {
    Env(String),
    File(String),
}

/// A resolved key. It is never printed: `Debug` is redacted, and there is no
/// `Display`.
pub struct Secret(String);

impl Secret {
    pub fn new(value: String) -> Secret {
        Secret(value)
    }

    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for Secret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Secret(<redacted>)")
    }
}

/// The port that turns a reference into a key (environment or file reads
/// live in the adapter, outside this crate).
pub trait SecretResolver {
    fn resolve(&self, secret: &SecretRef) -> Result<Secret>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenAiCompatible {
    pub endpoint: String,
    pub model: String,
    pub api_key: Option<SecretRef>,
    pub dims: usize,
    pub precision: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OpenRouter {
    pub model: String,
    pub api_key: SecretRef,
    pub dims: usize,
    pub precision: u8,
    /// OpenRouter routing preferences, passed through verbatim.
    pub provider: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum EmbedderConfig {
    OpenAiCompatible(OpenAiCompatible),
    OpenRouter(OpenRouter),
    Hashed,
}

/// A dense embedder after lowering: what the one request/response core needs.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    pub endpoint: String,
    /// The host label in the space id: the endpoint's host, or `openrouter`.
    pub host: String,
    pub model: String,
    pub api_key: Option<SecretRef>,
    pub dims: usize,
    pub precision: u8,
    pub provider: Option<Map<String, Value>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Lowered {
    Dense(Dense),
    Hashed,
}

impl Lowered {
    pub fn space(&self) -> Result<Space> {
        match self {
            Lowered::Hashed => Ok(Space::hashed()),
            Lowered::Dense(d) => Space::dense(&d.host, &d.model, d.dims),
        }
    }
}

impl EmbedderConfig {
    pub fn lower(&self) -> Result<Lowered> {
        Ok(match self {
            EmbedderConfig::Hashed => Lowered::Hashed,
            EmbedderConfig::OpenAiCompatible(c) => Lowered::Dense(Dense {
                endpoint: c.endpoint.clone(),
                host: endpoint_host(&c.endpoint)?,
                model: c.model.clone(),
                api_key: c.api_key.clone(),
                dims: c.dims,
                precision: c.precision,
                provider: None,
            }),
            EmbedderConfig::OpenRouter(c) => Lowered::Dense(Dense {
                endpoint: OPENROUTER_BASE_URL.to_string(),
                host: OPENROUTER_HOST.to_string(),
                model: c.model.clone(),
                api_key: Some(c.api_key.clone()),
                dims: c.dims,
                precision: c.precision,
                provider: c.provider.clone(),
            }),
        })
    }

    /// The space every vector of this embedder lives in.
    pub fn space(&self) -> Result<Space> {
        self.lower()?.space()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RerankerConfig {
    None,
    OpenAiCompatible {
        endpoint: String,
        model: String,
        api_key: Option<SecretRef>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddingConfig {
    pub embedder: EmbedderConfig,
    pub reranker: RerankerConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    embedder: RawTable,
    reranker: RawTable,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTable {
    kind: String,
    endpoint: Option<String>,
    model: Option<String>,
    api_key: Option<toml::Value>,
    dims: Option<usize>,
    precision: Option<u8>,
    provider: Option<toml::Table>,
}

/// Parse `embedding.toml`. Errors carry the toml message and line, never the
/// source text (a misplaced key must not be echoed).
pub fn parse(text: &str) -> Result<EmbeddingConfig> {
    let raw: RawFile = toml::from_str(text).map_err(|e| {
        let line = e
            .span()
            .map(|s| text[..s.start.min(text.len())].lines().count().max(1));
        match line {
            Some(n) => Error(format!("embedding.toml: line {n}: {}", e.message())),
            None => Error(format!("embedding.toml: {}", e.message())),
        }
    })?;
    Ok(EmbeddingConfig {
        embedder: embedder(raw.embedder)?,
        reranker: reranker(raw.reranker)?,
    })
}

fn embedder(t: RawTable) -> Result<EmbedderConfig> {
    const AT: &str = "[embedder]";
    let kind = t.kind.clone();
    Ok(match kind.as_str() {
        "hashed" => {
            only(AT, &t, &[])?;
            EmbedderConfig::Hashed
        }
        "openai_compatible" => {
            only(
                AT,
                &t,
                &["endpoint", "model", "api_key", "dims", "precision"],
            )?;
            let endpoint = need(AT, &kind, "endpoint", t.endpoint)?;
            endpoint_host(&endpoint)?;
            EmbedderConfig::OpenAiCompatible(OpenAiCompatible {
                endpoint,
                model: model(AT, &kind, t.model)?,
                api_key: t.api_key.map(|v| secret(AT, v)).transpose()?,
                dims: dims(AT, &kind, t.dims)?,
                precision: need(AT, &kind, "precision", t.precision)?,
            })
        }
        "openrouter" => {
            only(
                AT,
                &t,
                &["model", "api_key", "dims", "precision", "provider"],
            )?;
            let api_key = need(AT, &kind, "api_key", t.api_key)?;
            EmbedderConfig::OpenRouter(OpenRouter {
                model: model(AT, &kind, t.model)?,
                api_key: secret(AT, api_key)?,
                dims: dims(AT, &kind, t.dims)?,
                precision: need(AT, &kind, "precision", t.precision)?,
                provider: t.provider.map(provider).transpose()?,
            })
        }
        other => bail!(
            "embedding.toml: {AT} kind {other:?} is not one of \"openai_compatible\", \
             \"openrouter\", \"hashed\""
        ),
    })
}

fn reranker(t: RawTable) -> Result<RerankerConfig> {
    const AT: &str = "[reranker]";
    let kind = t.kind.clone();
    Ok(match kind.as_str() {
        "none" => {
            only(AT, &t, &[])?;
            RerankerConfig::None
        }
        "openai_compatible" => {
            only(AT, &t, &["endpoint", "model", "api_key"])?;
            let endpoint = need(AT, &kind, "endpoint", t.endpoint)?;
            endpoint_host(&endpoint)?;
            RerankerConfig::OpenAiCompatible {
                endpoint,
                model: model(AT, &kind, t.model)?,
                api_key: t.api_key.map(|v| secret(AT, v)).transpose()?,
            }
        }
        other => bail!(
            "embedding.toml: {AT} kind {other:?} is not one of \"none\", \"openai_compatible\""
        ),
    })
}

/// Fail on any field the kind does not take.
fn only(at: &str, t: &RawTable, allowed: &[&str]) -> Result<()> {
    let present = [
        ("endpoint", t.endpoint.is_some()),
        ("model", t.model.is_some()),
        ("api_key", t.api_key.is_some()),
        ("dims", t.dims.is_some()),
        ("precision", t.precision.is_some()),
        ("provider", t.provider.is_some()),
    ];
    for (name, set) in present {
        if set && !allowed.contains(&name) {
            bail!(
                "embedding.toml: {at} kind {:?} does not take `{name}`",
                t.kind
            );
        }
    }
    Ok(())
}

fn need<T>(at: &str, kind: &str, field: &str, value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        Error(format!(
            "embedding.toml: {at} kind {kind:?} requires `{field}` (there is no default)"
        ))
    })
}

fn model(at: &str, kind: &str, value: Option<String>) -> Result<String> {
    let m = need(at, kind, "model", value)?;
    if m.is_empty() || m.chars().any(char::is_whitespace) {
        bail!("embedding.toml: {at} `model` must be non-empty without blanks");
    }
    Ok(m)
}

fn dims(at: &str, kind: &str, value: Option<usize>) -> Result<usize> {
    let d = need(at, kind, "dims", value)?;
    if d == 0 {
        bail!("embedding.toml: {at} `dims` must be > 0");
    }
    Ok(d)
}

/// A key reference. Anything else (a literal string above all) fails, and
/// the message never contains the value.
fn secret(at: &str, value: toml::Value) -> Result<SecretRef> {
    let refused = || {
        Error(format!(
            "embedding.toml: {at} `api_key` must be {{ env = \"NAME\" }} or \
             {{ file = \"PATH\" }}; a literal key is refused (its value is not shown)"
        ))
    };
    let toml::Value::Table(t) = value else {
        return Err(refused());
    };
    if t.len() != 1 {
        return Err(refused());
    }
    let (k, v) = t.into_iter().next().ok_or_else(refused)?;
    let toml::Value::String(s) = v else {
        return Err(refused());
    };
    if s.is_empty() {
        return Err(refused());
    }
    match k.as_str() {
        "env" => Ok(SecretRef::Env(s)),
        "file" => Ok(SecretRef::File(s)),
        _ => Err(refused()),
    }
}

fn provider(t: toml::Table) -> Result<Map<String, Value>> {
    match serde_json::to_value(t) {
        Ok(Value::Object(m)) => Ok(m),
        Ok(_) => bail!("embedding.toml: `provider` is not a table"),
        Err(e) => bail!("embedding.toml: `provider`: {e}"),
    }
}

/// The host of an `http(s)://` endpoint, lower-cased (IPv6 kept bracketed).
/// An endpoint carrying credentials is refused; no message echoes the URL.
pub fn endpoint_host(endpoint: &str) -> Result<String> {
    let rest = endpoint
        .strip_prefix("https://")
        .or_else(|| endpoint.strip_prefix("http://"))
        .ok_or_else(|| Error("embedding.toml: `endpoint` must be an http(s):// URL".into()))?;
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.contains('@') {
        bail!(
            "embedding.toml: `endpoint` carries credentials; reference the key through `api_key`"
        );
    }
    let (host, port) = if authority.starts_with('[') {
        match authority.find(']') {
            Some(i) => (&authority[..=i], &authority[i + 1..]),
            None => bail!("embedding.toml: `endpoint` has an unterminated IPv6 host"),
        }
    } else {
        match authority.find(':') {
            Some(i) => (&authority[..i], &authority[i..]),
            None => (authority, ""),
        }
    };
    let port_ok = port.is_empty()
        || port
            .strip_prefix(':')
            .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    let host_ok = !host.is_empty()
        && host
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '[' | ']' | ':'));
    if !port_ok || !host_ok {
        bail!("embedding.toml: `endpoint` has no valid host[:port]");
    }
    Ok(host.to_ascii_lowercase())
}
