// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The embedder settings (owner requirement 2026-10-23: one typed config, resolved
//! once; two layers, project wins). Pure over `&str` so it builds for `no_std` and wasm: the host
//! reads the global layer (native `~/.latlas/config.toml`, browser OPFS) and the project layer (the
//! `[embedder]` table of the kg config) and hands both texts here.
//!
//! Owner: metrale-kg-derive.
//! Invariants:
//! - [`merge`] takes each key from the project layer when it sets it, else from the global layer;
//!   [`Provenance`] records which layer every resolved key came from.
//! - [`resolve`] defaults nothing (PCND). No layer naming an embedder at all is the declared
//!   [`Embedder::Unavailable`] mode; an embedder that is named but missing a key is `KgConfig`
//!   with the key's name.
//! - The API key is only ever a [`KeyRef`] (a file path or an environment variable name), never
//!   a value: a layer whose fields match the secrets denylist is refused.

use crate::bundle::Builder;
use crate::embed::Role;
use crate::id::{KgId, Label};
use crate::prelude::*;
use crate::secrets::secret_pattern;
use metrale_error::{ErrorCode, Result, error};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// 2026-10-23: The global layer's schema id.
pub const GLOBAL_SCHEMA: &str = "latlas-config/v1";

/// 2026-10-23: One layer's `[embedder]` table; every key optional so a layer may set a subset.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EmbedLayer {
    /// 2026-10-23: `http` (an OpenAI-compatible POST `/embeddings` API; DESIGN-latlas.md §3.5)
    /// or `unavailable`.
    pub mode: Option<String>,
    /// 2026-10-23: Why, for `mode = "unavailable"`.
    pub reason: Option<String>,
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub dimensions: Option<u32>,
    /// 2026-10-23: Extra request headers (`HTTP-Referer`, `X-Title`, ...); never a credential.
    pub headers: Option<BTreeMap<String, String>>,
    /// 2026-10-23: The key as a file path; the key is the first run of `[A-Za-z0-9_-]` in the
    /// file that starts with `key_prefix`.
    pub key_file: Option<String>,
    pub key_prefix: Option<String>,
    /// 2026-10-23: The key as an environment variable's name (its whole value is the key).
    pub key_env: Option<String>,
    pub timeout_ms: Option<u64>,
    pub batch: Option<u32>,
    pub max_attempts: Option<u32>,
    pub backoff_ms: Option<u64>,
    /// 2026-10-23: The rate limit: the least time between two requests.
    pub min_interval_ms: Option<u64>,
    /// 2026-10-23: The longest input (chars) the pipeline sends; a longer one is refused before
    /// sending, never truncated.
    pub max_input_chars: Option<u32>,
    /// 2026-10-23: The `input_type` an asymmetric model takes for each role.
    pub input_type: Option<InputTypes>,
}

/// 2026-10-23: The request's `input_type` per [`Role`] (`{ query = "query", document = "passage" }`
/// for nvidia/llama-nemotron-embed). One table, so a layer sets both or neither.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct InputTypes {
    pub query: String,
    pub document: String,
}

impl InputTypes {
    /// 2026-10-23: The value sent for `role`.
    pub fn of(&self, role: Role) -> &str {
        match role {
            Role::Query => &self.query,
            Role::Document => &self.document,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GlobalFile {
    schema: String,
    #[serde(default)]
    embedder: EmbedLayer,
}

/// 2026-10-23: Parse the global layer's text (`latlas-config/v1`).
pub fn parse_global(text: &str) -> Result<EmbedLayer> {
    let f: GlobalFile = toml::from_str(text)
        .map_err(|e| error!(ErrorCode::KgConfig, "(global layer)", e.to_string()))?;
    if f.schema != GLOBAL_SCHEMA {
        return Err(error!(
            ErrorCode::KgConfig,
            "(global layer) schema",
            format!("must be {GLOBAL_SCHEMA}")
        ));
    }
    Ok(f.embedder)
}

/// 2026-10-23: Where a resolved key came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Layer {
    Global,
    Project,
}

impl Layer {
    /// 2026-10-23: Its name in reports.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::Project => "project",
        }
    }
}

/// 2026-10-23: The layer of every key the merge set.
pub type Provenance = BTreeMap<&'static str, Layer>;

/// 2026-10-23: Project wins per key. The key reference (`key_file`, `key_prefix`, `key_env`) is
/// one setting: a project layer that names a key file or variable replaces the global reference
/// whole, so the two layers never combine into two references.
pub fn merge(global: &EmbedLayer, project: &EmbedLayer) -> (EmbedLayer, Provenance) {
    let mut prov = Provenance::new();
    let mut global = global.clone();
    if project.key_file.is_some() || project.key_env.is_some() {
        global.key_file = None;
        global.key_prefix = None;
        global.key_env = None;
    }
    let global = &global;
    macro_rules! pick {
        ($($f:ident),*) => {
            EmbedLayer { $($f: {
                let (v, l) = match (&project.$f, &global.$f) {
                    (Some(p), _) => (Some(p.clone()), Some(Layer::Project)),
                    (None, Some(g)) => (Some(g.clone()), Some(Layer::Global)),
                    (None, None) => (None, None),
                };
                if let Some(l) = l {
                    prov.insert(stringify!($f), l);
                }
                v
            }),* }
        };
    }
    let m = pick!(
        mode,
        reason,
        endpoint,
        model,
        dimensions,
        headers,
        key_file,
        key_prefix,
        key_env,
        timeout_ms,
        batch,
        max_attempts,
        backoff_ms,
        min_interval_ms,
        max_input_chars,
        input_type
    );
    (m, prov)
}

/// 2026-10-23: The API key's reference.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyRef {
    File { path: String, prefix: String },
    Env { name: String },
}

/// 2026-10-23: A resolved, immutable embedder configuration (ROTI).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbedSettings {
    pub endpoint: String,
    pub model: String,
    pub dimensions: u32,
    pub headers: BTreeMap<String, String>,
    pub key: KeyRef,
    pub timeout_ms: u64,
    pub batch: u32,
    pub max_attempts: u32,
    pub backoff_ms: u64,
    pub min_interval_ms: u64,
    pub max_input_chars: u32,
    pub input_type: InputTypes,
}

impl EmbedSettings {
    /// 2026-10-23: The vector space a cached vector belongs to: a change of model, dimension or
    /// document `input_type` makes every cached vector incomparable with a new query.
    pub fn space(&self) -> String {
        format!(
            "{}/{}/{}",
            self.model, self.dimensions, self.input_type.document
        )
    }
}

/// 2026-10-23: The embedder the merged layers declare.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Embedder {
    Available(EmbedSettings),
    Unavailable { reason: String },
}

fn need<T: Clone>(v: &Option<T>, key: &'static str) -> Result<T> {
    v.clone().ok_or_else(|| {
        error!(
            ErrorCode::KgConfig,
            format!("embedder.{key}"),
            "is required for mode http (no layer sets it)"
        )
    })
}

fn positive(v: u64, key: &'static str) -> Result<()> {
    if v == 0 {
        return Err(error!(
            ErrorCode::KgConfig,
            format!("embedder.{key}"),
            "must be positive"
        ));
    }
    Ok(())
}

/// 2026-10-23: The longest `key_prefix` (a prefix names the key's shape, e.g. `sk-or-`, and so
/// matches the denylist by design; anything longer would be key material).
pub const KEY_PREFIX_MAX: usize = 16;

/// 2026-10-23: Resolve the merged layer.
pub fn resolve(m: &EmbedLayer) -> Result<Embedder> {
    let texts = [&m.endpoint, &m.model, &m.key_file, &m.key_env, &m.reason];
    if m.key_prefix
        .as_ref()
        .is_some_and(|p| p.len() > KEY_PREFIX_MAX)
    {
        return Err(error!(
            ErrorCode::KgConfig,
            "embedder.key_prefix",
            format!("is longer than {KEY_PREFIX_MAX} characters, so it is a key, not a prefix")
        ));
    }
    let header_vals = m.headers.iter().flatten().flat_map(|(k, v)| [k, v]);
    for t in texts.into_iter().flatten().chain(header_vals) {
        if let Some(p) = secret_pattern(t) {
            return Err(error!(
                ErrorCode::KgConfig,
                "embedder",
                format!("holds a credential ({p}); give a key_file or key_env reference")
            ));
        }
    }
    let Some(mode) = m.mode.as_deref() else {
        if *m == EmbedLayer::default() {
            return Ok(Embedder::Unavailable {
                reason: "no layer configures an embedder".into(),
            });
        }
        return Err(error!(
            ErrorCode::KgConfig,
            "embedder.mode", "is required when any embedding key is set"
        ));
    };
    match mode {
        "unavailable" => Ok(Embedder::Unavailable {
            reason: need(&m.reason, "reason")?,
        }),
        "http" => {
            let key = match (&m.key_file, &m.key_env) {
                (Some(path), None) => KeyRef::File {
                    path: path.clone(),
                    prefix: need(&m.key_prefix, "key_prefix")?,
                },
                (None, Some(name)) => KeyRef::Env { name: name.clone() },
                (Some(_), Some(_)) => {
                    return Err(error!(
                        ErrorCode::KgConfig,
                        "embedder.key_file", "and key_env are both set; set exactly one"
                    ));
                }
                (None, None) => {
                    return Err(error!(
                        ErrorCode::KgConfig,
                        "embedder.key_file", "or key_env is required"
                    ));
                }
            };
            let s = EmbedSettings {
                endpoint: need(&m.endpoint, "endpoint")?,
                model: need(&m.model, "model")?,
                dimensions: need(&m.dimensions, "dimensions")?,
                headers: need(&m.headers, "headers")?,
                key,
                timeout_ms: need(&m.timeout_ms, "timeout_ms")?,
                batch: need(&m.batch, "batch")?,
                max_attempts: need(&m.max_attempts, "max_attempts")?,
                backoff_ms: need(&m.backoff_ms, "backoff_ms")?,
                min_interval_ms: need(&m.min_interval_ms, "min_interval_ms")?,
                max_input_chars: need(&m.max_input_chars, "max_input_chars")?,
                input_type: need(&m.input_type, "input_type")?,
            };
            if s.input_type.query.is_empty() || s.input_type.document.is_empty() {
                return Err(error!(
                    ErrorCode::KgConfig,
                    "embedder.input_type", "names a non-empty query and document"
                ));
            }
            positive(u64::from(s.max_input_chars), "max_input_chars")?;
            positive(u64::from(s.dimensions), "dimensions")?;
            positive(u64::from(s.batch), "batch")?;
            positive(u64::from(s.max_attempts), "max_attempts")?;
            positive(s.timeout_ms, "timeout_ms")?;
            if !s.endpoint.starts_with("https://")
                && !s.endpoint.starts_with("http://127.0.0.1")
                && !s.endpoint.starts_with("http://localhost")
            {
                return Err(error!(
                    ErrorCode::KgConfig,
                    "embedder.endpoint", "must be https (or a loopback http)"
                ));
            }
            Ok(Embedder::Available(s))
        }
        other => Err(error!(
            ErrorCode::KgConfig,
            "embedder.mode",
            format!("`{other}` is not http or unavailable")
        )),
    }
}

/// 2026-10-23: The project config's node: `config:<path>` with the keys its `[embedder]` layer
/// sets (references only; a key is never in a layer, [`resolve`] refuses one).
pub fn project_node(b: &mut Builder, path: &str, text: &str, layer: &EmbedLayer) {
    let Ok(id) = KgId::new(Label::Config, path) else {
        return;
    };
    let mut p = BTreeMap::new();
    let set = serde_json::to_value(layer).unwrap_or(Value::Null);
    let set: serde_json::Map<String, Value> = set
        .as_object()
        .into_iter()
        .flatten()
        .filter(|(_, v)| !v.is_null())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    p.insert("embedder".into(), Value::Object(set));
    let line = text
        .lines()
        .position(|l| l.trim() == "[embedder]")
        .map_or(1, |i| i + 1);
    b.define(id, format!("{path}:{line}"), text, p);
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLOBAL: &str = r#"schema = "latlas-config/v1"
[embedder]
mode = "http"
endpoint = "https://openrouter.ai/api/v1/embeddings"
model = "global-model"
dimensions = 2048
headers = { "X-Title" = "latlas" }
key_file = "/k/openrouter.embedder.txt"
key_prefix = "sk-or-"
timeout_ms = 70000
batch = 64
max_attempts = 5
backoff_ms = 2000
min_interval_ms = 0
max_input_chars = 8000
input_type = { query = "query", document = "passage" }
"#;

    fn layer(t: &str) -> EmbedLayer {
        toml::from_str(t).unwrap()
    }

    #[test]
    fn the_project_layer_wins_per_key_and_provenance_says_so() {
        let g = parse_global(GLOBAL).unwrap();
        let p = layer("model = \"project-model\"\nbatch = 8\n");
        let (m, prov) = merge(&g, &p);
        let Embedder::Available(s) = resolve(&m).unwrap() else {
            panic!("must be available")
        };
        assert_eq!(
            (s.model.as_str(), s.batch, s.dimensions),
            ("project-model", 8, 2048)
        );
        assert_eq!(prov["model"], Layer::Project);
        assert_eq!(prov["endpoint"], Layer::Global);
        assert_eq!(
            s.key,
            KeyRef::File {
                path: "/k/openrouter.embedder.txt".into(),
                prefix: "sk-or-".into()
            }
        );
        let (m, prov) = merge(&g, &layer("key_env = \"OPENROUTER_KEY\"\n"));
        let Embedder::Available(s) = resolve(&m).unwrap() else {
            panic!("must be available")
        };
        assert_eq!(
            s.key,
            KeyRef::Env {
                name: "OPENROUTER_KEY".into()
            }
        );
        assert!(!prov.contains_key("key_file"));
    }

    #[test]
    fn no_embedder_anywhere_is_the_declared_unavailable_mode() {
        let (m, _) = merge(&EmbedLayer::default(), &EmbedLayer::default());
        assert!(matches!(resolve(&m).unwrap(), Embedder::Unavailable { .. }));
        let p = layer("mode = \"unavailable\"\nreason = \"P0\"\n");
        let (m, _) = merge(&parse_global(GLOBAL).unwrap(), &p);
        assert_eq!(
            resolve(&m).unwrap(),
            Embedder::Unavailable {
                reason: "P0".into()
            }
        );
    }

    #[test]
    fn a_named_embedder_missing_a_key_fails_with_that_key() {
        let without: String = GLOBAL
            .lines()
            .filter(|l| !l.starts_with("min_interval_ms"))
            .collect::<Vec<_>>()
            .join("\n");
        let (m, _) = merge(&parse_global(&without).unwrap(), &EmbedLayer::default());
        let e = resolve(&m).unwrap_err().to_string();
        assert!(e.contains("embedder.min_interval_ms"), "{e}");
        let (m, _) = merge(&EmbedLayer::default(), &layer("model = \"x\"\n"));
        assert!(
            resolve(&m)
                .unwrap_err()
                .to_string()
                .contains("embedder.mode")
        );
    }

    #[test]
    fn an_inline_key_or_two_key_references_are_refused() {
        let (m, _) = merge(
            &parse_global(GLOBAL).unwrap(),
            &layer("headers = { Authorization = \"Bearer sk-or-v1-abc\" }\n"),
        );
        assert!(resolve(&m).unwrap_err().to_string().contains("credential"));
        let both =
            layer("key_env = \"OPENROUTER_KEY\"\nkey_file = \"/k/f\"\nkey_prefix = \"sk-or-\"\n");
        let (m, _) = merge(&parse_global(GLOBAL).unwrap(), &both);
        assert!(resolve(&m).unwrap_err().to_string().contains("both set"));
        assert!(parse_global("schema = \"other\"\n").is_err());
        let (m, _) = merge(
            &parse_global(GLOBAL).unwrap(),
            &layer("key_file = \"/k/f\"\nkey_prefix = \"sk-or-v1-0123456789abcdef\"\n"),
        );
        assert!(resolve(&m).unwrap_err().to_string().contains("key_prefix"));
    }
}
