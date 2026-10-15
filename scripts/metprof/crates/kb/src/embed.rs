// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the embedding port and the one embedder this crate ships.
//!
//! Every vector carries the id of the space it lives in: `kkb-embed/1` (the
//! hashed embedder below) or `openai_compatible:<host>:<model>@<dims>` (a
//! dense model, whose client lives outside this crate behind the same port).
//! Vectors of two spaces are never compared or stored together: every
//! operation that meets two spaces fails (ADDENDUM-EMBEDDINGS E1, E2).
//!
//! `kkb-embed/1` (DESIGN §2.7) is deterministic and model-free, identical on
//! the host and in WASM: the text is NFC-normalised and lower-cased; words
//! are the alphanumeric runs; a whitespace-separated name that the separators
//! `_`, `::` or `/` split is also kept whole; features are word unigrams,
//! word bigrams and character trigrams inside words of four or more
//! characters; each feature is FNV-1a 64 hashed to index `h % 256` with sign
//! bit 63, weighted `ln(1 + tf)`, and the vector is L2-normalised.

use std::collections::BTreeMap;
use std::fmt;

use unicode_normalization::UnicodeNormalization;

use crate::{Error, Result, bail};

pub const HASHED_SPACE: &str = "kkb-embed/1";
pub const HASHED_DIM: usize = 256;
/// The LatticeDB collection name of the hashed space.
pub const HASHED_COLLECTION: &str = "kkb@embed1";

/// The one dense kind a configuration lowers to (`openrouter` is sugar over it).
pub const DENSE_KIND: &str = "openai_compatible";

/// An embedding space id: `kkb-embed/1`, or `<kind>:<host>:<model>@<dims>`
/// for a dense model (ADDENDUM-EMBEDDINGS E2; `<host>` is the endpoint's host,
/// or `openrouter`).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Space {
    id: String,
    dims: usize,
}

impl Space {
    pub fn hashed() -> Space {
        Space {
            id: HASHED_SPACE.to_string(),
            dims: HASHED_DIM,
        }
    }

    /// A dense space; `host` is lower-case, bracketed when it is IPv6.
    pub fn dense(host: &str, model: &str, dims: usize) -> Result<Space> {
        let host_ok = if let Some(inner) = host.strip_prefix('[') {
            inner.ends_with(']') && inner.len() > 1
        } else {
            !host.is_empty() && !host.contains([':', '@', '/', '['])
        };
        if !host_ok
            || host
                .chars()
                .any(|c| c.is_whitespace() || c.is_ascii_uppercase())
        {
            bail!("dense space: host {host:?} is not a lower-case host name");
        }
        if model.is_empty() || model.chars().any(char::is_whitespace) || dims == 0 {
            bail!("dense space: model {model:?} must be non-empty without blanks, dims > 0");
        }
        Ok(Space {
            id: format!("{DENSE_KIND}:{host}:{model}@{dims}"),
            dims,
        })
    }

    /// A space from its id.
    pub fn parse(id: &str) -> Result<Space> {
        if id == HASHED_SPACE {
            return Ok(Space::hashed());
        }
        let unknown = || Error(format!("unknown embedding space {id:?}"));
        let rest = id
            .strip_prefix(DENSE_KIND)
            .and_then(|r| r.strip_prefix(':'))
            .ok_or_else(unknown)?;
        let host_end = if rest.starts_with('[') {
            rest.find(']').map(|i| i + 1)
        } else {
            rest.find(':')
        }
        .ok_or_else(unknown)?;
        let (host, tail) = rest.split_at(host_end);
        let (model, dims) = tail
            .strip_prefix(':')
            .and_then(|t| t.rsplit_once('@'))
            .ok_or_else(unknown)?;
        let dims = dims
            .parse::<usize>()
            .map_err(|_| Error(format!("embedding space {id:?}: dims is not a number")))?;
        let space = Space::dense(host, model, dims)?;
        if space.id != id {
            return Err(unknown());
        }
        Ok(space)
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Fail unless `other` is this space.
    pub fn require(&self, other: &Space) -> Result<()> {
        if self != other {
            bail!(
                "embedding spaces differ: {:?} vs {:?}; vectors of two spaces never mix",
                self.id,
                other.id
            );
        }
        Ok(())
    }
}

impl fmt::Display for Space {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.id)
    }
}

/// A unit vector and the space it lives in.
#[derive(Debug, Clone, PartialEq)]
pub struct Embedding {
    pub space: Space,
    pub vector: Vec<f32>,
}

impl Embedding {
    /// Cosine similarity; fails when the spaces differ.
    pub fn cosine(&self, other: &Embedding) -> Result<f32> {
        self.space.require(&other.space)?;
        let dot: f64 = self
            .vector
            .iter()
            .zip(&other.vector)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let na = norm(&self.vector);
        let nb = norm(&other.vector);
        if na == 0.0 || nb == 0.0 {
            bail!("cosine of a zero vector");
        }
        Ok((dot / (na * nb)) as f32)
    }
}

fn norm(v: &[f32]) -> f64 {
    v.iter()
        .map(|x| f64::from(*x) * f64::from(*x))
        .sum::<f64>()
        .sqrt()
}

/// The embedding port: texts in, one vector per text out, all in `space()`.
pub trait Embedder {
    fn space(&self) -> &Space;
    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>>;
}

/// `kkb-embed/1`.
pub struct Hashed {
    space: Space,
}

impl Hashed {
    pub fn new() -> Hashed {
        Hashed {
            space: Space::hashed(),
        }
    }
}

impl Default for Hashed {
    fn default() -> Self {
        Hashed::new()
    }
}

impl Embedder for Hashed {
    fn space(&self) -> &Space {
        &self.space
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Embedding>> {
        texts
            .iter()
            .map(|t| {
                Ok(Embedding {
                    space: self.space.clone(),
                    vector: hashed(t)?,
                })
            })
            .collect()
    }
}

/// FNV-1a, 64 bit.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn is_sep(c: char) -> bool {
    !c.is_alphanumeric()
}

/// The features of `text`, with their term frequencies.
pub fn features(text: &str) -> BTreeMap<String, u32> {
    let text: String = text.nfc().collect::<String>().to_lowercase();
    let mut tf: BTreeMap<String, u32> = BTreeMap::new();
    let mut add = |f: String| *tf.entry(f).or_insert(0) += 1;
    for chunk in text.split_whitespace() {
        let name = chunk.trim_matches(is_sep);
        let parts = name.split(is_sep).filter(|w| !w.is_empty()).count();
        if parts >= 2 && (name.contains('_') || name.contains("::") || name.contains('/')) {
            add(format!("n:{name}"));
        }
    }
    let words: Vec<&str> = text.split(is_sep).filter(|w| !w.is_empty()).collect();
    for w in &words {
        add(format!("w:{w}"));
        let chars: Vec<char> = w.chars().collect();
        if chars.len() >= 4 {
            for t in chars.windows(3) {
                add(format!("t:{}", t.iter().collect::<String>()));
            }
        }
    }
    for pair in words.windows(2) {
        add(format!("b:{} {}", pair[0], pair[1]));
    }
    tf
}

/// The `kkb-embed/1` vector of `text`; a text with no feature fails (no point
/// ever carries a zero vector).
pub fn hashed(text: &str) -> Result<Vec<f32>> {
    let mut v = [0f64; HASHED_DIM];
    for (f, n) in features(text) {
        let h = fnv1a64(f.as_bytes());
        let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
        v[(h % HASHED_DIM as u64) as usize] += sign * f64::from(n).ln_1p();
    }
    let n = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if n == 0.0 {
        bail!("kkb-embed/1: {text:?} has no feature to embed (or its features cancel)");
    }
    Ok(v.iter().map(|x| (x / n) as f32).collect())
}
