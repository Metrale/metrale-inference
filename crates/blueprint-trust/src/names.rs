// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The closed names of the trust model: key ids, suites, scopes and the signing
//! contexts.
//!
//! Owner: blueprint.
//! Invariants:
//! - A [`Kid`] is 1..=64 bytes of `[a-z0-9-]` (metrale-auth's `signing_kid` shape), checked on
//!   construction and on decode.
//! - Each [`Purpose`] has its own versioned FIPS 204 context; a signature made for one purpose
//!   never verifies as another.

use alloc::string::String;
use core::fmt;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

/// 2026-10-22: A key id, `[a-z0-9-]{1,64}`: scope and month, e.g. `kbp-blueprints-2026-10`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Kid(String);

impl Kid {
    /// 2026-10-22: The kid `s`, refused unless it is 1..=64 bytes of `[a-z0-9-]`.
    pub fn new(s: &str) -> Result<Self, Error> {
        let ok = (1..=64).contains(&s.len())
            && s.bytes()
                .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-');
        if ok {
            Ok(Self(String::from(s)))
        } else {
            Err(error!(
                ErrorCode::BlueprintKey,
                String::from(s),
                "a kid is 1-64 bytes of [a-z0-9-]"
            ))
        }
    }

    /// 2026-10-22: The kid's text.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for Kid {
    type Error = Error;
    fn try_from(s: String) -> Result<Self, Error> {
        Kid::new(&s)
    }
}

impl From<Kid> for String {
    fn from(k: Kid) -> String {
        k.0
    }
}

impl fmt::Display for Kid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// 2026-10-22: The algorithm pair. Root keys are ML-DSA-87, signing keys ML-DSA-65; both carry
/// an Ed25519 half. A new suite (e.g. libcrux) is a new variant, never a format change.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Suite {
    /// 2026-10-22: ML-DSA-65 (1952-byte key, 3309-byte signature) and Ed25519.
    MlDsa65Ed25519,
    /// 2026-10-22: ML-DSA-87 (2592-byte key, 4627-byte signature) and Ed25519.
    MlDsa87Ed25519,
}

impl Suite {
    /// 2026-10-22: The suite's display name.
    pub const fn name(self) -> &'static str {
        match self {
            Suite::MlDsa65Ed25519 => "mldsa65+ed25519",
            Suite::MlDsa87Ed25519 => "mldsa87+ed25519",
        }
    }

    /// 2026-10-22: The name of the ML-DSA half.
    pub const fn mldsa_name(self) -> &'static str {
        match self {
            Suite::MlDsa65Ed25519 => "ML-DSA-65",
            Suite::MlDsa87Ed25519 => "ML-DSA-87",
        }
    }

    /// 2026-10-22: The suite a display name spells.
    pub fn parse(s: &str) -> Option<Self> {
        [Suite::MlDsa65Ed25519, Suite::MlDsa87Ed25519]
            .into_iter()
            .find(|v| v.name() == s)
    }
}

/// 2026-10-22: What a certified key may sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Scope {
    /// 2026-10-22: Blueprint bodies.
    Blueprints,
    /// 2026-10-22: Kernel artifact manifests.
    Artifacts,
    /// 2026-10-22: The registry index.
    Index,
}

impl Scope {
    /// 2026-10-22: The scope's spelling.
    pub const fn name(self) -> &'static str {
        match self {
            Scope::Blueprints => "blueprints",
            Scope::Artifacts => "artifacts",
            Scope::Index => "index",
        }
    }

    /// 2026-10-22: The scope a spelling names.
    pub fn parse(s: &str) -> Option<Self> {
        [Scope::Blueprints, Scope::Artifacts, Scope::Index]
            .into_iter()
            .find(|v| v.name() == s)
    }
}

/// 2026-10-22: What a signature is for; each purpose is one versioned FIPS 204 context string.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purpose {
    /// 2026-10-22: A blueprint body.
    KbpBody,
    /// 2026-10-22: A sealed manifest (the per-customer sealed section of a `.kbp`).
    KbpSealed,
    /// 2026-10-22: A kernel artifact manifest.
    KernelArtifact,
    /// 2026-10-22: The registry index.
    KernelIndex,
    /// 2026-10-22: A key certificate (root only).
    KeyCert,
    /// 2026-10-22: The revocation list (root only).
    Revocations,
}

impl Purpose {
    /// 2026-10-22: The context string: ML-DSA's `ctx`, and the first part of the Ed25519
    /// message.
    pub const fn context(self) -> &'static [u8] {
        match self {
            Purpose::KbpBody => b"metrale/kbp-body/v1",
            Purpose::KbpSealed => b"metrale/kbp-sealed/v1",
            Purpose::KernelArtifact => b"metrale/kernel-artifact/v1",
            Purpose::KernelIndex => b"metrale/kernel-index/v1",
            Purpose::KeyCert => b"metrale/key-cert/v1",
            Purpose::Revocations => b"metrale/revocations/v1",
        }
    }
}
