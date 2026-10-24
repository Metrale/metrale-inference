// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Content identities: [`Sha256`] and the ids built from it.
//!
//! Owner: blueprint.
//! Invariants:
//! - [`ArtifactId`] is SHA-256 of the raw module bytes (never of a compressed form).
//! - [`BlueprintId`] is `sha256("kbp-body\0" ‖ body)` over the body section exactly as signed,
//!   so lite and full files of one build share it.
//! - Hex is lower case; [`Sha256::parse_hex`] accepts exactly 64 lower-case hex digits.

use alloc::string::String;
use core::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256 as Hasher};

/// 2026-10-22: A SHA-256 digest.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Sha256(pub [u8; 32]);

impl Sha256 {
    /// 2026-10-22: The digest of the concatenation of `parts`.
    pub fn of(parts: &[&[u8]]) -> Self {
        let mut h = Hasher::new();
        for p in parts {
            h.update(p);
        }
        Self(h.finalize().into())
    }

    /// 2026-10-22: Lower-case hex.
    pub fn hex(&self) -> String {
        use fmt::Write as _;
        let mut s = String::with_capacity(64);
        for b in self.0 {
            let _ = write!(s, "{b:02x}");
        }
        s
    }

    /// 2026-10-22: Parse 64 lower-case hex digits.
    pub fn parse_hex(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 64
            || !b
                .iter()
                .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(c))
        {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u8::from_str_radix(core::str::from_utf8(&b[2 * i..2 * i + 2]).ok()?, 16).ok()?;
        }
        Some(Self(out))
    }
}

impl fmt::Debug for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.hex())
    }
}

impl fmt::Display for Sha256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.hex())
    }
}

macro_rules! id_newtype {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub struct $name(pub Sha256);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                fmt::Display::fmt(&self.0, f)
            }
        }
    };
}

id_newtype!(
    /// 2026-10-22: SHA-256 of one compiled module's raw bytes (PTX text on NVIDIA).
    ArtifactId
);
id_newtype!(
    /// 2026-10-22: The identity of one signed blueprint body.
    BlueprintId
);
id_newtype!(
    /// 2026-10-22: The identity of a blueprint line across revisions.
    LineageId
);

impl ArtifactId {
    /// 2026-10-22: The id of `bytes`.
    pub fn of(bytes: &[u8]) -> Self {
        Self(Sha256::of(&[bytes]))
    }
}

impl BlueprintId {
    /// 2026-10-22: The id of a body section, as signed.
    pub fn of_body(body: &[u8]) -> Self {
        Self(Sha256::of(&[b"kbp-body\0", body]))
    }
}

impl LineageId {
    /// 2026-10-22: `sha256("kbp-lineage\0" ‖ lv(repo) ‖ lv(class) ‖ lv(name))`: length-prefixed
    /// so no two triples collide.
    pub fn of(repo: &str, class: &str, name: &str) -> Self {
        let lv =
            metrale_blueprint_trust::lv_cat(&[repo.as_bytes(), class.as_bytes(), name.as_bytes()]);
        Self(Sha256::of(&[b"kbp-lineage\0", &lv]))
    }
}

/// 2026-10-22: An incremental SHA-256, for inputs read in pieces (weight shards).
#[derive(Clone, Default)]
pub struct Sha256Stream(Hasher);

impl Sha256Stream {
    /// 2026-10-22: Feed `bytes`.
    pub fn update(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    /// 2026-10-22: The digest of everything fed.
    pub fn finish(self) -> Sha256 {
        Sha256(self.0.finalize().into())
    }
}
