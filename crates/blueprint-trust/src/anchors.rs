// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Trust anchors: the root public keys this binary trusts, and the revocation
//! state it was built with. This file is the only one a root rotation changes.
//!
//! Owner: blueprint.
//! Invariants:
//! - [`ANCHORS`] holds the production root `kbp-root-2026` (key ceremony of 2026-10-22). Its
//!   bytes are the `met blueprint key anchor` file committed at `anchors/kbp-root-2026.anchor`
//!   (SSOT), decoded at compile time: a file of any other shape fails the build. A caller that
//!   supplies an anchor itself runs the declared, reported mode `trust = dev-anchor`.
//! - [`TrustAnchors`] refuses two anchors with one kid.

use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

use crate::keys::PublicKeys;
use crate::names::{Kid, Suite};

/// 2026-10-22: A root key compiled into the binary.
#[derive(Clone, Copy, Debug)]
pub struct CompiledAnchor {
    /// 2026-10-22: Its kid, e.g. `kbp-root-2026`.
    pub kid: &'static str,
    /// 2026-10-22: Its suite.
    pub suite: Suite,
    /// 2026-10-22: The encoded ML-DSA verifying key.
    pub mldsa: &'static [u8],
    /// 2026-10-22: The Ed25519 verifying key.
    pub ed25519: [u8; 32],
}

/// 2026-10-22: The production root's public anchor file (postcard [`Anchor`], as written by
/// `met blueprint key anchor`).
pub const ROOT_2026_ANCHOR_FILE: &[u8] = include_bytes!("../anchors/kbp-root-2026.anchor");

/// 2026-10-22: The compiled root keys.
pub const ANCHORS: &[CompiledAnchor] = &[CompiledAnchor::from_anchor_file(ROOT_2026_ANCHOR_FILE)];

/// 2026-10-22: One postcard varint (LEB128, at most `u32`) at the front of `b`, and the rest.
#[allow(clippy::cast_lossless)] // 2026-10-22: `usize::from` is not callable in a const fn.
const fn varint(b: &[u8]) -> (usize, &[u8]) {
    let mut v = 0usize;
    let mut i = 0;
    while i < 5 && i < b.len() {
        let x = b[i];
        v |= ((x & 0x7f) as usize) << (7 * i);
        if x & 0x80 == 0 {
            return (v, b.split_at(i + 1).1);
        }
        i += 1;
    }
    panic!("anchor file: truncated or over-long varint");
}

impl CompiledAnchor {
    /// 2026-10-22: Decode an anchor file at compile time: the kid (varint length + UTF-8), the
    /// suite (its one-byte variant index), the ML-DSA key (varint length + bytes) and the 32
    /// Ed25519 bytes, nothing after. Any other shape is a build failure; the
    /// `compiled_anchors_equal_their_files` test checks the result against the postcard decode.
    pub const fn from_anchor_file(file: &'static [u8]) -> Self {
        let (kid_len, rest) = varint(file);
        let Some((kid, rest)) = rest.split_at_checked(kid_len) else {
            panic!("anchor file: truncated kid");
        };
        let Ok(kid) = core::str::from_utf8(kid) else {
            panic!("anchor file: kid is not UTF-8");
        };
        let Some((&suite, rest)) = rest.split_first() else {
            panic!("anchor file: no suite");
        };
        let suite = match suite {
            0 => Suite::MlDsa65Ed25519,
            1 => Suite::MlDsa87Ed25519,
            _ => panic!("anchor file: unknown suite"),
        };
        let (mldsa_len, rest) = varint(rest);
        let Some((mldsa, ed)) = rest.split_at_checked(mldsa_len) else {
            panic!("anchor file: truncated ML-DSA key");
        };
        let (Some(ed25519), 32) = (ed.first_chunk::<32>(), ed.len()) else {
            panic!("anchor file: the Ed25519 key is not the last 32 bytes");
        };
        Self {
            kid,
            suite,
            mldsa,
            ed25519: *ed25519,
        }
    }
}

/// 2026-10-22: The revocation serial this binary was built with; a list below it is a rollback.
pub const REVOCATIONS_FLOOR: u64 = 0;

/// 2026-10-22: The kids revoked as of [`REVOCATIONS_FLOOR`].
pub const EMBEDDED_REVOKED: &[&str] = &[];

/// 2026-10-22: A root key: its kid and public half.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Anchor {
    /// 2026-10-22: The root's kid.
    pub kid: Kid,
    /// 2026-10-22: Its public keys.
    pub keys: PublicKeys,
}

/// 2026-10-22: The anchors a verification trusts, at most one per kid.
#[derive(Clone, Debug, Default)]
pub struct TrustAnchors {
    list: Vec<Anchor>,
}

impl TrustAnchors {
    /// 2026-10-22: Trust exactly `anchors`; two with one kid are refused.
    pub fn new(anchors: Vec<Anchor>) -> Result<Self, Error> {
        for (i, a) in anchors.iter().enumerate() {
            if anchors[..i].iter().any(|b| b.kid == a.kid) {
                return Err(error!(
                    ErrorCode::BlueprintKey,
                    String::from(a.kid.as_str()),
                    "is listed twice as an anchor"
                ));
            }
        }
        Ok(Self { list: anchors })
    }

    /// 2026-10-22: The anchors compiled into this binary ([`ANCHORS`]).
    pub fn compiled() -> Result<Self, Error> {
        let mut list = Vec::with_capacity(ANCHORS.len());
        for a in ANCHORS {
            list.push(Anchor {
                kid: Kid::new(a.kid)?,
                keys: PublicKeys {
                    suite: a.suite,
                    mldsa: a.mldsa.to_vec(),
                    ed25519: a.ed25519,
                },
            });
        }
        Self::new(list)
    }

    /// 2026-10-22: The anchor with `kid`.
    pub fn get(&self, kid: &Kid) -> Option<&Anchor> {
        self.list.iter().find(|a| &a.kid == kid)
    }

    /// 2026-10-22: Whether no anchor is trusted.
    pub fn is_empty(&self) -> bool {
        self.list.is_empty()
    }
}
