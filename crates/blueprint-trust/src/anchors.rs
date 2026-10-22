// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Trust anchors: the root public keys this binary trusts, and the revocation
//! state it was built with. This file is the only one a root rotation changes.
//!
//! Owner: blueprint.
//! Invariants:
//! - [`ANCHORS`] is empty until the owner holds a root key ceremony (DESIGN.md §11 Q1). With no
//!   compiled anchor, every production verification is refused (`BlueprintUntrusted`); a caller
//!   that supplies an anchor itself runs the declared, reported mode `trust = dev-anchor`.
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

/// 2026-10-22: The compiled root keys. Empty: no production root exists yet.
pub const ANCHORS: &[CompiledAnchor] = &[];

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
