// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: What a root key signs: [`KeyCert`] for a scoped signing key and the
//! [`Revocations`] list. Each travels as its postcard bytes plus the hybrid signature over them,
//! so a verifier checks the signature before it decodes anything.
//!
//! Owner: blueprint.
//! Invariants: the signed bytes are the postcard encoding as written; nothing re-encodes them.

use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

use crate::keys::{HybridSig, PublicKeys, Signer};
use crate::names::{Kid, Purpose, Scope};

/// 2026-10-22: A signing key's certificate, issued by a root.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyCert {
    /// 2026-10-22: The certified key's id.
    pub kid: Kid,
    /// 2026-10-22: The certified key's public half.
    pub keys: PublicKeys,
    /// 2026-10-22: What it may sign.
    pub scope: Scope,
    /// 2026-10-22: First valid instant, Unix milliseconds (inclusive).
    pub not_before_ms: i64,
    /// 2026-10-22: End of validity, Unix milliseconds (exclusive).
    pub not_after_ms: i64,
    /// 2026-10-22: The issuer's certificate serial.
    pub serial: u64,
}

/// 2026-10-22: A certificate as it travels: the subject kid in clear (it selects the
/// certificate; it is checked against the decoded one after verification), the issuer, the
/// signed bytes and their signature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedCert {
    /// 2026-10-22: The certified kid, as the signed bytes state it.
    pub subject: Kid,
    /// 2026-10-22: The root that signed it.
    pub issuer: Kid,
    /// 2026-10-22: Postcard [`KeyCert`].
    pub cert: Vec<u8>,
    /// 2026-10-22: The issuer's signature over `cert` ([`Purpose::KeyCert`]).
    pub sig: HybridSig,
}

/// 2026-10-22: The revoked key ids, with a serial that only grows.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Revocations {
    /// 2026-10-22: Monotonic; a reader refuses one below its floor.
    pub serial: u64,
    /// 2026-10-22: Revoked kids.
    pub revoked: Vec<Kid>,
}

/// 2026-10-22: A revocation list as it travels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedRevocations {
    /// 2026-10-22: The root that signed it.
    pub issuer: Kid,
    /// 2026-10-22: Postcard [`Revocations`].
    pub body: Vec<u8>,
    /// 2026-10-22: The issuer's signature ([`Purpose::Revocations`]).
    pub sig: HybridSig,
}

fn encode<T: Serialize>(what: &'static str, v: &T) -> Result<Vec<u8>, Error> {
    postcard::to_allocvec(v).map_err(|_| error!(ErrorCode::BlueprintKey, what, "does not encode"))
}

/// 2026-10-22: `root` certifies `cert`.
pub fn certify(root: &Signer, cert: &KeyCert) -> Result<SignedCert, Error> {
    let bytes = encode("certificate", cert)?;
    let sig = root.sign(Purpose::KeyCert, &bytes)?;
    Ok(SignedCert {
        subject: cert.kid.clone(),
        issuer: root.kid().clone(),
        cert: bytes,
        sig,
    })
}

/// 2026-10-22: `root` signs the revocation list.
pub fn sign_revocations(root: &Signer, list: &Revocations) -> Result<SignedRevocations, Error> {
    let body = encode("revocations", list)?;
    let sig = root.sign(Purpose::Revocations, &body)?;
    Ok(SignedRevocations {
        issuer: root.kid().clone(),
        body,
        sig,
    })
}
