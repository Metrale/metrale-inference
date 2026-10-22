// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The trust check: a [`SigBlock`] over a payload, under a [`TrustPolicy`].
//!
//! Owner: blueprint.
//! Invariants (the order of [`verify_signed`], each step refusing with its own code):
//! 1. the block's kid selects a certificate in its chain (`BlueprintUntrusted`: unknown kid);
//! 2. the certificate's issuer is an anchor (`BlueprintUntrusted`);
//! 3. the certificate's signature verifies under that anchor (`BlueprintSignature`);
//! 4. only then is it decoded, and its kid must be the one that selected it;
//! 5. neither the key nor its issuer is revoked (`BlueprintRevoked`);
//! 6. `now` lies in its validity window (`BlueprintCertExpired`);
//! 7. its scope is the one needed (`BlueprintScope`);
//! 8. the payload signature verifies under the certified key (`BlueprintSignature`).

use alloc::string::String;
use alloc::vec::Vec;

use metrale_abstract_io::UnixMs;
use metrale_error::{Error, ErrorCode, error};
use serde::{Deserialize, Serialize};

use crate::anchors::{EMBEDDED_REVOKED, REVOCATIONS_FLOOR, TrustAnchors};
use crate::cert::{KeyCert, Revocations, SignedCert, SignedRevocations};
use crate::keys::{HybridSig, verify_hybrid};
use crate::names::{Kid, Purpose, Scope};

/// 2026-10-22: A signature as it travels: the signing kid, its certificate chain (one root-issued
/// certificate) and the hybrid signature.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigBlock {
    /// 2026-10-22: The signing key; selects its certificate in `chain`.
    pub kid: Kid,
    /// 2026-10-22: Root-issued certificates.
    pub chain: Vec<SignedCert>,
    /// 2026-10-22: The signature over the payload.
    pub sig: HybridSig,
}

/// 2026-10-22: A verified revocation state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Revoked {
    serial: u64,
    kids: Vec<Kid>,
}

impl Revoked {
    /// 2026-10-22: The state compiled into this binary.
    pub fn embedded() -> Result<Self, Error> {
        let kids = EMBEDDED_REVOKED
            .iter()
            .map(|k| Kid::new(k))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            serial: REVOCATIONS_FLOOR,
            kids,
        })
    }

    /// 2026-10-22: The revocation serial.
    pub fn serial(&self) -> u64 {
        self.serial
    }

    fn contains(&self, kid: &Kid) -> bool {
        self.kids.contains(kid)
    }
}

/// 2026-10-22: Verify a signed revocation list against `anchors`, refusing a serial below
/// `floor` (a replayed older list).
pub fn verify_revocations(
    anchors: &TrustAnchors,
    signed: &SignedRevocations,
    floor: u64,
) -> Result<Revoked, Error> {
    let anchor = anchors.get(&signed.issuer).ok_or_else(|| {
        error!(
            ErrorCode::BlueprintUntrusted,
            String::from(signed.issuer.as_str()),
            "no anchor has this kid"
        )
    })?;
    verify_hybrid(
        &anchor.keys,
        Purpose::Revocations,
        &signed.body,
        &signed.sig,
        "revocations",
        &signed.issuer,
    )?;
    let list: Revocations = postcard::from_bytes(&signed.body)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, "revocations", "postcard"))?;
    if list.serial < floor {
        return Err(error!(
            ErrorCode::BlueprintRollback,
            "revocations", list.serial, floor
        ));
    }
    Ok(Revoked {
        serial: list.serial,
        kids: list.revoked,
    })
}

/// 2026-10-22: What a verification trusts and when it runs.
#[derive(Clone, Copy, Debug)]
pub struct TrustPolicy<'a> {
    /// 2026-10-22: The trusted roots.
    pub anchors: &'a TrustAnchors,
    /// 2026-10-22: The revocation state.
    pub revoked: &'a Revoked,
    /// 2026-10-22: The instant certificates are judged at (from the caller's Clock port).
    pub now: UnixMs,
}

/// 2026-10-22: Who signed, once verified.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Verified {
    /// 2026-10-22: The signing key.
    pub kid: Kid,
    /// 2026-10-22: The root that certified it.
    pub anchor: Kid,
    /// 2026-10-22: The signing key's certificate.
    pub cert: KeyCert,
}

/// 2026-10-22: Verify `block` over `msg` for `purpose`, requiring a key of `scope`. See the
/// module invariants for the order.
pub fn verify_signed(
    policy: &TrustPolicy<'_>,
    block: &SigBlock,
    scope: Scope,
    purpose: Purpose,
    msg: &[u8],
    what: &'static str,
) -> Result<Verified, Error> {
    let kid = || String::from(block.kid.as_str());
    let leaf = block
        .chain
        .iter()
        .find(|c| c.subject == block.kid)
        .ok_or_else(|| {
            error!(
                ErrorCode::BlueprintUntrusted,
                kid(),
                "unknown kid: no certificate in the chain names it"
            )
        })?;
    let anchor = policy.anchors.get(&leaf.issuer).ok_or_else(|| {
        error!(
            ErrorCode::BlueprintUntrusted,
            kid(),
            "its issuer is not a trust anchor"
        )
    })?;
    verify_hybrid(
        &anchor.keys,
        Purpose::KeyCert,
        &leaf.cert,
        &leaf.sig,
        "certificate",
        &leaf.issuer,
    )?;
    let cert: KeyCert = postcard::from_bytes(&leaf.cert)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, "certificate", "postcard"))?;
    if cert.kid != block.kid {
        return Err(error!(
            ErrorCode::BlueprintUntrusted,
            kid(),
            "the certificate certifies another kid"
        ));
    }
    for k in [&cert.kid, &leaf.issuer] {
        if policy.revoked.contains(k) {
            return Err(error!(
                ErrorCode::BlueprintRevoked,
                String::from(k.as_str()),
                policy.revoked.serial()
            ));
        }
    }
    let now = policy.now.0;
    if now < cert.not_before_ms || now >= cert.not_after_ms {
        return Err(error!(
            ErrorCode::BlueprintCertExpired,
            kid(),
            cert.not_before_ms,
            cert.not_after_ms,
            now
        ));
    }
    if cert.scope != scope {
        return Err(error!(
            ErrorCode::BlueprintScope,
            kid(),
            cert.scope.name(),
            scope.name()
        ));
    }
    verify_hybrid(&cert.keys, purpose, msg, &block.sig, what, &block.kid)?;
    Ok(Verified {
        kid: block.kid.clone(),
        anchor: leaf.issuer.clone(),
        cert,
    })
}
