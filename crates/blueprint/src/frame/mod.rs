// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `.kbp` container: write an unsigned file, sign it, open (verify, then
//! decode) and convert between lite and full.
//!
//! Owner: blueprint.
//! Invariants:
//! - [`open`] runs: header → body hash → certificate chain → both signatures → only then any
//!   postcard decode of the body, locator or payload. Nothing but the fixed-width header is
//!   interpreted before the signature verifies (the signature block itself is decoded to be
//!   checked).
//! - [`rewrite`] copies the body and signature bytes verbatim and replaces only the locator
//!   and payload sections, so lite and full share the [`BlueprintId`] and the signature.
//! - The body is canonical: a decoded body re-encodes to the bytes it came from.

mod header;
mod locator;

use alloc::vec::Vec;

use metrale_blueprint_trust::{
    Purpose, Scope, SigBlock, SignedCert, Signer, TrustPolicy, Verified, verify_signed,
};
use metrale_error::{Error, ErrorCode, error};
use serde::Serialize;
use serde::de::DeserializeOwned;

pub use header::{
    FLAG_COMPRESSED, FLAG_FULL, HEADER_LEN, Header, KBP_FORMAT, MAX_ARTIFACT, MAX_BODY,
    MAX_BODY_INFLATED, MAX_LOCATOR, MAX_PAYLOAD, MAX_SIG, Sections,
};
pub use locator::{InlineArtifact, KernelSource, Unsigned};

use crate::ids::{BlueprintId, Sha256};
use crate::types::{BODY_SCHEMA, Blueprint};

/// 2026-10-22: The deflate level every writer uses; inflating does not depend on it.
const DEFLATE_LEVEL: u8 = 9;

pub(crate) fn encode<T: Serialize>(what: &'static str, v: &T) -> Result<Vec<u8>, Error> {
    postcard::to_allocvec(v)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, what, "does not encode"))
}

pub(crate) fn decode<T: DeserializeOwned>(what: &'static str, bytes: &[u8]) -> Result<T, Error> {
    match postcard::take_from_bytes::<T>(bytes) {
        Ok((v, [])) => Ok(v),
        Ok(_) => Err(error!(ErrorCode::BlueprintDecode, what, "trailing bytes")),
        Err(_) => Err(error!(ErrorCode::BlueprintDecode, what, "postcard")),
    }
}

pub(crate) fn deflate(bytes: &[u8]) -> Vec<u8> {
    miniz_oxide::deflate::compress_to_vec(bytes, DEFLATE_LEVEL)
}

pub(crate) fn inflate(what: &'static str, bytes: &[u8], limit: u64) -> Result<Vec<u8>, Error> {
    let lim = usize::try_from(limit).unwrap_or(usize::MAX);
    miniz_oxide::inflate::decompress_to_vec_with_limit(bytes, lim).map_err(|_| {
        error!(
            ErrorCode::BlueprintDecode,
            what, "deflate stream is corrupt or over its cap"
        )
    })
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

fn assemble(flags: u16, body: &[u8], locator: &[u8], payload: &[u8], sig: &[u8]) -> Vec<u8> {
    let h = Header {
        flags,
        body_len: len64(body.len()),
        body_sha256: Sha256::of(&[body]).0,
        locator_len: len64(locator.len()),
        payload_len: len64(payload.len()),
        sig_len: len64(sig.len()),
    };
    let mut out =
        Vec::with_capacity(HEADER_LEN + body.len() + locator.len() + payload.len() + sig.len());
    out.extend_from_slice(&h.write());
    for s in [body, locator, payload, sig] {
        out.extend_from_slice(s);
    }
    out
}

fn full_flag(payload: &[InlineArtifact]) -> u16 {
    if payload.is_empty() { 0 } else { FLAG_FULL }
}

/// 2026-10-22: An unsigned file: the deflated canonical body, the locator table and payload.
pub fn encode_unsigned(
    bp: &Blueprint,
    unsigned: &Unsigned,
    payload: &[InlineArtifact],
) -> Result<Vec<u8>, Error> {
    if unsigned.sources.len() != bp.kernels.modules.len() {
        return Err(error!(
            ErrorCode::BlueprintFrame,
            "the locator table must have one source per module"
        ));
    }
    let body = deflate(&encode("body", bp)?);
    let flags = FLAG_COMPRESSED | full_flag(payload);
    Ok(assemble(
        flags,
        &body,
        &encode("locator", unsigned)?,
        &encode("payload", &payload)?,
        &[],
    ))
}

fn check_body_hash(h: &Header, body: &[u8]) -> Result<(), Error> {
    let found = Sha256::of(&[body]);
    if found.0 != h.body_sha256 {
        return Err(error!(
            ErrorCode::BlueprintBodyHash,
            Sha256(h.body_sha256).hex(),
            found.hex()
        ));
    }
    Ok(())
}

/// 2026-10-22: Sign `file` (signed or not) with `signer`, whose certificate chain is `chain`.
/// The body hash is checked first; the old signature block, if any, is replaced.
pub fn sign(file: &[u8], signer: &Signer, chain: Vec<SignedCert>) -> Result<Vec<u8>, Error> {
    let (h, s) = Header::parse(file)?;
    let body = &file[s.body.clone()];
    check_body_hash(&h, body)?;
    let block = SigBlock {
        kid: signer.kid().clone(),
        chain,
        sig: signer.sign(Purpose::KbpBody, &h.signed_message(body))?,
    };
    Ok(assemble(
        h.flags,
        body,
        &file[s.locator],
        &file[s.payload],
        &encode("signature block", &block)?,
    ))
}

fn verify_body(
    policy: &TrustPolicy<'_>,
    h: &Header,
    body: &[u8],
    sig: &[u8],
) -> Result<Verified, Error> {
    if sig.is_empty() {
        return Err(error!(
            ErrorCode::BlueprintSignature,
            "body", "hybrid", "nobody (the file is unsigned)"
        ));
    }
    let block: SigBlock = decode("signature block", sig)?;
    verify_signed(
        policy,
        &block,
        Scope::Blueprints,
        Purpose::KbpBody,
        &h.signed_message(body),
        "body",
    )
}

/// 2026-10-22: A verified, decoded file.
#[derive(Clone, Debug)]
pub struct Opened {
    /// 2026-10-22: The body's identity.
    pub id: BlueprintId,
    /// 2026-10-22: The header.
    pub header: Header,
    /// 2026-10-22: The body.
    pub blueprint: Blueprint,
    /// 2026-10-22: The locator table and host resources.
    pub unsigned: Unsigned,
    /// 2026-10-22: Inline artifacts.
    pub payload: Vec<InlineArtifact>,
    /// 2026-10-22: Who signed it.
    pub signer: Verified,
}

fn decode_body(h: &Header, body: &[u8]) -> Result<Blueprint, Error> {
    let raw = if h.flags & FLAG_COMPRESSED != 0 {
        inflate("body", body, MAX_BODY_INFLATED)?
    } else {
        body.to_vec()
    };
    let schema = postcard::take_from_bytes::<u32>(&raw)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, "body", "no schema"))?
        .0;
    if schema > BODY_SCHEMA {
        return Err(error!(
            ErrorCode::BlueprintFormat,
            "BODY_SCHEMA", schema, BODY_SCHEMA
        ));
    }
    let bp: Blueprint = decode("body", &raw)?;
    if encode("body", &bp)? != raw {
        return Err(error!(ErrorCode::BlueprintDecode, "body", "not canonical"));
    }
    Ok(bp)
}

/// 2026-10-22: Verify `file` under `policy`, then decode it. See the module invariants.
pub fn open(file: &[u8], policy: &TrustPolicy<'_>) -> Result<Opened, Error> {
    let (h, s) = Header::parse(file)?;
    let body = &file[s.body.clone()];
    check_body_hash(&h, body)?;
    let signer = verify_body(policy, &h, body, &file[s.sig])?;
    let blueprint = decode_body(&h, body)?;
    let unsigned: Unsigned = decode("locator", &file[s.locator])?;
    let payload: Vec<InlineArtifact> = decode("payload", &file[s.payload])?;
    if unsigned.sources.len() != blueprint.kernels.modules.len() {
        return Err(error!(
            ErrorCode::BlueprintFrame,
            "the locator table must have one source per module"
        ));
    }
    if (h.flags & FLAG_FULL != 0) == payload.is_empty() {
        return Err(error!(
            ErrorCode::BlueprintFrame,
            "the full flag disagrees with the payload"
        ));
    }
    Ok(Opened {
        id: BlueprintId::of_body(body),
        header: h,
        blueprint,
        unsigned,
        payload,
        signer,
    })
}

/// 2026-10-22: Verify `file`, then write it with new locator and payload sections; the body
/// and signature bytes are copied verbatim.
pub fn rewrite(
    file: &[u8],
    policy: &TrustPolicy<'_>,
    unsigned: &Unsigned,
    payload: &[InlineArtifact],
) -> Result<Vec<u8>, Error> {
    let opened = open(file, policy)?;
    if unsigned.sources.len() != opened.blueprint.kernels.modules.len() {
        return Err(error!(
            ErrorCode::BlueprintFrame,
            "the locator table must have one source per module"
        ));
    }
    let (h, s) = Header::parse(file)?;
    let flags = (h.flags & !FLAG_FULL) | full_flag(payload);
    Ok(assemble(
        flags,
        &file[s.body],
        &encode("locator", unsigned)?,
        &encode("payload", &payload)?,
        &file[s.sig],
    ))
}
