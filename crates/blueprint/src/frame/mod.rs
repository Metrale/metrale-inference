// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The `.kbp` container: write an unsigned file, sign it, open (verify, then
//! decode) and convert between lite and full.
//!
//! Owner: blueprint.
//! Invariants:
//! - [`open`] runs: header → bounded decompression of the body (`crate::codec`; not a decode:
//!   the raw length is declared and capped) → raw body hash → certificate chain → both
//!   signatures → only then any postcard decode of the body, locator or payload. Nothing but
//!   the fixed-width header is interpreted before the signature verifies (the signature block
//!   itself is decoded to be checked).
//! - The body hash, the signed message and the [`BlueprintId`] cover the raw canonical body,
//!   never the stored bytes: the codec is outside identity ([`recode`]).
//! - [`rewrite`] copies the body and signature bytes verbatim and replaces only the locator
//!   and payload sections, so lite and full share the [`BlueprintId`] and the signature.
//! - Every decompression is bounded by a header or section length under its cap and by the
//!   caller's [`DecodeMemory`]; there is no default memory budget (PCND).
//! - The body is canonical: a decoded body re-encodes to the bytes it came from.

mod header;
mod locator;
mod payload;

use alloc::borrow::Cow;
use alloc::vec::Vec;

use metrale_blueprint_trust::{
    Purpose, Scope, SigBlock, SignedCert, Signer, TrustPolicy, Verified, verify_signed,
};
use metrale_error::{Error, ErrorCode, error};
use serde::Serialize;
use serde::de::DeserializeOwned;

pub use header::{
    FLAG_FULL, FLAG_SOLID, HEADER_LEN, Header, KBP_FORMAT, MAX_ARTIFACT, MAX_BODY, MAX_BODY_RAW,
    MAX_LOCATOR, MAX_PAYLOAD, MAX_PAYLOAD_RAW, MAX_SIG, Sections,
};
pub use locator::{InlineArtifact, KernelSource, Unsigned};
pub use payload::PayloadEntry;

use crate::codec::{Codec, DecodeMemory};
use crate::ids::{BlueprintId, Sha256};
use crate::types::{BODY_SCHEMA, Blueprint, OLDEST_BODY_SCHEMA};

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

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// 2026-10-22: The body section as written: stored bytes, their codec, and the raw body's
/// length and hash.
struct Body<'a> {
    stored: Cow<'a, [u8]>,
    codec: Codec,
    raw_len: u64,
    raw_sha256: [u8; 32],
}

impl<'a> Body<'a> {
    /// 2026-10-22: `raw` stored under `codec`.
    fn encode(raw: &[u8], codec: Codec) -> Self {
        Self {
            stored: Cow::Owned(codec.compress(raw)),
            codec,
            raw_len: len64(raw.len()),
            raw_sha256: Sha256::of(&[raw]).0,
        }
    }

    /// 2026-10-22: An existing file's body section, copied byte for byte.
    fn verbatim(h: &Header, stored: &'a [u8]) -> Self {
        Self {
            stored: Cow::Borrowed(stored),
            codec: h.codec,
            raw_len: h.body_raw_len,
            raw_sha256: h.body_sha256,
        }
    }
}

fn assemble(flags: u8, body: Body<'_>, locator: &[u8], payload: &[u8], sig: &[u8]) -> Vec<u8> {
    let stored = &body.stored[..];
    let h = Header {
        flags,
        codec: body.codec,
        body_len: len64(stored.len()),
        body_raw_len: body.raw_len,
        body_sha256: body.raw_sha256,
        locator_len: len64(locator.len()),
        payload_len: len64(payload.len()),
        sig_len: len64(sig.len()),
    };
    let mut out =
        Vec::with_capacity(HEADER_LEN + stored.len() + locator.len() + payload.len() + sig.len());
    out.extend_from_slice(&h.write());
    for s in [stored, locator, payload, sig] {
        out.extend_from_slice(s);
    }
    out
}

/// 2026-10-22: An unsigned file: the canonical body and the payload stored under
/// [`Codec::WRITE`], and the locator table.
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
    let (flags, payload) = payload::encode_payload(payload, Codec::WRITE)?;
    Ok(assemble(
        flags,
        Body::encode(&encode("body", bp)?, Codec::WRITE),
        &encode("locator", unsigned)?,
        &payload,
        &[],
    ))
}

fn raw_body(h: &Header, stored: &[u8], memory: DecodeMemory) -> Result<Vec<u8>, Error> {
    h.codec
        .decompress("body", stored, h.body_raw_len, MAX_BODY_RAW, memory)
}

fn check_body_hash(h: &Header, raw: &[u8]) -> Result<(), Error> {
    let found = Sha256::of(&[raw]);
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
/// The body hash is checked first (the body decodes under `memory`); the old signature block,
/// if any, is replaced.
pub fn sign(
    file: &[u8],
    signer: &Signer,
    chain: Vec<SignedCert>,
    memory: DecodeMemory,
) -> Result<Vec<u8>, Error> {
    let (h, s) = Header::parse(file)?;
    let stored = &file[s.body.clone()];
    let raw = raw_body(&h, stored, memory)?;
    check_body_hash(&h, &raw)?;
    let block = SigBlock {
        kid: signer.kid().clone(),
        chain,
        sig: signer.sign(Purpose::KbpBody, &h.signed_message(&raw))?,
    };
    Ok(assemble(
        h.flags,
        Body::verbatim(&h, stored),
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

fn decode_body(raw: &[u8]) -> Result<Blueprint, Error> {
    let schema = postcard::take_from_bytes::<u32>(raw)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, "body", "no schema"))?
        .0;
    if !(OLDEST_BODY_SCHEMA..=BODY_SCHEMA).contains(&schema) {
        return Err(error!(
            ErrorCode::BlueprintFormat,
            "BODY_SCHEMA",
            schema,
            alloc::format!("{OLDEST_BODY_SCHEMA}..={BODY_SCHEMA}")
        ));
    }
    let bp: Blueprint = decode("body", raw)?;
    if encode("body", &bp)? != raw {
        return Err(error!(ErrorCode::BlueprintDecode, "body", "not canonical"));
    }
    Ok(bp)
}

/// 2026-10-22: Verify `file` under `policy`, then decode it, every kzx block within `memory`.
/// See the module invariants.
pub fn open(file: &[u8], policy: &TrustPolicy<'_>, memory: DecodeMemory) -> Result<Opened, Error> {
    let (h, s) = Header::parse(file)?;
    let raw = raw_body(&h, &file[s.body.clone()], memory)?;
    check_body_hash(&h, &raw)?;
    let signer = verify_body(policy, &h, &raw, &file[s.sig])?;
    let blueprint = decode_body(&raw)?;
    let unsigned: Unsigned = decode("locator", &file[s.locator])?;
    let payload = payload::decode_payload(h.flags, &file[s.payload], memory)?;
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
        id: BlueprintId::of_body(&raw),
        header: h,
        blueprint,
        unsigned,
        payload,
        signer,
    })
}

/// 2026-10-22: Verify `file`, then write it with new locator and payload sections (the payload
/// under [`Codec::WRITE`]); the body and signature bytes are copied verbatim.
pub fn rewrite(
    file: &[u8],
    policy: &TrustPolicy<'_>,
    memory: DecodeMemory,
    unsigned: &Unsigned,
    payload: &[InlineArtifact],
) -> Result<Vec<u8>, Error> {
    let opened = open(file, policy, memory)?;
    if unsigned.sources.len() != opened.blueprint.kernels.modules.len() {
        return Err(error!(
            ErrorCode::BlueprintFrame,
            "the locator table must have one source per module"
        ));
    }
    let (h, s) = Header::parse(file)?;
    let (flags, payload) = payload::encode_payload(payload, Codec::WRITE)?;
    Ok(assemble(
        flags,
        Body::verbatim(&h, &file[s.body]),
        &encode("locator", unsigned)?,
        &payload,
        &file[s.sig],
    ))
}

/// 2026-10-22: Verify `file`, then write it with its body and its payload stored under `codec`
/// (a per-entry payload becomes one solid block); the locator and signature bytes are copied
/// verbatim. The id and the signature do not change, because neither covers stored bytes.
pub fn recode(
    file: &[u8],
    policy: &TrustPolicy<'_>,
    memory: DecodeMemory,
    codec: Codec,
) -> Result<Vec<u8>, Error> {
    let opened = open(file, policy, memory)?;
    let (flags, payload) = payload::encode_payload(&opened.payload, codec)?;
    let (_, s) = Header::parse(file)?;
    Ok(assemble(
        flags,
        Body::encode(&encode("body", &opened.blueprint)?, codec),
        &file[s.locator],
        &payload,
        &file[s.sig],
    ))
}
