// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the KKB1 byte layout and its header.
//!
//! ```text
//! "KKB1" | u32 LE header_len | header (canonical JSON) | ciphertext ‖ 16-byte GCM tag
//! ```
//!
//! The header is canonical JSON: keys in byte order (the struct fields below
//! are declared in that order, so serde emits them sorted whatever map
//! features are unified into `serde_json`), no whitespace, byte fields in
//! padded standard base64 (RFC 4648 §4), `plaintext.sha256` in lowercase hex.
//! [`parse`] re-serializes the parsed header and refuses any input whose
//! bytes differ, so exactly one header encoding exists for each header value.
//! The header bytes are the AES-GCM AAD.

use serde::{Deserialize, Serialize};

use crate::error::{EnvelopeError, Result};
use crate::primitives::{MLKEM768_CT_LEN, NONCE_LEN, TAG_LEN};

/// The four magic bytes.
pub const MAGIC: [u8; 4] = *b"KKB1";
/// The header format version this crate writes and reads.
pub const VERSION: u32 = 1;
/// The AEAD named in every header.
pub const ALG: &str = "A256GCM";
/// The per-file key derivation named in every header.
pub const KDF: &str = "HKDF-SHA256";
/// The DEK wrap scheme named in `wrapped_dek.scheme`.
pub const WRAP_SCHEME: &str = "hybrid-x25519-mlkem768-hkdf-a256gcm";
/// Length of a wrapped DEK: 32-byte DEK ‖ 16-byte tag.
pub const WRAPPED_DEK_CT_LEN: usize = 32 + TAG_LEN;
const PREFIX_LEN: usize = 8;
const KID_MAX: usize = 64;
const PATH_MAX: usize = 255;
const INFO_PREFIX: &str = "kkb/v1/";

/// The KKB1 header. Field order is the canonical (sorted) key order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Header {
    pub alg: String,
    pub created_at: u64,
    pub info: String,
    pub kdf: String,
    pub kid: String,
    #[serde(with = "b64_array")]
    pub nonce: [u8; NONCE_LEN],
    pub plaintext: PlaintextInfo,
    pub publisher_build: String,
    pub v: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wrapped_dek: Option<WrappedDek>,
}

/// What the ciphertext decrypts to, checked after decryption.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PlaintextInfo {
    pub encoding: String,
    pub len: u64,
    #[serde(with = "hex32")]
    pub sha256: [u8; 32],
}

/// The publish DEK, wrapped to the KeyRing's hybrid public key.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WrappedDek {
    #[serde(with = "b64_array")]
    pub ct: [u8; WRAPPED_DEK_CT_LEN],
    #[serde(with = "b64_array")]
    pub ct_mlkem: [u8; MLKEM768_CT_LEN],
    #[serde(with = "b64_array")]
    pub epk_x25519: [u8; 32],
    #[serde(with = "b64_array")]
    pub nonce: [u8; NONCE_LEN],
    pub scheme: String,
}

impl Header {
    /// The canonical header bytes (also the AEAD AAD).
    pub fn to_canonical_bytes(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| EnvelopeError::HeaderJson(e.to_string()))
    }

    /// Checks the fields a reader relies on: version, algorithms, kid, and
    /// that `info` is `kkb/v1/<kid>/<valid path>`.
    pub fn validate(&self) -> Result<()> {
        if self.v != VERSION {
            return Err(EnvelopeError::UnsupportedVersion(self.v));
        }
        expect_eq("alg", &self.alg, ALG)?;
        expect_eq("kdf", &self.kdf, KDF)?;
        validate_kid(&self.kid)?;
        let path = self
            .info
            .strip_prefix(INFO_PREFIX)
            .and_then(|rest| rest.strip_prefix(self.kid.as_str()))
            .and_then(|rest| rest.strip_prefix('/'))
            .ok_or(EnvelopeError::BadInfo)?;
        validate_path(path).map_err(|_| EnvelopeError::BadInfo)?;
        if let Some(w) = &self.wrapped_dek {
            expect_eq("wrapped_dek.scheme", &w.scheme, WRAP_SCHEME)?;
        }
        Ok(())
    }
}

fn expect_eq(field: &'static str, value: &str, want: &str) -> Result<()> {
    if value == want {
        Ok(())
    } else {
        Err(EnvelopeError::Unsupported {
            field,
            value: value.to_owned(),
        })
    }
}

/// A kid is 1..=64 bytes of `[A-Za-z0-9._-]` and is not `.` or `..`.
pub fn validate_kid(kid: &str) -> Result<()> {
    let ok = !kid.is_empty()
        && kid.len() <= KID_MAX
        && kid != "."
        && kid != ".."
        && kid
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b));
    if ok {
        Ok(())
    } else {
        Err(EnvelopeError::InvalidKid)
    }
}

/// A path is 1..=255 bytes of `[A-Za-z0-9._-/]`, relative, with no empty,
/// `.` or `..` segment.
pub fn validate_path(path: &str) -> Result<()> {
    let ok = !path.is_empty()
        && path.len() <= PATH_MAX
        && path
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._-/".contains(&b))
        && path
            .split('/')
            .all(|seg| !seg.is_empty() && seg != "." && seg != "..");
    if ok {
        Ok(())
    } else {
        Err(EnvelopeError::InvalidPath)
    }
}

/// The HKDF info string and header `info`: `kkb/v1/<kid>/<path>`.
pub fn info_string(kid: &str, path: &str) -> Result<String> {
    validate_kid(kid)?;
    validate_path(path)?;
    Ok(format!("{INFO_PREFIX}{kid}/{path}"))
}

/// A parsed envelope: the validated header, its exact bytes, and the body.
#[derive(Debug)]
pub struct Parsed<'a> {
    pub header: Header,
    pub header_bytes: &'a [u8],
    pub ciphertext_and_tag: &'a [u8],
}

/// Assembles `MAGIC | len | header | body`.
pub fn encode(header_bytes: &[u8], ciphertext_and_tag: &[u8]) -> Result<Vec<u8>> {
    let len = u32::try_from(header_bytes.len()).map_err(|_| EnvelopeError::HeaderTooLarge)?;
    let mut out = Vec::with_capacity(PREFIX_LEN + header_bytes.len() + ciphertext_and_tag.len());
    out.extend_from_slice(&MAGIC);
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(ciphertext_and_tag);
    Ok(out)
}

/// Splits and validates an envelope without any key: what a key server needs
/// to read `kid` and `wrapped_dek`. Authenticity is established only by `open`.
pub fn parse(bytes: &[u8]) -> Result<Parsed<'_>> {
    let magic = bytes
        .get(..MAGIC.len())
        .ok_or(EnvelopeError::Truncated("magic"))?;
    if magic != MAGIC {
        return Err(EnvelopeError::BadMagic);
    }
    let len_bytes: [u8; 4] = bytes
        .get(MAGIC.len()..PREFIX_LEN)
        .and_then(|s| s.try_into().ok())
        .ok_or(EnvelopeError::Truncated("header length"))?;
    let header_len = usize::try_from(u32::from_le_bytes(len_bytes))
        .map_err(|_| EnvelopeError::HeaderTooLarge)?;
    let header_end = PREFIX_LEN
        .checked_add(header_len)
        .ok_or(EnvelopeError::HeaderTooLarge)?;
    let header_bytes = bytes
        .get(PREFIX_LEN..header_end)
        .ok_or(EnvelopeError::Truncated("header"))?;
    let ciphertext_and_tag = &bytes[header_end..];
    if ciphertext_and_tag.len() < TAG_LEN {
        return Err(EnvelopeError::Truncated("ciphertext tag"));
    }
    let header: Header = serde_json::from_slice(header_bytes)
        .map_err(|e| EnvelopeError::HeaderJson(e.to_string()))?;
    if header.to_canonical_bytes()? != header_bytes {
        return Err(EnvelopeError::NonCanonicalHeader);
    }
    header.validate()?;
    Ok(Parsed {
        header,
        header_bytes,
        ciphertext_and_tag,
    })
}

mod b64_array {
    use base64::Engine;
    use base64::engine::general_purpose::STANDARD;
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    pub fn serialize<S: Serializer, const N: usize>(v: &[u8; N], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&STANDARD.encode(v))
    }

    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        d: D,
    ) -> Result<[u8; N], D::Error> {
        let text = String::deserialize(d)?;
        let bytes = STANDARD.decode(text.as_bytes()).map_err(D::Error::custom)?;
        let len = bytes.len();
        bytes
            .try_into()
            .map_err(|_| D::Error::custom(format!("expected {N} bytes, got {len}")))
    }
}

mod hex32 {
    use serde::{Deserialize, Deserializer, Serializer, de::Error};

    const DIGITS: &[u8; 16] = b"0123456789abcdef";

    pub fn serialize<S: Serializer>(v: &[u8; 32], s: S) -> Result<S::Ok, S::Error> {
        let text: String = v
            .iter()
            .flat_map(|b| [DIGITS[usize::from(b >> 4)], DIGITS[usize::from(b & 0x0f)]])
            .map(char::from)
            .collect();
        s.serialize_str(&text)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 32], D::Error> {
        let text = String::deserialize(d)?;
        let raw = text.as_bytes();
        if raw.len() != 64 {
            return Err(D::Error::custom("sha256 must be 64 lowercase hex digits"));
        }
        let nibble = |c: u8| {
            DIGITS
                .iter()
                .position(|&d| d == c)
                .and_then(|p| u8::try_from(p).ok())
                .ok_or_else(|| D::Error::custom("sha256 must be lowercase hex"))
        };
        let mut out = [0u8; 32];
        for (o, pair) in out.iter_mut().zip(raw.chunks_exact(2)) {
            *o = (nibble(pair[0])? << 4) | nibble(pair[1])?;
        }
        Ok(out)
    }
}
