// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: sealing and opening one file under a publish DEK.
//!
//! `K_file = HKDF-SHA256(ikm = DEK, salt = kid, info = "kkb/v1/<kid>/<path>")`;
//! the body is `AES-256-GCM(K_file, nonce, aad = canonical header bytes)`
//! with a fresh random 96-bit nonce from the [`Entropy`] port. After a
//! successful decryption the plaintext length and sha256 are checked against
//! the (authenticated) header.

use core::fmt;

use zeroize::Zeroizing;

use crate::error::{EnvelopeError, Result};
use crate::format::{self, ALG, Header, KDF, PlaintextInfo, VERSION, WrappedDek};
use crate::ports::{Clock, Entropy};
use crate::primitives::{
    KEY_LEN, NONCE_LEN, aes256gcm_open, aes256gcm_seal, hkdf_sha256_key, sha256,
};

/// A publish data-encryption key: 32 bytes, wiped on drop, never printed.
#[derive(Clone)]
pub struct Dek(Zeroizing<[u8; KEY_LEN]>);

impl Dek {
    /// Wraps existing key bytes.
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// A fresh DEK drawn from `entropy`.
    pub fn generate(entropy: &mut dyn Entropy) -> Result<Self> {
        let mut key = Zeroizing::new([0u8; KEY_LEN]);
        entropy.fill(key.as_mut_slice())?;
        Ok(Self(key))
    }

    /// The raw key bytes.
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }
}

impl fmt::Debug for Dek {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Dek(<redacted>)")
    }
}

/// `K_file = HKDF-SHA256(DEK, salt = kid, info = "kkb/v1/<kid>/<path>")`.
pub fn derive_file_key(dek: &Dek, kid: &str, path: &str) -> Result<Zeroizing<[u8; KEY_LEN]>> {
    let info = format::info_string(kid, path)?;
    hkdf_sha256_key(kid.as_bytes(), dek.as_bytes(), info.as_bytes())
}

/// Everything about one file except its bytes and the keys.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SealRequest {
    pub kid: String,
    pub path: String,
    /// The plaintext encoding label, e.g. `deflate+lattice-jsonl/1`.
    /// Compression is the caller's job; this crate only records the label.
    pub encoding: String,
    pub publisher_build: String,
    /// The publish DEK wrapped to the KeyRing, carried in this file's header.
    pub wrapped_dek: Option<WrappedDek>,
}

/// Seals `plaintext` into a KKB1 envelope.
pub fn seal(
    dek: &Dek,
    request: &SealRequest,
    plaintext: &[u8],
    entropy: &mut dyn Entropy,
    clock: &dyn Clock,
) -> Result<Vec<u8>> {
    if request.encoding.is_empty() {
        return Err(EnvelopeError::MissingField("encoding"));
    }
    if request.publisher_build.is_empty() {
        return Err(EnvelopeError::MissingField("publisher_build"));
    }
    let info = format::info_string(&request.kid, &request.path)?;
    let key = derive_file_key(dek, &request.kid, &request.path)?;
    let mut nonce = [0u8; NONCE_LEN];
    entropy.fill(&mut nonce)?;
    let len = u64::try_from(plaintext.len()).map_err(|_| EnvelopeError::PlaintextTooLarge)?;
    let header = Header {
        alg: ALG.to_owned(),
        created_at: clock.now_unix_seconds()?,
        info,
        kdf: KDF.to_owned(),
        kid: request.kid.clone(),
        nonce,
        plaintext: PlaintextInfo {
            encoding: request.encoding.clone(),
            len,
            sha256: sha256(plaintext),
        },
        publisher_build: request.publisher_build.clone(),
        v: VERSION,
        wrapped_dek: request.wrapped_dek.clone(),
    };
    header.validate()?;
    let header_bytes = header.to_canonical_bytes()?;
    let body = aes256gcm_seal(&key, &nonce, &header_bytes, plaintext)?;
    format::encode(&header_bytes, &body)
}

/// What the reader expects the envelope to be: the kid it fetched the DEK
/// for, and the path it asked for. Both are bound into the file key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expected<'a> {
    pub kid: &'a str,
    pub path: &'a str,
}

/// A successfully opened envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opened {
    pub header: Header,
    pub plaintext: Vec<u8>,
}

/// Opens a KKB1 envelope sealed under `dek` for `expected`.
pub fn open(dek: &Dek, expected: &Expected<'_>, bytes: &[u8]) -> Result<Opened> {
    let parsed = format::parse(bytes)?;
    let header = parsed.header;
    if header.kid != expected.kid {
        return Err(EnvelopeError::KidMismatch {
            expected: expected.kid.to_owned(),
            found: header.kid,
        });
    }
    let info = format::info_string(expected.kid, expected.path)?;
    if header.info != info {
        return Err(EnvelopeError::InfoMismatch {
            expected: info,
            found: header.info,
        });
    }
    let key = derive_file_key(dek, expected.kid, expected.path)?;
    let plaintext = aes256gcm_open(
        &key,
        &header.nonce,
        parsed.header_bytes,
        parsed.ciphertext_and_tag,
    )?;
    let actual = u64::try_from(plaintext.len()).map_err(|_| EnvelopeError::PlaintextTooLarge)?;
    if actual != header.plaintext.len {
        return Err(EnvelopeError::PlaintextLength {
            expected: header.plaintext.len,
            actual,
        });
    }
    if sha256(&plaintext) != header.plaintext.sha256 {
        return Err(EnvelopeError::PlaintextDigest);
    }
    Ok(Opened { header, plaintext })
}
