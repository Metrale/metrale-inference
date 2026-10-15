// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the four primitives the envelope is built from, each a thin
//! call into a RustCrypto crate. They are public so the known-answer tests
//! (NIST CAVP AES-GCM, RFC 5869 HKDF, FIPS 203 ML-KEM-768) exercise exactly
//! the code paths `seal`, `open`, `wrap_dek` and `unwrap_dek` use.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Key, Nonce};
use hkdf::Hkdf;
use ml_kem::kem::{Decapsulate, KeyExport};
use ml_kem::ml_kem_768;
use sha2::{Digest, Sha256};
use zeroize::Zeroizing;

use crate::error::{EnvelopeError, Result};

/// AES-256-GCM key length.
pub const KEY_LEN: usize = 32;
/// AES-GCM nonce length (96 bits, NIST SP 800-38D §8.2.2 random construction).
pub const NONCE_LEN: usize = 12;
/// AES-GCM tag length.
pub const TAG_LEN: usize = 16;
/// ML-KEM-768 encapsulation key length (FIPS 203 Table 3).
pub const MLKEM768_EK_LEN: usize = 1184;
/// ML-KEM-768 ciphertext length (FIPS 203 Table 3).
pub const MLKEM768_CT_LEN: usize = 1088;
/// ML-KEM seed `d ‖ z` length (FIPS 203 §7.1, the 64-byte private seed form).
pub const MLKEM_SEED_LEN: usize = 64;
/// ML-KEM encapsulation randomness `m` length (FIPS 203 Algorithm 17).
pub const MLKEM_M_LEN: usize = 32;

/// SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// HKDF-SHA256 (RFC 5869) extract-then-expand into `okm`.
pub fn hkdf_sha256(salt: &[u8], ikm: &[u8], info: &[u8], okm: &mut [u8]) -> Result<()> {
    Hkdf::<Sha256>::new(Some(salt), ikm)
        .expand(info, okm)
        .map_err(|_| EnvelopeError::BadLength {
            field: "hkdf output",
            expected: 255 * 32,
            actual: okm.len(),
        })
}

/// HKDF-SHA256 into a 32-byte key that is wiped on drop.
pub fn hkdf_sha256_key(salt: &[u8], ikm: &[u8], info: &[u8]) -> Result<Zeroizing<[u8; KEY_LEN]>> {
    let mut key = Zeroizing::new([0u8; KEY_LEN]);
    hkdf_sha256(salt, ikm, info, key.as_mut_slice())?;
    Ok(key)
}

/// AES-256-GCM encrypt: returns `ciphertext ‖ 16-byte tag`.
pub fn aes256gcm_seal(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>> {
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
        .encrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| EnvelopeError::PlaintextTooLarge)
}

/// AES-256-GCM decrypt of `ciphertext ‖ tag`. Any authentication failure is
/// [`EnvelopeError::Decrypt`] and returns no plaintext.
pub fn aes256gcm_open(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Result<Vec<u8>> {
    Aes256Gcm::new(Key::<Aes256Gcm>::from_slice(key))
        .decrypt(
            Nonce::from_slice(nonce),
            Payload {
                msg: ciphertext_and_tag,
                aad,
            },
        )
        .map_err(|_| EnvelopeError::Decrypt)
}

/// ML-KEM-768 decapsulation key from the 64-byte seed `d ‖ z` (FIPS 203
/// Algorithm 16, `ML-KEM.KeyGen_internal`).
pub fn mlkem768_from_seed(seed: &[u8; MLKEM_SEED_LEN]) -> ml_kem_768::DecapsulationKey {
    ml_kem_768::DecapsulationKey::from_seed((*seed).into())
}

/// The encapsulation key of a decapsulation key, serialized (FIPS 203 `ek`).
pub fn mlkem768_ek_bytes(dk: &ml_kem_768::DecapsulationKey) -> [u8; MLKEM768_EK_LEN] {
    dk.encapsulation_key().to_bytes().into()
}

/// Parses an ML-KEM-768 encapsulation key, applying the FIPS 203 §7.2
/// modulus check (every coefficient below q).
pub fn mlkem768_parse_ek(ek: &[u8; MLKEM768_EK_LEN]) -> Result<ml_kem_768::EncapsulationKey> {
    ml_kem_768::EncapsulationKey::new(&(*ek).into())
        .map_err(|_| EnvelopeError::InvalidPublicKey("ML-KEM-768 encapsulation key"))
}

/// ML-KEM-768 encapsulation with caller-supplied randomness `m` (FIPS 203
/// Algorithm 17, `ML-KEM.Encaps_internal`), after the §7.2 input check on
/// `ek`. Returns `(c, K)`.
pub fn mlkem768_encapsulate(
    ek: &[u8; MLKEM768_EK_LEN],
    m: &[u8; MLKEM_M_LEN],
) -> Result<([u8; MLKEM768_CT_LEN], Zeroizing<[u8; 32]>)> {
    let (ct, shared) = mlkem768_parse_ek(ek)?.encapsulate_deterministic(&(*m).into());
    Ok((ct.into(), Zeroizing::new(shared.into())))
}

/// ML-KEM-768 decapsulation (FIPS 203 Algorithm 18). A ciphertext that was
/// not produced for this key yields the implicit-rejection key, never an error.
pub fn mlkem768_decapsulate(
    dk: &ml_kem_768::DecapsulationKey,
    ct: &[u8; MLKEM768_CT_LEN],
) -> Zeroizing<[u8; 32]> {
    Zeroizing::new(dk.decapsulate(&(*ct).into()).into())
}
