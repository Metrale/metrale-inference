// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The primitives sealed kernels are built from, each a thin call into a RustCrypto
//! crate. Public so the known-answer tests (FIPS 203 ML-KEM-768 ACVP, NIST CAVP AES-256-GCM,
//! RFC 5869 HKDF, RFC 7748 X25519) exercise exactly the paths the wrap and the artifact seal use
//! (the KKB1 envelope's `primitives.rs` shape).
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - No randomness here: ML-KEM encapsulation takes `m`, X25519 takes the ephemeral secret.
//! - ML-KEM decapsulation never fails (FIPS 203 implicit rejection); a wrong key or ciphertext
//!   yields an unrelated shared secret, which the AEAD then refuses.
//! - Secret outputs are `Zeroizing`.

use aes_gcm::aead::{Aead, KeyInit, Payload};
use aes_gcm::{Aes256Gcm, Nonce};
use alloc::vec::Vec;
use hkdf::Hkdf;
use metrale_error::{Error, ErrorCode, error};
use ml_kem::kem::{Decapsulate, KeyExport};
use ml_kem::ml_kem_768;
use sha2::{Digest, Sha256};
use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroizing;

/// 2026-10-22: AES-256-GCM key length.
pub const KEY_LEN: usize = 32;
/// 2026-10-22: AES-GCM nonce length (96 bits).
pub const NONCE_LEN: usize = 12;
/// 2026-10-22: AES-GCM tag length.
pub const TAG_LEN: usize = 16;
/// 2026-10-22: X25519 key and shared-secret length.
pub const X25519_LEN: usize = 32;
/// 2026-10-22: ML-KEM-768 encapsulation key length (FIPS 203 Table 3).
pub const MLKEM768_EK_LEN: usize = 1184;
/// 2026-10-22: ML-KEM-768 ciphertext length (FIPS 203 Table 3).
pub const MLKEM768_CT_LEN: usize = 1088;
/// 2026-10-22: ML-KEM seed `d ‖ z` length (FIPS 203 §7.1).
pub const MLKEM_SEED_LEN: usize = 64;
/// 2026-10-22: ML-KEM encapsulation randomness `m` length (FIPS 203 Algorithm 17).
pub const MLKEM_M_LEN: usize = 32;

/// 2026-10-22: SHA-256 of `data`.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    Sha256::digest(data).into()
}

/// 2026-10-22: HKDF-SHA-256 (RFC 5869) extract-then-expand into `okm`; an `okm` longer than
/// 255 blocks is refused.
pub fn hkdf_sha256(salt: &[u8], ikm: &[u8], info: &[u8], okm: &mut [u8]) -> Result<(), Error> {
    Hkdf::<Sha256>::new(Some(salt), ikm)
        .expand(info, okm)
        .map_err(|_| {
            error!(
                ErrorCode::InvalidArgument,
                "hkdf output length", "longer than 255 SHA-256 blocks"
            )
        })
}

/// 2026-10-22: HKDF-SHA-256 into an `N`-byte array, wiped on drop. `N` is checked against
/// HKDF's 255-block limit at compile time, so the expansion cannot fail.
pub fn hkdf_sha256_array<const N: usize>(
    salt: &[u8],
    ikm: &[u8],
    info: &[u8],
) -> Zeroizing<[u8; N]> {
    const { assert!(N <= 255 * 32, "HKDF-SHA-256 expands to at most 255 blocks") };
    let mut okm = Zeroizing::new([0u8; N]);
    Hkdf::<Sha256>::new(Some(salt), ikm)
        .expand(info, okm.as_mut_slice())
        .expect("N is within HKDF's limit (checked at compile time)");
    okm
}

/// 2026-10-22: AES-256-GCM encrypt: `ciphertext ‖ 16-byte tag`. Refused only above the GCM
/// per-message cap (2^36 − 32 bytes), far above any artifact cap.
pub fn aes256gcm_seal(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, Error> {
    Aes256Gcm::new(&(*key).into())
        .encrypt(
            &Nonce::from(*nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| {
            error!(
                ErrorCode::InvalidArgument,
                "AES-256-GCM plaintext", "over the per-message cap"
            )
        })
}

/// 2026-10-22: AES-256-GCM decrypt of `ciphertext ‖ tag`: `None` on any authentication failure,
/// with no plaintext. The caller names the refusal (an opaque code of its own).
pub fn aes256gcm_open(
    key: &[u8; KEY_LEN],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    ciphertext_and_tag: &[u8],
) -> Option<Zeroizing<Vec<u8>>> {
    Aes256Gcm::new(&(*key).into())
        .decrypt(
            &Nonce::from(*nonce),
            Payload {
                msg: ciphertext_and_tag,
                aad,
            },
        )
        .ok()
        .map(Zeroizing::new)
}

/// 2026-10-22: The X25519 public key of a secret (RFC 7748 `X25519(k, 9)`).
pub fn x25519_public(secret: &[u8; X25519_LEN]) -> [u8; X25519_LEN] {
    PublicKey::from(&StaticSecret::from(*secret)).to_bytes()
}

/// 2026-10-22: X25519 Diffie-Hellman (RFC 7748 `X25519(k, u)`): `None` when the result is not
/// contributory (a low-order `u` gives the all-zero secret).
pub fn x25519_shared(
    secret: &[u8; X25519_LEN],
    their_public: &[u8; X25519_LEN],
) -> Option<Zeroizing<[u8; X25519_LEN]>> {
    let shared = StaticSecret::from(*secret).diffie_hellman(&PublicKey::from(*their_public));
    if !shared.was_contributory() {
        return None;
    }
    Some(Zeroizing::new(shared.to_bytes()))
}

/// 2026-10-22: The ML-KEM-768 decapsulation key of a seed `d ‖ z` (FIPS 203 Algorithm 16).
pub fn mlkem768_from_seed(seed: &[u8; MLKEM_SEED_LEN]) -> ml_kem_768::DecapsulationKey {
    ml_kem_768::DecapsulationKey::from_seed((*seed).into())
}

/// 2026-10-22: The serialized encapsulation key of a decapsulation key (FIPS 203 `ek`).
pub fn mlkem768_ek_bytes(dk: &ml_kem_768::DecapsulationKey) -> [u8; MLKEM768_EK_LEN] {
    dk.encapsulation_key().to_bytes().into()
}

/// 2026-10-22: Parse an encapsulation key, applying the FIPS 203 §7.2 modulus check.
pub fn mlkem768_parse_ek(
    ek: &[u8; MLKEM768_EK_LEN],
) -> Result<ml_kem_768::EncapsulationKey, Error> {
    ml_kem_768::EncapsulationKey::new(&(*ek).into()).map_err(|_| {
        error!(
            ErrorCode::BlueprintKey,
            "ML-KEM-768 encapsulation key", "fails the FIPS 203 modulus check"
        )
    })
}

/// 2026-10-22: ML-KEM-768 encapsulation with caller-supplied `m` (FIPS 203 Algorithm 17), after
/// the §7.2 check of `ek`: `(c, K)`.
pub fn mlkem768_encapsulate(
    ek: &[u8; MLKEM768_EK_LEN],
    m: &[u8; MLKEM_M_LEN],
) -> Result<([u8; MLKEM768_CT_LEN], Zeroizing<[u8; 32]>), Error> {
    let (ct, shared) = mlkem768_parse_ek(ek)?.encapsulate_deterministic(&(*m).into());
    Ok((ct.into(), Zeroizing::new(shared.into())))
}

/// 2026-10-22: ML-KEM-768 decapsulation (FIPS 203 Algorithm 18); never fails (implicit
/// rejection).
pub fn mlkem768_decapsulate(
    dk: &ml_kem_768::DecapsulationKey,
    ct: &[u8; MLKEM768_CT_LEN],
) -> Zeroizing<[u8; 32]> {
    Zeroizing::new(dk.decapsulate(&(*ct).into()).into())
}
