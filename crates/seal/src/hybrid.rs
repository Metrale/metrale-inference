// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The hybrid recipient key: [`HybridSecret`] (X25519 secret ‖ ML-KEM-768 seed
//! `d ‖ z`, 96 bytes), [`HybridPublic`] (X25519 public ‖ ML-KEM-768 `ek`, 1216 bytes) and its
//! [`Fingerprint`] (SHA-256 of the public bytes). The KKB1 envelope's `KeyRingSecret` /
//! `KeyRingPublic`, moved into a core crate so a customer key and a store KeyRing key are one
//! type (DESIGN-encrypted-kernels.md §1.5).
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - A recipient key file holds a 32-byte seed; [`HybridSecret::from_seed`] expands it with
//!   HKDF-SHA-256 under `metrale/kbp-recipient/v1`, so equal seeds give equal keys everywhere.
//! - Secret bytes are `Zeroizing`; `Debug` of a secret prints nothing of it, `Debug` of a public
//!   key prints only its fingerprint.
//! - A public key is parsed with the FIPS 203 §7.2 modulus check on `ek`.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use metrale_error::{Error, ErrorCode, error};
use zeroize::Zeroizing;

use crate::labels;
use crate::primitives::{
    MLKEM_SEED_LEN, MLKEM768_EK_LEN, X25519_LEN, hkdf_sha256_array, mlkem768_ek_bytes,
    mlkem768_from_seed, mlkem768_parse_ek, sha256, x25519_public,
};

/// 2026-10-22: The recipient seed length: what a `kbp-recipient-key v1` file holds.
pub const SEED_LEN: usize = 32;
/// 2026-10-22: The serialized [`HybridSecret`] length.
pub const SECRET_LEN: usize = X25519_LEN + MLKEM_SEED_LEN;
/// 2026-10-22: The serialized [`HybridPublic`] length.
pub const PUBLIC_LEN: usize = X25519_LEN + MLKEM768_EK_LEN;

/// 2026-10-22: Lower-case hex of `bytes`.
pub fn hex(bytes: &[u8]) -> String {
    use fmt::Write as _;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// 2026-10-22: SHA-256 of a [`HybridPublic`]'s bytes: the name of a recipient.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fingerprint(pub [u8; 32]);

impl Fingerprint {
    /// 2026-10-22: Lower-case hex.
    pub fn hex(&self) -> String {
        hex(&self.0)
    }
}

impl fmt::Debug for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.hex())
    }
}

impl fmt::Display for Fingerprint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sha256:{}", self.hex())
    }
}

/// 2026-10-22: The recipient's hybrid secret key. Wiped on drop, never printed.
pub struct HybridSecret {
    x25519: Zeroizing<[u8; X25519_LEN]>,
    mlkem_seed: Zeroizing<[u8; MLKEM_SEED_LEN]>,
}

impl fmt::Debug for HybridSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("HybridSecret(<redacted>)")
    }
}

impl HybridSecret {
    /// 2026-10-22: The key a 32-byte recipient seed expands to:
    /// `HKDF-SHA256(salt = "metrale/kbp-recipient/v1", ikm = seed, info =
    /// "hybrid-x25519-mlkem768", 96)`.
    pub fn from_seed(seed: &[u8; SEED_LEN]) -> Self {
        let okm = hkdf_sha256_array::<SECRET_LEN>(labels::RECIPIENT, seed, labels::RECIPIENT_INFO);
        Self::from_bytes(&okm)
    }

    /// 2026-10-22: A secret from its 96-byte serialization (the KKB1 KeyRing form).
    pub fn from_bytes(bytes: &[u8; SECRET_LEN]) -> Self {
        let mut x25519 = Zeroizing::new([0u8; X25519_LEN]);
        let mut mlkem_seed = Zeroizing::new([0u8; MLKEM_SEED_LEN]);
        x25519.copy_from_slice(&bytes[..X25519_LEN]);
        mlkem_seed.copy_from_slice(&bytes[X25519_LEN..]);
        Self { x25519, mlkem_seed }
    }

    /// 2026-10-22: The 96-byte serialization: X25519 secret ‖ ML-KEM seed `d ‖ z`.
    pub fn to_bytes(&self) -> Zeroizing<[u8; SECRET_LEN]> {
        let mut out = Zeroizing::new([0u8; SECRET_LEN]);
        out[..X25519_LEN].copy_from_slice(self.x25519.as_slice());
        out[X25519_LEN..].copy_from_slice(self.mlkem_seed.as_slice());
        out
    }

    /// 2026-10-22: The matching public key.
    pub fn public(&self) -> HybridPublic {
        HybridPublic {
            x25519: x25519_public(&self.x25519),
            mlkem_ek: mlkem768_ek_bytes(&mlkem768_from_seed(&self.mlkem_seed)),
        }
    }

    pub(crate) fn x25519(&self) -> &[u8; X25519_LEN] {
        &self.x25519
    }

    pub(crate) fn mlkem_seed(&self) -> &[u8; MLKEM_SEED_LEN] {
        &self.mlkem_seed
    }
}

/// 2026-10-22: The recipient's hybrid public key: the value a sealer wraps to.
#[derive(Clone, PartialEq, Eq)]
pub struct HybridPublic {
    x25519: [u8; X25519_LEN],
    mlkem_ek: [u8; MLKEM768_EK_LEN],
}

impl fmt::Debug for HybridPublic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HybridPublic({})", self.fingerprint())
    }
}

impl HybridPublic {
    /// 2026-10-22: Parse 1216 bytes, applying the FIPS 203 §7.2 modulus check to `ek`.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        if bytes.len() != PUBLIC_LEN {
            return Err(error!(
                ErrorCode::BlueprintKey,
                "recipient public key",
                alloc::format!("{} bytes, not {PUBLIC_LEN}", bytes.len())
            ));
        }
        let mut x25519 = [0u8; X25519_LEN];
        let mut mlkem_ek = [0u8; MLKEM768_EK_LEN];
        x25519.copy_from_slice(&bytes[..X25519_LEN]);
        mlkem_ek.copy_from_slice(&bytes[X25519_LEN..]);
        mlkem768_parse_ek(&mlkem_ek)?;
        Ok(Self { x25519, mlkem_ek })
    }

    /// 2026-10-22: The 1216-byte serialization: X25519 public ‖ ML-KEM-768 `ek`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(PUBLIC_LEN);
        out.extend_from_slice(&self.x25519);
        out.extend_from_slice(&self.mlkem_ek);
        out
    }

    /// 2026-10-22: SHA-256 of [`Self::to_bytes`].
    pub fn fingerprint(&self) -> Fingerprint {
        Fingerprint(sha256(&self.to_bytes()))
    }

    pub(crate) fn x25519(&self) -> &[u8; X25519_LEN] {
        &self.x25519
    }

    pub(crate) fn mlkem_ek(&self) -> &[u8; MLKEM768_EK_LEN] {
        &self.mlkem_ek
    }
}
