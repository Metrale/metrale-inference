// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: One sealed artifact (DESIGN-encrypted-kernels.md §2.4):
//!
//! ```text
//! okm    = HKDF-SHA256(salt = "metrale/kbp-seal/v1", ikm = CEK, info = lv(BlueprintId, ArtifactId), 44)
//! K, N   = okm[0..32], okm[32..44]
//! sealed = AES-256-GCM(K, N, AAD = lv("metrale/kbp-seal/v1", BlueprintId, ArtifactId, module,
//!                                     arch, [codec id], raw_len LE), compressed bytes)
//! ```
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - Each `(K, N)` encrypts exactly one message: two artifacts never share a key or a nonce,
//!   and two CEKs never share either, so no nonce is stored and none can be reused.
//! - Sealing is a pure function of `(CEK, binding, bytes)`.
//! - [`open_artifact`] refuses a ciphertext longer than its cap before any AEAD work, and every
//!   authentication failure is the one opaque `SealedOpen`.

use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use zeroize::Zeroizing;

use crate::labels::{self, lv_cat};
use crate::primitives::{
    KEY_LEN, NONCE_LEN, TAG_LEN, aes256gcm_open, aes256gcm_seal, hkdf_sha256_array,
};
use crate::wrap::Cek;

/// 2026-10-22: What one sealed artifact is bound to; every field is from the signed body.
#[derive(Clone, Copy, Debug)]
pub struct ArtifactBinding<'a> {
    /// 2026-10-22: The `BlueprintId` digest.
    pub blueprint: &'a [u8; 32],
    /// 2026-10-22: The `ArtifactId` digest (SHA-256 of the raw module bytes).
    pub artifact: &'a [u8; 32],
    /// 2026-10-22: The module name.
    pub module: &'a str,
    /// 2026-10-22: The module's arch.
    pub arch: &'a str,
    /// 2026-10-22: The inner codec id the sealed bytes are compressed with.
    pub codec: u8,
    /// 2026-10-22: The raw module length.
    pub raw_len: u64,
}

impl ArtifactBinding<'_> {
    /// 2026-10-22: The AEAD associated data.
    pub fn aad(&self) -> Vec<u8> {
        lv_cat(&[
            labels::SEAL,
            self.blueprint,
            self.artifact,
            self.module.as_bytes(),
            self.arch.as_bytes(),
            &[self.codec],
            &self.raw_len.to_le_bytes(),
        ])
    }
}

/// 2026-10-22: The key and nonce of one artifact. Wiped on drop.
pub struct ArtifactKey(Zeroizing<[u8; KEY_LEN + NONCE_LEN]>);

impl ArtifactKey {
    /// 2026-10-22: Derive the key and nonce of `artifact` in `blueprint` under `cek`.
    pub fn derive(cek: &Cek, blueprint: &[u8; 32], artifact: &[u8; 32]) -> Self {
        Self(hkdf_sha256_array::<{ KEY_LEN + NONCE_LEN }>(
            labels::SEAL,
            cek.as_bytes(),
            &lv_cat(&[blueprint, artifact]),
        ))
    }

    /// 2026-10-22: The AES-256 key.
    pub fn key(&self) -> &[u8; KEY_LEN] {
        self.0[..KEY_LEN]
            .try_into()
            .expect("the okm is KEY_LEN + NONCE_LEN bytes")
    }

    /// 2026-10-22: The GCM nonce.
    pub fn nonce(&self) -> &[u8; NONCE_LEN] {
        self.0[KEY_LEN..]
            .try_into()
            .expect("the okm is KEY_LEN + NONCE_LEN bytes")
    }
}

/// 2026-10-22: Seal `compressed` under `cek` for `binding`: ciphertext ‖ tag.
pub fn seal_artifact(
    cek: &Cek,
    binding: &ArtifactBinding<'_>,
    compressed: &[u8],
) -> Result<Vec<u8>, Error> {
    let k = ArtifactKey::derive(cek, binding.blueprint, binding.artifact);
    aes256gcm_seal(k.key(), k.nonce(), &binding.aad(), compressed)
}

/// 2026-10-22: Open `sealed` under `cek` for `binding`; a ciphertext whose compressed length
/// exceeds `max_compressed` is refused before the AEAD (`BlueprintTooLarge`), any
/// authentication failure is `SealedOpen` naming the module.
pub fn open_artifact(
    cek: &Cek,
    binding: &ArtifactBinding<'_>,
    sealed: &[u8],
    max_compressed: u64,
) -> Result<Zeroizing<Vec<u8>>, Error> {
    let refuse = || error!(ErrorCode::SealedOpen, String::from(binding.module));
    let compressed = sealed.len().checked_sub(TAG_LEN).ok_or_else(refuse)?;
    let compressed = u64::try_from(compressed).unwrap_or(u64::MAX);
    if compressed > max_compressed {
        return Err(error!(
            ErrorCode::BlueprintTooLarge,
            alloc::format!("sealed module {}", binding.module),
            compressed,
            max_compressed
        ));
    }
    let k = ArtifactKey::derive(cek, binding.blueprint, binding.artifact);
    aes256gcm_open(k.key(), k.nonce(), &binding.aad(), sealed).ok_or_else(refuse)
}
