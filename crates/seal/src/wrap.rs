// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The content key ([`Cek`]), the randomness of one seal run ([`SealRandom`]) and
//! the hybrid wrap of a key to a recipient (DESIGN-encrypted-kernels.md §1.1; the KKB1
//! envelope's `wrap.rs` construction):
//!
//! ```text
//! epk          = X25519(esk, 9)
//! ss_X         = X25519(esk, pk_X)                      (must be contributory)
//! (ct_M, ss_M) = ML-KEM-768.Encaps_internal(ek, m)
//! KEK          = HKDF-SHA256(salt = label.salt, ikm = ss_M ‖ ss_X ‖ ct_M ‖ epk ‖ pk_X, info = label.info)
//! ct           = AES-256-GCM(KEK, nonce, aad = lv(label.salt, label.info), key)
//! ```
//!
//! Owner: blueprint (sealed kernels).
//! Invariants:
//! - X-Wing ordering: the ML-KEM secret first, then the X25519 secret, the ML-KEM ciphertext
//!   and both X25519 public values, so the KEK stays secret while either component KEM holds.
//! - Randomness is a value ([`WrapRandom`]), drawn by the host in one `Entropy::fill` per run.
//! - Every unwrap failure (non-contributory point, wrong key, altered field, other label) is
//!   the one opaque `SealedUnwrap`; ML-KEM's implicit rejection never errors on its own.

use alloc::vec::Vec;
use core::fmt;

use metrale_abstract_io::Entropy;
use metrale_error::{Error, ErrorCode, error};
use zeroize::Zeroizing;

use crate::hybrid::{Fingerprint, HybridPublic, HybridSecret};
use crate::labels::{self, lv_cat};
use crate::primitives::{
    KEY_LEN, MLKEM_M_LEN, MLKEM768_CT_LEN, NONCE_LEN, TAG_LEN, X25519_LEN, aes256gcm_open,
    aes256gcm_seal, hkdf_sha256_array, mlkem768_decapsulate, mlkem768_encapsulate,
    mlkem768_from_seed, x25519_public, x25519_shared,
};

/// 2026-10-22: The wrapped key's AEAD output length: key ‖ tag.
pub const WRAPPED_CT_LEN: usize = KEY_LEN + TAG_LEN;

/// 2026-10-22: A 32-byte content-encryption key. Wiped on drop, never printed, never cloned.
pub struct Cek(Zeroizing<[u8; KEY_LEN]>);

impl fmt::Debug for Cek {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Cek(<redacted>)")
    }
}

impl Cek {
    /// 2026-10-22: A key from its bytes.
    pub fn from_bytes(bytes: [u8; KEY_LEN]) -> Self {
        Self(Zeroizing::new(bytes))
    }

    /// 2026-10-22: The key bytes (for the derivations and the independent-decrypt tests).
    pub fn as_bytes(&self) -> &[u8; KEY_LEN] {
        &self.0
    }

    /// 2026-10-22: Whether two keys are equal, compared without an early exit (the hot-swap
    /// refusal of a reused CEK, DESIGN-encrypted-kernels.md §7).
    pub fn same_as(&self, other: &Cek) -> bool {
        self.0
            .iter()
            .zip(other.0.iter())
            .fold(0u8, |a, (x, y)| a | (x ^ y))
            == 0
    }
}

/// 2026-10-22: The randomness of one wrap.
pub struct WrapRandom {
    /// 2026-10-22: The ephemeral X25519 secret.
    pub esk: Zeroizing<[u8; X25519_LEN]>,
    /// 2026-10-22: The ML-KEM encapsulation randomness.
    pub m: Zeroizing<[u8; MLKEM_M_LEN]>,
    /// 2026-10-22: The AES-GCM nonce of the wrap (one message per KEK).
    pub nonce: [u8; NONCE_LEN],
}

/// 2026-10-22: Bytes of one [`WrapRandom`].
pub const WRAP_RANDOM_LEN: usize = X25519_LEN + MLKEM_M_LEN + NONCE_LEN;

/// 2026-10-22: The randomness of one seal run: the CEK and one [`WrapRandom`] per recipient.
pub struct SealRandom {
    /// 2026-10-22: The content key.
    pub cek: Cek,
    /// 2026-10-22: One per recipient, in recipient order.
    pub wraps: Vec<WrapRandom>,
}

impl SealRandom {
    /// 2026-10-22: Split `bytes` (exactly `32 + recipients × 76`) into a run's randomness.
    pub fn from_bytes(bytes: &[u8], recipients: usize) -> Result<Self, Error> {
        if bytes.len() != Self::len(recipients) {
            return Err(error!(
                ErrorCode::InvalidArgument,
                "seal randomness", "its length is not 32 + 76 per recipient"
            ));
        }
        let take = |at: usize| {
            let mut a = [0u8; 32];
            a.copy_from_slice(&bytes[at..at + 32]);
            Zeroizing::new(a)
        };
        let cek = Cek(take(0));
        let wraps = (0..recipients)
            .map(|i| {
                let at = KEY_LEN + i * WRAP_RANDOM_LEN;
                let mut nonce = [0u8; NONCE_LEN];
                nonce.copy_from_slice(&bytes[at + 64..at + WRAP_RANDOM_LEN]);
                WrapRandom {
                    esk: take(at),
                    m: take(at + 32),
                    nonce,
                }
            })
            .collect();
        Ok(Self { cek, wraps })
    }

    /// 2026-10-22: The byte count of a run for `recipients`.
    pub const fn len(recipients: usize) -> usize {
        KEY_LEN + recipients * WRAP_RANDOM_LEN
    }

    /// 2026-10-22: Draw a run's randomness for `recipients` from `entropy` in one `fill` (BPPT).
    pub fn draw<E: Entropy>(entropy: &E, recipients: usize) -> Result<Self, Error> {
        let mut bytes = Zeroizing::new(alloc::vec![0u8; Self::len(recipients)]);
        entropy.fill(&mut bytes)?;
        Self::from_bytes(&bytes, recipients)
    }
}

/// 2026-10-22: A key wrapped to one recipient.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrappedCek {
    /// 2026-10-22: The ephemeral X25519 public key.
    pub epk: [u8; X25519_LEN],
    /// 2026-10-22: The ML-KEM-768 ciphertext.
    pub ct_mlkem: [u8; MLKEM768_CT_LEN],
    /// 2026-10-22: The AES-GCM nonce.
    pub nonce: [u8; NONCE_LEN],
    /// 2026-10-22: The wrapped key ‖ tag.
    pub ct: [u8; WRAPPED_CT_LEN],
}

/// 2026-10-22: What a wrap is bound to: the HKDF salt, the HKDF info, and the recipient whose
/// fingerprint names the refusal. The AAD is `lv(salt, info)`.
#[derive(Clone, Debug)]
pub struct WrapLabel {
    salt: &'static [u8],
    info: Vec<u8>,
    recipient: Fingerprint,
}

impl WrapLabel {
    /// 2026-10-22: The `.kbp` CEK wrap: salt `metrale/kbp-cek/v1`, info
    /// `lv(BlueprintId, recipient fingerprint)`.
    pub fn kbp_cek(blueprint: &[u8; 32], recipient: Fingerprint) -> Self {
        Self {
            salt: labels::CEK,
            info: lv_cat(&[blueprint, &recipient.0]),
            recipient,
        }
    }

    fn aad(&self) -> Vec<u8> {
        lv_cat(&[self.salt, &self.info])
    }
}

fn derive_kek(
    label: &WrapLabel,
    ss_mlkem: &[u8; 32],
    ss_x25519: &[u8; 32],
    ct_mlkem: &[u8; MLKEM768_CT_LEN],
    epk: &[u8; X25519_LEN],
    pk_x25519: &[u8; X25519_LEN],
) -> Zeroizing<[u8; KEY_LEN]> {
    let mut ikm = Zeroizing::new(Vec::with_capacity(64 + MLKEM768_CT_LEN + 2 * X25519_LEN));
    for part in [&ss_mlkem[..], ss_x25519, ct_mlkem, epk, pk_x25519] {
        ikm.extend_from_slice(part);
    }
    hkdf_sha256_array::<KEY_LEN>(label.salt, &ikm, &label.info)
}

/// 2026-10-22: Wrap `key` to `to` under `label`, with the randomness `rnd`. A recipient X25519
/// key of low order (a non-contributory exchange) is refused (`BlueprintKey`).
pub fn wrap_key(
    label: &WrapLabel,
    key: &Cek,
    to: &HybridPublic,
    rnd: &WrapRandom,
) -> Result<WrappedCek, Error> {
    let epk = x25519_public(&rnd.esk);
    let ss_x = x25519_shared(&rnd.esk, to.x25519()).ok_or_else(|| {
        error!(
            ErrorCode::BlueprintKey,
            alloc::format!("recipient {}", to.fingerprint()),
            "its X25519 public key is of low order (non-contributory)"
        )
    })?;
    let (ct_mlkem, ss_m) = mlkem768_encapsulate(to.mlkem_ek(), &rnd.m)?;
    let kek = derive_kek(label, &ss_m, &ss_x, &ct_mlkem, &epk, to.x25519());
    let sealed = aes256gcm_seal(&kek, &rnd.nonce, &label.aad(), key.as_bytes())?;
    let mut ct = [0u8; WRAPPED_CT_LEN];
    ct.copy_from_slice(&sealed);
    Ok(WrappedCek {
        epk,
        ct_mlkem,
        nonce: rnd.nonce,
        ct,
    })
}

/// 2026-10-22: Recover the key wrapped under `label` with `with`. Every failure is
/// `SealedUnwrap` naming the label's recipient.
pub fn unwrap_key(
    label: &WrapLabel,
    wrapped: &WrappedCek,
    with: &HybridSecret,
) -> Result<Cek, Error> {
    let refuse = || error!(ErrorCode::SealedUnwrap, label.recipient.hex());
    let ss_x = x25519_shared(with.x25519(), &wrapped.epk).ok_or_else(refuse)?;
    let ss_m = mlkem768_decapsulate(&mlkem768_from_seed(with.mlkem_seed()), &wrapped.ct_mlkem);
    let pk_x = x25519_public(with.x25519());
    let kek = derive_kek(label, &ss_m, &ss_x, &wrapped.ct_mlkem, &wrapped.epk, &pk_x);
    let key = aes256gcm_open(&kek, &wrapped.nonce, &label.aad(), &wrapped.ct).ok_or_else(refuse)?;
    if key.len() != KEY_LEN {
        return Err(refuse());
    }
    let mut out = Zeroizing::new([0u8; KEY_LEN]);
    out.copy_from_slice(&key);
    Ok(Cek(out))
}
