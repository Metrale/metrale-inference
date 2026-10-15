// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: the hybrid DEK wrap (DESIGN §7.1, QUESTIONS Q4 option a).
//!
//! The KeyRing holds an X25519 secret key and an ML-KEM-768 decapsulation
//! key. To wrap a publish DEK for key id `kid`:
//!
//! ```text
//! esk, m, nonce   <- Entropy
//! epk             =  X25519(esk, 9)
//! ss_X            =  X25519(esk, pk_X)              (must be contributory)
//! (ct_M, ss_M)    =  ML-KEM-768.Encaps_internal(ek, m)
//! KEK             =  HKDF-SHA256(salt = WRAP_SCHEME,
//!                                ikm  = ss_M ‖ ss_X ‖ ct_M ‖ epk ‖ pk_X,
//!                                info = "kkb/v1/wrap/<kid>")
//! ct              =  AES-256-GCM(KEK, nonce, aad = "kkb/v1/wrap/<kid>", DEK)
//! ```
//!
//! The combiner follows the X-Wing ordering (ML-KEM secret first, then the
//! X25519 secret, ciphertext and both X25519 public values), so the KEK stays
//! secret while either component KEM holds. Unwrap recomputes the KEK from
//! `ss_X = X25519(sk_X, epk)` and `ss_M = Decaps(dk, ct_M)`; every failure is
//! the single [`EnvelopeError::Unwrap`].

use core::fmt;

use x25519_dalek::{PublicKey, StaticSecret};
use zeroize::Zeroizing;

use crate::error::{EnvelopeError, Result};
use crate::format::{WRAP_SCHEME, WrappedDek, validate_kid};
use crate::ports::Entropy;
use crate::primitives::{
    KEY_LEN, MLKEM_M_LEN, MLKEM_SEED_LEN, MLKEM768_CT_LEN, MLKEM768_EK_LEN, NONCE_LEN,
    aes256gcm_open, aes256gcm_seal, hkdf_sha256_key, mlkem768_decapsulate, mlkem768_ek_bytes,
    mlkem768_encapsulate, mlkem768_from_seed, mlkem768_parse_ek, sha256,
};
use crate::seal::Dek;

const X25519_LEN: usize = 32;
/// Serialized [`KeyRingSecret`]: X25519 secret ‖ ML-KEM-768 seed `d ‖ z`.
pub const KEYRING_SECRET_LEN: usize = X25519_LEN + MLKEM_SEED_LEN;
/// Serialized [`KeyRingPublic`]: X25519 public ‖ ML-KEM-768 encapsulation key.
pub const KEYRING_PUBLIC_LEN: usize = X25519_LEN + MLKEM768_EK_LEN;

/// The KeyRing's hybrid secret key. Wiped on drop, never printed.
#[derive(Clone)]
pub struct KeyRingSecret {
    x25519: Zeroizing<[u8; X25519_LEN]>,
    mlkem_seed: Zeroizing<[u8; MLKEM_SEED_LEN]>,
}

impl fmt::Debug for KeyRingSecret {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("KeyRingSecret(<redacted>)")
    }
}

impl KeyRingSecret {
    /// A fresh keypair drawn from `entropy`.
    pub fn generate(entropy: &mut dyn Entropy) -> Result<Self> {
        let mut bytes = Zeroizing::new([0u8; KEYRING_SECRET_LEN]);
        entropy.fill(bytes.as_mut_slice())?;
        Ok(Self::from_bytes(&bytes))
    }

    /// Restores a secret key from its 96-byte serialization.
    pub fn from_bytes(bytes: &[u8; KEYRING_SECRET_LEN]) -> Self {
        let mut x25519 = Zeroizing::new([0u8; X25519_LEN]);
        let mut mlkem_seed = Zeroizing::new([0u8; MLKEM_SEED_LEN]);
        x25519.copy_from_slice(&bytes[..X25519_LEN]);
        mlkem_seed.copy_from_slice(&bytes[X25519_LEN..]);
        Self { x25519, mlkem_seed }
    }

    /// The 96-byte serialization: X25519 secret ‖ ML-KEM seed `d ‖ z`.
    pub fn to_bytes(&self) -> Zeroizing<[u8; KEYRING_SECRET_LEN]> {
        let mut out = Zeroizing::new([0u8; KEYRING_SECRET_LEN]);
        out[..X25519_LEN].copy_from_slice(self.x25519.as_slice());
        out[X25519_LEN..].copy_from_slice(self.mlkem_seed.as_slice());
        out
    }

    /// The matching public key.
    pub fn public(&self) -> KeyRingPublic {
        let x25519 = PublicKey::from(&StaticSecret::from(*self.x25519)).to_bytes();
        let mlkem_ek = mlkem768_ek_bytes(&mlkem768_from_seed(&self.mlkem_seed));
        KeyRingPublic { x25519, mlkem_ek }
    }
}

/// The KeyRing's hybrid public key, the value a publisher pins.
#[derive(Clone, PartialEq, Eq)]
pub struct KeyRingPublic {
    x25519: [u8; X25519_LEN],
    mlkem_ek: [u8; MLKEM768_EK_LEN],
}

impl fmt::Debug for KeyRingPublic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fp = self.fingerprint();
        write!(f, "KeyRingPublic(sha256:")?;
        fp.iter().try_for_each(|b| write!(f, "{b:02x}"))?;
        f.write_str(")")
    }
}

impl KeyRingPublic {
    /// Parses a 1216-byte public key, applying the FIPS 203 §7.2 modulus
    /// check to the ML-KEM encapsulation key.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() != KEYRING_PUBLIC_LEN {
            return Err(EnvelopeError::BadLength {
                field: "keyring public key",
                expected: KEYRING_PUBLIC_LEN,
                actual: bytes.len(),
            });
        }
        let mut x25519 = [0u8; X25519_LEN];
        let mut mlkem_ek = [0u8; MLKEM768_EK_LEN];
        x25519.copy_from_slice(&bytes[..X25519_LEN]);
        mlkem_ek.copy_from_slice(&bytes[X25519_LEN..]);
        mlkem768_parse_ek(&mlkem_ek)?;
        Ok(Self { x25519, mlkem_ek })
    }

    /// The 1216-byte serialization: X25519 public ‖ ML-KEM-768 `ek`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(KEYRING_PUBLIC_LEN);
        out.extend_from_slice(&self.x25519);
        out.extend_from_slice(&self.mlkem_ek);
        out
    }

    /// SHA-256 of [`Self::to_bytes`]: the fingerprint a publisher pins.
    pub fn fingerprint(&self) -> [u8; 32] {
        sha256(&self.to_bytes())
    }
}

/// The AAD and HKDF info of a wrap for `kid`.
fn wrap_label(kid: &str) -> Result<String> {
    validate_kid(kid)?;
    Ok(format!("kkb/v1/wrap/{kid}"))
}

fn derive_kek(
    label: &str,
    ss_mlkem: &[u8; 32],
    ss_x25519: &[u8; 32],
    ct_mlkem: &[u8; MLKEM768_CT_LEN],
    epk: &[u8; X25519_LEN],
    pk_x25519: &[u8; X25519_LEN],
) -> Result<Zeroizing<[u8; KEY_LEN]>> {
    let mut ikm = Zeroizing::new(Vec::with_capacity(
        32 + 32 + MLKEM768_CT_LEN + 2 * X25519_LEN,
    ));
    ikm.extend_from_slice(ss_mlkem);
    ikm.extend_from_slice(ss_x25519);
    ikm.extend_from_slice(ct_mlkem);
    ikm.extend_from_slice(epk);
    ikm.extend_from_slice(pk_x25519);
    hkdf_sha256_key(WRAP_SCHEME.as_bytes(), &ikm, label.as_bytes())
}

/// Wraps `dek` for `kid` to the KeyRing public key `to`.
pub fn wrap_dek(
    dek: &Dek,
    kid: &str,
    to: &KeyRingPublic,
    entropy: &mut dyn Entropy,
) -> Result<WrappedDek> {
    let label = wrap_label(kid)?;
    let mut esk = Zeroizing::new([0u8; X25519_LEN]);
    let mut m = Zeroizing::new([0u8; MLKEM_M_LEN]);
    let mut nonce = [0u8; NONCE_LEN];
    entropy.fill(esk.as_mut_slice())?;
    entropy.fill(m.as_mut_slice())?;
    entropy.fill(&mut nonce)?;

    let esk = StaticSecret::from(*esk);
    let epk = PublicKey::from(&esk).to_bytes();
    let shared_x = esk.diffie_hellman(&PublicKey::from(to.x25519));
    if !shared_x.was_contributory() {
        return Err(EnvelopeError::NonContributory);
    }
    let (ct_mlkem, ss_mlkem) = mlkem768_encapsulate(&to.mlkem_ek, &m)?;
    let kek = derive_kek(
        &label,
        &ss_mlkem,
        shared_x.as_bytes(),
        &ct_mlkem,
        &epk,
        &to.x25519,
    )?;
    let ct_vec = aes256gcm_seal(&kek, &nonce, label.as_bytes(), dek.as_bytes())?;
    let ct = ct_vec
        .as_slice()
        .try_into()
        .map_err(|_| EnvelopeError::BadLength {
            field: "wrapped_dek.ct",
            expected: KEY_LEN + 16,
            actual: ct_vec.len(),
        })?;
    Ok(WrappedDek {
        ct,
        ct_mlkem,
        epk_x25519: epk,
        nonce,
        scheme: WRAP_SCHEME.to_owned(),
    })
}

/// Recovers the DEK wrapped for `kid` with the KeyRing secret key `with`.
pub fn unwrap_dek(wrapped: &WrappedDek, kid: &str, with: &KeyRingSecret) -> Result<Dek> {
    if wrapped.scheme != WRAP_SCHEME {
        return Err(EnvelopeError::Unsupported {
            field: "wrapped_dek.scheme",
            value: wrapped.scheme.clone(),
        });
    }
    let label = wrap_label(kid)?;
    let sk_x = StaticSecret::from(*with.x25519);
    let pk_x = PublicKey::from(&sk_x).to_bytes();
    let shared_x = sk_x.diffie_hellman(&PublicKey::from(wrapped.epk_x25519));
    if !shared_x.was_contributory() {
        return Err(EnvelopeError::Unwrap);
    }
    let dk = mlkem768_from_seed(&with.mlkem_seed);
    let ss_mlkem = mlkem768_decapsulate(&dk, &wrapped.ct_mlkem);
    let kek = derive_kek(
        &label,
        &ss_mlkem,
        shared_x.as_bytes(),
        &wrapped.ct_mlkem,
        &wrapped.epk_x25519,
        &pk_x,
    )?;
    let dek = Zeroizing::new(
        aes256gcm_open(&kek, &wrapped.nonce, label.as_bytes(), &wrapped.ct)
            .map_err(|_| EnvelopeError::Unwrap)?,
    );
    let bytes: [u8; KEY_LEN] = dek
        .as_slice()
        .try_into()
        .map_err(|_| EnvelopeError::Unwrap)?;
    Ok(Dek::from_bytes(bytes))
}
