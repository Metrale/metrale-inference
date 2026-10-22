// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Hybrid keys: a [`Signer`] expanded once from two 32-byte seeds, the public half
//! ([`PublicKeys`]) and [`verify_hybrid`].
//!
//! Owner: blueprint.
//! Invariants:
//! - ML-DSA signs `msg` with the purpose's context (`sign_deterministic(msg, ctx)`); Ed25519
//!   signs [`lv_cat`]`([ctx, msg])`, metrale-auth's length-prefixed layout. Both are
//!   deterministic, so signing the same bytes twice gives the same signature.
//! - [`verify_hybrid`] refuses unless both halves verify; the ML-DSA half is checked first.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use ed25519_dalek::Signer as _;
use metrale_error::{Error, ErrorCode, error};
use ml_dsa::signature::Keypair;
use ml_dsa::{
    B32, EncodedVerifyingKey, MlDsa65, MlDsa87, MlDsaParams, Signature, SigningKey, VerifyingKey,
};
use serde::{Deserialize, Serialize};

use crate::names::{Kid, Purpose, Suite};

/// 2026-10-22: Length-value concatenation: each part as a big-endian `u32` length then its
/// bytes (metrale-auth `encoding::lv_cat`), so no two part lists share an encoding.
pub fn lv_cat(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len() + 4).sum());
    for p in parts {
        // 2026-10-22: Every caller's parts are far below 4 GiB (bodies are capped at 4 MiB).
        let len = u32::try_from(p.len()).unwrap_or(u32::MAX);
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(p);
    }
    out
}

/// 2026-10-22: The public half of a hybrid key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PublicKeys {
    /// 2026-10-22: Which ML-DSA parameter set the key is.
    pub suite: Suite,
    /// 2026-10-22: The encoded ML-DSA verifying key.
    pub mldsa: Vec<u8>,
    /// 2026-10-22: The Ed25519 verifying key.
    pub ed25519: [u8; 32],
}

/// 2026-10-22: A hybrid signature: both halves, encoded.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct HybridSig {
    /// 2026-10-22: The encoded ML-DSA signature.
    pub mldsa: Vec<u8>,
    /// 2026-10-22: The 64-byte Ed25519 signature.
    pub ed25519: Vec<u8>,
}

enum MlDsaKey {
    L65(Box<SigningKey<MlDsa65>>),
    L87(Box<SigningKey<MlDsa87>>),
}

/// 2026-10-22: A hybrid signing key, expanded once from its seeds. The seeds themselves stay
/// with the caller (the host crate zeroizes them).
pub struct Signer {
    kid: Kid,
    suite: Suite,
    mldsa: MlDsaKey,
    ed25519: ed25519_dalek::SigningKey,
}

fn sign_mldsa<P: MlDsaParams>(k: &SigningKey<P>, msg: &[u8], ctx: &[u8]) -> Result<Vec<u8>, Error> {
    k.expanded_key()
        .sign_deterministic(msg, ctx)
        .map(|s| s.encode().to_vec())
        .map_err(|_| error!(ErrorCode::BlueprintKey, "ml-dsa", "signing failed"))
}

impl Signer {
    /// 2026-10-22: The signer `kid` of `suite` for the two seeds.
    pub fn from_seeds(
        kid: Kid,
        suite: Suite,
        mldsa_seed: &[u8; 32],
        ed25519_seed: &[u8; 32],
    ) -> Self {
        let xi = B32::from(*mldsa_seed);
        let mldsa = match suite {
            Suite::MlDsa65Ed25519 => MlDsaKey::L65(Box::new(SigningKey::<MlDsa65>::from_seed(&xi))),
            Suite::MlDsa87Ed25519 => MlDsaKey::L87(Box::new(SigningKey::<MlDsa87>::from_seed(&xi))),
        };
        Self {
            kid,
            suite,
            mldsa,
            ed25519: ed25519_dalek::SigningKey::from_bytes(ed25519_seed),
        }
    }

    /// 2026-10-22: The key id.
    pub fn kid(&self) -> &Kid {
        &self.kid
    }

    /// 2026-10-22: The public half.
    pub fn public(&self) -> PublicKeys {
        let mldsa = match &self.mldsa {
            MlDsaKey::L65(k) => k.verifying_key().encode().to_vec(),
            MlDsaKey::L87(k) => k.verifying_key().encode().to_vec(),
        };
        PublicKeys {
            suite: self.suite,
            mldsa,
            ed25519: self.ed25519.verifying_key().to_bytes(),
        }
    }

    /// 2026-10-22: Sign `msg` for `purpose`: deterministic, so equal input gives equal bytes.
    pub fn sign(&self, purpose: Purpose, msg: &[u8]) -> Result<HybridSig, Error> {
        let ctx = purpose.context();
        let mldsa = match &self.mldsa {
            MlDsaKey::L65(k) => sign_mldsa(k, msg, ctx)?,
            MlDsaKey::L87(k) => sign_mldsa(k, msg, ctx)?,
        };
        let ed25519 = self.ed25519.sign(&lv_cat(&[ctx, msg])).to_bytes().to_vec();
        Ok(HybridSig { mldsa, ed25519 })
    }
}

fn verify_mldsa<P: MlDsaParams>(pk: &[u8], msg: &[u8], ctx: &[u8], sig: &[u8]) -> bool {
    let Ok(enc) = EncodedVerifyingKey::<P>::try_from(pk) else {
        return false;
    };
    let Ok(sig) = Signature::<P>::try_from(sig) else {
        return false;
    };
    VerifyingKey::<P>::decode(&enc).verify_with_context(msg, ctx, &sig)
}

fn verify_ed25519(pk: &[u8; 32], msg: &[u8], sig: &[u8]) -> bool {
    let Ok(sig) = <[u8; 64]>::try_from(sig) else {
        return false;
    };
    let Ok(vk) = ed25519_dalek::VerifyingKey::from_bytes(pk) else {
        return false;
    };
    vk.verify_strict(msg, &ed25519_dalek::Signature::from_bytes(&sig))
        .is_ok()
}

/// 2026-10-22: Check `sig` over `msg` for `purpose` under `pk`; `what` and `kid` name the
/// refusal. Both halves must verify.
pub fn verify_hybrid(
    pk: &PublicKeys,
    purpose: Purpose,
    msg: &[u8],
    sig: &HybridSig,
    what: &'static str,
    kid: &Kid,
) -> Result<(), Error> {
    let ctx = purpose.context();
    let mldsa_ok = match pk.suite {
        Suite::MlDsa65Ed25519 => verify_mldsa::<MlDsa65>(&pk.mldsa, msg, ctx, &sig.mldsa),
        Suite::MlDsa87Ed25519 => verify_mldsa::<MlDsa87>(&pk.mldsa, msg, ctx, &sig.mldsa),
    };
    if !mldsa_ok {
        return Err(error!(
            ErrorCode::BlueprintSignature,
            what,
            pk.suite.mldsa_name(),
            String::from(kid.as_str())
        ));
    }
    if !verify_ed25519(&pk.ed25519, &lv_cat(&[ctx, msg]), &sig.ed25519) {
        return Err(error!(
            ErrorCode::BlueprintSignature,
            what,
            "Ed25519",
            String::from(kid.as_str())
        ));
    }
    Ok(())
}
