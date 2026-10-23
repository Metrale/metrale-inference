// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Verifying the sealed section (DESIGN-encrypted-kernels.md §3 step 2): the block
//! decodes, its signature verifies, and only then does the manifest decode and bind.
//!
//! Owner: blueprint.
//! Invariants (the order of [`open_sealed`], after the body signature has verified):
//! 1. the block decodes (bytes and a signature block only);
//! 2. `verify_signed(KbpSealed, Blueprints)` over `"kbp-sealed-v1\0" ‖ manifest`;
//! 3. the manifest's schema, then its canonical decode;
//! 4. `manifest.blueprint == BlueprintId` of the body (`SealedManifest`: a section transplanted
//!    onto another body is refused here);
//! 5. recipients and artifacts non-empty, strictly sorted, every module in range, every
//!    `ct_len == compressed_len + 16`, every compressed length under `MAX_ARTIFACT`, and no
//!    sealed module sourced `internal` (the binary would hold plaintext the manifest says is
//!    sealed).

use alloc::vec::Vec;

use metrale_blueprint_trust::{
    Purpose, Scope, SigBlock, SignedCert, Signer, TrustPolicy, Verified, lv_cat, verify_signed,
};
use metrale_error::{Error, ErrorCode, error};

use super::sealed::{
    SEALED_SCHEMA, SEALED_SIG_PREFIX, SealedArtifact, SealedBlock, SealedManifest,
};
use super::{KernelSource, MAX_ARTIFACT, Unsigned, decode, encode};
use crate::ids::{BlueprintId, Sha256};
use crate::types::Blueprint;

/// 2026-10-22: The AEAD tag every ciphertext carries beyond its compressed bytes.
pub const SEAL_TAG_LEN: u64 = 16;

/// 2026-10-22: A verified sealed section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sealed {
    /// 2026-10-22: The manifest.
    pub manifest: SealedManifest,
    /// 2026-10-22: Who signed it.
    pub signer: Verified,
}

impl Sealed {
    /// 2026-10-22: The manifest entry of module `index`, if that module is sealed.
    pub fn artifact(&self, index: usize) -> Option<&SealedArtifact> {
        let i = u32::try_from(index).ok()?;
        self.manifest
            .artifacts
            .binary_search_by_key(&i, |a| a.module)
            .ok()
            .map(|at| &self.manifest.artifacts[at])
    }
}

/// 2026-10-22: The opaque customer tag of `customer_id`: `sha256("kbp-customer\0" ‖
/// lv(customer_id))`.
pub fn customer_tag(customer_id: &str) -> Sha256 {
    Sha256::of(&[b"kbp-customer\0", &lv_cat(&[customer_id.as_bytes()])])
}

/// 2026-10-22: The bytes the sealed signature covers.
pub fn signed_message(manifest: &[u8]) -> Vec<u8> {
    let mut m = Vec::with_capacity(SEALED_SIG_PREFIX.len() + manifest.len());
    m.extend_from_slice(SEALED_SIG_PREFIX);
    m.extend_from_slice(manifest);
    m
}

/// 2026-10-22: The sealed section bytes for `manifest`, signed by `signer` with `chain`.
pub fn sign_sealed(
    manifest: &SealedManifest,
    signer: &Signer,
    chain: Vec<SignedCert>,
) -> Result<Vec<u8>, Error> {
    let bytes = encode("sealed manifest", manifest)?;
    let sig = signer.sign(Purpose::KbpSealed, &signed_message(&bytes))?;
    encode(
        "sealed section",
        &SealedBlock {
            manifest: bytes,
            sig: SigBlock {
                kid: signer.kid().clone(),
                chain,
                sig,
            },
        },
    )
}

fn bad(why: &'static str) -> Error {
    error!(ErrorCode::SealedManifest, why)
}

fn decode_manifest(bytes: &[u8]) -> Result<SealedManifest, Error> {
    let schema = postcard::take_from_bytes::<u32>(bytes)
        .map_err(|_| error!(ErrorCode::BlueprintDecode, "sealed manifest", "no schema"))?
        .0;
    if !(1..=SEALED_SCHEMA).contains(&schema) {
        return Err(error!(
            ErrorCode::BlueprintFormat,
            "SEALED_SCHEMA",
            schema,
            alloc::format!("1..={SEALED_SCHEMA}")
        ));
    }
    let m: SealedManifest = decode("sealed manifest", bytes)?;
    if encode("sealed manifest", &m)? != bytes {
        return Err(error!(
            ErrorCode::BlueprintDecode,
            "sealed manifest", "not canonical"
        ));
    }
    Ok(m)
}

fn strictly_sorted<T, K: Ord>(items: &[T], key: impl Fn(&T) -> K) -> bool {
    items.windows(2).all(|w| key(&w[0]) < key(&w[1]))
}

/// 2026-10-22: Check `m` against the body it claims and the locator table beside it.
pub fn check_manifest(
    m: &SealedManifest,
    id: &BlueprintId,
    bp: &Blueprint,
    unsigned: &Unsigned,
) -> Result<(), Error> {
    if m.blueprint != *id {
        return Err(bad("it is bound to another blueprint body"));
    }
    if m.recipients.is_empty() || !strictly_sorted(&m.recipients, |r| r.fingerprint) {
        return Err(bad("recipients must be non-empty, sorted and unique"));
    }
    if m.artifacts.is_empty() || !strictly_sorted(&m.artifacts, |a| a.module) {
        return Err(bad(
            "artifacts must be non-empty, sorted by module and unique",
        ));
    }
    for a in &m.artifacts {
        let src = usize::try_from(a.module)
            .ok()
            .filter(|i| *i < bp.kernels.modules.len())
            .and_then(|i| unsigned.sources.get(i))
            .ok_or_else(|| bad("an artifact names a module the body does not have"))?;
        if a.compressed_len.checked_add(SEAL_TAG_LEN) != Some(a.ct_len) {
            return Err(bad(
                "an artifact's ciphertext length is not its compressed length + 16",
            ));
        }
        if a.compressed_len > MAX_ARTIFACT {
            return Err(error!(
                ErrorCode::BlueprintTooLarge,
                "sealed artifact", a.compressed_len, MAX_ARTIFACT
            ));
        }
        if matches!(src, KernelSource::Internal) {
            return Err(bad(
                "a sealed module's source is internal (the binary would hold its plaintext)",
            ));
        }
    }
    Ok(())
}

/// 2026-10-22: Verify and decode the sealed `section` of a file whose body (`id`, `bp`) and
/// locator table (`unsigned`) have already verified and decoded; `None` for an unsealed file
/// (the header guarantees an empty section). See the module invariants for the order.
pub fn open_sealed(
    policy: &TrustPolicy<'_>,
    section: &[u8],
    id: &BlueprintId,
    bp: &Blueprint,
    unsigned: &Unsigned,
) -> Result<Option<Sealed>, Error> {
    if section.is_empty() {
        return Ok(None);
    }
    let block: SealedBlock = decode("sealed section", section)?;
    let signer = verify_signed(
        policy,
        &block.sig,
        Scope::Blueprints,
        Purpose::KbpSealed,
        &signed_message(&block.manifest),
        "sealed manifest",
    )?;
    let manifest = decode_manifest(&block.manifest)?;
    check_manifest(&manifest, id, bp, unsigned)?;
    Ok(Some(Sealed { manifest, signer }))
}
