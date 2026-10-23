// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Sealing a verified plain blueprint for its recipients (DESIGN-encrypted-kernels.md
//! §2.4, §6 step 5): every module compressed, then sealed under the run's content key; the key
//! wrapped to each recipient; the manifest signed; the file written with the body and its
//! signature untouched.
//!
//! Owner: blueprint.
//! Invariants:
//! - Compress, then seal: identity (`ArtifactId`, `BlueprintId`) stays over the raw bytes.
//! - The run's randomness arrives as one value ([`SealRandom`], one `Entropy::fill` by the
//!   host) and is consumed here: the CEK and every per-artifact key are dropped (zeroized)
//!   before [`seal`] returns, and none of them is in its output.
//! - Recipients are unique by fingerprint and written sorted; there is no escrow recipient
//!   beyond those the caller names (owner decision 2026-10-22).

use alloc::vec::Vec;

use metrale_blueprint_trust::{SignedCert, Signer, TrustPolicy};
use metrale_error::{Error, ErrorCode, error};
use metrale_seal::{
    ArtifactBinding, Fingerprint, HybridPublic, SealRandom, SealSuite, WrapLabel, seal_artifact,
    wrap_key,
};

use crate::codec::{Codec, DecodeMemory};
use crate::frame::sealed_check::{SEAL_TAG_LEN, sign_sealed};
use crate::frame::{
    self, InlineArtifact, KernelSource, Recipient, SEALED_SCHEMA, SealedArtifact, SealedManifest,
    Unsigned, customer_tag,
};
use crate::ids::Sha256;
use crate::resolve::{inline_all, resolve_modules};
use crate::resolve_sealed::ciphertext_url;

/// 2026-10-22: Where a sealed file's ciphertexts go.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement<'a> {
    /// 2026-10-22: Inline: a single-file sealed full `.kbp`.
    Inline,
    /// 2026-10-22: The public content-addressed store under `base`: a sealed lite `.kbp`, the
    /// ciphertexts returned for upload.
    Remote {
        /// 2026-10-22: The store base; a ciphertext is at `<base>/<ct_sha256 hex>`.
        base: &'a str,
    },
}

/// 2026-10-22: What to seal for whom.
#[derive(Clone, Copy, Debug)]
pub struct SealRequest<'a> {
    /// 2026-10-22: The recipients' public keys (at least one).
    pub recipients: &'a [HybridPublic],
    /// 2026-10-22: The customer id; only its opaque tag is written.
    pub customer_id: &'a str,
    /// 2026-10-22: Unix ms from the host's Clock.
    pub issued_ms: i64,
    /// 2026-10-22: The inner codec each module is compressed with before sealing.
    pub codec: Codec,
    /// 2026-10-22: Inline or remote ciphertexts.
    pub placement: Placement<'a>,
}

/// 2026-10-22: A sealed file and, for a remote placement, the ciphertexts by their hash.
#[derive(Clone, Debug)]
pub struct SealedOutput {
    /// 2026-10-22: The sealed `.kbp`.
    pub file: Vec<u8>,
    /// 2026-10-22: `(ct_sha256, ciphertext)` for the store; empty for an inline placement.
    pub ciphertexts: Vec<(Sha256, Vec<u8>)>,
    /// 2026-10-22: The recipients' fingerprints, sorted.
    pub recipients: Vec<Fingerprint>,
}

fn sorted_recipients(keys: &[HybridPublic]) -> Result<Vec<(Fingerprint, &HybridPublic)>, Error> {
    let mut out: Vec<(Fingerprint, &HybridPublic)> =
        keys.iter().map(|k| (k.fingerprint(), k)).collect();
    out.sort_by_key(|(fp, _)| *fp);
    if out.is_empty() || out.windows(2).any(|w| w[0].0 == w[1].0) {
        return Err(error!(
            ErrorCode::SealedManifest,
            "recipients must be at least one key, each listed once"
        ));
    }
    Ok(out)
}

fn len64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

/// 2026-10-22: Seal the plain file `file` (verified under `policy`) for `req`, with the run's
/// randomness `random` (one wrap per recipient) and the manifest signed by `signer` / `chain`.
/// `internal` supplies embedded modules for a lite input.
pub fn seal<'b, F>(
    file: &[u8],
    policy: &TrustPolicy<'_>,
    memory: DecodeMemory,
    internal: F,
    req: &SealRequest<'_>,
    random: SealRandom,
    signer: &Signer,
    chain: Vec<SignedCert>,
) -> Result<SealedOutput, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
{
    let opened = frame::open(file, policy, memory)?;
    if opened.sealed.is_some() {
        return Err(error!(
            ErrorCode::SealedManifest,
            "the file is already sealed"
        ));
    }
    if req.customer_id.is_empty() {
        return Err(error!(
            ErrorCode::SealedManifest,
            "the customer id is empty"
        ));
    }
    let recipients = sorted_recipients(req.recipients)?;
    if random.wraps.len() != recipients.len() {
        return Err(error!(
            ErrorCode::InvalidArgument,
            "seal randomness", "it must hold one wrap per recipient"
        ));
    }
    let plain = resolve_modules(&opened, internal)?;
    let blueprint = &opened.id.0.0;
    let mut artifacts = Vec::with_capacity(plain.len());
    let mut cts = Vec::with_capacity(plain.len());
    for (i, (m, (_, raw))) in opened
        .blueprint
        .kernels
        .modules
        .iter()
        .zip(&plain)
        .enumerate()
    {
        let comp = req.codec.compress(raw);
        let binding = ArtifactBinding {
            blueprint,
            artifact: &m.artifact.0.0,
            module: &m.name,
            arch: &m.arch,
            codec: req.codec.id(),
            raw_len: m.len,
        };
        let ct = seal_artifact(&random.cek, &binding, &comp)?;
        artifacts.push(SealedArtifact {
            module: u32::try_from(i)
                .map_err(|_| error!(ErrorCode::BlueprintFrame, "too many modules"))?,
            compressed_len: len64(comp.len()),
            ct_len: len64(comp.len()) + SEAL_TAG_LEN,
            ct_sha256: Sha256::of(&[&ct]),
        });
        cts.push(ct);
    }
    let wrapped = recipients
        .iter()
        .zip(&random.wraps)
        .map(|((fp, pk), rnd)| {
            Ok(Recipient {
                fingerprint: *fp,
                wrap: wrap_key(&WrapLabel::kbp_cek(blueprint, *fp), &random.cek, pk, rnd)?,
            })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    drop(random);
    let manifest = SealedManifest {
        schema: SEALED_SCHEMA,
        blueprint: opened.id,
        suite: SealSuite::HybridX25519MlKem768HkdfA256Gcm,
        codec: req.codec,
        customer: customer_tag(req.customer_id),
        issued_ms: req.issued_ms,
        recipients: wrapped,
        artifacts,
    };
    let section = sign_sealed(&manifest, signer, chain)?;
    let (sources, payload, ciphertexts) = place(&opened.blueprint, &manifest, cts, req.placement)?;
    let unsigned = Unsigned {
        sources,
        resources: opened.unsigned.resources.clone(),
    };
    let file = frame::write_sealed(file, &opened, &unsigned, &payload, &manifest, &section)?;
    Ok(SealedOutput {
        file,
        ciphertexts,
        recipients: recipients.iter().map(|(fp, _)| *fp).collect(),
    })
}

type Placed = (
    Vec<KernelSource>,
    Vec<InlineArtifact>,
    Vec<(Sha256, Vec<u8>)>,
);

fn place(
    bp: &crate::types::Blueprint,
    manifest: &SealedManifest,
    cts: Vec<Vec<u8>>,
    placement: Placement<'_>,
) -> Result<Placed, Error> {
    match placement {
        Placement::Inline => {
            let slices: Vec<&[u8]> = cts.iter().map(Vec::as_slice).collect();
            let (sources, mut payload) = inline_all(&slices)?;
            for (p, m) in payload.iter_mut().zip(&bp.kernels.modules) {
                p.artifact = m.artifact;
            }
            Ok((sources, payload, Vec::new()))
        }
        Placement::Remote { base } => {
            let sources = manifest
                .artifacts
                .iter()
                .map(|a| KernelSource::Remote {
                    url: ciphertext_url(base, &a.ct_sha256),
                    locator_sha256: a.ct_sha256,
                })
                .collect();
            let out = manifest
                .artifacts
                .iter()
                .map(|a| a.ct_sha256)
                .zip(cts)
                .collect();
            Ok((sources, Vec::new(), out))
        }
    }
}
