// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Resolving sealed modules (DESIGN-encrypted-kernels.md §3 steps 4-6): the
//! recipient unwraps the content key, and each sealed module's ciphertext is checked, opened,
//! decompressed and checked again; and the lite/full forms of a sealed file.
//!
//! Owner: blueprint.
//! Invariants (the order per sealed module, each step refusing with its own code):
//! 1. the recipient key's fingerprint is listed (`SealedNotRecipient`), then the hybrid unwrap
//!    (`SealedUnwrap`, opaque: wrong key, altered wrap and other body look alike);
//! 2. the ciphertext's length and SHA-256 equal the signed `ct_len` / `ct_sha256`
//!    (`SealedCiphertextHash`) **before any AEAD**: no decryption oracle on attacker bytes;
//! 3. AES-256-GCM open under the derived key, nonce and AAD (`SealedOpen`, opaque);
//! 4. bounded decompression to exactly the signed raw length under `MAX_ARTIFACT`;
//! 5. `ArtifactId::of(plain) == ModuleSpec.artifact` (`ArtifactHashMismatch`).
//! - Plaintext exists only in the returned buffers; nothing here writes, caches or keeps it,
//!   and a lite/full conversion moves ciphertext only.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};
use metrale_seal::{ArtifactBinding, Cek, HybridSecret, WrapLabel, open_artifact, unwrap_key};

use crate::codec::DecodeMemory;
use crate::frame::{
    InlineArtifact, KernelSource, MAX_ARTIFACT, Opened, Sealed, SealedArtifact, Unsigned,
};
use crate::ids::Sha256;
use crate::resolve::{checked, inline_all, source_bytes};
use crate::types::ModuleSpec;

fn sealed_of(opened: &Opened) -> Result<&Sealed, Error> {
    opened
        .sealed
        .as_ref()
        .ok_or_else(|| error!(ErrorCode::SealedManifest, "the file is not sealed"))
}

/// 2026-10-22: The content key of a sealed file, for the recipient holding `secret`.
pub fn unseal(opened: &Opened, secret: &HybridSecret) -> Result<Cek, Error> {
    let sealed = sealed_of(opened)?;
    let fp = secret.public().fingerprint();
    let r = sealed
        .manifest
        .recipients
        .iter()
        .find(|r| r.fingerprint == fp)
        .ok_or_else(|| error!(ErrorCode::SealedNotRecipient, fp.hex()))?;
    unwrap_key(&WrapLabel::kbp_cek(&opened.id.0.0, fp), &r.wrap, secret)
}

/// 2026-10-22: The ciphertext of sealed module `m`, checked against its signed length and hash.
fn ciphertext<'b, F, C>(
    opened: &Opened,
    m: &ModuleSpec,
    src: &KernelSource,
    a: &SealedArtifact,
    internal: &F,
    store: &C,
) -> Result<Cow<'b, [u8]>, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
    C: Fn(&Sha256) -> Option<Vec<u8>>,
{
    let bytes = source_bytes(m, src, opened, internal, store, Some(a))?;
    let found = Sha256::of(&[&bytes]);
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if len != a.ct_len || found != a.ct_sha256 {
        return Err(error!(
            ErrorCode::SealedCiphertextHash,
            m.name.clone(),
            alloc::format!("{} bytes {}", a.ct_len, a.ct_sha256),
            alloc::format!("{len} bytes {found}")
        ));
    }
    Ok(bytes)
}

/// 2026-10-22: Check every sealed module's ciphertext against the signed manifest without a key
/// (what a keyless reader can verify); the count of sealed modules checked.
pub fn verify_ciphertexts<'b, F, C>(opened: &Opened, internal: F, store: C) -> Result<u32, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
    C: Fn(&Sha256) -> Option<Vec<u8>>,
{
    let sealed = sealed_of(opened)?;
    let modules = &opened.blueprint.kernels.modules;
    let mut n = 0u32;
    for (i, (m, src)) in modules.iter().zip(&opened.unsigned.sources).enumerate() {
        if let Some(a) = sealed.artifact(i) {
            ciphertext(opened, m, src, a, &internal, &store)?;
            n += 1;
        }
    }
    Ok(n)
}

/// 2026-10-22: Every module's verified plaintext, in module order: plain modules as
/// `resolve_modules` does, sealed ones opened under `cek` (module invariants), every kzx
/// decode within `memory`.
pub fn resolve_unsealed<'b, F, C>(
    opened: &Opened,
    internal: F,
    store: C,
    cek: &Cek,
    memory: DecodeMemory,
) -> Result<Vec<(String, Cow<'b, [u8]>)>, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
    C: Fn(&Sha256) -> Option<Vec<u8>>,
{
    let sealed = sealed_of(opened)?;
    let codec = sealed.manifest.codec;
    let modules = &opened.blueprint.kernels.modules;
    let mut out = Vec::with_capacity(modules.len());
    for (i, (m, src)) in modules.iter().zip(&opened.unsigned.sources).enumerate() {
        let Some(a) = sealed.artifact(i) else {
            let bytes = source_bytes(m, src, opened, &internal, &store, None)?;
            out.push((m.name.clone(), checked(m, bytes)?));
            continue;
        };
        let ct = ciphertext(opened, m, src, a, &internal, &store)?;
        let binding = ArtifactBinding {
            blueprint: &opened.id.0.0,
            artifact: &m.artifact.0.0,
            module: &m.name,
            arch: &m.arch,
            codec: codec.id(),
            raw_len: m.len,
        };
        let comp = open_artifact(cek, &binding, &ct, a.compressed_len)?;
        let plain = codec.decompress("sealed module", &comp, m.len, MAX_ARTIFACT, memory)?;
        out.push((m.name.clone(), checked(m, Cow::Owned(plain))?));
    }
    Ok(out)
}

/// 2026-10-22: The address of a ciphertext under a store `base`: `<base>/<ct_sha256 hex>`. The
/// store is public and content-addressed (owner decision 2026-10-22): the path names the
/// ciphertext hash only, so `base` must not name a customer, module or blueprint.
pub fn ciphertext_url(base: &str, ct: &Sha256) -> String {
    alloc::format!("{}/{}", base.trim_end_matches('/'), ct.hex())
}

/// 2026-10-22: The lite form of a sealed file: no payload, plain modules `internal`, sealed
/// modules `remote` at [`ciphertext_url`] under `base` (the ciphertext bytes themselves go to the
/// store).
pub fn lite_sealed(opened: &Opened, base: &str) -> Result<(Unsigned, Vec<InlineArtifact>), Error> {
    let sealed = sealed_of(opened)?;
    let sources = (0..opened.blueprint.kernels.modules.len())
        .map(|i| match sealed.artifact(i) {
            Some(a) => KernelSource::Remote {
                url: ciphertext_url(base, &a.ct_sha256),
                locator_sha256: a.ct_sha256,
            },
            None => KernelSource::Internal,
        })
        .collect();
    Ok((
        Unsigned {
            sources,
            resources: opened.unsigned.resources.clone(),
        },
        Vec::new(),
    ))
}

/// 2026-10-22: The full form of a sealed file: plain modules' verified plaintext and sealed
/// modules' verified **ciphertext** inline (ciphertexts not inline come from `store`).
pub fn full_sealed<'b, F, C>(
    opened: &Opened,
    internal: F,
    store: C,
) -> Result<(Unsigned, Vec<InlineArtifact>), Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
    C: Fn(&Sha256) -> Option<Vec<u8>>,
{
    let sealed = sealed_of(opened)?;
    let modules = &opened.blueprint.kernels.modules;
    let mut bytes: Vec<Cow<'b, [u8]>> = Vec::with_capacity(modules.len());
    for (i, (m, src)) in modules.iter().zip(&opened.unsigned.sources).enumerate() {
        bytes.push(match sealed.artifact(i) {
            Some(a) => ciphertext(opened, m, src, a, &internal, &store)?,
            None => checked(m, source_bytes(m, src, opened, &internal, &store, None)?)?,
        });
    }
    let slices: Vec<&[u8]> = bytes.iter().map(|b| &**b).collect();
    let (sources, mut payload) = inline_all(&slices)?;
    // 2026-10-22: An inline entry names the plaintext it stands for, sealed or not
    // (`InlineArtifact.artifact` is informational; the signed hashes are what is checked).
    for (p, m) in payload.iter_mut().zip(modules) {
        p.artifact = m.artifact;
    }
    Ok((
        Unsigned {
            sources,
            resources: opened.unsigned.resources.clone(),
        },
        payload,
    ))
}
