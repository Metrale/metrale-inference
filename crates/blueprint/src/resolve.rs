// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Kernel resolution: each module's bytes from its source, checked against the
//! signed hash; and the lite ↔ full conversions built on it.
//!
//! Owner: blueprint.
//! Invariants:
//! - Every source is checked the same way: the bytes' SHA-256 must equal the signed
//!   `ModuleSpec.artifact` (`ArtifactHashMismatch` names the module), whatever the locator says.
//! - Phase 1 resolves `internal` and `inline`; `disk` and `remote` parse and are refused with
//!   `KernelSourceNotInPhase`, except that a sealed module's ciphertext may come from the
//!   caller's content-addressed `store` (by its signed hash; `crate::resolve_sealed`).
//! - The caller supplies the embedded modules through `internal` and stored ciphertexts through
//!   `store` (no I/O here).
//! - A sealed module never resolves here: without a recipient key it is `SealedKeyRequired`
//!   (`crate::resolve_sealed` opens it), and [`full`] inlines its ciphertext, never plaintext.

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};

use crate::frame::{InlineArtifact, KernelSource, Opened, SealedArtifact, Unsigned};
use crate::ids::{ArtifactId, Sha256};
use crate::types::ModuleSpec;

/// 2026-10-22: A ciphertext store that holds nothing: for files whose sealed modules are inline.
pub fn no_store(_: &Sha256) -> Option<Vec<u8>> {
    None
}

/// 2026-10-22: `bytes` if they hash to `m`'s signed artifact id, else `ArtifactHashMismatch`.
pub(crate) fn checked<'a>(m: &ModuleSpec, bytes: Cow<'a, [u8]>) -> Result<Cow<'a, [u8]>, Error> {
    let found = ArtifactId::of(&bytes);
    if found != m.artifact {
        return Err(error!(
            ErrorCode::ArtifactHashMismatch,
            m.name.clone(),
            m.artifact.0.hex(),
            found.0.hex()
        ));
    }
    Ok(bytes)
}

fn inline_bytes(m: &ModuleSpec, payload: &[InlineArtifact], entry: u32) -> Result<Vec<u8>, Error> {
    let e = usize::try_from(entry)
        .ok()
        .and_then(|i| payload.get(i))
        .ok_or_else(|| {
            error!(
                ErrorCode::KernelMissing,
                m.name.clone(),
                m.artifact.0.hex(),
                "inline"
            )
        })?;
    Ok(e.bytes.clone())
}

/// 2026-10-22: The bytes module `m`'s source yields, unchecked: plaintext for a plain module,
/// ciphertext for a sealed one (`sealed`, whose `disk`/`remote` bytes come from `store` by the
/// signed hash).
pub(crate) fn source_bytes<'b, F, C>(
    m: &ModuleSpec,
    src: &KernelSource,
    opened: &Opened,
    internal: &F,
    store: &C,
    sealed: Option<&SealedArtifact>,
) -> Result<Cow<'b, [u8]>, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
    C: Fn(&Sha256) -> Option<Vec<u8>>,
{
    let not_in_phase = |source: &'static str, phase: u32| {
        error!(
            ErrorCode::KernelSourceNotInPhase,
            source,
            m.name.clone(),
            phase
        )
    };
    match (src, sealed) {
        (KernelSource::Internal, _) => Ok(Cow::Borrowed(internal(&m.name).ok_or_else(|| {
            error!(
                ErrorCode::KernelMissing,
                m.name.clone(),
                m.artifact.0.hex(),
                "internal"
            )
        })?)),
        (KernelSource::Inline { entry }, _) => {
            Ok(Cow::Owned(inline_bytes(m, &opened.payload, *entry)?))
        }
        (KernelSource::Disk { .. } | KernelSource::Remote { .. }, Some(a)) => {
            store(&a.ct_sha256).map(Cow::Owned).ok_or_else(|| {
                error!(
                    ErrorCode::KernelMissing,
                    m.name.clone(),
                    a.ct_sha256.hex(),
                    src.name()
                )
            })
        }
        (KernelSource::Disk { .. }, None) => Err(not_in_phase("disk", 2)),
        (KernelSource::Remote { .. }, None) => Err(not_in_phase("remote", 3)),
    }
}

/// 2026-10-22: Every module's verified bytes, in module order. `internal` returns the
/// binary's embedded bytes for a module name. A sealed module is `SealedKeyRequired`.
pub fn resolve_modules<'b, F>(
    opened: &Opened,
    internal: F,
) -> Result<Vec<(String, Cow<'b, [u8]>)>, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
{
    let modules = &opened.blueprint.kernels.modules;
    let mut out = Vec::with_capacity(modules.len());
    for (i, (m, src)) in modules.iter().zip(&opened.unsigned.sources).enumerate() {
        if opened.sealed.as_ref().and_then(|s| s.artifact(i)).is_some() {
            return Err(error!(ErrorCode::SealedKeyRequired, m.name.clone()));
        }
        let bytes = source_bytes(m, src, opened, &internal, &no_store, None)?;
        out.push((m.name.clone(), checked(m, bytes)?));
    }
    Ok(out)
}

/// 2026-10-22: The lite form of a plain file: no payload, every module `internal`. A sealed
/// file's lite form is [`crate::resolve_sealed::lite_sealed`] (`frame::rewrite` refuses an
/// `internal` sealed module).
pub fn lite(opened: &Opened) -> (Unsigned, Vec<InlineArtifact>) {
    let sources = opened
        .blueprint
        .kernels
        .modules
        .iter()
        .map(|_| KernelSource::Internal)
        .collect();
    (
        Unsigned {
            sources,
            resources: opened.unsigned.resources.clone(),
        },
        Vec::new(),
    )
}

/// 2026-10-22: Inline sources and payload entries for `modules`' bytes, in module order.
pub fn inline_all(modules: &[&[u8]]) -> Result<(Vec<KernelSource>, Vec<InlineArtifact>), Error> {
    let mut payload = Vec::with_capacity(modules.len());
    let mut sources = Vec::with_capacity(modules.len());
    for (i, bytes) in modules.iter().enumerate() {
        let entry =
            u32::try_from(i).map_err(|_| error!(ErrorCode::BlueprintFrame, "too many modules"))?;
        payload.push(InlineArtifact::of(bytes));
        sources.push(KernelSource::Inline { entry });
    }
    Ok((sources, payload))
}

/// 2026-10-22: The full form of a plain file: every module's verified bytes inline (the frame
/// stores the payload under `Codec::WRITE`). A sealed file's is
/// [`crate::resolve_sealed::full_sealed`].
pub fn full<'b, F>(opened: &Opened, internal: F) -> Result<(Unsigned, Vec<InlineArtifact>), Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
{
    let resolved = resolve_modules(opened, internal)?;
    let bytes: Vec<&[u8]> = resolved.iter().map(|(_, b)| &**b).collect();
    let (sources, payload) = inline_all(&bytes)?;
    Ok((
        Unsigned {
            sources,
            resources: opened.unsigned.resources.clone(),
        },
        payload,
    ))
}
