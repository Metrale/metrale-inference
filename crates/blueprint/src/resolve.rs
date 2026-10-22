// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Kernel resolution: each module's bytes from its source, checked against the
//! signed hash; and the lite ↔ full conversions built on it.
//!
//! Owner: blueprint.
//! Invariants:
//! - Every source is checked the same way: the bytes' SHA-256 must equal the signed
//!   `ModuleSpec.artifact` (`ArtifactHashMismatch` names the module), whatever the locator says.
//! - Phase 1 resolves `internal` and `inline`; `disk` and `remote` parse and are refused with
//!   `KernelSourceNotInPhase`.
//! - The caller supplies the embedded modules through `internal` (no I/O here).

use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use metrale_error::{Error, ErrorCode, error};

use crate::frame::{InlineArtifact, KernelSource, Opened, Unsigned};
use crate::ids::ArtifactId;
use crate::types::ModuleSpec;

fn checked<'a>(m: &ModuleSpec, bytes: Cow<'a, [u8]>) -> Result<Cow<'a, [u8]>, Error> {
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

/// 2026-10-22: Every module's verified bytes, in module order. `internal` returns the
/// binary's embedded bytes for a module name.
pub fn resolve_modules<'b, F>(
    opened: &Opened,
    internal: F,
) -> Result<Vec<(String, Cow<'b, [u8]>)>, Error>
where
    F: Fn(&str) -> Option<&'b [u8]>,
{
    let modules = &opened.blueprint.kernels.modules;
    let mut out = Vec::with_capacity(modules.len());
    for (m, src) in modules.iter().zip(&opened.unsigned.sources) {
        let bytes = match src {
            KernelSource::Internal => Cow::Borrowed(internal(&m.name).ok_or_else(|| {
                error!(
                    ErrorCode::KernelMissing,
                    m.name.clone(),
                    m.artifact.0.hex(),
                    "internal"
                )
            })?),
            KernelSource::Inline { entry } => Cow::Owned(inline_bytes(m, &opened.payload, *entry)?),
            KernelSource::Disk { .. } => {
                return Err(error!(
                    ErrorCode::KernelSourceNotInPhase,
                    "disk",
                    m.name.clone(),
                    2u32
                ));
            }
            KernelSource::Remote { .. } => {
                return Err(error!(
                    ErrorCode::KernelSourceNotInPhase,
                    "remote",
                    m.name.clone(),
                    3u32
                ));
            }
        };
        out.push((m.name.clone(), checked(m, bytes)?));
    }
    Ok(out)
}

/// 2026-10-22: The lite form: no payload, every module `internal`.
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

/// 2026-10-22: The full form: every module's verified bytes inline (the frame stores the
/// payload under `Codec::WRITE`).
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
