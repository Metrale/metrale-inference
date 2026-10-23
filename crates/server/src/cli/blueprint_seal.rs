// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint seal | open`, the sealed parts of `verify | lite | full | show`, and
//! `met blueprint key gen-recipient | recipient-fingerprint` (DESIGN-encrypted-kernels.md §1.5,
//! §6 step 5): the I/O side of `metrale_blueprint::sealing` and `resolve_sealed`.
//!
//! Owner: server CLI (blueprint).
//! Invariants:
//! - `seal` is one process: it draws the run's randomness in one `Entropy::fill`, hands it to the
//!   pure sealer, which consumes and zeroizes the content key; nothing it prints or writes holds
//!   the CEK or a per-artifact key (the leak test, `blueprint_seal_tests.rs`).
//! - Every report goes to the caller's writer, so the leak test sees all of it.
//! - Plaintext is written only by `open`, only into a directory it creates; never to a cache.
//! - A recipient key on a plain file is refused, and a sealed file without one is verified by
//!   ciphertext hash only, said so (declared, never guessed: PCND).

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use metrale_abstract_io::{Entropy, UnixMs};
use metrale_blueprint::codec::Codec;
use metrale_blueprint::frame::{self, KernelSource, Opened};
use metrale_blueprint::ids::Sha256;
use metrale_blueprint::resolve_sealed::{
    full_sealed, lite_sealed, resolve_unsealed, unseal, verify_ciphertexts,
};
use metrale_blueprint::sealing::{CiphertextPlacement, SealRequest, seal};
use metrale_blueprint::types::ArtifactKind;
use metrale_blueprint::{frame::InlineArtifact, frame::Unsigned, resolve};
use metrale_blueprint_sign::read_signer;
use metrale_blueprint_sign::recipient::{gen_recipient, read_recipient_key, read_recipient_public};
use metrale_blueprint_trust::TrustPolicy;
use metrale_seal::SealRandom;

use super::blueprint::{DECODE_MEMORY, internal_of, open_at, write_kbp_to};
use super::blueprint_args::{SealArgs, SealedArgs, TrustArgs};

/// 2026-10-22: The ciphertexts a sealed file's non-inline sealed modules need, read by hash from
/// `dir` (`<dir>/<ct_sha256 hex>`); empty without a directory. A missing or unreadable file
/// fails here, with its path.
pub(crate) fn store(o: &Opened, dir: Option<&Path>) -> Result<BTreeMap<Sha256, Vec<u8>>> {
    let mut out = BTreeMap::new();
    let (Some(sealed), Some(dir)) = (&o.sealed, dir) else {
        return Ok(out);
    };
    for a in &sealed.manifest.artifacts {
        let src = &o.unsigned.sources[usize::try_from(a.module)?];
        if !matches!(src, KernelSource::Inline { .. }) {
            let p = dir.join(a.ct_sha256.hex());
            let bytes = std::fs::read(&p).with_context(|| format!("reading {}", p.display()))?;
            out.insert(a.ct_sha256, bytes);
        }
    }
    Ok(out)
}

/// 2026-10-22: The `show`/`verify` lines of a sealed file (empty for a plain one).
pub(crate) fn sealed_summary(o: &Opened) -> String {
    let Some(s) = &o.sealed else {
        return String::new();
    };
    let m = &s.manifest;
    let mut out = format!(
        "  sealed    {} of {} modules, {} ciphertext bytes, inner codec {}, {}\n            customer {}, sealed at {} ms, signed by {} under {}\n",
        m.artifacts.len(),
        o.blueprint.kernels.modules.len(),
        m.artifacts.iter().map(|a| a.ct_len).sum::<u64>(),
        m.codec.name(),
        m.suite.name(),
        m.customer,
        m.issued_ms,
        s.signer.kid,
        s.signer.anchor,
    );
    for r in &m.recipients {
        out += &format!("            recipient {}\n", r.fingerprint);
    }
    out
}

/// 2026-10-22: `verify` of a sealed file: every ciphertext by hash, and with a recipient key
/// every module decrypted and checked; the report line.
pub(crate) fn verify_sealed(o: &Opened, a: &SealedArgs) -> Result<String> {
    if o.sealed.is_none() {
        if a.recipient_key.is_some() || a.ciphertexts.is_some() {
            bail!(
                "--recipient-key and --ciphertexts are for sealed files; this file is not sealed"
            );
        }
        return Ok(String::new());
    }
    let internal = internal_of(o)?;
    let st = store(o, a.ciphertexts.as_deref())?;
    let fetch = |h: &Sha256| st.get(h).cloned();
    let n = verify_ciphertexts(o, &internal, fetch)?;
    let Some(key) = &a.recipient_key else {
        return Ok(format!(
            "sealed: {n} ciphertexts verified by hash; plaintext not checked (no --recipient-key)"
        ));
    };
    let secret = read_recipient_key(key)?;
    let cek = unseal(o, &secret)?;
    let modules = resolve_unsealed(o, &internal, fetch, &cek, DECODE_MEMORY)?;
    Ok(format!(
        "sealed: {n} ciphertexts verified by hash; {} modules decrypted and checked by hash for recipient {}",
        modules.len(),
        secret.public().fingerprint()
    ))
}

/// 2026-10-22: `lite` / `full` of a sealed file: the new locator table and payload.
pub(crate) fn sealed_form(
    o: &Opened,
    remote_base: Option<&str>,
    ciphertexts: Option<&Path>,
    full: bool,
) -> Result<(Unsigned, Vec<InlineArtifact>)> {
    if full {
        let internal = internal_of(o)?;
        let st = store(o, ciphertexts)?;
        return Ok(full_sealed(o, &internal, |h: &Sha256| st.get(h).cloned())?);
    }
    let base = remote_base.context("a sealed file's lite form needs --remote-base")?;
    Ok(lite_sealed(o, base)?)
}

/// 2026-10-22: `met blueprint seal`, with the run's randomness from `entropy` and the manifest's
/// time `now`; every report line goes to `out`.
pub(crate) fn seal_cmd<E: Entropy, W: Write>(
    a: &SealArgs,
    entropy: &E,
    now: UnixMs,
    out: &mut W,
) -> Result<()> {
    let (file, o, anchors, revoked) = open_at(&a.input, &a.trust, now, out)?;
    let recipients = a
        .recipient_public
        .iter()
        .map(|p| read_recipient_public(p))
        .collect::<Result<Vec<_>, _>>()?;
    let (signer, cert) = read_signer(&a.key, &a.cert)?;
    let placement = match &a.remote_base {
        Some(base) => CiphertextPlacement::Remote { base },
        None => CiphertextPlacement::Inline,
    };
    if let Some(dir) = &a.ciphertext_dir {
        std::fs::create_dir(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    let req = SealRequest {
        recipients: &recipients,
        customer_id: &a.customer_id,
        issued_ms: now.0,
        codec: Codec::WRITE,
        placement,
    };
    let policy = TrustPolicy {
        anchors: &anchors,
        revoked: &revoked,
        now,
    };
    let internal = internal_of(&o)?;
    let random = SealRandom::draw(entropy, recipients.len())?;
    let sealed = seal(
        &file,
        &policy,
        DECODE_MEMORY,
        internal,
        &req,
        random,
        &signer,
        vec![cert],
    )?;
    if let Some(dir) = &a.ciphertext_dir {
        for (h, ct) in &sealed.ciphertexts {
            let p = dir.join(h.hex());
            std::fs::write(&p, ct).with_context(|| format!("writing {}", p.display()))?;
        }
        writeln!(
            out,
            "wrote {} ciphertexts to {}",
            sealed.ciphertexts.len(),
            dir.display()
        )?;
    }
    write_kbp_to(&a.out, &sealed.file, out)?;
    let opened = frame::open(&sealed.file, &policy, DECODE_MEMORY)?;
    write!(
        out,
        "sealed blueprint {}\n{}",
        opened.id,
        sealed_summary(&opened)
    )?;
    Ok(())
}

fn module_file(name: &str, kind: ArtifactKind) -> Result<String> {
    let ok = !name.is_empty()
        && !name.starts_with('.')
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'.');
    if !ok {
        bail!("module name {name:?} is not a safe file name");
    }
    let ext = match kind {
        ArtifactKind::PtxText => "ptx",
        ArtifactKind::AmdCodeObject => "co",
        ArtifactKind::ScaleObject => "o",
        ArtifactKind::AppleGpuLibrary => "metallib",
    };
    Ok(format!("{name}.{ext}"))
}

/// 2026-10-22: `met blueprint open`: verify, unwrap, decrypt and check every module, then write
/// them into `out_dir`, which this creates.
pub(crate) fn open_cmd<W: Write>(
    input: &Path,
    trust: &TrustArgs,
    recipient_key: &Path,
    ciphertexts: Option<&Path>,
    out_dir: &Path,
    now: UnixMs,
    out: &mut W,
) -> Result<()> {
    let (_, o, _, _) = open_at(input, trust, now, out)?;
    if o.sealed.is_none() {
        bail!("{} is not sealed", input.display());
    }
    let secret = read_recipient_key(recipient_key)?;
    let cek = unseal(&o, &secret)?;
    let internal = internal_of(&o)?;
    let st = store(&o, ciphertexts)?;
    let modules = resolve_unsealed(
        &o,
        internal,
        |h: &Sha256| st.get(h).cloned(),
        &cek,
        DECODE_MEMORY,
    )?;
    drop(cek);
    let names = o
        .blueprint
        .kernels
        .modules
        .iter()
        .map(|m| module_file(&m.name, m.kind))
        .collect::<Result<Vec<_>>>()?;
    std::fs::create_dir(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    for (file, (_, bytes)) in names.iter().zip(&modules) {
        let p = out_dir.join(file);
        std::fs::write(&p, bytes).with_context(|| format!("writing {}", p.display()))?;
    }
    writeln!(
        out,
        "opened {}: {} modules written to {}",
        o.id,
        modules.len(),
        out_dir.display()
    )?;
    Ok(())
}

/// 2026-10-22: `met blueprint key gen-recipient`, drawing the seed from `entropy`.
pub(crate) fn gen_recipient_cmd<E: Entropy, W: Write>(
    entropy: &E,
    key: &Path,
    public: &Path,
    out: &mut W,
) -> Result<()> {
    let fp = gen_recipient(entropy, key, public)?;
    writeln!(
        out,
        "wrote recipient key {} (mode 600) and its public half {}\nfingerprint {fp}\n\
         The seed in {} is the only copy: a lost seed or a lost sealed blueprint means the kernels cannot be recovered, by design.",
        key.display(),
        public.display(),
        key.display()
    )?;
    Ok(())
}

/// 2026-10-22: `met blueprint key recipient-fingerprint`.
pub(crate) fn recipient_fingerprint_cmd<W: Write>(public: &Path, out: &mut W) -> Result<()> {
    writeln!(out, "{}", read_recipient_public(public)?.fingerprint())?;
    Ok(())
}

/// 2026-10-22: A plain file's `lite`/`full`, refusing sealed-only flags (PCND both ways).
pub(crate) fn plain_form(
    o: &Opened,
    remote_base: Option<&str>,
    ciphertexts: Option<&Path>,
    full: bool,
) -> Result<(Unsigned, Vec<InlineArtifact>)> {
    if remote_base.is_some() || ciphertexts.is_some() {
        bail!("--remote-base and --ciphertexts are for sealed files; this file is not sealed");
    }
    if full {
        return Ok(resolve::full(o, internal_of(o)?)?);
    }
    Ok(resolve::lite(o))
}

#[cfg(test)]
#[path = "blueprint_seal_tests.rs"]
mod blueprint_seal_tests;
