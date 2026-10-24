// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Recipient key files (DESIGN-encrypted-kernels.md §1.5): the only place a
//! recipient secret is generated, written or read from disk.
//!
//! ```text
//! kbp-recipient-key v1                 kbp-recipient-public v1
//! fingerprint <64 hex>                 fingerprint <64 hex>
//! seed <64 hex>                        public <base64 of 1216 bytes>
//! ```
//!
//! Owner: blueprint.
//! Invariants:
//! - The seed is drawn from the caller's `Entropy` port, written once to a new file created with
//!   mode 600 (never over an existing file), and held only in `Zeroizing` buffers.
//! - Reading refuses a key file readable by group or others, and checks the fingerprint line
//!   against the key the seed (or the public bytes) defines, so a pasted-wrong file is refused.
//! - Nothing here prints; the caller reports the fingerprint.

use std::io::Write as _;
use std::path::Path;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use metrale_abstract_io::Entropy;
use metrale_error::{Error, ErrorCode, error};
use metrale_seal::{Fingerprint, HybridPublic, HybridSecret, SEED_LEN};
use zeroize::Zeroizing;

use crate::{bad, check_mode, io, seed};

/// 2026-10-22: The first line of a recipient key file.
pub const RECIPIENT_KEY_MAGIC: &str = "kbp-recipient-key v1";
/// 2026-10-22: The first line of a recipient public key file.
pub const RECIPIENT_PUBLIC_MAGIC: &str = "kbp-recipient-public v1";

fn fields<'a>(
    path: &Path,
    text: &'a str,
    magic: &str,
) -> Result<(&'a str, &'a str, &'a str), Error> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    if lines.next() != Some(magic) {
        return Err(bad(
            path,
            &format!("not a recipient file (first line must be `{magic}`)"),
        ));
    }
    let mut fp = None;
    let mut value = None;
    for line in lines {
        let (k, v) = line
            .split_once(' ')
            .ok_or_else(|| bad(path, "a line is `field value`"))?;
        let slot = match k {
            "fingerprint" => &mut fp,
            "seed" | "public" => &mut value,
            _ => return Err(bad(path, "unknown field")),
        };
        if slot.replace((k, v)).is_some() {
            return Err(bad(path, "a field is given twice"));
        }
    }
    match (fp, value) {
        (Some((_, fp)), Some((k, v))) => Ok((fp, k, v)),
        _ => Err(bad(path, "fingerprint and the key line are both required")),
    }
}

fn check_fingerprint(path: &Path, stated: &str, actual: Fingerprint) -> Result<(), Error> {
    if stated != actual.hex() {
        return Err(bad(path, "the fingerprint line does not match the key"));
    }
    Ok(())
}

/// 2026-10-22: The recipient secret a `kbp-recipient-key v1` file holds.
pub fn read_recipient_key(path: &Path) -> Result<HybridSecret, Error> {
    check_mode(path)?;
    let text = Zeroizing::new(std::fs::read_to_string(path).map_err(|e| io(path, &e))?);
    let (fp, k, v) = fields(path, &text, RECIPIENT_KEY_MAGIC)?;
    if k != "seed" {
        return Err(bad(path, "a recipient key file holds a `seed` line"));
    }
    let secret = HybridSecret::from_seed(&*seed(path, v)?);
    check_fingerprint(path, fp, secret.public().fingerprint())?;
    Ok(secret)
}

/// 2026-10-22: The recipient public key a `kbp-recipient-public v1` file holds.
pub fn read_recipient_public(path: &Path) -> Result<HybridPublic, Error> {
    let text = std::fs::read_to_string(path).map_err(|e| io(path, &e))?;
    let (fp, k, v) = fields(path, &text, RECIPIENT_PUBLIC_MAGIC)?;
    if k != "public" {
        return Err(bad(path, "a recipient public file holds a `public` line"));
    }
    let bytes = STANDARD
        .decode(v)
        .map_err(|_| bad(path, "the public key is not standard base64"))?;
    let public = HybridPublic::from_bytes(&bytes)?;
    check_fingerprint(path, fp, public.fingerprint())?;
    Ok(public)
}

/// 2026-10-22: The text of a public key file for `public`.
pub fn public_text(public: &HybridPublic) -> String {
    format!(
        "{RECIPIENT_PUBLIC_MAGIC}\nfingerprint {}\npublic {}\n",
        public.fingerprint().hex(),
        STANDARD.encode(public.to_bytes())
    )
}

fn create_new(path: &Path, mode: u32) -> Result<std::fs::File, Error> {
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        o.mode(mode);
    }
    #[cfg(not(unix))]
    let _ = mode;
    o.open(path).map_err(|e| io(path, &e))
}

/// 2026-10-22: Generate a recipient key from one 32-byte draw of `entropy`: the key file at
/// `key` (new, mode 600) and its public half at `public` (new, mode 644). Neither path may
/// exist. Returns the fingerprint.
pub fn gen_recipient<E: Entropy>(
    entropy: &E,
    key: &Path,
    public: &Path,
) -> Result<Fingerprint, Error> {
    let mut s = Zeroizing::new([0u8; SEED_LEN]);
    entropy.fill(s.as_mut_slice())?;
    let secret = HybridSecret::from_seed(&s);
    let pk = secret.public();
    let fp = pk.fingerprint();
    if public.exists() {
        return Err(error!(
            ErrorCode::Io,
            format!("{}: exists; refusing to overwrite", public.display())
        ));
    }
    let seed_hex = Zeroizing::new(metrale_seal::hex(s.as_slice()));
    let text = Zeroizing::new(format!(
        "{RECIPIENT_KEY_MAGIC}\nfingerprint {}\nseed {}\n",
        fp.hex(),
        seed_hex.as_str()
    ));
    create_new(key, 0o600)?
        .write_all(text.as_bytes())
        .map_err(|e| io(key, &e))?;
    create_new(public, 0o644)?
        .write_all(public_text(&pk).as_bytes())
        .map_err(|e| io(public, &e))?;
    Ok(fp)
}
