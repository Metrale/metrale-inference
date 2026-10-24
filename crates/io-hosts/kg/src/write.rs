// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: Atomic SSOT writes for `kg_record` (design §4.1 invariant 4): a temp file beside the
//! target, flushed, then renamed over it, only if the target still holds the text the record was
//! built from. Other agents append to the same files, so a concurrent change is refused (the
//! caller rebuilds from the new text), never overwritten.
//!
//! Owner: metrale-io-host-kg.
//! Invariants:
//! - The target is either its old bytes or its new bytes; a failure leaves it untouched and
//!   removes the temp file.
//! - The temp file lives in the target's directory, so the rename never crosses a file system.

use std::fs;
use std::io::Write;
use std::path::Path;

use metrale_error::{ErrorCode, Result, error};

/// 2026-10-23: Replace `path`'s contents `expected` with `new`, or refuse.
pub fn replace(path: &Path, expected: &str, new: &str) -> Result<()> {
    let shown = path.display().to_string();
    let werr = |what: &str, e: &std::io::Error| {
        error!(
            ErrorCode::KgRecordWrite,
            shown.clone(),
            format!("{what}: {}", e.kind())
        )
    };
    let current = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && expected.is_empty() => String::new(),
        Err(e) => return Err(werr("read", &e)),
    };
    if current != expected {
        return Err(error!(
            ErrorCode::KgRecordWrite,
            shown, "changed since the record was built; rebuild it from the new text"
        ));
    }
    let dir = path.parent().ok_or_else(|| {
        error!(
            ErrorCode::KgRecordWrite,
            shown.clone(),
            "has no parent directory"
        )
    })?;
    let name = path
        .file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned());
    let tmp = dir.join(format!(".{name}.kg-tmp-{}", std::process::id()));
    let write = || -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(new.as_bytes())?;
        f.sync_all()?;
        if let Ok(meta) = fs::metadata(path) {
            fs::set_permissions(&tmp, meta.permissions())?;
        }
        Ok(())
    };
    if let Err(e) = write() {
        let _ = fs::remove_file(&tmp);
        return Err(werr("temp write", &e));
    }
    let still = fs::read_to_string(path).unwrap_or_default();
    if still != current {
        let _ = fs::remove_file(&tmp);
        return Err(error!(
            ErrorCode::KgRecordWrite,
            shown, "changed during the write; rebuild it from the new text"
        ));
    }
    fs::rename(&tmp, path).map_err(|e| {
        let _ = fs::remove_file(&tmp);
        werr("rename", &e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("kg-write-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn a_matching_file_is_replaced_and_no_temp_remains() {
        let d = dir("ok");
        let p = d.join("L.tsv");
        fs::write(&p, "a\n").unwrap();
        replace(&p, "a\n", "a\nb\n").unwrap();
        assert_eq!(fs::read_to_string(&p).unwrap(), "a\nb\n");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1);
        fs::remove_dir_all(&d).unwrap();
    }

    #[test]
    fn a_concurrently_changed_file_is_refused_and_left_byte_identical() {
        let d = dir("race");
        let p = d.join("L.tsv");
        fs::write(&p, "a\nother-agent\n").unwrap();
        let e = replace(&p, "a\n", "a\nmine\n").unwrap_err();
        assert!(e.to_string().contains("changed since"), "{e}");
        assert_eq!(fs::read_to_string(&p).unwrap(), "a\nother-agent\n");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1);
        fs::remove_dir_all(&d).unwrap();
    }
}
