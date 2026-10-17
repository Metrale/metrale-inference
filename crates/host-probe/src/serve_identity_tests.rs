// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for reading this process's identity.
//! 2026-10-17: Moved from metrale-bench with the host half of the identity.
//!
//! Owner: host-probe (serve identity).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: The identity carries the lever digest of this process.
#[test]
fn the_identity_carries_this_process_lever_digest() {
    let id = this_process();
    assert_eq!(id.env_sha256, fingerprint(&process_levers()));
    assert_eq!(id.env_sha256.len(), 64);
}

#[test]
fn a_file_digest_is_its_bytes() {
    let dir = std::env::temp_dir().join(format!("serve-identity-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let p = dir.join("bin");
    std::fs::write(&p, b"hello").unwrap();
    assert_eq!(
        file_sha256(&p).unwrap(),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
    std::fs::write(&p, b"hellO").unwrap();
    assert_ne!(
        file_sha256(&p).unwrap(),
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    );
    // 2026-09-30: Several read chunks (64 KiB each) hash as the bytes do in one call.
    let big: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&p, &big).unwrap();
    assert_eq!(file_sha256(&p).unwrap(), hex_lower(&Sha256::digest(&big)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_missing_file_is_an_error_naming_it() {
    let p = std::env::temp_dir().join(format!("serve-identity-absent-{}", std::process::id()));
    let err = file_sha256(&p).unwrap_err();
    assert!(format!("{err:#}").contains("opening"), "{err:#}");
}
