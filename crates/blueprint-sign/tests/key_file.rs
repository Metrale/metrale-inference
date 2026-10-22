// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Key files: a well-formed file signs; a file open to group/others, a missing or
//! repeated field and a short seed are refused; the certificate must be the key's. Seeds are
//! fixed test values written to a temp dir.

use std::path::{Path, PathBuf};

use metrale_blueprint_sign::{read_key, sign_kbp, write_public};
use metrale_blueprint_trust::{KeyCert, Kid, Scope, Signer, Suite, certify};
use metrale_error::ErrorCode;

fn write_key(dir: &Path, name: &str, body: &str, mode: u32) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, body).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    p
}

fn key_text(kid: &str, m: u8, e: u8) -> String {
    format!(
        "kbp-key v1\nkid {kid}\nsuite mldsa65+ed25519\nmldsa-seed {}\ned25519-seed {}\n",
        format!("{m:02x}").repeat(32),
        format!("{e:02x}").repeat(32)
    )
}

#[test]
fn a_key_file_yields_the_signer_its_seeds_define() {
    let d = tempfile::tempdir().unwrap();
    let p = write_key(d.path(), "k", &key_text("test-blueprints", 3, 4), 0o600);
    let s = read_key(&p).unwrap();
    let direct = Signer::from_seeds(
        Kid::new("test-blueprints").unwrap(),
        Suite::MlDsa65Ed25519,
        &[3; 32],
        &[4; 32],
    );
    assert_eq!(s.public(), direct.public());
}

#[test]
fn malformed_and_exposed_key_files_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let good = key_text("k", 1, 2);
    #[cfg(unix)]
    assert_eq!(
        read_key(&write_key(d.path(), "open", &good, 0o644))
            .map(|_| ())
            .unwrap_err()
            .code(),
        ErrorCode::BlueprintKey
    );
    for (name, text) in [
        ("magic", good.replace("kbp-key v1", "kbp-key v2")),
        ("missing", good.replace("suite mldsa65+ed25519\n", "")),
        ("twice", format!("{good}kid k\n")),
        ("short", good.replace(&"02".repeat(32), &"02".repeat(31))),
        ("hex", good.replace(&"02".repeat(32), &"zz".repeat(32))),
        ("kid", good.replace("kid k", "kid K")),
    ] {
        let err = read_key(&write_key(d.path(), name, &text, 0o600))
            .map(|_| ())
            .unwrap_err();
        assert_eq!(err.code(), ErrorCode::BlueprintKey, "{name}: {err}");
    }
}

#[test]
fn a_certificate_for_another_kid_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let key = write_key(d.path(), "k", &key_text("test-blueprints", 3, 4), 0o600);
    let root = Signer::from_seeds(
        Kid::new("test-root").unwrap(),
        Suite::MlDsa87Ed25519,
        &[1; 32],
        &[2; 32],
    );
    let other = Signer::from_seeds(
        Kid::new("other").unwrap(),
        Suite::MlDsa65Ed25519,
        &[5; 32],
        &[6; 32],
    );
    let cert = certify(
        &root,
        &KeyCert {
            kid: other.kid().clone(),
            keys: other.public(),
            scope: Scope::Blueprints,
            not_before_ms: 0,
            not_after_ms: 1,
            serial: 1,
        },
    )
    .unwrap();
    let cert_path = d.path().join("c");
    write_public(&cert_path, &cert).unwrap();
    assert_eq!(
        sign_kbp(&[], &key, &cert_path).unwrap_err().code(),
        ErrorCode::BlueprintKey
    );
}
