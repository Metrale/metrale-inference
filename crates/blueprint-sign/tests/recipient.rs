// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Recipient key files: generation draws one 32-byte seed from the Entropy port and
//! writes a mode-600 key file and its public half, never over an existing file; both read back
//! to the key the seed defines; an exposed file, a wrong fingerprint line, a wrong magic, a
//! repeated field and a malformed public key are refused. The entropy is the seeded host-mem
//! ChaCha20 port (a test host, not a mock of the crypto).

use std::path::Path;

use metrale_abstract_io::Entropy as _;
use metrale_blueprint_sign::recipient::{gen_recipient, read_recipient_key, read_recipient_public};
use metrale_error::ErrorCode;
use metrale_io_host_mem::SeededEntropy;
use metrale_seal::HybridSecret;

fn make_key(
    dir: &Path,
) -> (
    std::path::PathBuf,
    std::path::PathBuf,
    metrale_seal::Fingerprint,
) {
    let (k, p) = (dir.join("key.txt"), dir.join("key.pub"));
    let fp = gen_recipient(&SeededEntropy::new(1, 0), &k, &p).unwrap();
    (k, p, fp)
}

#[test]
fn a_generated_key_reads_back_to_the_key_its_seed_defines() {
    let d = tempfile::tempdir().unwrap();
    let (k, p, fp) = make_key(d.path());
    let mut seed = [0u8; 32];
    SeededEntropy::new(1, 0).fill(&mut seed).unwrap();
    let expected = HybridSecret::from_seed(&seed);
    let secret = read_recipient_key(&k).unwrap();
    assert_eq!(secret.public(), expected.public());
    assert_eq!(read_recipient_public(&p).unwrap(), expected.public());
    assert_eq!(fp, expected.public().fingerprint());
    let text = std::fs::read_to_string(&k).unwrap();
    assert!(text.starts_with("kbp-recipient-key v1\nfingerprint "));
    assert!(
        text.contains(&metrale_seal::hex(&seed)),
        "the file holds the seed"
    );
    assert!(
        !std::fs::read_to_string(&p)
            .unwrap()
            .contains(&metrale_seal::hex(&seed))
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&k).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn generation_never_overwrites() {
    let d = tempfile::tempdir().unwrap();
    let (k, p, _) = make_key(d.path());
    let before = std::fs::read(&k).unwrap();
    assert!(gen_recipient(&SeededEntropy::new(2, 0), &k, &d.path().join("other.pub")).is_err());
    assert!(gen_recipient(&SeededEntropy::new(2, 0), &d.path().join("other.txt"), &p).is_err());
    assert!(
        !d.path().join("other.txt").exists(),
        "nothing written when the public path exists"
    );
    assert_eq!(std::fs::read(&k).unwrap(), before);
}

fn write(dir: &Path, name: &str, text: &str, mode: u32) -> std::path::PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    #[cfg(not(unix))]
    let _ = mode;
    p
}

#[test]
fn malformed_exposed_and_mismatched_files_are_refused() {
    let d = tempfile::tempdir().unwrap();
    let (k, p, _) = make_key(d.path());
    let key = std::fs::read_to_string(&k).unwrap();
    let public = std::fs::read_to_string(&p).unwrap();
    let other_fp = HybridSecret::from_seed(&[9; 32])
        .public()
        .fingerprint()
        .hex();
    let fp_line = key.lines().nth(1).unwrap().to_string();
    #[cfg(unix)]
    assert_eq!(
        read_recipient_key(&write(d.path(), "open", &key, 0o644))
            .unwrap_err()
            .code(),
        ErrorCode::BlueprintKey
    );
    for (name, text) in [
        ("magic", key.replace("kbp-recipient-key v1", "kbp-key v1")),
        (
            "fingerprint",
            key.replace(&fp_line, &format!("fingerprint {other_fp}")),
        ),
        ("twice", format!("{key}seed {}\n", "00".repeat(32))),
        ("missing", key.replace(&format!("{fp_line}\n"), "")),
        ("public-in-key", key.replace("seed ", "public ")),
    ] {
        let e = read_recipient_key(&write(d.path(), name, &text, 0o600)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::BlueprintKey, "{name}: {e}");
    }
    let pub_fp = public.lines().nth(1).unwrap().to_string();
    for (name, text) in [
        (
            "pmagic",
            public.replace("kbp-recipient-public v1", "kbp-recipient-key v1"),
        ),
        (
            "pfingerprint",
            public.replace(&pub_fp, &format!("fingerprint {other_fp}")),
        ),
        ("pbase64", public.replace("public ", "public !")),
        ("pshort", {
            use base64::Engine as _;
            let pk = HybridSecret::from_seed(&[9; 32]).public();
            let b = base64::engine::general_purpose::STANDARD.encode(&pk.to_bytes()[..1215]);
            format!(
                "kbp-recipient-public v1\nfingerprint {}\npublic {b}\n",
                pk.fingerprint().hex()
            )
        }),
    ] {
        let e = read_recipient_public(&write(d.path(), name, &text, 0o644)).unwrap_err();
        assert_eq!(e.code(), ErrorCode::BlueprintKey, "{name}: {e}");
    }
}
