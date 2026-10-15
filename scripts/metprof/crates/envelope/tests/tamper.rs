// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: every altered, truncated, relabeled or mis-addressed envelope
//! fails `open` with a typed error, and the error names the right cause.

mod common;

use common::{CREATED_AT, FixedClock, KID, PATH, StreamEntropy, request};
use metprof_envelope::primitives::{aes256gcm_seal, sha256};
use metprof_envelope::{
    Dek, EnvelopeError, Expected, Header, derive_file_key, format, open, parse, seal,
};

const EXPECTED: Expected<'static> = Expected {
    kid: KID,
    path: PATH,
};
const PT: &[u8] = b"lattice rows";

fn sealed() -> (Dek, Vec<u8>) {
    let mut entropy = StreamEntropy::new(b"tamper");
    let dek = Dek::generate(&mut entropy).expect("dek");
    let env = seal(
        &dek,
        &request(None),
        PT,
        &mut entropy,
        &FixedClock(CREATED_AT),
    )
    .expect("seal");
    (dek, env)
}

fn header_len(env: &[u8]) -> usize {
    usize::try_from(u32::from_le_bytes(env[4..8].try_into().expect("4"))).expect("usize")
}

/// Replaces the header text, fixing up the length prefix, body untouched.
fn with_header_text(env: &[u8], edit: impl Fn(&str) -> String) -> Vec<u8> {
    let hl = header_len(env);
    let text = std::str::from_utf8(&env[8..8 + hl]).expect("utf8");
    let edited = edit(text);
    assert_ne!(edited, text, "the edit must change the header");
    format::encode(edited.as_bytes(), &env[8 + hl..]).expect("encode")
}

/// A correctly encrypted envelope for an arbitrary (possibly lying) header.
fn encrypt_with_header(dek: &Dek, header: &Header, pt: &[u8]) -> Vec<u8> {
    let bytes = header.to_canonical_bytes().expect("json");
    let key = derive_file_key(dek, KID, PATH).expect("key");
    let body = aes256gcm_seal(&key, &header.nonce, &bytes, pt).expect("seal");
    format::encode(&bytes, &body).expect("encode")
}

#[test]
fn untampered_control_opens() {
    let (dek, env) = sealed();
    assert_eq!(open(&dek, &EXPECTED, &env).expect("open").plaintext, PT);
}

#[test]
fn semantically_valid_header_edit_fails_authentication() {
    let (dek, env) = sealed();
    let bad = with_header_text(&env, |t| t.replace("1700000000", "1700000001"));
    assert!(parse(&bad).is_ok(), "the edited header is well formed");
    assert_eq!(open(&dek, &EXPECTED, &bad), Err(EnvelopeError::Decrypt));
}

#[test]
fn ciphertext_and_tag_byte_flips_fail_authentication() {
    let (dek, env) = sealed();
    let start = 8 + header_len(&env);
    for i in start..env.len() {
        let mut bad = env.clone();
        bad[i] ^= 0x01;
        assert_eq!(
            open(&dek, &EXPECTED, &bad),
            Err(EnvelopeError::Decrypt),
            "byte {i}"
        );
    }
}

#[test]
fn every_single_byte_flip_anywhere_is_rejected() {
    let (dek, env) = sealed();
    for i in 0..env.len() {
        for mask in [0x01u8, 0x80, 0xff] {
            let mut bad = env.clone();
            bad[i] ^= mask;
            assert!(
                open(&dek, &EXPECTED, &bad).is_err(),
                "byte {i} mask {mask:#x}"
            );
        }
    }
}

#[test]
fn every_truncation_is_rejected_with_the_right_error() {
    let (dek, env) = sealed();
    let hl = header_len(&env);
    for cut in 0..env.len() {
        let err = open(&dek, &EXPECTED, &env[..cut]).expect_err("truncated");
        let want = match cut {
            0..4 => EnvelopeError::Truncated("magic"),
            4..8 => EnvelopeError::Truncated("header length"),
            c if c < 8 + hl => EnvelopeError::Truncated("header"),
            c if c < 8 + hl + 16 => EnvelopeError::Truncated("ciphertext tag"),
            _ => EnvelopeError::Decrypt,
        };
        assert_eq!(err, want, "cut {cut}");
    }
}

#[test]
fn oversized_header_length_is_truncation_not_a_panic() {
    let (dek, mut env) = sealed();
    env[4..8].copy_from_slice(&u32::MAX.to_le_bytes());
    assert_eq!(
        open(&dek, &EXPECTED, &env),
        Err(EnvelopeError::Truncated("header"))
    );
}

#[test]
fn bad_magic_is_rejected() {
    let (dek, mut env) = sealed();
    env[3] = b'2';
    assert_eq!(open(&dek, &EXPECTED, &env), Err(EnvelopeError::BadMagic));
}

#[test]
fn wrong_kid_and_wrong_path_are_named() {
    let (dek, env) = sealed();
    let other_kid = Expected {
        kid: "kid-test-02",
        path: PATH,
    };
    assert!(matches!(
        open(&dek, &other_kid, &env),
        Err(EnvelopeError::KidMismatch { .. })
    ));
    let other_path = Expected {
        kid: KID,
        path: "lattice/other.jsonl",
    };
    assert!(matches!(
        open(&dek, &other_path, &env),
        Err(EnvelopeError::InfoMismatch { .. })
    ));
}

#[test]
fn consistently_relabeled_kid_and_path_fail_authentication() {
    let (dek, env) = sealed();
    let relabeled = with_header_text(&env, |t| t.replace(KID, "kid-test-02"));
    let as_other = Expected {
        kid: "kid-test-02",
        path: PATH,
    };
    assert_eq!(
        open(&dek, &as_other, &relabeled),
        Err(EnvelopeError::Decrypt)
    );
    let moved = with_header_text(&env, |t| t.replace("levers.jsonl", "levers.jsonx"));
    let as_moved = Expected {
        kid: KID,
        path: "lattice/levers.jsonx",
    };
    assert_eq!(open(&dek, &as_moved, &moved), Err(EnvelopeError::Decrypt));
}

#[test]
fn wrong_dek_fails_authentication() {
    let (_, env) = sealed();
    assert_eq!(
        open(&Dek::from_bytes([9; 32]), &EXPECTED, &env),
        Err(EnvelopeError::Decrypt)
    );
}

#[test]
fn non_canonical_headers_are_rejected_before_decryption() {
    let (dek, env) = sealed();
    let spaced = with_header_text(&env, |t| t.replacen(',', ", ", 1));
    let null_wrap = with_header_text(&env, |t| {
        t.replace(r#""v":1}"#, r#""v":1,"wrapped_dek":null}"#)
    });
    let reordered = with_header_text(&env, |t| {
        let (alg, rest) = t[1..].split_once(',').expect("comma");
        format!("{{{},{alg}}}", &rest[..rest.len() - 1])
    });
    for bad in [spaced, null_wrap, reordered] {
        assert_eq!(
            open(&dek, &EXPECTED, &bad),
            Err(EnvelopeError::NonCanonicalHeader)
        );
    }
}

#[test]
fn schema_violations_are_header_errors() {
    let (dek, env) = sealed();
    let unknown = with_header_text(&env, |t| t.replace(r#""v":1}"#, r#""v":1,"x":0}"#));
    let upper_hex = with_header_text(&env, |t| {
        let digest = format::info_string(KID, PATH)
            .map(|_| sha_hex(PT))
            .expect("info");
        t.replace(&digest, &digest.to_uppercase())
    });
    let short_nonce = with_header_text(&env, |t| {
        let n = t
            .split(r#""nonce":""#)
            .nth(1)
            .and_then(|r| r.split('"').next())
            .expect("nonce");
        t.replace(n, "AAAA")
    });
    for bad in [unknown, upper_hex, short_nonce] {
        assert!(matches!(
            open(&dek, &EXPECTED, &bad),
            Err(EnvelopeError::HeaderJson(_))
        ));
    }
}

#[test]
fn unsupported_version_alg_kdf_and_bad_info_are_typed() {
    let (dek, env) = sealed();
    let base = parse(&env).expect("parse").header;
    let mut v2 = base.clone();
    v2.v = 2;
    assert_eq!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &v2, PT)),
        Err(EnvelopeError::UnsupportedVersion(2))
    );
    let mut alg = base.clone();
    alg.alg = "A128GCM".to_owned();
    assert!(matches!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &alg, PT)),
        Err(EnvelopeError::Unsupported { field: "alg", .. })
    ));
    let mut kdf = base.clone();
    kdf.kdf = "HKDF-SHA512".to_owned();
    assert!(matches!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &kdf, PT)),
        Err(EnvelopeError::Unsupported { field: "kdf", .. })
    ));
    for info in [
        "kkb/v1/other/lattice/levers.jsonl",
        "kkb/v2/kid-test-01/x",
        "kkb/v1/kid-test-01/../x",
    ] {
        let mut h = base.clone();
        h.info = info.to_owned();
        assert_eq!(
            open(&dek, &EXPECTED, &encrypt_with_header(&dek, &h, PT)),
            Err(EnvelopeError::BadInfo),
            "{info}"
        );
    }
}

#[test]
fn authentic_but_lying_plaintext_metadata_is_caught() {
    let (dek, env) = sealed();
    let base = parse(&env).expect("parse").header;
    let mut digest = base.clone();
    digest.plaintext.sha256 = sha256(b"something else");
    assert_eq!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &digest, PT)),
        Err(EnvelopeError::PlaintextDigest)
    );
    let mut len = base.clone();
    len.plaintext.len += 1;
    assert_eq!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &len, PT)),
        Err(EnvelopeError::PlaintextLength {
            expected: 13,
            actual: 12
        })
    );
    assert_eq!(
        open(&dek, &EXPECTED, &encrypt_with_header(&dek, &base, PT))
            .expect("control")
            .plaintext,
        PT
    );
}

#[test]
fn seal_rejects_bad_kids_paths_and_missing_fields() {
    let dek = Dek::from_bytes([1; 32]);
    let mut entropy = StreamEntropy::new(b"inputs");
    let clock = FixedClock(CREATED_AT);
    let long_kid = "k".repeat(65);
    for kid in ["", ".", "..", "a/b", "a b", long_kid.as_str()] {
        let mut r = request(None);
        r.kid = kid.to_owned();
        assert_eq!(
            seal(&dek, &r, PT, &mut entropy, &clock),
            Err(EnvelopeError::InvalidKid),
            "{kid:?}"
        );
    }
    let long_path = "p".repeat(256);
    for path in [
        "",
        "/abs",
        "a//b",
        "a/./b",
        "a/../b",
        "a/",
        "a\\b",
        "a?b",
        long_path.as_str(),
    ] {
        let mut r = request(None);
        r.path = path.to_owned();
        assert_eq!(
            seal(&dek, &r, PT, &mut entropy, &clock),
            Err(EnvelopeError::InvalidPath),
            "{path:?}"
        );
    }
    let mut r = request(None);
    r.encoding.clear();
    assert_eq!(
        seal(&dek, &r, PT, &mut entropy, &clock),
        Err(EnvelopeError::MissingField("encoding"))
    );
    let mut r = request(None);
    r.publisher_build.clear();
    assert_eq!(
        seal(&dek, &r, PT, &mut entropy, &clock),
        Err(EnvelopeError::MissingField("publisher_build"))
    );
}

fn sha_hex(data: &[u8]) -> String {
    sha256(data).iter().map(|b| format!("{b:02x}")).collect()
}
