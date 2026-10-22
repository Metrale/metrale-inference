// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The container: canonical encoding, verification before decode, lite ↔ full,
//! and one refusal per failure with its specific code.

mod common;

use common::*;
use metrale_abstract_io::UnixMs;
use metrale_blueprint::codec::Codec;
use metrale_blueprint::frame::{self, InlineArtifact, KernelSource, MAX_BODY, Opened};
use metrale_blueprint::resolve;
use metrale_blueprint::types::Blueprint;
use metrale_blueprint_trust::{
    KeyCert, Kid, Purpose, Revocations, Scope, SigBlock, Signer, Suite, TrustPolicy, certify,
    sign_revocations, verify_revocations,
};
use metrale_error::ErrorCode;

fn open(w: &World, file: &[u8]) -> Result<Opened, ErrorCode> {
    frame::open(file, &w.policy()).map_err(|e| e.code())
}

#[test]
fn encoding_is_canonical_and_deterministic() {
    let bp = blueprint();
    let a = postcard::to_allocvec(&bp).unwrap();
    let b = postcard::to_allocvec(&postcard::from_bytes::<Blueprint>(&a).unwrap()).unwrap();
    assert_eq!(a, b, "decode then encode is byte-identical");
    let w = world();
    assert_eq!(
        w.signed(),
        w.signed(),
        "two builds of one input are byte-identical, signature included"
    );
}

#[test]
fn a_signed_file_opens_to_the_same_blueprint() {
    let w = world();
    let o = open(&w, &w.signed()).unwrap();
    assert_eq!(o.blueprint, blueprint());
    assert_eq!(o.signer.kid.as_str(), "test-blueprints");
    assert_eq!(o.signer.anchor.as_str(), "test-root");
    assert_eq!(o.unsigned, unsigned(&blueprint()));
}

#[test]
fn lite_and_full_share_the_id_and_the_signature() {
    let w = world();
    let lite = w.signed();
    let o = open(&w, &lite).unwrap();
    let (u, p) = resolve::full(&o, internal).unwrap();
    let full = frame::rewrite(&lite, &w.policy(), &u, &p).unwrap();
    let f = open(&w, &full).unwrap();
    assert_eq!(f.id, o.id);
    assert!(
        f.unsigned
            .sources
            .iter()
            .all(|s| matches!(s, KernelSource::Inline { .. }))
    );
    // 2026-10-22: Full resolves without the binary's modules: the payload alone suffices.
    let none = |_: &str| None;
    let bytes = resolve::resolve_modules(&f, none).unwrap();
    assert_eq!(&*bytes[0].1, MOD_A);
    assert_eq!(&*bytes[1].1, MOD_B);
    let (u, p) = resolve::lite(&f);
    let back = frame::rewrite(&full, &w.policy(), &u, &p).unwrap();
    assert_eq!(
        back, lite,
        "full then lite is the original lite file, byte for byte"
    );
}

#[test]
fn a_flipped_body_byte_is_a_body_hash_error() {
    let w = world();
    let mut f = stored(&w);
    assert!(open(&w, &f).is_ok(), "the control opens");
    let at = body_range(&f).start + 8;
    f[at] ^= 1;
    assert_eq!(open(&w, &f).err(), Some(ErrorCode::BlueprintBodyHash));
}

/// 2026-10-22: Decode never runs before verification: garbage in the body with the header
/// hash patched to match reports the signature, not a decode error.
#[test]
fn a_malformed_body_under_a_bad_signature_reports_the_signature() {
    let w = world();
    let mut f = stored(&w);
    let body = body_range(&f);
    for b in &mut f[body] {
        *b = 0xff;
    }
    rehash(&mut f);
    assert_eq!(open(&w, &f).err(), Some(ErrorCode::BlueprintSignature));
}

#[test]
fn an_unsigned_file_is_refused() {
    let w = world();
    let bp = blueprint();
    let f = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
    assert_eq!(open(&w, &f).err(), Some(ErrorCode::BlueprintSignature));
}

fn resign(w: &World, f: &[u8], purpose: Purpose) -> Vec<u8> {
    // 2026-10-22: Sign the correct message for the wrong purpose, then splice the block in.
    let (h, s) = frame::Header::parse(f).unwrap();
    let block = SigBlock {
        kid: w.leaf.kid().clone(),
        chain: vec![w.cert()],
        sig: w
            .leaf
            .sign(
                purpose,
                &h.signed_message(
                    &h.codec
                        .decompress(
                            "body",
                            &f[s.body.clone()],
                            h.body_raw_len,
                            frame::MAX_BODY_RAW,
                        )
                        .unwrap(),
                ),
            )
            .unwrap(),
    };
    let sig = postcard::to_allocvec(&block).unwrap();
    let mut out = f[..s.sig.start].to_vec();
    out.extend_from_slice(&sig);
    out[72..80].copy_from_slice(&(sig.len() as u64).to_le_bytes());
    out
}

#[test]
fn a_signature_under_another_context_is_refused() {
    let w = world();
    let f = w.signed();
    assert!(
        open(&w, &resign(&w, &f, Purpose::KbpBody)).is_ok(),
        "the control verifies"
    );
    for p in [
        Purpose::KernelArtifact,
        Purpose::KernelIndex,
        Purpose::KeyCert,
        Purpose::Revocations,
    ] {
        assert_eq!(
            open(&w, &resign(&w, &f, p)).err(),
            Some(ErrorCode::BlueprintSignature),
            "{p:?}"
        );
    }
}

#[test]
fn trust_refusals_reach_open_with_their_codes() {
    let w = world();
    let f = w.signed();
    // 2026-10-22: Unknown kid: signed by a key with no certificate in the chain.
    let stranger = Signer::from_seeds(
        Kid::new("stranger").unwrap(),
        Suite::MlDsa65Ed25519,
        &[8; 32],
        &[8; 32],
    );
    let g = frame::sign(&f, &stranger, vec![w.cert()]).unwrap();
    assert_eq!(open(&w, &g).err(), Some(ErrorCode::BlueprintUntrusted));
    // 2026-10-22: Revoked kid.
    let list = sign_revocations(
        &w.root,
        &Revocations {
            serial: 2,
            revoked: vec![w.leaf.kid().clone()],
        },
    )
    .unwrap();
    let revoked = verify_revocations(&w.anchors, &list, 0).unwrap();
    let p = TrustPolicy {
        revoked: &revoked,
        ..w.policy()
    };
    assert_eq!(
        frame::open(&f, &p).map(|_| ()).map_err(|e| e.code()),
        Err(ErrorCode::BlueprintRevoked)
    );
    // 2026-10-22: Expired certificate.
    let late = TrustPolicy {
        now: UnixMs(NOW + 1000),
        ..w.policy()
    };
    assert_eq!(
        frame::open(&f, &late).map(|_| ()).map_err(|e| e.code()),
        Err(ErrorCode::BlueprintCertExpired)
    );
    // 2026-10-22: Rollback to a lower revocation serial.
    let old = sign_revocations(
        &w.root,
        &Revocations {
            serial: 1,
            revoked: vec![],
        },
    )
    .unwrap();
    assert_eq!(
        verify_revocations(&w.anchors, &old, 2)
            .map(|_| ())
            .map_err(|e| e.code()),
        Err(ErrorCode::BlueprintRollback)
    );
    // 2026-10-22: A key certified for artifacts may not sign a blueprint.
    let art = certify(
        &w.root,
        &KeyCert {
            kid: w.leaf.kid().clone(),
            keys: w.leaf.public(),
            scope: Scope::Artifacts,
            not_before_ms: NOW - 1,
            not_after_ms: NOW + 1,
            serial: 2,
        },
    )
    .unwrap();
    assert_eq!(
        open(&w, &frame::sign(&f, &w.leaf, vec![art]).unwrap()).err(),
        Some(ErrorCode::BlueprintScope)
    );
}

#[test]
fn truncated_trailing_and_oversized_files_are_refused() {
    let w = world();
    let f = w.signed();
    assert_eq!(
        open(&w, &f[..40]).err(),
        Some(ErrorCode::BlueprintFrame),
        "truncated header"
    );
    assert_eq!(
        open(&w, &f[..f.len() - 1]).err(),
        Some(ErrorCode::BlueprintFrame),
        "truncated section"
    );
    let mut t = f.clone();
    t.push(0);
    assert_eq!(
        open(&w, &t).err(),
        Some(ErrorCode::BlueprintFrame),
        "trailing byte"
    );
    let mut big = f.clone();
    big[8..16].copy_from_slice(&(MAX_BODY + 1).to_le_bytes());
    assert_eq!(
        open(&w, &big).err(),
        Some(ErrorCode::BlueprintTooLarge),
        "body over its cap"
    );
    let mut huge = f.clone();
    huge[64..72].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(
        open(&w, &huge).err(),
        Some(ErrorCode::BlueprintTooLarge),
        "payload over its cap"
    );
    let mut magic = f.clone();
    magic[0] = b'X';
    assert_eq!(open(&w, &magic).err(), Some(ErrorCode::BlueprintFrame));
    let mut fmt = f;
    fmt[4] = 2;
    assert_eq!(
        open(&w, &fmt).err(),
        Some(ErrorCode::BlueprintFormat),
        "unknown KBP_FORMAT"
    );
}

#[test]
fn a_newer_body_schema_is_refused() {
    let w = world();
    let mut bp = blueprint();
    bp.schema = metrale_blueprint::BODY_SCHEMA + 1;
    let f = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
    let f = frame::sign(&f, &w.leaf, vec![w.cert()]).unwrap();
    assert_eq!(open(&w, &f).err(), Some(ErrorCode::BlueprintFormat));
}

/// 2026-10-22: A body older than [`OLDEST_BODY_SCHEMA`] (schema 1 carried the mirrored circuit
/// records) is refused by its schema field, never decoded as the new shape.
#[test]
fn a_body_schema_older_than_the_reader_decodes_is_refused() {
    let w = world();
    let mut bp = blueprint();
    bp.schema = metrale_blueprint::OLDEST_BODY_SCHEMA - 1;
    let f = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
    let f = frame::sign(&f, &w.leaf, vec![w.cert()]).unwrap();
    assert_eq!(open(&w, &f).err(), Some(ErrorCode::BlueprintFormat));
    bp.schema = metrale_blueprint::OLDEST_BODY_SCHEMA;
    let f = frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap();
    let f = frame::sign(&f, &w.leaf, vec![w.cert()]).unwrap();
    assert_eq!(open(&w, &f).err(), None);
}

#[test]
fn a_tampered_inline_kernel_is_a_hash_mismatch() {
    let w = world();
    let lite = w.signed();
    let o = open(&w, &lite).unwrap();
    let (u, mut p) = resolve::full(&o, internal).unwrap();
    let mut evil = MOD_A.to_vec();
    evil.extend_from_slice(b"// planted\n");
    p[0] = InlineArtifact {
        artifact: p[0].artifact,
        ..InlineArtifact::pack(&evil, Codec::WRITE).unwrap()
    };
    let full = frame::rewrite(&lite, &w.policy(), &u, &p).unwrap();
    let f = open(&w, &full).expect("the payload is unsigned, so the file still opens");
    let err = resolve::resolve_modules(&f, |_| None).unwrap_err();
    assert_eq!(err.code(), ErrorCode::ArtifactHashMismatch);
    assert!(err.to_string().contains("add"), "names the module: {err}");
}

#[test]
fn a_wrong_internal_module_is_a_hash_mismatch_and_a_missing_one_is_missing() {
    let w = world();
    let o = open(&w, &w.signed()).unwrap();
    let swapped = |n: &str| internal(if n == "add" { "scale" } else { "add" });
    assert_eq!(
        resolve::resolve_modules(&o, swapped).unwrap_err().code(),
        ErrorCode::ArtifactHashMismatch
    );
    assert_eq!(
        resolve::resolve_modules(&o, |_| None).unwrap_err().code(),
        ErrorCode::KernelMissing
    );
}

#[test]
fn disk_and_remote_sources_parse_but_are_not_in_phase_1() {
    let w = world();
    let lite = w.signed();
    let o = open(&w, &lite).unwrap();
    for src in [
        KernelSource::Disk {
            path: "/cache/x".into(),
        },
        KernelSource::Remote {
            url: "https://x".into(),
            locator_sha256: metrale_blueprint::Sha256::of(&[b"l"]),
        },
    ] {
        let mut u = o.unsigned.clone();
        u.sources[0] = src;
        let f = open(&w, &frame::rewrite(&lite, &w.policy(), &u, &[]).unwrap()).unwrap();
        assert_eq!(
            resolve::resolve_modules(&f, internal).unwrap_err().code(),
            ErrorCode::KernelSourceNotInPhase
        );
    }
}

#[test]
fn transport_round_trips_and_refuses_garbage() {
    use metrale_blueprint::transport::{from_text, to_text};
    let w = world();
    let f = w.signed();
    let t = to_text(&f);
    assert!(t.starts_with("kbp1."));
    assert_eq!(from_text(&format!("  {t}\n")).unwrap(), f);
    assert_eq!(
        from_text("kbp2.AAAA").unwrap_err().code(),
        ErrorCode::BlueprintFrame
    );
    assert_eq!(
        from_text("kbp1.@@@@").unwrap_err().code(),
        ErrorCode::BlueprintFrame
    );
}
