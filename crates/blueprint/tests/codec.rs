// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The compression seam: identity and signatures never depend on the codec, and
//! decompression of untrusted input is bounded, each refusal with its own code.

mod common;

use common::*;
use metrale_blueprint::codec::Codec;
use metrale_blueprint::frame::{self, InlineArtifact, MAX_ARTIFACT, MAX_BODY_RAW};
use metrale_blueprint::resolve;
use metrale_error::ErrorCode;

fn code(r: Result<impl Sized, metrale_error::Error>) -> Option<ErrorCode> {
    r.err().map(|e| e.code())
}

#[test]
fn a_recoded_file_keeps_its_id_and_its_signature() {
    let w = world();
    let lite = w.signed();
    let o = frame::open(&lite, &w.policy()).unwrap();
    let raw = frame::recode(&lite, &w.policy(), Codec::Stored).unwrap();
    assert_ne!(raw, lite, "the stored bytes differ");
    assert_eq!(raw[7], Codec::Stored.id());
    let r = frame::open(&raw, &w.policy()).unwrap();
    assert_eq!(
        (r.id, &r.blueprint, &r.signer),
        (o.id, &o.blueprint, &o.signer)
    );
    assert_eq!(
        frame::recode(&raw, &w.policy(), Codec::Deflate).unwrap(),
        lite,
        "recoding back is the original file, byte for byte"
    );

    let (u, p) = resolve::full(&o, internal).unwrap();
    let full = frame::rewrite(&lite, &w.policy(), &u, &p).unwrap();
    let full_raw = frame::recode(&full, &w.policy(), Codec::Stored).unwrap();
    let f = frame::open(&full_raw, &w.policy()).unwrap();
    assert_eq!(f.id, o.id);
    assert!(f.payload.iter().all(|a| a.codec == Codec::Stored));
    let ids = |p: &[InlineArtifact]| p.iter().map(|a| a.artifact).collect::<Vec<_>>();
    assert_eq!(ids(&f.payload), ids(&p), "artifact ids are over raw bytes");
    assert_eq!(
        resolve::resolve_modules(&f, |_| None).unwrap().len(),
        2,
        "stored artifacts still resolve and hash-check"
    );
}

#[test]
fn a_corrupt_body_stream_is_a_codec_error() {
    let w = world();
    let mut f = w.signed();
    assert_eq!(f[7], Codec::Deflate.id());
    let body = body_range(&f);
    for b in &mut f[body] {
        *b = 0xff;
    }
    assert_eq!(
        code(frame::open(&f, &w.policy())),
        Some(ErrorCode::BlueprintCodec)
    );
}

/// 2026-10-22: The declared raw length is the bound: above the cap it is refused before any
/// work; a stream that yields more or fewer bytes than declared is refused.
#[test]
fn decompression_is_bounded_by_the_declared_raw_length() {
    let w = world();
    let f = w.signed();
    let raw_len = u64::from_le_bytes(f[16..24].try_into().unwrap());
    let with_len = |n: u64| {
        let mut g = f.clone();
        g[16..24].copy_from_slice(&n.to_le_bytes());
        code(frame::open(&g, &w.policy()))
    };
    assert_eq!(with_len(raw_len), None, "the control opens");
    assert_eq!(
        with_len(MAX_BODY_RAW + 1),
        Some(ErrorCode::BlueprintTooLarge)
    );
    assert_eq!(
        with_len(raw_len - 1),
        Some(ErrorCode::BlueprintCodec),
        "longer than declared"
    );
    assert_eq!(
        with_len(raw_len + 1),
        Some(ErrorCode::BlueprintCodec),
        "shorter than declared"
    );

    let bomb = vec![0u8; 1 << 20];
    for codec in [Codec::Stored, Codec::Deflate] {
        let stored = codec.compress(&bomb);
        assert_eq!(
            code(codec.decompress("t", &stored, 1 << 20, (1 << 20) - 1)),
            Some(ErrorCode::BlueprintTooLarge),
            "{codec:?}"
        );
        assert_eq!(
            codec.decompress("t", &stored, 1 << 20, 1 << 20).unwrap(),
            bomb
        );
        assert_eq!(
            code(codec.decompress("t", &stored, 1 << 10, 1 << 20)),
            Some(ErrorCode::BlueprintCodec),
            "{codec:?}"
        );
    }
    let over = InlineArtifact {
        raw_len: MAX_ARTIFACT + 1,
        ..InlineArtifact::pack(MOD_A, Codec::WRITE).unwrap()
    };
    assert_eq!(code(over.unpack()), Some(ErrorCode::BlueprintTooLarge));
}

#[test]
fn an_unknown_codec_id_is_refused() {
    let w = world();
    let mut f = w.signed();
    f[7] = 9;
    assert_eq!(
        code(frame::open(&f, &w.policy())),
        Some(ErrorCode::BlueprintFormat)
    );
    assert_eq!(
        postcard::from_bytes::<Codec>(&[9]).map_err(|_| ()),
        Err(()),
        "the wire form refuses it too"
    );
    for c in [Codec::Stored, Codec::Deflate] {
        assert_eq!(postcard::to_allocvec(&c).unwrap(), [c.id()]);
        assert_eq!(Codec::from_id(c.id()).unwrap(), c);
    }
}
