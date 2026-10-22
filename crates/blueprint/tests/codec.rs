// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: The compression seam: kzx is the write codec, identity and signatures never
//! depend on the codec, and files written under deflate (per-entry payloads included) stay
//! readable and re-encode to kzx under the same id and signature.

mod common;

use common::*;
use metrale_blueprint::codec::Codec;
use metrale_blueprint::frame::{
    self, FLAG_FULL, FLAG_SOLID, Header, InlineArtifact, KernelSource, PayloadEntry, Unsigned,
};
use metrale_blueprint::resolve;
use metrale_error::ErrorCode;

fn ids(p: &[InlineArtifact]) -> Vec<metrale_blueprint::ArtifactId> {
    p.iter().map(|a| a.artifact).collect()
}

#[test]
fn kzx_is_the_write_codec_and_the_payload_is_one_solid_block() {
    let w = world();
    let lite = w.signed();
    assert_eq!(lite[7], Codec::Kzx.id(), "the body is written under kzx");
    assert_eq!(lite[6], 0, "a lite file carries no payload flag");
    let f = full(&w);
    assert_eq!(f[6], FLAG_FULL | FLAG_SOLID);
    let p = &f[payload_range(&f)];
    assert_eq!(p[0], Codec::Kzx.id(), "the solid payload is kzx");
    let raw_len = u64::from_le_bytes(p[1..9].try_into().unwrap());
    assert_eq!(raw_len, (MOD_A.len() + MOD_B.len()) as u64);
    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    assert_eq!(o.payload[0].bytes, MOD_A);
    assert_eq!(o.payload[1].bytes, MOD_B);
}

#[test]
fn kzx_round_trips_bodies_and_kernels_byte_for_byte() {
    let w = world();
    let body = postcard::to_allocvec(&blueprint()).unwrap();
    let mut solid = MOD_A.repeat(40);
    solid.extend_from_slice(&MOD_B.repeat(40));
    for raw in [&body[..], &solid[..], MOD_A, b"", b"x"] {
        let stored = Codec::Kzx.compress(raw);
        assert_eq!(stored, Codec::Kzx.compress(raw), "deterministic");
        let n = raw.len() as u64;
        assert_eq!(Codec::Kzx.decompress("t", &stored, n, n, MEM).unwrap(), raw);
    }
    assert!(
        Codec::Kzx.compress(&solid).len() * 10 < solid.len(),
        "it compresses"
    );
    let o = frame::open(&w.signed(), &w.policy(), MEM).unwrap();
    assert_eq!(
        o.blueprint,
        blueprint(),
        "a kzx body opens to the same blueprint"
    );
}

#[test]
fn a_recoded_file_keeps_its_id_and_its_signature() {
    let w = world();
    let lite = w.signed();
    let o = frame::open(&lite, &w.policy(), MEM).unwrap();
    for codec in [Codec::Stored, Codec::Deflate] {
        let other = frame::recode(&lite, &w.policy(), MEM, codec).unwrap();
        assert_ne!(other, lite, "{codec:?}: the stored bytes differ");
        assert_eq!(other[7], codec.id());
        let r = frame::open(&other, &w.policy(), MEM).unwrap();
        assert_eq!(
            (r.id, &r.blueprint, &r.signer),
            (o.id, &o.blueprint, &o.signer)
        );
        assert_eq!(
            frame::recode(&other, &w.policy(), MEM, Codec::Kzx).unwrap(),
            lite,
            "{codec:?} back to kzx is the original file, byte for byte"
        );
    }

    let f = full(&w);
    let fo = frame::open(&f, &w.policy(), MEM).unwrap();
    for codec in [Codec::Stored, Codec::Deflate] {
        let other = frame::recode(&f, &w.policy(), MEM, codec).unwrap();
        assert_eq!(other[payload_range(&other)][0], codec.id());
        let r = frame::open(&other, &w.policy(), MEM).unwrap();
        assert_eq!((r.id, &r.signer), (fo.id, &fo.signer));
        assert_eq!(
            ids(&r.payload),
            ids(&fo.payload),
            "artifact ids are over raw bytes"
        );
        assert_eq!(resolve::resolve_modules(&r, |_| None).unwrap().len(), 2);
        assert_eq!(
            frame::recode(&other, &w.policy(), MEM, Codec::Kzx).unwrap(),
            f
        );
    }
}

/// 2026-10-22: A full file as Phase 1 wrote it (deflate body, per-entry deflate payload, flag
/// bit 1 clear), assembled by hand and signed.
fn phase1_full(w: &World) -> Vec<u8> {
    let lite = frame::recode(&w.signed(), &w.policy(), MEM, Codec::Deflate).unwrap();
    let (mut h, s) = Header::parse(&lite).unwrap();
    let mut u: Unsigned = postcard::from_bytes(&lite[s.locator.clone()]).unwrap();
    u.sources = vec![
        KernelSource::Inline { entry: 0 },
        KernelSource::Inline { entry: 1 },
    ];
    let entry = |raw: &[u8]| PayloadEntry {
        artifact: metrale_blueprint::ArtifactId::of(raw),
        raw_len: raw.len() as u64,
        codec: Codec::Deflate,
        stored: Codec::Deflate.compress(raw),
    };
    let locator = postcard::to_allocvec(&u).unwrap();
    let payload = postcard::to_allocvec(&vec![entry(MOD_A), entry(MOD_B)]).unwrap();
    h.flags = FLAG_FULL;
    h.locator_len = locator.len() as u64;
    h.payload_len = payload.len() as u64;
    h.sig_len = 0;
    let mut file = h.write().to_vec();
    file.extend_from_slice(&lite[s.body]);
    file.extend_from_slice(&locator);
    file.extend_from_slice(&payload);
    frame::sign(&file, &w.leaf, vec![w.cert()], MEM).unwrap()
}

#[test]
fn a_phase1_deflate_file_stays_readable_and_reencodes_to_kzx_under_its_signature() {
    let w = world();
    let old = phase1_full(&w);
    assert_eq!((old[6], old[7]), (FLAG_FULL, Codec::Deflate.id()));
    let o = frame::open(&old, &w.policy(), MEM).unwrap();
    let bytes = resolve::resolve_modules(&o, |_| None).unwrap();
    assert_eq!((&*bytes[0].1, &*bytes[1].1), (MOD_A, MOD_B));

    let new = frame::recode(&old, &w.policy(), MEM, Codec::Kzx).unwrap();
    assert_eq!((new[6], new[7]), (FLAG_FULL | FLAG_SOLID, Codec::Kzx.id()));
    let sig = |f: &[u8]| f[Header::parse(f).unwrap().1.sig].to_vec();
    assert_eq!(
        sig(&new),
        sig(&old),
        "the signature block is copied, not re-made"
    );
    let n = frame::open(&new, &w.policy(), MEM).expect("the old signature verifies");
    assert_eq!(
        (n.id, &n.signer, &n.blueprint),
        (o.id, &o.signer, &o.blueprint)
    );
    assert_eq!(n.id, frame::open(&w.signed(), &w.policy(), MEM).unwrap().id);
    assert_eq!(resolve::resolve_modules(&n, |_| None).unwrap().len(), 2);
    assert!(new.len() < old.len(), "{} < {}", new.len(), old.len());
}

#[test]
fn an_unknown_codec_id_is_refused() {
    let w = world();
    let mut f = w.signed();
    f[7] = 9;
    assert_eq!(
        frame::open(&f, &w.policy(), MEM).err().map(|e| e.code()),
        Some(ErrorCode::BlueprintFormat)
    );
    let mut g = full(&w);
    let at = payload_range(&g).start;
    g[at] = 9;
    assert_eq!(
        frame::open(&g, &w.policy(), MEM).err().map(|e| e.code()),
        Some(ErrorCode::BlueprintFormat),
        "the solid payload's codec id too"
    );
    assert_eq!(
        postcard::from_bytes::<Codec>(&[9]).map_err(|_| ()),
        Err(()),
        "the wire form refuses it too"
    );
    for c in [Codec::Stored, Codec::Deflate, Codec::Kzx] {
        assert_eq!(postcard::to_allocvec(&c).unwrap(), [c.id()]);
        assert_eq!(Codec::from_id(c.id()).unwrap(), c);
    }
    assert_eq!(Codec::Kzx.id(), 2);
}
