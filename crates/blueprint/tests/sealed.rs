// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Sealed files, the positive side (DESIGN-encrypted-kernels.md §8 E2): one identity
//! across plain lite, plain full, sealed lite and sealed full; a keyless open sees the plain
//! blueprint; every recipient opens every module to its plaintext; recode and lite ↔ full keep
//! both signatures; remote ciphertexts are named by their hash only; sealing is deterministic
//! given its randomness; the card states the sealed section. Keys are fixed test seeds.

mod common;

use std::collections::BTreeMap;

use common::*;
use metrale_blueprint::codec::Codec;
use metrale_blueprint::frame::{self, FLAG_SEALED, KernelSource, Opened};
use metrale_blueprint::ids::Sha256;
use metrale_blueprint::resolve::{self, no_store};
use metrale_blueprint::resolve_sealed::{
    full_sealed, lite_sealed, resolve_unsealed, unseal, verify_ciphertexts,
};
use metrale_blueprint::sealing::Placement;
use metrale_error::ErrorCode;

fn open(w: &World, f: &[u8]) -> Opened {
    frame::open(f, &w.policy(), MEM).unwrap()
}

fn sig_bytes(f: &[u8]) -> Vec<u8> {
    let (_, s) = frame::Header::parse(f).unwrap();
    f[s.sig].to_vec()
}

fn sealed_bytes(f: &[u8]) -> Vec<u8> {
    let (_, s) = frame::Header::parse(f).unwrap();
    f[s.sealed].to_vec()
}

const BASE: &str = "https://kernels.example/sealed/";

#[test]
fn plain_and_sealed_lite_and_full_share_one_identity_and_one_body_signature() {
    let w = world();
    let plain_lite = w.signed();
    let plain_full = full(&w);
    let sealed_full = sealed_full(&w);
    let remote = seal_for(
        &w,
        &plain_lite,
        &[&recipient(7)],
        "acme",
        Placement::Remote { base: BASE },
        0,
    );
    let files = [&plain_lite, &plain_full, &sealed_full, &remote.file];
    let opened: Vec<Opened> = files.iter().map(|f| open(&w, f)).collect();
    for (o, f) in opened.iter().zip(files) {
        assert_eq!(o.id, opened[0].id);
        assert_eq!(
            o.blueprint, opened[0].blueprint,
            "a keyless open sees the plain body"
        );
        assert_eq!(sig_bytes(f), sig_bytes(&plain_lite), "body signature bytes");
    }
    assert!(opened[0].sealed.is_none() && opened[1].sealed.is_none());
    assert_eq!(
        opened[2].sealed, opened[3].sealed,
        "one manifest for both placements"
    );
    assert_eq!(sealed_bytes(&sealed_full), sealed_bytes(&remote.file));
    assert_ne!(sealed_full[6] & FLAG_SEALED, 0);
}

fn plaintexts(w: &World, f: &[u8], key: &metrale_seal::HybridSecret) -> Vec<Vec<u8>> {
    let o = open(w, f);
    let cek = unseal(&o, key).unwrap();
    resolve_unsealed(&o, internal, no_store, &cek, MEM)
        .unwrap()
        .into_iter()
        .map(|(_, b)| b.into_owned())
        .collect()
}

#[test]
fn every_recipient_opens_every_module_to_its_plaintext() {
    let w = world();
    let (a, b) = (recipient(7), recipient(8));
    let out = seal_for(&w, &w.signed(), &[&b, &a], "acme", Placement::Inline, 0);
    assert_eq!(out.recipients.len(), 2);
    assert!(out.recipients[0] < out.recipients[1], "written sorted");
    for key in [&a, &b] {
        assert_eq!(
            plaintexts(&w, &out.file, key),
            vec![MOD_A.to_vec(), MOD_B.to_vec()]
        );
    }
    let o = open(&w, &out.file);
    let p = &o.payload;
    assert!(
        p.iter().all(|e| e.bytes != MOD_A && e.bytes != MOD_B),
        "the payload is ciphertext"
    );
    assert_eq!(p[0].artifact, o.blueprint.kernels.modules[0].artifact);
}

#[test]
fn without_a_key_a_sealed_module_is_key_required_and_its_ciphertext_still_verifies() {
    let w = world();
    let o = open(&w, &sealed_full(&w));
    let err = resolve::resolve_modules(&o, internal).unwrap_err();
    assert_eq!(err.code(), ErrorCode::SealedKeyRequired);
    assert!(err.to_string().contains("add"), "{err}");
    assert_eq!(verify_ciphertexts(&o, internal, no_store).unwrap(), 2);
}

#[test]
fn recode_keeps_sealed_entries_and_both_signatures() {
    let w = world();
    let f = sealed_full(&w);
    let before = open(&w, &f);
    for codec in [Codec::Stored, Codec::Kzx, Codec::Deflate] {
        let g = frame::recode(&f, &w.policy(), MEM, codec).unwrap();
        let after = open(&w, &g);
        assert_eq!(
            after.payload, before.payload,
            "{codec:?}: ciphertext byte-identical"
        );
        assert_eq!(sealed_bytes(&g), sealed_bytes(&f));
        assert_eq!(sig_bytes(&g), sig_bytes(&f));
        assert_eq!(after.id, before.id);
        assert_eq!(
            plaintexts(&w, &g, &recipient(7)),
            vec![MOD_A.to_vec(), MOD_B.to_vec()]
        );
    }
}

#[test]
fn lite_and_full_of_a_sealed_file_keep_both_signatures_and_round_trip() {
    let w = world();
    let f = sealed_full(&w);
    let o = open(&w, &f);
    let (u, p) = lite_sealed(&o, BASE).unwrap();
    let lite = frame::rewrite(&f, &w.policy(), MEM, &u, &p).unwrap();
    assert_eq!(sig_bytes(&lite), sig_bytes(&f));
    assert_eq!(sealed_bytes(&lite), sealed_bytes(&f));
    let l = open(&w, &lite);
    assert!(l.payload.is_empty());
    let manifest = &o.sealed.as_ref().unwrap().manifest;
    let store: BTreeMap<Sha256, Vec<u8>> = manifest
        .artifacts
        .iter()
        .zip(&o.payload)
        .map(|(a, e)| (a.ct_sha256, e.bytes.clone()))
        .collect();
    let fetch = |h: &Sha256| store.get(h).cloned();
    let (u, p) = full_sealed(&l, internal, fetch).unwrap();
    assert_eq!(
        frame::rewrite(&lite, &w.policy(), MEM, &u, &p).unwrap(),
        f,
        "byte for byte"
    );
    let (u, p) = resolve::lite(&o);
    assert_eq!(
        frame::rewrite(&f, &w.policy(), MEM, &u, &p)
            .unwrap_err()
            .code(),
        ErrorCode::SealedManifest,
        "a sealed module may not become internal"
    );
}

#[test]
fn remote_ciphertexts_are_named_by_their_hash_only() {
    let w = world();
    let out = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme-corp",
        Placement::Remote { base: BASE },
        0,
    );
    let o = open(&w, &out.file);
    let manifest = &o.sealed.as_ref().unwrap().manifest;
    assert_eq!(out.ciphertexts.len(), 2);
    for ((src, a), (h, ct)) in o
        .unsigned
        .sources
        .iter()
        .zip(&manifest.artifacts)
        .zip(&out.ciphertexts)
    {
        let KernelSource::Remote {
            url,
            locator_sha256,
        } = src
        else {
            panic!("{src:?}")
        };
        assert_eq!(
            url,
            &format!("https://kernels.example/sealed/{}", a.ct_sha256.hex())
        );
        assert!(
            !url.contains("acme") && !url.contains("add") && !url.contains("toy"),
            "{url}"
        );
        assert_eq!((locator_sha256, h), (&a.ct_sha256, &a.ct_sha256));
        assert_eq!(Sha256::of(&[ct]), *h);
    }
    let store: BTreeMap<Sha256, Vec<u8>> = out.ciphertexts.iter().cloned().collect();
    let cek = unseal(&o, &recipient(7)).unwrap();
    let plain =
        resolve_unsealed(&o, internal, |h: &Sha256| store.get(h).cloned(), &cek, MEM).unwrap();
    assert_eq!(&*plain[1].1, MOD_B);
    assert_eq!(
        resolve_unsealed(&o, internal, no_store, &cek, MEM)
            .unwrap_err()
            .code(),
        ErrorCode::KernelMissing
    );
    assert!(
        !out.file.windows(9).any(|x| x == b"acme-corp"),
        "the customer id is not in the file"
    );
}

#[test]
fn sealing_is_deterministic_in_its_randomness_and_fresh_per_run() {
    let w = world();
    let a = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        Placement::Inline,
        0,
    )
    .file;
    let b = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        Placement::Inline,
        0,
    )
    .file;
    let c = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        Placement::Inline,
        1,
    )
    .file;
    assert_eq!(a, b);
    let (pa, pc) = (open(&w, &a).payload, open(&w, &c).payload);
    assert_ne!(pa[0].bytes, pc[0].bytes, "a new run is a new CEK");
}

#[test]
fn sealing_refuses_what_it_cannot_state() {
    let w = world();
    let sealed = sealed_full(&w);
    let pk = [recipient(7).public()];
    let try_seal =
        |input: &[u8], keys: Vec<metrale_seal::HybridPublic>, customer: &str, n: usize| {
            metrale_blueprint::sealing::seal(
                input,
                &w.policy(),
                MEM,
                internal,
                &metrale_blueprint::sealing::SealRequest {
                    recipients: &keys,
                    customer_id: customer,
                    issued_ms: NOW,
                    codec: Codec::Kzx,
                    placement: Placement::Inline,
                },
                run_random(0, n),
                &w.leaf,
                vec![w.cert()],
            )
            .map(|_| ())
            .unwrap_err()
            .code()
        };
    assert_eq!(
        try_seal(&sealed, pk.to_vec(), "acme", 1),
        ErrorCode::SealedManifest,
        "resealing"
    );
    assert_eq!(
        try_seal(&w.signed(), vec![], "acme", 0),
        ErrorCode::SealedManifest,
        "no recipient"
    );
    let twice = vec![pk[0].clone(), pk[0].clone()];
    assert_eq!(
        try_seal(&w.signed(), twice, "acme", 2),
        ErrorCode::SealedManifest,
        "twice"
    );
    assert_eq!(
        try_seal(&w.signed(), pk.to_vec(), "", 1),
        ErrorCode::SealedManifest,
        "no customer"
    );
    assert_eq!(
        try_seal(&w.signed(), pk.to_vec(), "acme", 2),
        ErrorCode::InvalidArgument,
        "random"
    );
}
