// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Sealed files, the negative side (DESIGN-encrypted-kernels.md §3, §8 E2): each
//! attack is refused at its step with its own code. A flipped manifest byte is a signature
//! error; a section signed for another body is `SealedManifest`; flag and section must agree;
//! stripping the section leaves ciphertext that fails `ArtifactHashMismatch`; an `internal`
//! sealed module is refused; swapped, garbage, other-customer and other-blueprint ciphertexts are
//! `SealedCiphertextHash` before any decryption; a non-recipient and an altered wrap are refused
//! and yield no bytes; no key material reaches the file.

mod common;

use common::*;
use metrale_blueprint::frame::sealed_check::sign_sealed;
use metrale_blueprint::frame::{
    self, FLAG_FULL, FLAG_SEALED, FLAG_SOLID, KernelSource, Opened, SealedBlock, Unsigned,
};
use metrale_blueprint::resolve::{self, no_store};
use metrale_blueprint::resolve_sealed::{resolve_unsealed, unseal};
use metrale_blueprint::sealing::Placement;
use metrale_error::ErrorCode;
use metrale_seal::{ArtifactKey, Cek, hex};

fn code(w: &World, f: &[u8]) -> ErrorCode {
    frame::open(f, &w.policy(), MEM)
        .map(|_| ())
        .unwrap_err()
        .code()
}

fn open(w: &World, f: &[u8]) -> Opened {
    frame::open(f, &w.policy(), MEM).unwrap()
}

fn resolve_code(w: &World, f: &[u8]) -> (ErrorCode, String) {
    let o = open(w, f);
    let cek = unseal(&o, &recipient(7)).unwrap();
    let e = resolve_unsealed(&o, internal, no_store, &cek, MEM)
        .map(|_| ())
        .unwrap_err();
    (e.code(), e.to_string())
}

fn section(f: &[u8]) -> std::ops::Range<usize> {
    frame::Header::parse(f).unwrap().1.sealed
}

#[test]
fn a_flipped_manifest_byte_is_a_signature_error() {
    let w = world();
    let f = sealed_full(&w);
    let s = section(&f);
    // 2026-10-22: Past the manifest's varint length and schema, inside the manifest bytes.
    for at in [s.start + 4, s.start + 40, s.start + 300] {
        let mut g = f.clone();
        g[at] ^= 1;
        assert_eq!(code(&w, &g), ErrorCode::BlueprintSignature, "byte {at}");
    }
}

/// 2026-10-22: Sabotage target (DESIGN-encrypted-kernels.md §8 E2): with the
/// `manifest.blueprint == id` check deleted from `check_manifest`, both `assert_eq!`s here get
/// `Ok` instead of `SealedManifest` and go red.
#[test]
fn a_sealed_section_bound_to_another_body_is_refused() {
    let w = world();
    let f = sealed_full(&w);
    // 2026-10-22: Another body: revision 2 of the same blueprint (other id, same modules).
    let mut bp = blueprint();
    bp.revision = 2;
    let other = frame::sign(
        &frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap(),
        &w.leaf,
        vec![w.cert()],
        MEM,
    )
    .unwrap();
    let other_full = {
        let o = open(&w, &other);
        let (u, p) = resolve::full(&o, internal).unwrap();
        frame::rewrite(&other, &w.policy(), MEM, &u, &p).unwrap()
    };
    // 2026-10-22: Transplant the whole sealed tail (locator, ciphertext payload, section).
    let (_, s) = frame::Header::parse(&f).unwrap();
    let mut g = splice(&other_full, Part::Sealed, &f[s.sealed.clone()], f[6]);
    g = splice(&g, Part::Locator, &f[s.locator.clone()], f[6]);
    g = splice(&g, Part::Payload, &f[s.payload.clone()], f[6]);
    assert_eq!(code(&w, &g), ErrorCode::SealedManifest, "transplanted");
    // 2026-10-22: A manifest CI really signed, but for the other body.
    let mut m = open(&w, &f).sealed.unwrap().manifest;
    m.blueprint = open(&w, &other).id;
    let resigned = sign_sealed(&m, &w.leaf, vec![w.cert()]).unwrap();
    assert_eq!(
        code(&w, &splice(&f, Part::Sealed, &resigned, f[6])),
        ErrorCode::SealedManifest
    );
}

#[test]
fn a_newer_sealed_schema_is_refused_after_its_signature() {
    let w = world();
    let f = sealed_full(&w);
    let mut m = open(&w, &f).sealed.unwrap().manifest;
    m.schema = 2;
    let resigned = sign_sealed(&m, &w.leaf, vec![w.cert()]).unwrap();
    assert_eq!(
        code(&w, &splice(&f, Part::Sealed, &resigned, f[6])),
        ErrorCode::BlueprintFormat
    );
}

#[test]
fn the_sealed_flag_and_the_sealed_section_must_agree() {
    let w = world();
    let f = sealed_full(&w);
    let flags = f[6];
    assert_eq!(
        code(&w, &splice(&f, Part::Sealed, &[], flags)),
        ErrorCode::BlueprintFrame,
        "flag, no section"
    );
    let s = f[section(&f)].to_vec();
    assert_eq!(
        code(&w, &splice(&f, Part::Sealed, &s, flags & !FLAG_SEALED)),
        ErrorCode::BlueprintFrame,
        "section, no flag"
    );
}

#[test]
fn stripping_the_sealed_section_leaves_ciphertext_that_fails_its_hash() {
    let w = world();
    let f = sealed_full(&w);
    let plain = splice(&f, Part::Sealed, &[], FLAG_FULL | FLAG_SOLID);
    let o = open(&w, &plain);
    assert!(o.sealed.is_none(), "a downgraded file opens as plain");
    assert_eq!(
        resolve::resolve_modules(&o, internal).unwrap_err().code(),
        ErrorCode::ArtifactHashMismatch
    );
    // 2026-10-22: The remote form: stripped, its sources point at ciphertext it cannot resolve.
    let remote = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        Placement::Remote { base: "https://s" },
        0,
    );
    let o = open(&w, &splice(&remote.file, Part::Sealed, &[], 0));
    assert_eq!(
        resolve::resolve_modules(&o, internal).unwrap_err().code(),
        ErrorCode::KernelSourceNotInPhase
    );
}

#[test]
fn a_sealed_module_sourced_internal_is_refused_at_open() {
    let w = world();
    let f = sealed_full(&w);
    let mut u: Unsigned =
        postcard::from_bytes(&f[frame::Header::parse(&f).unwrap().1.locator]).unwrap();
    u.sources[1] = KernelSource::Internal;
    let g = splice(&f, Part::Locator, &postcard::to_allocvec(&u).unwrap(), f[6]);
    assert_eq!(code(&w, &g), ErrorCode::SealedManifest);
}

fn swap_sources(f: &[u8]) -> Vec<u8> {
    let mut u: Unsigned =
        postcard::from_bytes(&f[frame::Header::parse(f).unwrap().1.locator]).unwrap();
    u.sources.swap(0, 1);
    splice(f, Part::Locator, &postcard::to_allocvec(&u).unwrap(), f[6])
}

#[test]
fn ciphertexts_swapped_between_modules_are_refused_before_decryption() {
    let w = world();
    let (c, msg) = resolve_code(&w, &swap_sources(&sealed_full(&w)));
    assert_eq!(c, ErrorCode::SealedCiphertextHash);
    assert!(
        msg.contains("sealed module add "),
        "names the module: {msg}"
    );
}

/// 2026-10-22: `w`'s sealed full with module 0's inline ciphertext replaced by `ct`.
fn with_ciphertext(w: &World, f: &[u8], ct: Vec<u8>) -> Vec<u8> {
    let o = open(w, f);
    let mut payload = o.payload.clone();
    payload[0].bytes = ct;
    frame::rewrite(f, &w.policy(), MEM, &o.unsigned, &payload).unwrap()
}

/// 2026-10-22: Sabotage target (DESIGN-encrypted-kernels.md §8 E2): with the pre-AEAD length
/// and hash check deleted from `resolve_sealed::ciphertext`, the code here is `SealedOpen`
/// (the AEAD ran on attacker bytes) and the `assert_eq!` goes red.
#[test]
fn garbage_ciphertext_is_a_hash_refusal_not_a_decryption_attempt() {
    let w = world();
    let f = sealed_full(&w);
    let len = open(&w, &f).payload[0].bytes.len();
    let (c, _) = resolve_code(&w, &with_ciphertext(&w, &f, vec![0x5a; len]));
    assert_eq!(c, ErrorCode::SealedCiphertextHash);
}

#[test]
fn another_customers_or_another_blueprints_ciphertext_is_refused() {
    let w = world();
    let f = sealed_full(&w);
    let theirs = seal_for(
        &w,
        &w.signed(),
        &[&recipient(9)],
        "globex",
        Placement::Inline,
        50,
    )
    .file;
    let replay = with_ciphertext(&w, &f, open(&w, &theirs).payload[0].bytes.clone());
    assert_eq!(
        resolve_code(&w, &replay).0,
        ErrorCode::SealedCiphertextHash,
        "other customer"
    );
    let mut bp = blueprint();
    bp.revision = 2;
    let other = frame::sign(
        &frame::encode_unsigned(&bp, &unsigned(&bp), &[]).unwrap(),
        &w.leaf,
        vec![w.cert()],
        MEM,
    )
    .unwrap();
    let other_sealed = seal_for(&w, &other, &[&recipient(7)], "acme", Placement::Inline, 0).file;
    let cross = with_ciphertext(&w, &f, open(&w, &other_sealed).payload[0].bytes.clone());
    assert_eq!(
        resolve_code(&w, &cross).0,
        ErrorCode::SealedCiphertextHash,
        "other blueprint"
    );
}

#[test]
fn a_non_recipient_and_an_altered_wrap_are_refused_and_yield_no_bytes() {
    let w = world();
    let f = sealed_full(&w);
    let o = open(&w, &f);
    let e = unseal(&o, &recipient(8)).unwrap_err();
    assert_eq!(e.code(), ErrorCode::SealedNotRecipient);
    assert!(
        e.to_string()
            .contains(&recipient(8).public().fingerprint().hex())
    );
    // 2026-10-22: The wrap another blueprint's run made for this same recipient.
    let theirs = seal_for(
        &w,
        &w.signed(),
        &[&recipient(7)],
        "acme",
        Placement::Inline,
        99,
    )
    .file;
    let mut forged = o.clone();
    forged.sealed.as_mut().unwrap().manifest.recipients[0].wrap =
        open(&w, &theirs).sealed.unwrap().manifest.recipients[0]
            .wrap
            .clone();
    forged.sealed.as_mut().unwrap().manifest.recipients[0]
        .wrap
        .ct[0] ^= 1;
    assert_eq!(
        unseal(&forged, &recipient(7)).unwrap_err().code(),
        ErrorCode::SealedUnwrap
    );
    // 2026-10-22: A wrong CEK opens nothing.
    let wrong = Cek::from_bytes([0xee; 32]);
    let e = resolve_unsealed(&o, internal, no_store, &wrong, MEM)
        .map(|_| ())
        .unwrap_err();
    assert_eq!(e.code(), ErrorCode::SealedOpen);
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|x| x == needle)
}

#[test]
fn no_key_material_reaches_the_sealed_file() {
    let w = world();
    let f = sealed_full(&w);
    let o = open(&w, &f);
    let cek = run_random(0, 1).cek;
    assert!(
        unseal(&o, &recipient(7)).unwrap().same_as(&cek),
        "the CEK is the seeded one"
    );
    let mut secrets = vec![cek.as_bytes().to_vec()];
    for m in &o.blueprint.kernels.modules {
        let k = ArtifactKey::derive(&cek, &o.id.0.0, &m.artifact.0.0);
        secrets.push(k.key().to_vec());
        secrets.push(k.nonce().to_vec());
    }
    // 2026-10-22: Positive control: the search finds a secret that is present.
    assert!(contains(
        &[&b"x"[..], &secrets[0], b"y"].concat(),
        &secrets[0]
    ));
    let block: SealedBlock = postcard::from_bytes(&f[section(&f)]).unwrap();
    for s in &secrets {
        assert!(!contains(&f, s), "raw secret in the file");
        assert!(!contains(&f, hex(s).as_bytes()), "hex secret in the file");
        assert!(!contains(&block.manifest, s));
    }
}
