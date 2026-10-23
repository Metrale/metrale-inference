// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `met blueprint seal | open | verify --recipient-key` on the GLM recipe and the
//! kernels this binary embeds, through the command functions (the Entropy port is the seeded
//! host-mem ChaCha20 host, the clock a fixed instant; no crypto is mocked):
//! - the CEK leak test (DESIGN-encrypted-kernels.md §8 E2): with the seeded run the CEK and
//!   every per-artifact key and nonce are known, and neither the command's whole report, nor an
//!   error, nor any output file (sealed full, sealed lite, every ciphertext) holds one of them,
//!   raw or in hex;
//! - plain lite, sealed full and sealed lite share the `BlueprintId`; both recipients open every
//!   module to the embedded bytes; and the GLM sizes are reported.

use std::path::{Path, PathBuf};

use metrale_abstract_io::UnixMs;
use metrale_blueprint::frame;
use metrale_blueprint_sign::write_public;
use metrale_blueprint_trust::{Anchor, Revoked, TrustPolicy};
use metrale_io_host_mem::SeededEntropy;
use metrale_seal::{ArtifactKey, SealRandom, hex};

use super::super::blueprint::DECODE_MEMORY;
use super::super::blueprint_args::{OutArgs, SealArgs, SealedArgs, TrustArgs};
use super::super::blueprint_build::blueprint_tests::{Glm, glm};
use super::*;

const NOW: UnixMs = UnixMs(1_760_000_000_000);
const RUN_SEED: u64 = 42;

fn write_mode(p: &Path, text: &str, mode: u32) {
    std::fs::write(p, text).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(mode)).unwrap();
    }
    #[cfg(not(unix))]
    let _ = mode;
}

struct Fixture {
    _dir: tempfile::TempDir,
    root: PathBuf,
    g: Glm,
    input: PathBuf,
    trust: TrustArgs,
    key: PathBuf,
    cert: PathBuf,
    pubs: Vec<PathBuf>,
    secrets: Vec<PathBuf>,
}

fn fixture() -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().to_path_buf();
    let g = glm(&root);
    let input = root.join("plain.kbp");
    std::fs::write(&input, &g.lite).unwrap();
    let anchor = root.join("root.anchor");
    write_public(
        &anchor,
        &Anchor {
            kid: g.root.kid().clone(),
            keys: g.root.public(),
        },
    )
    .unwrap();
    // 2026-10-22: The test leaf's seeds ([3; 32], [4; 32], `blueprint_tests::keys`).
    let key = root.join("leaf.key");
    write_mode(
        &key,
        &format!(
            "kbp-key v1\nkid test-blueprints\nsuite mldsa65+ed25519\nmldsa-seed {}\ned25519-seed {}\n",
            "03".repeat(32),
            "04".repeat(32)
        ),
        0o600,
    );
    let cert = root.join("leaf.cert");
    write_public(&cert, &g.cert).unwrap();
    let (mut pubs, mut secrets) = (Vec::new(), Vec::new());
    for i in 0..2u64 {
        let (k, p) = (
            root.join(format!("r{i}.key")),
            root.join(format!("r{i}.pub")),
        );
        let mut log = Vec::new();
        gen_recipient_cmd(&SeededEntropy::new(100 + i, 0), &k, &p, &mut log).unwrap();
        assert!(String::from_utf8(log).unwrap().contains("the only copy"));
        secrets.push(k);
        pubs.push(p);
    }
    Fixture {
        trust: TrustArgs {
            trust_anchor: Some(anchor),
            revocations: None,
        },
        _dir: dir,
        root,
        g,
        input,
        key,
        cert,
        pubs,
        secrets,
    }
}

fn seal_args(f: &Fixture, out: &str, remote: Option<&str>) -> SealArgs {
    SealArgs {
        input: f.input.clone(),
        trust: f.trust.clone(),
        recipient_public: f.pubs.clone(),
        customer_id: "acme-corp".into(),
        key: f.key.clone(),
        cert: f.cert.clone(),
        remote_base: remote.map(String::from),
        ciphertext_dir: remote.map(|_| f.root.join(format!("{out}.ct"))),
        out: OutArgs {
            output: f.root.join(out),
            text: false,
        },
    }
}

/// 2026-10-22: Every secret of the seeded run, raw and as lower/upper hex.
fn secrets_of_run(f: &Fixture, id: &metrale_blueprint::BlueprintId) -> Vec<Vec<u8>> {
    let random = SealRandom::draw(&SeededEntropy::new(RUN_SEED, 0), 2).unwrap();
    let cek = random.cek;
    let mut raw = vec![cek.as_bytes().to_vec()];
    for w in &random.wraps {
        raw.push(w.esk.to_vec());
        raw.push(w.m.to_vec());
    }
    for m in &f.g.bp.kernels.modules {
        let k = ArtifactKey::derive(&cek, &id.0.0, &m.artifact.0.0);
        raw.push(k.key().to_vec());
        raw.push(k.nonce().to_vec());
    }
    let mut out = Vec::new();
    for r in raw {
        out.push(hex(&r).into_bytes());
        out.push(hex(&r).to_uppercase().into_bytes());
        out.push(r);
    }
    out
}

/// 2026-10-22: The length of the first secret found in `hay`, if any.
fn found(hay: &[u8], secrets: &[Vec<u8>]) -> Option<usize> {
    secrets
        .iter()
        .find(|s| hay.windows(s.len()).any(|w| w == s.as_slice()))
        .map(Vec::len)
}

fn assert_clean(what: &str, hay: &[u8], secrets: &[Vec<u8>]) {
    if let Some(len) = found(hay, secrets) {
        panic!("{what} holds key material ({len} bytes)");
    }
}

#[test]
fn sealing_glm_leaks_no_key_material_and_every_recipient_opens_it() {
    let f = fixture();
    let anchors = &f.g.anchors;
    let revoked = Revoked::embedded().unwrap();
    let policy = TrustPolicy {
        anchors,
        revoked: &revoked,
        now: NOW,
    };
    let plain = frame::open(&f.g.lite, &policy, DECODE_MEMORY).unwrap();
    let secrets = secrets_of_run(&f, &plain.id);
    // 2026-10-22: Positive control for the search: the seeded run's CEK hex is found where it is.
    let cek_hex = secrets[0].clone();
    assert_eq!(
        found(&[b"x".as_slice(), &cek_hex].concat(), &secrets),
        Some(64)
    );
    let mut report = Vec::new();
    seal_cmd(
        &seal_args(&f, "full.kbp", None),
        &SeededEntropy::new(RUN_SEED, 0),
        NOW,
        &mut report,
    )
    .unwrap();
    let mut remote_report = Vec::new();
    let remote = seal_args(&f, "lite.kbp", Some("https://kernels.example/sealed"));
    seal_cmd(
        &remote,
        &SeededEntropy::new(RUN_SEED, 0),
        NOW,
        &mut remote_report,
    )
    .unwrap();
    let full = std::fs::read(f.root.join("full.kbp")).unwrap();
    let lite = std::fs::read(f.root.join("lite.kbp")).unwrap();
    assert_clean("the seal report", &report, &secrets);
    assert_clean("the remote seal report", &remote_report, &secrets);
    assert_clean("the sealed full file", &full, &secrets);
    assert_clean("the sealed lite file", &lite, &secrets);
    let ct_dir = remote.ciphertext_dir.clone().unwrap();
    let mut ct_bytes = 0u64;
    for e in std::fs::read_dir(&ct_dir).unwrap() {
        let bytes = std::fs::read(e.unwrap().path()).unwrap();
        ct_bytes += bytes.len() as u64;
        assert_clean("a ciphertext file", &bytes, &secrets);
    }
    // 2026-10-22: An error raised after the run's randomness is drawn (a recipient listed twice).
    let mut twice = seal_args(&f, "twice.kbp", None);
    twice.recipient_public.push(f.pubs[0].clone());
    let err = seal_cmd(
        &twice,
        &SeededEntropy::new(RUN_SEED, 0),
        NOW,
        &mut Vec::new(),
    )
    .unwrap_err();
    assert_clean("an error", format!("{err:?} {err:#}").as_bytes(), &secrets);

    let sealed = frame::open(&full, &policy, DECODE_MEMORY).unwrap();
    let sealed_lite = frame::open(&lite, &policy, DECODE_MEMORY).unwrap();
    assert_eq!(
        (sealed.id, sealed_lite.id),
        (plain.id, plain.id),
        "one identity"
    );
    let known = SealRandom::draw(&SeededEntropy::new(RUN_SEED, 0), 2)
        .unwrap()
        .cek;
    let real = unseal(&sealed, &read_recipient_key(&f.secrets[0]).unwrap()).unwrap();
    assert!(
        real.same_as(&known),
        "the secrets searched for are the run's real ones"
    );
    let text = String::from_utf8(report).unwrap();
    assert!(
        text.contains(&format!("sealed blueprint {}", plain.id)),
        "{text}"
    );
    for (i, k) in f.secrets.iter().enumerate() {
        let line = verify_sealed(
            &sealed,
            &SealedArgs {
                recipient_key: Some(k.clone()),
                ciphertexts: None,
            },
        )
        .unwrap();
        assert!(
            line.contains(&format!(
                "{} modules decrypted",
                f.g.bp.kernels.modules.len()
            )),
            "{line}"
        );
        let out_dir = f.root.join(format!("open{i}"));
        let input = f.root.join(if i == 0 { "full.kbp" } else { "lite.kbp" });
        open_cmd(
            &input,
            &f.trust,
            k,
            Some(&ct_dir),
            &out_dir,
            NOW,
            &mut Vec::new(),
        )
        .unwrap();
        let internal = internal_of(&plain).unwrap();
        for m in &f.g.bp.kernels.modules {
            let got = std::fs::read(out_dir.join(format!("{}.ptx", m.name))).unwrap();
            assert_eq!(Some(got.as_slice()), internal(&m.name), "{}", m.name);
        }
    }
    let keyless = verify_sealed(
        &sealed,
        &SealedArgs {
            recipient_key: None,
            ciphertexts: None,
        },
    )
    .unwrap();
    assert!(keyless.contains("plaintext not checked"), "{keyless}");
    let (u, p) = resolve::full(&plain, internal_of(&plain).unwrap()).unwrap();
    let plain_full = frame::rewrite(&f.g.lite, &policy, DECODE_MEMORY, &u, &p).unwrap();
    eprintln!(
        "GLM {}: plain lite {} | plain full {} | sealed full {} | sealed lite {} + {} ciphertext bytes in {} files | sealed section {} | raw modules {}",
        plain.id,
        f.g.lite.len(),
        plain_full.len(),
        full.len(),
        lite.len(),
        ct_bytes,
        sealed.sealed.as_ref().unwrap().manifest.artifacts.len(),
        sealed.header.sealed_len,
        f.g.bp.kernels.modules.iter().map(|m| m.len).sum::<u64>()
    );
}

#[test]
fn sealed_only_flags_are_refused_on_plain_files_and_lite_needs_a_base() {
    let f = fixture();
    let revoked = Revoked::embedded().unwrap();
    let policy = TrustPolicy {
        anchors: &f.g.anchors,
        revoked: &revoked,
        now: NOW,
    };
    let plain = frame::open(&f.g.lite, &policy, DECODE_MEMORY).unwrap();
    let with_key = SealedArgs {
        recipient_key: Some(f.secrets[0].clone()),
        ciphertexts: None,
    };
    assert!(verify_sealed(&plain, &with_key).is_err());
    assert!(plain_form(&plain, Some("https://x"), None, false).is_err());
    seal_cmd(
        &seal_args(&f, "full.kbp", None),
        &SeededEntropy::new(1, 0),
        NOW,
        &mut Vec::new(),
    )
    .unwrap();
    let sealed = frame::open(
        &std::fs::read(f.root.join("full.kbp")).unwrap(),
        &policy,
        DECODE_MEMORY,
    )
    .unwrap();
    assert!(
        sealed_form(&sealed, None, None, false).is_err(),
        "lite of a sealed file needs a base"
    );
    let open_twice = open_cmd(
        &f.root.join("full.kbp"),
        &f.trust,
        &f.secrets[0],
        None,
        &f.root,
        NOW,
        &mut Vec::new(),
    );
    assert!(
        open_twice.is_err(),
        "open never writes into an existing directory"
    );
}
