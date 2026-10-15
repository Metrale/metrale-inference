// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: ML-KEM-768 (FIPS 203) known answers, run through the crate's
//! own `mlkem768_from_seed`, `mlkem768_ek_bytes`, `mlkem768_encapsulate` and
//! `mlkem768_decapsulate` (the calls the KeyRing wrap makes). The vectors are
//! the NIST ACVP-Server files vendored by RustCrypto/KEMs at ml-kem v0.3.2;
//! tests/data/mlkem768_acvp.txt cites the exact URLs, digests and selection.

mod common;

use std::collections::HashMap;

use common::{hex, hex_array};
use metprof_envelope::primitives::{
    mlkem768_decapsulate, mlkem768_ek_bytes, mlkem768_encapsulate, mlkem768_from_seed,
};
#[allow(deprecated)]
use ml_kem::ExpandedKeyEncoding;
use ml_kem::kem::KeyExport;
use ml_kem::ml_kem_768::DecapsulationKey;

const DATA: &str = include_str!("data/mlkem768_acvp.txt");

fn records(kind: &str) -> Vec<HashMap<&'static str, &'static str>> {
    DATA.split("\n\n")
        .map(|block| {
            block
                .lines()
                .filter(|l| !l.starts_with('#') && !l.is_empty())
                .map(|l| l.split_once(" = ").expect("name = value"))
                .collect::<HashMap<_, _>>()
        })
        .filter(|r| r.get("kind") == Some(&kind))
        .collect()
}

#[test]
fn data_file_has_the_selected_vectors() {
    assert_eq!(records("keygen").len(), 3);
    assert_eq!(records("encap").len(), 3);
    assert_eq!(records("decap-key").len(), 1);
    assert_eq!(records("decap").len(), 10);
}

#[test]
fn keygen_from_seed_matches_acvp_ek_and_dk() {
    for r in records("keygen") {
        let mut seed = hex(r["d"]);
        seed.extend(hex(r["z"]));
        let dk = mlkem768_from_seed(&seed.try_into().expect("64-byte seed"));
        assert_eq!(
            mlkem768_ek_bytes(&dk).to_vec(),
            hex(r["ek"]),
            "tcId {}",
            r["tcId"]
        );
        #[allow(deprecated)]
        let expanded = dk.to_expanded_bytes();
        assert_eq!(expanded.to_vec(), hex(r["dk"]), "tcId {}", r["tcId"]);
    }
}

#[test]
fn encapsulate_with_given_m_matches_acvp_c_and_k() {
    for r in records("encap") {
        let (c, k) = mlkem768_encapsulate(&hex_array(r["ek"]), &hex_array(r["m"])).expect("encaps");
        assert_eq!(c.to_vec(), hex(r["c"]), "tcId {}", r["tcId"]);
        assert_eq!(k.to_vec(), hex(r["k"]), "tcId {}", r["tcId"]);
    }
}

#[test]
fn decapsulate_matches_acvp_k_including_implicit_rejection() {
    let key = &records("decap-key")[0];
    assert_eq!(key["parameterSet"], "ML-KEM-768");
    let dk_bytes = hex(key["dk"]);
    #[allow(deprecated)]
    let dk =
        DecapsulationKey::from_expanded_bytes(dk_bytes.as_slice().try_into().expect("2400 bytes"))
            .expect("valid expanded dk");
    let cases = records("decap");
    let rejected = cases
        .iter()
        .filter(|r| r["reason"] == "modify ciphertext")
        .count();
    assert_eq!(
        rejected, 5,
        "both the valid and the implicit-rejection paths are covered"
    );
    for r in cases {
        let k = mlkem768_decapsulate(&dk, &hex_array(r["c"]));
        assert_eq!(
            k.to_vec(),
            hex(r["k"]),
            "tcId {} ({})",
            r["tcId"],
            r["reason"]
        );
    }
}

#[test]
fn seed_form_round_trips_through_ek_bytes() {
    let r = &records("keygen")[0];
    let mut seed = hex(r["d"]);
    seed.extend(hex(r["z"]));
    let dk = mlkem768_from_seed(&seed.try_into().expect("64-byte seed"));
    let ek = mlkem768_ek_bytes(&dk);
    assert_eq!(ek.to_vec(), dk.encapsulation_key().to_bytes().to_vec());
    let (c, k) = mlkem768_encapsulate(&ek, &[7u8; 32]).expect("encaps");
    assert_eq!(*mlkem768_decapsulate(&dk, &c), *k);
}
