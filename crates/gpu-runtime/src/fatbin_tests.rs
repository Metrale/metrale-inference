// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests of the fatbin PTX reader on real nvcc 13 output (`fatbin_fixtures/`,
//! whose README gives the commands) and on corruptions of it.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::*;

const FATBIN: &[u8] = include_bytes!("fatbin_fixtures/pdl_global.fatbin");
const FATBIN_ZSTD: &[u8] = include_bytes!("fatbin_fixtures/pdl_global_zstd.fatbin");
const PTX: &str = include_str!("fatbin_fixtures/pdl_global.ptx");

/// 2026-10-24: Comments dropped and whitespace runs collapsed, blank lines removed: the form
/// in which nvcc's fatbin PTX and its `--ptx` output agree.
fn normalised(ptx: &str) -> Vec<String> {
    ptx.lines()
        .map(|l| {
            l.split("//")
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|l| !l.is_empty())
        .collect()
}

#[test]
fn the_fatbin_ptx_is_the_ptx_build_of_the_same_source() {
    assert!(is_fatbin(FATBIN));
    let texts = ptx_texts(FATBIN).unwrap();
    assert_eq!(texts.len(), 1);
    assert!(texts[0].contains(".target sm_121f"), "{}", &texts[0][..200]);
    assert_eq!(normalised(texts[0]), normalised(PTX));
}

#[cfg(feature = "cuda")]
#[test]
fn the_registry_metadata_read_through_a_fatbin_matches_the_ptx_modules() {
    let text = ptx_texts(FATBIN).unwrap()[0];
    let globals = crate::ptx_globals::defined_globals(text);
    assert!(globals.contains("fixture_counter"), "{globals:?}");
    assert_eq!(globals, crate::ptx_globals::defined_globals(PTX));
    let waits = crate::ptx_pdl::waiting_entries(text);
    assert_eq!(waits.into_iter().collect::<Vec<_>>(), ["fixture_waits"]);
    assert_eq!(crate::ptx_pdl::waiting_entries(PTX).len(), 1);
}

#[test]
fn a_compressed_ptx_payload_is_refused_not_skipped() {
    assert!(is_fatbin(FATBIN_ZSTD));
    let err = ptx_texts(FATBIN_ZSTD).unwrap_err();
    assert!(err.contains("compressed"), "{err}");
}

#[test]
fn ptx_text_and_other_blobs_are_not_fatbins() {
    assert!(!is_fatbin(PTX.as_bytes()));
    assert!(!is_fatbin(b"\x7fELF\x02\x01\x01"));
    assert!(!is_fatbin(&[0x50, 0xed, 0x55]));
    assert!(ptx_texts(PTX.as_bytes()).is_err());
}

#[test]
fn every_truncation_is_an_error_never_a_panic_or_a_partial_read() {
    for len in 4..FATBIN.len() {
        // 2026-10-24: The container header states its size, so any shorter blob is malformed.
        let cut = &FATBIN[..len];
        match ptx_texts(cut) {
            Err(_) => {}
            Ok(t) => panic!(
                "a {len}-byte prefix of {} parsed: {} bytes",
                FATBIN.len(),
                t[0].len()
            ),
        }
    }
}

#[test]
fn a_fatbin_with_no_ptx_entry_is_refused() {
    let mut only_elf = FATBIN.to_vec();
    let header = usize::from(read_u16(&only_elf, 6).unwrap());
    let mut e = header;
    while e < only_elf.len() {
        let eh = read_u32(&only_elf, e + 4).unwrap() as usize;
        let size = read_u64(&only_elf, e + 8).unwrap() as usize;
        if read_u16(&only_elf, e) == Some(KIND_PTX) {
            only_elf[e] = 2;
        }
        e += eh + size;
    }
    let err = ptx_texts(&only_elf).unwrap_err();
    assert!(err.contains("no PTX"), "{err}");
}

#[test]
fn an_entry_that_claims_more_than_its_container_is_refused() {
    let mut long = FATBIN.to_vec();
    let first = usize::from(read_u16(&long, 6).unwrap());
    long[first + 8..first + 16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(ptx_texts(&long).is_err());
}
