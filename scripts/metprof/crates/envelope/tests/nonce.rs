// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: nonce uniqueness. Under one DEK and one (kid, path) the file
//! key is fixed, so every seal must carry a fresh nonce. The property runs
//! over many seals with the real OS entropy; a constant-entropy control
//! proves the check detects a repeat.

mod common;

use std::collections::HashSet;

use common::{CREATED_AT, ConstantEntropy, FixedClock, StreamEntropy, request};
use metprof_envelope::{Dek, Entropy, OsEntropy, parse, seal};

const SEALS: usize = 4096;

fn distinct_nonces(dek: &Dek, entropy: &mut dyn Entropy) -> usize {
    let mut seen = HashSet::new();
    for i in 0..SEALS {
        let env = seal(
            dek,
            &request(None),
            &i.to_le_bytes(),
            entropy,
            &FixedClock(CREATED_AT),
        )
        .expect("seal");
        seen.insert(parse(&env).expect("parse").header.nonce);
    }
    seen.len()
}

#[test]
fn os_entropy_nonces_are_unique_over_many_seals() {
    let dek = Dek::generate(&mut OsEntropy).expect("dek");
    assert_eq!(distinct_nonces(&dek, &mut OsEntropy), SEALS);
}

#[test]
fn constant_entropy_control_is_detected() {
    let dek = Dek::from_bytes([3; 32]);
    assert_eq!(distinct_nonces(&dek, &mut ConstantEntropy(0)), 1);
}

#[test]
fn nonce_is_exactly_the_entropy_port_output() {
    let dek = Dek::from_bytes([5; 32]);
    let mut source = StreamEntropy::new(b"nonce");
    let mut twin = source.clone();
    let env = seal(
        &dek,
        &request(None),
        b"p",
        &mut source,
        &FixedClock(CREATED_AT),
    )
    .expect("seal");
    let mut want = [0u8; 12];
    twin.fill(&mut want).expect("fill");
    assert_eq!(parse(&env).expect("parse").header.nonce, want);
}
