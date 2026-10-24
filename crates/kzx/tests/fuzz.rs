// SPDX-License-Identifier: MIT OR Apache-2.0

//! Fuzz-style tests with a fixed seed: random streams, mutated valid streams and forged
//! headers never panic, never exceed the caps and always terminate. Every case is bounded
//! by `Limits`, so a bug that lost the bound would show up as a slow or allocating test, not
//! as a hang in production.

#[path = "../examples/bench/synth.rs"]
mod synth;

use kzx::{Error, Limits};

const LIMITS: Limits = Limits {
    max_output: 1 << 20,
    max_memory: 64 << 20,
};

fn check(packed: &[u8]) {
    match kzx::decompress(packed, &LIMITS) {
        Ok(out) => {
            let info = kzx::inspect(packed).expect("a decodable stream inspects");
            assert_eq!(
                out.len() as u64,
                info.total_raw,
                "output length must equal the declaration"
            );
            assert!(info.total_raw <= LIMITS.max_output);
        }
        Err(Error::OutputCapExceeded { declared, cap }) => assert!(declared > cap),
        Err(Error::MemoryCapExceeded { needed, cap }) => assert!(needed > cap),
        Err(_) => {}
    }
}

#[test]
fn random_bytes_never_panic_and_always_terminate() {
    let mut rng = synth::Rng::new(0xf00d);
    for _ in 0..3000 {
        let n = rng.below(96) as usize;
        let mut data = rng.bytes(n);
        if rng.below(2) == 0 && data.len() >= 4 {
            data[..3].copy_from_slice(b"kzx");
            data[3] = 1;
        }
        check(&data);
    }
}

#[test]
fn mutated_valid_streams_never_panic_or_exceed_caps() {
    let mut rng = synth::Rng::new(0xbeef);
    let inputs = [
        synth::ptx_module(&mut rng, 0, 1, 60),
        synth::elf_object(&mut rng, 100, 2),
        synth::postcard_body(&mut rng, 40, 20, 8),
        rng.bytes(700),
    ];
    let valid: Vec<Vec<u8>> = inputs.iter().map(|i| kzx::compress(i)).collect();
    let multi = kzx::compress_blocks(&inputs.iter().map(|i| i.as_slice()).collect::<Vec<_>>());
    for base in valid.iter().chain(std::iter::once(&multi)) {
        check(base);
        for _ in 0..150 {
            let mut m = base.clone();
            match rng.below(5) {
                0 => {
                    let at = rng.below(m.len() as u64) as usize;
                    m[at] ^= 1 << rng.below(8);
                }
                1 => {
                    let at = rng.below(m.len() as u64) as usize;
                    m[at] = (rng.next() >> 8) as u8;
                }
                2 => {
                    let cut = rng.below(m.len() as u64) as usize;
                    m.truncate(cut);
                }
                3 => {
                    let extra = rng.below(40) as usize;
                    m.extend(rng.bytes(extra));
                }
                _ => {
                    // Header-targeted: lengths, codec, memory parameter.
                    let at = 8 + rng.below(20) as usize;
                    if at < m.len() {
                        m[at] = (rng.next() >> 8) as u8;
                    }
                }
            }
            check(&m);
        }
    }
}

#[test]
fn forged_headers_are_refused_without_allocating() {
    let mut rng = synth::Rng::new(0xcafe);
    for _ in 0..500 {
        let mut h = Vec::new();
        h.extend_from_slice(b"kzx\x01");
        h.extend_from_slice(&1u32.to_le_bytes());
        h.push((rng.below(5)) as u8);
        h.push((rng.below(64)) as u8);
        h.extend_from_slice(&0u16.to_le_bytes());
        h.extend_from_slice(&rng.next().to_le_bytes());
        h.extend_from_slice(&rng.next().to_le_bytes());
        let extra = rng.below(16) as usize;
        h.extend(rng.bytes(extra));
        check(&h);
    }
}

#[test]
fn every_ptx_decode_stops_at_the_declared_length() {
    // Markers in a corrupt PTX stream could emit up to 20 digits each; the output is still
    // exactly the declared length.
    let mut rng = synth::Rng::new(0x1234);
    let raw = synth::ptx_module(&mut rng, 2, 1, 80);
    let packed = kzx::compress(&raw);
    for i in 0..200 {
        let mut m = packed.clone();
        let at = 28 + (i * 7919) % (m.len() - 28);
        m[at] ^= 0xff;
        if let Ok(out) = kzx::decompress(&m, &LIMITS) {
            assert_eq!(out.len(), raw.len());
        }
    }
}
