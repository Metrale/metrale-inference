// SPDX-License-Identifier: MIT OR Apache-2.0

//! Negative tests: every refusal is typed and happens before any decoding work.

#[path = "../examples/bench/synth.rs"]
mod synth;

use kzx::{Error, Limits, Profile};

const LIMITS: Limits = Limits {
    max_output: 1 << 24,
    max_memory: 1 << 30,
};

fn sample() -> Vec<u8> {
    let mut rng = synth::Rng::new(21);
    synth::ptx_module(&mut rng, 0, 1, 200)
}

#[test]
fn bad_magic_and_version_are_refused() {
    let mut packed = kzx::compress(&sample());
    assert_eq!(
        kzx::decompress(b"", &LIMITS),
        Err(Error::Truncated { needed: 8 })
    );
    assert_eq!(
        kzx::decompress(b"kz", &LIMITS),
        Err(Error::Truncated { needed: 6 })
    );
    let mut bad = packed.clone();
    bad[0] = b'K';
    assert_eq!(kzx::decompress(&bad, &LIMITS), Err(Error::BadMagic));
    packed[3] = 9;
    assert_eq!(
        kzx::decompress(&packed, &LIMITS),
        Err(Error::UnsupportedVersion { found: 9 })
    );
}

#[test]
fn output_cap_is_checked_from_the_header_before_any_work() {
    let raw = sample();
    let packed = kzx::compress(&raw);
    let declared = raw.len() as u64;
    let tight = Limits {
        max_output: declared - 1,
        max_memory: 1 << 30,
    };
    assert_eq!(
        kzx::decompress(&packed, &tight),
        Err(Error::OutputCapExceeded {
            declared,
            cap: declared - 1
        })
    );
    let exact = Limits {
        max_output: declared,
        max_memory: 1 << 30,
    };
    assert!(kzx::decompress(&packed, &exact).is_ok());
    // A forged header declaring 2^60 bytes is refused the same way, with no allocation.
    let mut forged = packed.clone();
    forged[12..20].copy_from_slice(&(1u64 << 60).to_le_bytes());
    assert_eq!(
        kzx::decompress(&forged, &LIMITS),
        Err(Error::OutputCapExceeded {
            declared: 1 << 60,
            cap: 1 << 24
        })
    );
}

#[test]
fn memory_cap_is_checked_before_allocation() {
    let raw = sample();
    let packed = kzx::compress_with(&raw, Profile::Ptx, 20).unwrap();
    let needed = kzx::memory_bytes(Profile::Ptx, 20);
    assert!(
        needed > 100 << 20,
        "mem_log2 20 should need >100 MiB, needs {needed}"
    );
    let small = Limits {
        max_output: 1 << 24,
        max_memory: needed - 1,
    };
    assert_eq!(
        kzx::decompress(&packed, &small),
        Err(Error::MemoryCapExceeded {
            needed,
            cap: needed - 1
        })
    );
    let enough = Limits {
        max_output: 1 << 24,
        max_memory: needed,
    };
    assert!(kzx::decompress(&packed, &enough).is_ok());
}

#[test]
fn block_header_fields_are_validated() {
    let packed = kzx::compress(&sample());
    let mut unknown = packed.clone();
    unknown[8] = 77;
    assert_eq!(
        kzx::decompress(&unknown, &LIMITS),
        Err(Error::UnknownCodec { id: 77 })
    );
    let mut mem = packed.clone();
    mem[9] = 3;
    assert_eq!(
        kzx::decompress(&mem, &LIMITS),
        Err(Error::BadMemoryParameter { mem_log2: 3 })
    );
    mem[9] = 40;
    assert_eq!(
        kzx::decompress(&mem, &LIMITS),
        Err(Error::BadMemoryParameter { mem_log2: 40 })
    );
    let mut reserved = packed.clone();
    reserved[10] = 1;
    assert_eq!(
        kzx::decompress(&reserved, &LIMITS),
        Err(Error::ReservedNonZero)
    );
    let mut trailing = packed.clone();
    trailing.extend_from_slice(b"xyz");
    assert_eq!(
        kzx::decompress(&trailing, &LIMITS),
        Err(Error::TrailingBytes { count: 3 })
    );
    let truncated = &packed[..packed.len() - 5];
    assert_eq!(
        kzx::decompress(truncated, &LIMITS),
        Err(Error::Truncated { needed: 5 })
    );
    let mut count = packed.clone();
    count[4..8].copy_from_slice(&(70_000u32).to_le_bytes());
    assert_eq!(
        kzx::decompress(&count, &LIMITS),
        Err(Error::TooManyBlocks { count: 70_000 })
    );
}

#[test]
fn stored_blocks_must_have_equal_lengths() {
    let packed = kzx::compress(b"tiny");
    let info = kzx::inspect(&packed).unwrap();
    assert_eq!(info.blocks[0].profile, Profile::Stored);
    let mut bad = packed.clone();
    bad[12..20].copy_from_slice(&3u64.to_le_bytes());
    assert_eq!(
        kzx::decompress(&bad, &LIMITS),
        Err(Error::StoredLengthMismatch)
    );
}

#[test]
fn encoder_parameters_are_validated() {
    assert_eq!(
        kzx::compress_with(b"abc", Profile::Generic, 9),
        Err(Error::InvalidParameter)
    );
    assert_eq!(
        kzx::compress_with(b"abc", Profile::Generic, 27),
        Err(Error::InvalidParameter)
    );
    assert!(kzx::compress_with(b"abc", Profile::Generic, 10).is_ok());
}

#[test]
fn a_corrupt_body_decodes_to_the_declared_length_without_panicking() {
    let raw = sample();
    let packed = kzx::compress(&raw);
    let mut corrupt = packed.clone();
    let mid = 20 + (corrupt.len() - 20) / 2;
    corrupt[mid] ^= 0x55;
    let out = kzx::decompress(&corrupt, &LIMITS).unwrap();
    assert_eq!(out.len(), raw.len());
    assert!(
        out != raw,
        "a flipped byte must change the output (the caller's hash catches it)"
    );
}

#[test]
fn errors_display_their_fields() {
    let e = Error::OutputCapExceeded {
        declared: 10,
        cap: 5,
    };
    assert_eq!(
        e.to_string(),
        "kzx declares 10 output bytes, above the cap of 5"
    );
    assert_eq!(Error::BadMagic.to_string(), "not a kzx stream: bad magic");
}
