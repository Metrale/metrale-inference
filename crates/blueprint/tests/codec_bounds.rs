// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Decompression of untrusted bytes is bounded and every refusal is typed: tampered
//! kzx streams never panic and never yield other kernels, the declared raw lengths and the
//! caller's model-memory budget are hard caps checked before any work.

mod common;

use common::*;
use metrale_blueprint::codec::{Codec, DecodeMemory};
use metrale_blueprint::frame::{self, MAX_ARTIFACT, MAX_BODY_RAW, MAX_PAYLOAD_RAW};
use metrale_blueprint::resolve;
use metrale_error::ErrorCode;

fn code(r: Result<impl Sized, metrale_error::Error>) -> Option<ErrorCode> {
    r.err().map(|e| e.code())
}

/// 2026-10-22: The most model memory any kzx block of `stream` needs.
fn needed(stream: &[u8]) -> u64 {
    let info = kzx::inspect(stream).unwrap();
    info.blocks
        .iter()
        .map(|b| kzx::memory_bytes(b.profile, b.mem_log2))
        .max()
        .unwrap()
}

/// 2026-10-22: A flipped byte is refused with a typed code, or (a coder tail byte that does not
/// change the decoded bytes) opens to the identical body under the same id; never a panic.
#[test]
fn every_tampered_body_byte_is_a_typed_refusal_or_decodes_identically() {
    let w = world();
    let f = w.signed();
    let id = frame::open(&f, &w.policy(), MEM).unwrap().id;
    let mut identical = 0;
    for i in body_range(&f) {
        let mut g = f.clone();
        g[i] ^= 0x5a;
        match frame::open(&g, &w.policy(), MEM) {
            Err(e) => assert!(
                matches!(
                    e.code(),
                    ErrorCode::BlueprintCodec
                        | ErrorCode::BlueprintBodyHash
                        | ErrorCode::BlueprintTooLarge
                ),
                "byte {i}: {e}"
            ),
            Ok(o) => {
                assert_eq!((o.id, &o.blueprint), (id, &blueprint()), "byte {i}");
                identical += 1;
            }
        }
    }
    assert!(
        identical <= 4,
        "only coder tail bytes may be inert: {identical}"
    );
    let mut g = f.clone();
    let body = body_range(&g);
    g[body].fill(0xff);
    assert_eq!(
        code(frame::open(&g, &w.policy(), MEM)),
        Some(ErrorCode::BlueprintCodec),
        "not a kzx stream at all"
    );
}

#[test]
fn every_tampered_payload_byte_is_refused_or_caught_by_the_kernel_hash_or_inert() {
    let w = world();
    let f = full(&w);
    for i in payload_range(&f) {
        let mut g = f.clone();
        g[i] ^= 0x5a;
        match frame::open(&g, &w.policy(), MEM) {
            Err(e) => assert!(
                matches!(
                    e.code(),
                    ErrorCode::BlueprintCodec
                        | ErrorCode::BlueprintTooLarge
                        | ErrorCode::BlueprintFrame
                        | ErrorCode::BlueprintDecode
                        | ErrorCode::BlueprintFormat
                ),
                "byte {i}: {e}"
            ),
            Ok(o) => match resolve::resolve_modules(&o, |_| None) {
                // 2026-10-22: The payload is unsigned; tampered bytes that still decode must
                // fail the signed per-module hash, or be a coder tail byte that leaves the
                // decoded kernels identical.
                Err(e) => assert_eq!(e.code(), ErrorCode::ArtifactHashMismatch, "byte {i}"),
                Ok(m) => assert_eq!((&*m[0].1, &*m[1].1), (MOD_A, MOD_B), "byte {i}"),
            },
        }
    }
}

/// 2026-10-22: The header's raw length is the output cap handed to kzx: a stream declaring more
/// is refused before any work; one yielding less than declared is refused.
#[test]
fn the_declared_body_length_caps_kzx_output() {
    let w = world();
    let f = w.signed();
    let raw_len = u64::from_le_bytes(f[16..24].try_into().unwrap());
    let with_len = |n: u64| {
        let mut g = f.clone();
        g[16..24].copy_from_slice(&n.to_le_bytes());
        code(frame::open(&g, &w.policy(), MEM))
    };
    assert_eq!(with_len(raw_len), None, "the control opens");
    assert_eq!(
        with_len(MAX_BODY_RAW + 1),
        Some(ErrorCode::BlueprintTooLarge)
    );
    assert_eq!(
        with_len(raw_len - 1),
        Some(ErrorCode::BlueprintTooLarge),
        "the stream declares more than the header allows"
    );
    assert_eq!(
        with_len(raw_len + 1),
        Some(ErrorCode::BlueprintCodec),
        "the stream yields fewer bytes than declared"
    );

    let bomb = vec![0u8; 1 << 20];
    for codec in [Codec::Stored, Codec::Deflate, Codec::Kzx] {
        let stored = codec.compress(&bomb);
        assert_eq!(
            code(codec.decompress("t", &stored, 1 << 20, (1 << 20) - 1, MEM)),
            Some(ErrorCode::BlueprintTooLarge),
            "{codec:?}"
        );
        assert_eq!(
            codec
                .decompress("t", &stored, 1 << 20, 1 << 20, MEM)
                .unwrap(),
            bomb
        );
        let short = code(codec.decompress("t", &stored, 1 << 10, 1 << 20, MEM));
        let want = if codec == Codec::Kzx {
            ErrorCode::BlueprintTooLarge
        } else {
            ErrorCode::BlueprintCodec
        };
        assert_eq!(
            short,
            Some(want),
            "{codec:?}: a 1 MiB stream under a 1 KiB length"
        );
    }
}

#[test]
fn the_model_memory_budget_is_a_hard_cap() {
    let w = world();
    let lite = w.signed();
    let body = &lite[body_range(&lite)];
    let need = needed(body);
    assert!(need <= MEM.0, "a lite body fits the wasm budget: {need}");
    assert!(frame::open(&lite, &w.policy(), DecodeMemory(need)).is_ok());
    let e = frame::open(&lite, &w.policy(), DecodeMemory(need - 1)).unwrap_err();
    assert_eq!(e.code(), ErrorCode::BlueprintTooLarge);
    assert!(e.to_string().contains("model memory"), "{e}");

    let f = full(&w);
    let p = &f[payload_range(&f)];
    let index_len = u64::from_le_bytes(p[9..17].try_into().unwrap()) as usize;
    let need_p = needed(&p[17 + index_len..]);
    let budget = DecodeMemory(need.max(need_p));
    assert!(frame::open(&f, &w.policy(), budget).is_ok());
    if need_p > need {
        assert_eq!(
            code(frame::open(&f, &w.policy(), DecodeMemory(need_p - 1))),
            Some(ErrorCode::BlueprintTooLarge),
            "the payload block is held to the same budget"
        );
    }
    let stored = Codec::Kzx.compress(&MOD_A.repeat(100));
    let n = (MOD_A.len() * 100) as u64;
    assert_eq!(
        code(Codec::Kzx.decompress("t", &stored, n, n, DecodeMemory(needed(&stored) - 1))),
        Some(ErrorCode::BlueprintTooLarge)
    );
}

#[test]
fn the_solid_payload_lengths_are_checked_before_decoding() {
    let w = world();
    let f = full(&w);
    let at = payload_range(&f).start;
    let patched = |off: usize, v: u64| {
        let mut g = f.clone();
        g[at + off..at + off + 8].copy_from_slice(&v.to_le_bytes());
        code(frame::open(&g, &w.policy(), MEM))
    };
    let raw_len = u64::from_le_bytes(f[at + 1..at + 9].try_into().unwrap());
    assert_eq!(patched(1, raw_len), None, "the control opens");
    assert_eq!(
        patched(1, MAX_PAYLOAD_RAW + 1),
        Some(ErrorCode::BlueprintTooLarge)
    );
    assert_eq!(
        patched(1, raw_len + 1),
        Some(ErrorCode::BlueprintFrame),
        "the index no longer sums to the raw length"
    );
    assert_eq!(
        patched(9, u64::MAX),
        Some(ErrorCode::BlueprintFrame),
        "index past the end"
    );

    let o = frame::open(&f, &w.policy(), MEM).unwrap();
    let mut g = f.clone();
    let index = postcard::to_allocvec(&vec![
        (o.payload[0].artifact, MAX_ARTIFACT + 1),
        (o.payload[1].artifact, 0u64),
    ])
    .unwrap();
    let old_len = u64::from_le_bytes(f[at + 9..at + 17].try_into().unwrap()) as usize;
    g.splice(at + 17..at + 17 + old_len, index.iter().copied());
    g[at + 9..at + 17].copy_from_slice(&(index.len() as u64).to_le_bytes());
    let delta = index.len() as i64 - old_len as i64;
    let pl = u64::from_le_bytes(g[64..72].try_into().unwrap()) as i64 + delta;
    g[64..72].copy_from_slice(&(pl as u64).to_le_bytes());
    assert_eq!(
        code(frame::open(&g, &w.policy(), MEM)),
        Some(ErrorCode::BlueprintTooLarge),
        "one artifact over its cap"
    );
}
