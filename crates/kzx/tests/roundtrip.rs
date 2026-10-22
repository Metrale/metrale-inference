// SPDX-License-Identifier: MIT OR Apache-2.0

//! Byte-exact round trips over the synthetic corpus, every profile, every framing path.

#[path = "../examples/bench/synth.rs"]
mod synth;

use kzx::{Limits, Profile};

const LIMITS: Limits = Limits {
    max_output: 1 << 30,
    max_memory: 1 << 30,
};

fn roundtrip(raw: &[u8]) -> Vec<u8> {
    let packed = kzx::compress(raw);
    let back = kzx::decompress(&packed, &LIMITS).expect("decompress");
    assert_eq!(back.len(), raw.len(), "length differs");
    assert!(back == raw, "bytes differ");
    packed
}

#[test]
fn synthetic_ptx_modules_round_trip_and_sniff_as_ptx() {
    let mut rng = synth::Rng::new(11);
    for i in 0..3 {
        let m = synth::ptx_module(&mut rng, i, 2, 300);
        assert_eq!(kzx::sniff(&m), Profile::Ptx);
        let packed = roundtrip(&m);
        let info = kzx::inspect(&packed).unwrap();
        assert_eq!(info.blocks[0].profile, Profile::Ptx);
        assert!(
            packed.len() * 8 < m.len(),
            "{} -> {} should compress 8x+",
            m.len(),
            packed.len()
        );
    }
}

#[test]
fn synthetic_elf_objects_round_trip_and_sniff_as_elf() {
    let mut rng = synth::Rng::new(12);
    let o = synth::elf_object(&mut rng, 2000, 8);
    assert_eq!(kzx::sniff(&o), Profile::Elf);
    let packed = roundtrip(&o);
    assert!(packed.len() * 3 < o.len());
}

#[test]
fn synthetic_postcard_bodies_round_trip_under_the_generic_model() {
    let mut rng = synth::Rng::new(13);
    let b = synth::postcard_body(&mut rng, 400, 200, 100);
    assert_eq!(kzx::sniff(&b), Profile::Generic);
    let packed = roundtrip(&b);
    assert!(packed.len() * 2 < b.len());
}

#[test]
fn every_profile_round_trips_every_class() {
    let mut rng = synth::Rng::new(14);
    let inputs = [
        synth::ptx_module(&mut rng, 0, 1, 120),
        synth::elf_object(&mut rng, 300, 3),
        synth::postcard_body(&mut rng, 100, 50, 20),
        rng.bytes(5000),
    ];
    for raw in &inputs {
        for profile in [
            Profile::Stored,
            Profile::Generic,
            Profile::Ptx,
            Profile::Elf,
        ] {
            for mem in [kzx::MIN_MEM_LOG2, 14] {
                let packed = kzx::compress_with(raw, profile, mem).unwrap();
                let back = kzx::decompress(&packed, &LIMITS).unwrap();
                assert!(
                    back == *raw,
                    "{profile:?} mem {mem} failed on {} bytes",
                    raw.len()
                );
            }
        }
    }
}

#[test]
fn ptx_transform_edge_cases_are_exact() {
    let cases: [&[u8]; 12] = [
        b"",
        b"%",
        b"%r",
        b"%r0 %r00 %r007 %r1 %r18446744073709551615 %r18446744073709551616 %r9999999999999999999",
        b"$L__BB0_1:\n$L__BB12_345:\n$9 $_9 x9 %%9 %r9%r10",
        b"\x01\x02\x01\x01\x02\x02%r\x01\x02 7 %f1\x01",
        b"%r1,%r2,%r3,%r4,%r5,%r6,%r7,%r8,%r9,%r10,%r11,%r12",
        b"abc%r12def %rd99999 %p0 %f00 %f0f",
        b"0123456789%r1 123%r2 %r3abc %r4_5 %r6.7",
        b"%ab_cd_ef_gh_ij_kl_mn_op_qr_st_uv_wx_yz_ab_cd_ef_gh_ij_kl_mn_op_qr_st_uv_wx_yz_ab_cd_ef_gh_ij_kl_mn_op_qr_st_uv_wx_yz_ab_cd_ef_gh_ij_kl_mn_op_qr_st_uv_wx_yz_ab_cd_ef_gh_ij_kl_mn_op_qr_st_uv_wx_yz99",
        b"\n.version 9.0\n.target sm_90a\n%r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1 %r1",
        b"%r4294967296 %r4294967295 %r18446744073709551615 %r1 %r18446744073709551615 %r0",
    ];
    for raw in cases {
        for profile in [Profile::Ptx, Profile::Generic] {
            let packed = kzx::compress_with(raw, profile, kzx::MIN_MEM_LOG2).unwrap();
            let back = kzx::decompress(&packed, &LIMITS).unwrap();
            assert!(
                back == raw,
                "{profile:?} failed on {:?}",
                String::from_utf8_lossy(raw)
            );
        }
    }
}

#[test]
fn multi_block_streams_round_trip_block_by_block() {
    let mut rng = synth::Rng::new(15);
    let body = synth::postcard_body(&mut rng, 200, 100, 40);
    let ptx = synth::ptx_module(&mut rng, 1, 1, 200);
    let tiny = b"hi";
    let empty: &[u8] = b"";
    let packed = kzx::compress_blocks(&[&body, &ptx, tiny, empty]);
    let info = kzx::inspect(&packed).unwrap();
    assert_eq!(info.blocks.len(), 4);
    assert_eq!(info.blocks[2].profile, Profile::Stored);
    assert_eq!(info.blocks[3].raw_len, 0);
    let parts = kzx::decompress_blocks(&packed, &LIMITS).unwrap();
    assert_eq!(
        parts,
        vec![body.clone(), ptx.clone(), tiny.to_vec(), Vec::new()]
    );
    let joined = kzx::decompress(&packed, &LIMITS).unwrap();
    assert_eq!(joined, [body, ptx, tiny.to_vec()].concat());
}

#[test]
fn compression_is_deterministic_across_calls() {
    let mut rng = synth::Rng::new(16);
    let m = synth::ptx_module(&mut rng, 3, 2, 400);
    let a = kzx::compress(&m);
    let b = kzx::compress(&m);
    assert_eq!(a, b);
    let c = kzx::compress_blocks(&[&m, &m]);
    let d = kzx::compress_blocks(&[&m, &m]);
    assert_eq!(c, d);
}

#[test]
fn boundary_lengths_round_trip() {
    let mut rng = synth::Rng::new(17);
    for n in [0usize, 1, 2, 23, 24, 25, 255, 256, 4095, 4096, 4097] {
        let raw = rng.bytes(n);
        roundtrip(&raw);
        let text: Vec<u8> = raw.iter().map(|b| b'a' + (b % 26)).collect();
        roundtrip(&text);
    }
}
