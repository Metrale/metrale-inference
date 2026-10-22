// SPDX-License-Identifier: MIT OR Apache-2.0

//! Golden bytes: the compressed form of fixed synthetic fixtures is pinned, so any change to a
//! model, a table or the framing is a visible format change. The pins are the length, the
//! first 16 bytes and an FNV-1a hash of the whole stream (so the file carries no corpus).

#[path = "../examples/bench/synth.rs"]
mod synth;

use kzx::Profile;

fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

struct Pin {
    name: &'static str,
    len: usize,
    head: [u8; 16],
    fnv: u64,
}

fn check(pin: &Pin, packed: &[u8]) {
    let mut head = [0u8; 16];
    head.copy_from_slice(&packed[..16]);
    let fnv = fnv1a(packed);
    assert!(
        packed.len() == pin.len && head == pin.head && fnv == pin.fnv,
        "golden `{}` changed: len {} head {:02x?} fnv {:#018x}",
        pin.name,
        packed.len(),
        head,
        fnv
    );
}

#[test]
fn ptx_fixture_bytes_are_pinned() {
    let mut rng = synth::Rng::new(0x9001);
    let raw = synth::ptx_module(&mut rng, 7, 2, 500);
    let packed = kzx::compress(&raw);
    check(
        &Pin {
            name: "ptx",
            len: 948,
            head: [
                0x6b, 0x7a, 0x78, 0x01, 0x01, 0x00, 0x00, 0x00, 0x02, 0x0b, 0x00, 0x00, 0x04, 0xa0,
                0x00, 0x00,
            ],
            fnv: 0xce1f_6cfe_d0e8_2e46,
        },
        &packed,
    );
}

#[test]
fn elf_fixture_bytes_are_pinned() {
    let mut rng = synth::Rng::new(0x9002);
    let raw = synth::elf_object(&mut rng, 3000, 10);
    let packed = kzx::compress(&raw);
    check(
        &Pin {
            name: "elf",
            len: 1870,
            head: [
                0x6b, 0x7a, 0x78, 0x01, 0x01, 0x00, 0x00, 0x00, 0x03, 0x0b, 0x00, 0x00, 0x88, 0xbd,
                0x00, 0x00,
            ],
            fnv: 0x6287_9ccc_011d_cd9d,
        },
        &packed,
    );
}

#[test]
fn body_fixture_bytes_are_pinned() {
    let mut rng = synth::Rng::new(0x9003);
    let raw = synth::postcard_body(&mut rng, 600, 300, 150);
    let packed = kzx::compress_with(&raw, Profile::Generic, 16).unwrap();
    check(
        &Pin {
            name: "body",
            len: 13958,
            head: [
                0x6b, 0x7a, 0x78, 0x01, 0x01, 0x00, 0x00, 0x00, 0x01, 0x10, 0x00, 0x00, 0xd6, 0xb6,
                0x00, 0x00,
            ],
            fnv: 0xa395_2116_16e6_9426,
        },
        &packed,
    );
}

#[test]
fn multi_block_fixture_bytes_are_pinned() {
    let mut rng = synth::Rng::new(0x9004);
    let body = synth::postcard_body(&mut rng, 100, 60, 30);
    let ptx = synth::ptx_module(&mut rng, 1, 1, 150);
    let packed = kzx::compress_blocks(&[&body, &ptx, b"tiny", b""]);
    check(
        &Pin {
            name: "multi",
            len: 3857,
            head: [
                0x6b, 0x7a, 0x78, 0x01, 0x04, 0x00, 0x00, 0x00, 0x01, 0x0a, 0x00, 0x00, 0xb4, 0x20,
                0x00, 0x00,
            ],
            fnv: 0x082f_6260_90ae_f797,
        },
        &packed,
    );
}
