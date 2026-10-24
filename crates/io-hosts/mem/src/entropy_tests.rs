// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `SeededEntropy` is reproducible per `(seed, stream)` and
//! diverges when either changes.

use metrale_abstract_io::Entropy;

use super::SeededEntropy;

fn draw(source: &SeededEntropy) -> [u8; 32] {
    let mut buf = [0u8; 32];
    source.fill(&mut buf).expect("SeededEntropy never errors");
    buf
}

#[test]
fn same_seed_and_stream_give_the_same_bytes() {
    let a = SeededEntropy::new(7, 0);
    let b = SeededEntropy::new(7, 0);
    let first = draw(&a);
    assert_eq!(first, draw(&b));
    let second = draw(&a);
    assert_eq!(second, draw(&b));
    assert_ne!(first, second, "successive draws advance the stream");
}

#[test]
fn different_seed_differs() {
    assert_ne!(
        draw(&SeededEntropy::new(7, 0)),
        draw(&SeededEntropy::new(8, 0))
    );
}

#[test]
fn different_stream_differs() {
    assert_ne!(
        draw(&SeededEntropy::new(7, 0)),
        draw(&SeededEntropy::new(7, 1))
    );
}

/// Catches: a `seed32` that returns zeros or a fixed array instead of drawing, or that draws
/// more or less than 32 bytes (the next draw would then differ from the reference stream).
#[test]
fn seed32_is_the_next_32_bytes_of_the_stream() {
    let (a, b) = (SeededEntropy::new(7, 0), SeededEntropy::new(7, 0));
    assert_eq!(a.seed32().expect("SeededEntropy never errors"), draw(&b));
    assert_eq!(draw(&a), draw(&b), "seed32 consumed exactly 32 bytes");
}
