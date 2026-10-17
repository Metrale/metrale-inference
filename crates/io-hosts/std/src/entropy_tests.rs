// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: `OsEntropy` fills the whole buffer.

use metrale_abstract_io::Entropy;

use super::OsEntropy;

#[test]
fn fills_32_bytes_not_all_zero() {
    let mut buf = [0u8; 32];
    OsEntropy.fill(&mut buf).expect("OS entropy available");
    // 2026-10-17: All-zero output from a working source has probability 2^-256.
    assert!(buf.iter().any(|&b| b != 0));
}
