// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: The checks a kernel's shared-memory index map must pass, computed from the byte
//! addresses each lane touches: bank conflicts per access phase, and vector alignment.
//!
//! Owner: metrale-layout.
//! Invariants: 32 banks of 4 bytes; a vector access of `w` bytes is served in phases of
//! `128 / w` lanes (a phase moves 128 bytes), as the hardware serves wide shared-memory
//! accesses.

/// 2026-10-05: Banks.
const BANKS: u64 = 32;
/// 2026-10-05: Bytes per bank word.
const WORD: u64 = 4;

/// 2026-10-05: The worst phase of an access.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BankReport {
    /// 2026-10-05: Lanes per phase.
    pub lanes_per_phase: usize,
    /// 2026-10-05: The most distinct words any one bank serves in one phase (1: conflict-free).
    pub ways: u64,
}

/// 2026-10-05: Bank conflicts of one warp access: `addr[lane]` is lane `lane`'s byte address,
/// each lane reading `width` bytes (4, 8 or 16).
pub fn bank_conflicts(addr: &[u64], width: u64) -> BankReport {
    let lanes_per_phase = (128 / width.max(WORD)).max(1) as usize;
    let mut ways = 1;
    for phase in addr.chunks(lanes_per_phase) {
        let mut per_bank: std::collections::BTreeMap<u64, std::collections::BTreeSet<u64>> =
            Default::default();
        for &a in phase {
            for w in 0..width.div_ceil(WORD) {
                let word = a / WORD + w;
                per_bank.entry(word % BANKS).or_default().insert(word);
            }
        }
        ways = ways.max(per_bank.values().map(|s| s.len() as u64).max().unwrap_or(1));
    }
    BankReport {
        lanes_per_phase,
        ways,
    }
}

/// 2026-10-05: Every lane's access of `width` bytes starts on a multiple of `width`.
pub fn vectors_aligned(addr: &[u64], width: u64) -> bool {
    addr.iter().all(|a| a.is_multiple_of(width))
}
