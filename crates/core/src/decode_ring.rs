// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: Constants of the per-sequence SSM decode-rollback ring that more than one
//! crate reads (model-layers sizes the ring, model-engine names its cold tier, the server
//! validates `--ssm-decode-ring-slots`). Moved from metrale-kernels, which only the binary
//! links, so those crates no longer depend on the kernel table.
//!
//! Owner: metrale-core.
//! Invariants: none beyond the values.

/// 2026-09-25: Deepest per-sequence decode-rollback ring (boundary SSM-state
/// snapshots): the default depth, the top of the auto-fit ladder, and the
/// largest value `--ssm-decode-ring-slots` accepts. A model without SSM
/// layers gets no ring.
pub const DECODE_ROLLBACK_RING_SLOTS: usize = 8;

/// 2026-09-25: Domain separator mixed with the model fingerprint to form the
/// SSM decode-ring cold-tier namespace (`derive_decode_ns_salted` in
/// metrale-model-engine), which keeps decode keys apart from other key
/// spaces derived from the same fingerprint.
pub const DECODE_DOMAIN: u64 = 0xD3C0_DE12_A5B6_C7D8;
