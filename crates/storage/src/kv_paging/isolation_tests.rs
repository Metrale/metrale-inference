// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KV paging isolation tests without hardware. The peer is a `Residency`,
//! the type the real peer's `SharedPaging` holds, over an in-memory arena and swap,
//! keyed by the wire key alone; clients fold keys as `KvPagingBackend::block_key` does.
//!
//! Owner: storage, KV paging tier.
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::num::NonZeroU64;

use super::ns::{FNV_OFFSET, KV_DOMAIN, derive_kv_ns, fnv1a_64, wire_key};
use crate::group::{GroupKey, GroupLayout, KvKind};
use crate::snapshot_swap::{MemSwapStore, Residency, VecSlotArena};

/// 2026-09-25: Blob size of the mock peer; the keys under test do not depend on it.
const BB: usize = 64;

fn layout() -> GroupLayout {
    GroupLayout::new(4, 16, 2, 16, 128, 2, 4096)
}

type MockPeer = Residency<VecSlotArena, MemSwapStore>;

/// 2026-09-25: A peer with `slots` arena slots over an in-memory swap with no disk
/// cap (`Residency::new`), so nothing is ever dropped.
fn peer(slots: usize) -> MockPeer {
    Residency::new(VecSlotArena::new(BB, slots), MemSwapStore::new(BB)).unwrap()
}

/// 2026-09-25: One simulated KV paging client; `key` folds keys as
/// `KvPagingBackend::block_key` does.
struct Client {
    ns: NonZeroU64,
}

impl Client {
    fn new(fp: u64, salt: u64) -> Self {
        Self::with_codec(fp, salt, CODEC_A)
    }
    fn with_codec(fp: u64, salt: u64, codec_salt: u64) -> Self {
        Self {
            ns: derive_kv_ns(fp, &layout(), 2, 16, 128, salt, codec_salt),
        }
    }
    fn key(&self, layer: u32, block: u32) -> u64 {
        let base = layout()
            .group_id(GroupKey::new(layer, block, 0, KvKind::K))
            .0;
        wire_key(self.ns, base)
    }
    fn put(&self, peer: &mut MockPeer, layer: u32, block: u32, tag: u8) {
        peer.put_blob(self.key(layer, block), &[tag; BB]).unwrap();
    }
    fn get(&self, peer: &mut MockPeer, layer: u32, block: u32) -> Option<Vec<u8>> {
        let mut out = vec![0u8; BB];
        peer.get_blob(self.key(layer, block), &mut out)
            .unwrap()
            .then_some(out)
    }
}

// 2026-09-25: The hybrid and dense fingerprints metrale-model-engine's
// `fingerprint_tests.rs` pins.
const FP_A: u64 = 0x5629_922c_51a1_6a10;
const FP_B: u64 = 0x971e_b3b4_bd13_22f1;
const SALT: u64 = 0x0BAD_5EED;
// 2026-10-18: Two `KvCacheConfig::spill_salt` values; the namespace treats them as opaque.
const CODEC_A: u64 = 0xC0DE_0000_0000_000A;
const CODEC_B: u64 = 0xC0DE_0000_0000_000B;

#[test]
fn two_models_do_not_cross_serve() {
    let mut peer = peer(8);
    let a = Client::new(FP_A, SALT);
    let b = Client::new(FP_B, SALT);
    a.put(&mut peer, 1, 3, 0xAA);
    assert_eq!(
        b.get(&mut peer, 1, 3),
        None,
        "model B must MISS model A's KV block for the same (layer, block)"
    );
    assert_eq!(a.get(&mut peer, 1, 3), Some(vec![0xAA; BB]));
}

// 2026-09-25: `GroupKey.block` is a client-local index, so two clients of one model
// use the same block ids; only the salt keeps one from reading the other's block. A
// restarted client with no pinned salt draws a new one and misses its earlier blocks.
#[test]
fn same_model_two_salts_do_not_cross_serve() {
    let mut peer = peer(8);
    let c1 = Client::new(FP_A, 0x1111);
    let c2 = Client::new(FP_A, 0x2222);
    assert_ne!(c1.ns, c2.ns, "the salt must separate same-model clients");
    c1.put(&mut peer, 0, 0, 0xC1);
    assert_eq!(
        c2.get(&mut peer, 0, 0),
        None,
        "same model + same block id + different client ⇒ MISS, never the other \
         client's bytes"
    );
    c2.put(&mut peer, 0, 0, 0xC2);
    assert_eq!(c1.get(&mut peer, 0, 0), Some(vec![0xC1; BB]));
    assert_eq!(c2.get(&mut peer, 0, 0), Some(vec![0xC2; BB]));
}

// 2026-09-25: Same model and salt give the same keys, so a client that pins
// `METRALE_KV_PAGING_SALT` finds its blocks again after reconnecting.
#[test]
fn same_model_same_salt_round_trips_across_instances() {
    let mut peer = peer(8);
    let c1 = Client::new(FP_A, SALT);
    let c2 = Client::new(FP_A, SALT);
    assert_eq!(c1.ns, c2.ns);
    c1.put(&mut peer, 2, 5, 0x77);
    assert_eq!(c2.get(&mut peer, 2, 5), Some(vec![0x77; BB]));
}

// 2026-09-25: The peer's registry keys arenas by `(kind, blob_bytes)`
// (`cache_peer::registry`), so an SSM lookup never reaches the KV arena, even for an
// equal key and blob size. `KV_DOMAIN` in the namespace separates the keys as well.
#[test]
fn cross_kind_registry_never_mixes() {
    let mut registry: HashMap<(u8, usize), MockPeer> = HashMap::new();
    registry.insert((0, BB), peer(4));
    registry.insert((1, BB), peer(4));
    let key = 0xDEAD_BEEF_DEAD_BEEFu64;
    registry
        .get_mut(&(1, BB))
        .unwrap()
        .put_blob(key, &[0x4B; BB])
        .unwrap();
    let mut out = vec![0u8; BB];
    assert!(
        !registry
            .get_mut(&(0, BB))
            .unwrap()
            .get_blob(key, &mut out)
            .unwrap(),
        "an SSM lookup must never see a KV entry, even with equal key AND blob_bytes"
    );
    assert!(
        registry
            .get_mut(&(1, BB))
            .unwrap()
            .get_blob(key, &mut out)
            .unwrap()
    );
    assert_eq!(out, vec![0x4B; BB]);
}

// 2026-09-25: With more blocks than arena slots and no disk cap, the peer spills to
// swap and faults every block back unchanged; a KV block that is dropped cannot be
// recovered (`kv_miss_error`).
#[test]
fn kv_blocks_survive_spill_and_fault_byte_identical() {
    let mut peer = peer(4);
    let c = Client::new(FP_A, SALT);
    let pat = |layer: u32, block: u32| ((layer * 31 + block) & 0xFF) as u8;
    for layer in 0..2u32 {
        for block in 0..16u32 {
            c.put(&mut peer, layer, block, pat(layer, block));
        }
    }
    for layer in 0..2u32 {
        for block in 0..16u32 {
            assert_eq!(
                c.get(&mut peer, layer, block),
                Some(vec![pat(layer, block); BB]),
                "block (L{layer},B{block}) must fault back byte-identical, never drop"
            );
        }
    }
    assert!(
        peer.stats().spills_to_disk > 0,
        "the test must force spills"
    );
    assert!(peer.stats().faults_from_disk > 0);
    assert_eq!(
        peer.stats().disk_evictions,
        0,
        "miss-proof: nothing dropped"
    );
}

// 2026-09-25: `HighSpeedSwap::alloc_disk_block_id` reuses freed ids, so one wire key
// is written again with new bytes; a later read must return the new bytes.
#[test]
fn disk_id_reuse_overwrites_in_place() {
    let mut peer = peer(4);
    let c = Client::new(FP_A, SALT);
    c.put(&mut peer, 0, 7, 0x01);
    c.put(&mut peer, 0, 7, 0x02);
    assert_eq!(c.get(&mut peer, 0, 7), Some(vec![0x02; BB]));
}

// 2026-10-18: Two codecs can share a byte layout (nvfp4 and turbo4), so a model served under
// one must miss the blocks it spilled under the other: the codec salt separates them.
#[test]
fn two_codecs_of_one_model_do_not_cross_serve() {
    let mut peer = peer(8);
    let a = Client::with_codec(FP_A, SALT, CODEC_A);
    let b = Client::with_codec(FP_A, SALT, CODEC_B);
    assert_ne!(a.ns, b.ns);
    a.put(&mut peer, 1, 3, 0xA4);
    assert_eq!(
        b.get(&mut peer, 1, 3),
        None,
        "a block spilled under one codec must MISS under another"
    );
    assert_eq!(a.get(&mut peer, 1, 3), Some(vec![0xA4; BB]));
}

/// 2026-10-18: The version-1 namespace encoding (no codec salt) for 2-byte elements, block 16
/// and head_dim 128, kept here as the reference for blocks spilled before the bump.
fn derive_kv_ns_v1(fp: u64, l: &GroupLayout, salt: u64) -> NonZeroU64 {
    let mut buf = Vec::new();
    for (tag, v) in [
        (0x00u8, 1u64),
        (0x01, fp),
        (0x02, KV_DOMAIN),
        (0x03, 2),
        (0x04, 16),
        (0x05, 128),
        (0x06, u64::from(l.num_layers)),
        (0x07, u64::from(l.num_blocks)),
        (0x08, u64::from(l.num_kv_heads)),
        (0x09, l.fs_block_size),
        (0x0a, l.group_stride),
        (0x0b, salt),
    ] {
        buf.push(tag);
        buf.extend_from_slice(&v.to_le_bytes());
    }
    NonZeroU64::new(fnv1a_64(&buf)).unwrap_or(NonZeroU64::new(FNV_OFFSET).unwrap())
}

// 2026-10-18: The reference above is the old encoding: it reproduces the golden namespace
// version 1 pinned (80 layers, 4096 blocks, 8 KV heads, block 16, head_dim 128).
#[test]
fn the_v1_reference_reproduces_the_v1_golden() {
    let l = GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096);
    let ns = derive_kv_ns_v1(0x5629_922c_51a1_6a10, &l, 0xD00D_F00D_0000_0001);
    assert_eq!(ns.get(), 0x74b4_02b0_f421_5375);
}

// 2026-10-18: A block a version-1 client spilled is never read by a version-2 client of the
// same model, salt and layout: the bump makes it a miss, never a misread.
#[test]
fn a_version_1_spill_is_never_read() {
    let mut peer = peer(8);
    let old = Client {
        ns: derive_kv_ns_v1(FP_A, &layout(), SALT),
    };
    let new = Client::new(FP_A, SALT);
    assert_ne!(old.ns, new.ns);
    for block in 0..4 {
        old.put(&mut peer, 0, block, 0x01);
    }
    for block in 0..4 {
        assert_eq!(new.get(&mut peer, 0, block), None, "block {block}");
        assert_eq!(old.get(&mut peer, 0, block), Some(vec![0x01; BB]));
    }
}
