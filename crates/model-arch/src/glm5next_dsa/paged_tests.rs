// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Host tests of the paged indexer address math: the flat arm is the kernel's old
//! `raw * D`, the paged arm places every row of every block in its own non-overlapping slot
//! inside its block, keys and gates never meet, and the kernels carry the same formula.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;

/// 2026-10-09: GLM-5.3: index_head_dim 128, FP8 latent of 512 B per token, so the V side of
/// a 64-token block is 32,768 B.
const D: usize = 128;
const BS: usize = 64;
const STRIDE: usize = BS * 512;

fn glm() -> PagedIndexerLayout {
    PagedIndexerLayout::new(BS, STRIDE, D).expect("GLM-5.3 fits the FP8 V side exactly")
}

/// 2026-10-09: Without a table the offset is `raw * D`, the expression every flat launch
/// used before the paged path existed.
#[test]
fn the_flat_arm_is_raw_times_d() {
    for raw in [0usize, 1, 63, 64, 4095, 262_143] {
        assert_eq!(row_elem(raw, D, None, BS, 0).unwrap(), raw * D);
    }
}

/// 2026-10-09: Over a scrambled table, each (logical position) maps to a distinct D-element
/// key row; the row lies inside its physical block's key region, and the gate rows (shifted by
/// the key region) lie inside the block's stride and never overlap a key row. A formula that
/// ignored the table, swapped slot and block, or used the token stride as the block stride
/// fails one of these.
#[test]
fn paged_rows_are_disjoint_and_stay_in_their_block() {
    let l = glm();
    let table: Vec<u32> = vec![7, 2, 9, 0, 5];
    let blk = l.blk_elems();
    let gate_elems = l.gate_offset_bytes() / 2;
    let mut keys = std::collections::BTreeSet::new();
    for raw in 0..table.len() * BS {
        let e = row_elem(raw, D, Some(&table), BS, blk).unwrap();
        let block = table[raw / BS] as usize;
        let base = block * blk;
        assert!(
            e >= base && e + D <= base + gate_elems,
            "key row {raw} leaves its block"
        );
        let g = e + gate_elems;
        assert!(g + D <= base + blk, "gate row {raw} leaves its block");
        assert_eq!(e % D, 0, "rows are D-aligned");
        assert!(keys.insert(e), "row {raw} collides");
        assert_eq!(l.key_row_bytes(&table, raw).unwrap(), e * 2);
    }
    let gates: std::collections::BTreeSet<usize> = keys.iter().map(|k| k + gate_elems).collect();
    assert!(keys.is_disjoint(&gates), "a gate row overlaps a key row");
}

/// 2026-10-09: Two sequences sharing a prefix block read the same address for the shared
/// positions and different addresses past it: the property prefix caching relies on.
#[test]
fn a_shared_prefix_block_is_one_address_for_both_sequences() {
    let blk = glm().blk_elems();
    let a: Vec<u32> = vec![3, 11];
    let b: Vec<u32> = vec![3, 12];
    for raw in 0..BS {
        assert_eq!(
            row_elem(raw, D, Some(&a), BS, blk).unwrap(),
            row_elem(raw, D, Some(&b), BS, blk).unwrap()
        );
    }
    for raw in BS..2 * BS {
        assert_ne!(
            row_elem(raw, D, Some(&a), BS, blk).unwrap(),
            row_elem(raw, D, Some(&b), BS, blk).unwrap()
        );
    }
}

#[test]
fn a_row_past_the_table_is_an_error() {
    let blk = glm().blk_elems();
    let err = row_elem(BS * 2, D, Some(&[1, 2]), BS, blk).unwrap_err();
    assert!(err.to_string().contains("logical block 2"), "{err}");
}

/// 2026-10-09: The V side must hold 4 * D bytes per token: the FP8 latent (512 B) fits exactly,
/// BF16 (1024 B) fits, NVFP4 (288 B) and an odd stride are refused.
#[test]
fn the_layout_refuses_a_v_side_too_small_or_odd() {
    assert!(PagedIndexerLayout::new(BS, BS * 1024, D).is_ok());
    let nvfp4 = BS * (512 / 2 + 512 / 16);
    assert!(PagedIndexerLayout::new(BS, nvfp4, D).is_err());
    assert!(PagedIndexerLayout::new(BS, STRIDE - 1, D).is_err());
    assert!(PagedIndexerLayout::new(BS, STRIDE + 1, D).is_err());
    assert!(PagedIndexerLayout::new(0, STRIDE, D).is_err());
}

/// 2026-10-09: The device descriptor of a paged cache has no validity buffer and a gate base
/// one key region past the pool base; a flat one has no table.
#[test]
fn the_device_descriptors_match_the_layout() {
    let l = glm();
    let pool = DevicePtr(0x1000_0000);
    let p = IndexerRowsDev::paged(&l, pool, DevicePtr(0x2000)).unwrap();
    assert_eq!(p.k, pool);
    assert_eq!(p.gate, pool.offset(BS * D * 2));
    assert_eq!(p.valid, DevicePtr::NULL);
    assert_eq!(
        (p.block_size as usize, p.blk_elems as usize),
        (BS, STRIDE / 2)
    );
    assert!(p.is_paged());
    assert!(IndexerRowsDev::paged(&l, pool, DevicePtr::NULL).is_err());
    let f = IndexerRowsDev::flat(DevicePtr(1), DevicePtr(2), DevicePtr(3));
    assert!(!f.is_paged());
    assert_eq!((f.block_size, f.blk_elems), (0, 0));
}

/// 2026-10-09: [`row_elem`] is the host twin of the kernels' `dsa_row_elem`; both kernel
/// sources must carry the same two arms, and the reads they replaced must be gone.
#[test]
fn both_kernel_sources_carry_the_same_address_formula() {
    for src in [
        include_str!("../../../../kernels/gb10/common/dsa_indexer.cu"),
        include_str!("../../../../kernels/b300/common/dsa_indexer.cu"),
    ] {
        assert!(src.contains("if (bt == nullptr) return (size_t)raw * D;"));
        assert!(src.contains("return (size_t)bt[raw / bs] * blk_elems + (size_t)(raw % bs) * D;"));
        assert!(src.contains("return valid == nullptr || valid[raw] != 0;"));
        assert!(
            !src.contains("gate[(size_t)raw * D + d]"),
            "an unpaged gate read is left"
        );
        assert!(
            !src.contains("k[(size_t)raw * D + d]"),
            "an unpaged key read is left"
        );
        assert!(
            !src.contains("valid_keys[t] != 0"),
            "an unpaged validity read is left"
        );
    }
}
