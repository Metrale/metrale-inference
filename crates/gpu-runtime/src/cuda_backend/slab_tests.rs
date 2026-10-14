// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Tests of the slab layout and the member-release bookkeeping.

use super::{SLAB_ALIGN, SlabRelease, SlabTable, slab_layout};

#[test]
fn layout_aligns_every_member_and_never_overlaps() {
    let sizes = [819_200usize, 4, 102_400, 0, 1, SLAB_ALIGN, SLAB_ALIGN + 1];
    let (offsets, total) = slab_layout(&sizes);
    assert_eq!(offsets.len(), sizes.len());
    for (i, &o) in offsets.iter().enumerate() {
        assert_eq!(o % SLAB_ALIGN, 0, "member {i} misaligned");
        let end = o + sizes[i].max(1);
        let next = offsets.get(i + 1).copied().unwrap_or(total);
        assert!(end <= next, "member {i} overlaps the next");
    }
    // 2026-10-13: Zero-size members still get distinct pointers.
    assert_ne!(offsets[3], offsets[4]);
    // 2026-10-13: Waste is below one alignment unit per member.
    let used: usize = sizes.iter().map(|s| s.max(&1)).sum();
    assert!(total - used < SLAB_ALIGN * sizes.len());
}

#[test]
fn layout_of_nothing_is_empty() {
    assert_eq!(slab_layout(&[]), (vec![], 0));
}

#[test]
fn base_is_released_once_with_the_last_member_in_any_order() {
    let mut t = SlabTable::default();
    let a = t.insert(0x1000_0000, &[0, 256, 512], 768);
    let b = t.insert(0x2000_0000, &[0], 256);
    assert_eq!(t.live_slabs(), 2);
    assert_eq!(t.release(a[1]), SlabRelease::StillLive);
    assert_eq!(t.release(a[2]), SlabRelease::StillLive);
    assert_eq!(t.release(b[0]), SlabRelease::LastMember(0x2000_0000));
    assert_eq!(t.release(a[0]), SlabRelease::LastMember(0x1000_0000));
    assert_eq!(t.live_slabs(), 0);
}

#[test]
fn a_freed_member_is_no_longer_a_member() {
    let mut t = SlabTable::default();
    let a = t.insert(0x1000_0000, &[0, 256], 512);
    assert_eq!(t.release(a[0]), SlabRelease::StillLive);
    // 2026-10-13: A second free of the same pointer is not swallowed as a member free: the caller
    // hands it to the driver, which reports the double free as it would without slabs.
    assert_eq!(t.release(a[0]), SlabRelease::NotMember);
    assert_eq!(t.release(a[1]), SlabRelease::LastMember(0x1000_0000));
}

#[test]
fn a_pointer_never_carved_is_not_a_member() {
    let mut t = SlabTable::default();
    t.insert(0x1000_0000, &[0, 256], 512);
    // 2026-10-13: Inside the slab but not a member start.
    assert_eq!(t.release(0x1000_0080), SlabRelease::NotMember);
    assert_eq!(t.release(0xdead_0000), SlabRelease::NotMember);
    assert_eq!(t.live_slabs(), 1);
}

/// 2026-10-14: A slab whose small members sit beside larger ones, laid out as `alloc_slab_impl`
/// lays it out at a 2 MiB-aligned base.
fn mixed_slab(t: &mut SlabTable, base: u64) -> Vec<(u64, usize)> {
    const MIB: usize = 1 << 20;
    let sizes = [MIB, 3 * MIB, 819_200, 5 * MIB + 7, 4096, 9 * MIB];
    let (offsets, total) = slab_layout(&sizes);
    let members = t.insert(base, &offsets, total);
    members.into_iter().zip(sizes).collect()
}

#[test]
fn a_slab_reads_as_one_allocation_and_its_members_as_none() {
    use super::super::alloc_ledger::{SMALL_ALLOC_CHUNK, chunk_slack};
    let mut t = SlabTable::default();
    let base = 64 * SMALL_ALLOC_CHUNK;
    let members = mixed_slab(&mut t, base);
    let loose = (8 * SMALL_ALLOC_CHUNK, 4096usize);
    let ledger: Vec<(u64, usize)> = members.iter().copied().chain([loose]).collect();
    // 2026-10-14: Read member by member, the chunks the small members share with the larger
    // ones count as slack, though the slab fills them.
    assert!(chunk_slack(ledger.iter().copied(), SMALL_ALLOC_CHUNK) > SMALL_ALLOC_CHUNK as usize);
    let placed = t.driver_allocations(ledger.into_iter());
    let (_, total) = slab_layout(&members.iter().map(|m| m.1).collect::<Vec<_>>());
    assert_eq!(placed.len(), 2);
    assert!(placed.contains(&loose));
    assert!(placed.contains(&(base, total)));
    // 2026-10-14: Whole, only the loose small allocation leaves its chunk's rest.
    assert_eq!(
        chunk_slack(placed.into_iter(), SMALL_ALLOC_CHUNK),
        SMALL_ALLOC_CHUNK as usize - loose.1
    );
}

#[test]
fn a_slab_under_a_chunk_is_one_small_allocation() {
    use super::super::alloc_ledger::{SMALL_ALLOC_CHUNK, chunk_slack};
    let mut t = SlabTable::default();
    let base = 3 * SMALL_ALLOC_CHUNK;
    let sizes = [1000usize, 3000];
    let (offsets, total) = slab_layout(&sizes);
    let members = t.insert(base, &offsets, total);
    let placed = t.driver_allocations(members.into_iter().zip(sizes));
    assert_eq!(placed, vec![(base, total)]);
    assert_eq!(
        chunk_slack(placed.into_iter(), SMALL_ALLOC_CHUNK),
        SMALL_ALLOC_CHUNK as usize - total
    );
}

#[test]
fn a_released_slab_leaves_the_driver_view() {
    let mut t = SlabTable::default();
    let members = mixed_slab(&mut t, 0x4000_0000);
    for &(m, _) in &members {
        t.release(m);
    }
    assert!(t.driver_allocations(std::iter::empty()).is_empty());
}
