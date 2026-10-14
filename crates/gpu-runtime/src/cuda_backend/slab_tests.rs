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
    let a = t.insert(0x1000_0000, &[0, 256, 512]);
    let b = t.insert(0x2000_0000, &[0]);
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
    let a = t.insert(0x1000_0000, &[0, 256]);
    assert_eq!(t.release(a[0]), SlabRelease::StillLive);
    // 2026-10-13: A second free of the same pointer is not swallowed as a member free: the caller
    // hands it to the driver, which reports the double free as it would without slabs.
    assert_eq!(t.release(a[0]), SlabRelease::NotMember);
    assert_eq!(t.release(a[1]), SlabRelease::LastMember(0x1000_0000));
}

#[test]
fn a_pointer_never_carved_is_not_a_member() {
    let mut t = SlabTable::default();
    t.insert(0x1000_0000, &[0, 256]);
    // 2026-10-13: Inside the slab but not a member start.
    assert_eq!(t.release(0x1000_0080), SlabRelease::NotMember);
    assert_eq!(t.release(0xdead_0000), SlabRelease::NotMember);
    assert_eq!(t.live_slabs(), 1);
}
