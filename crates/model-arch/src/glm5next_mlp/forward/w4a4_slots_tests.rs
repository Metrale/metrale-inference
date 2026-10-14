// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The slot GEMV's launch split on a recording mock backend. Each launch is checked
//! by replaying the kernel's own addressing (`w4a4_moe_slots` in `w4a4_gemv_mx_moe.cu`: slot
//! `blockIdx.y`, row `slot / act_div`, row stride `ka / 2`, `ka / 16`, 4 bytes) over every slot
//! of the launch and comparing with the one-launch addressing of the same global slot.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockGpuBackend, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::w4a4::tests::{ptr_arg, u32_arg};
use super::{CUDA_MAX_GRID_Y, W4a4SlotsCall, launch_w4a4_slots, slot_pieces};

const KERN: KernelHandle = KernelHandle(0x77);

/// 2026-10-14: Distinct, far-apart bases so a wrong offset cannot land on another buffer.
fn call(n: usize, k: usize, slots: usize, act_div: usize) -> W4a4SlotsCall {
    W4a4SlotsCall {
        aq: DevicePtr(1 << 40),
        as_: DevicePtr(2 << 40),
        ag: DevicePtr(3 << 40),
        ids: DevicePtr(4 << 40),
        packed_ptrs: DevicePtr(5 << 40),
        scale_ptrs: DevicePtr(6 << 40),
        scale2_vals: DevicePtr(7 << 40),
        out: DevicePtr(8 << 40),
        n,
        k,
        slots,
        act_div,
        num_experts: 288,
    }
}

fn run(c: &W4a4SlotsCall) -> Vec<MockLaunch> {
    let gpu = MockGpuBackend::new();
    launch_w4a4_slots(&gpu, KERN, c, 0).expect("valid call launches");
    gpu.launches_snapshot()
}

/// 2026-10-14: What the kernel reads and writes for block row `y` of launch `l`, as addresses:
/// (aq, as, ag, ids entry, out row).
fn kernel_addrs(l: &MockLaunch, y: usize, ka: usize) -> [u64; 5] {
    let (n, act_div) = (u32_arg(l, 8) as usize, u32_arg(l, 10) as usize);
    let row = (y / act_div) as u64;
    [
        ptr_arg(l, 0).0 + row * (ka / 2) as u64,
        ptr_arg(l, 1).0 + row * (ka / 16) as u64,
        ptr_arg(l, 2).0 + row * 4,
        ptr_arg(l, 3).0 + y as u64 * 4,
        ptr_arg(l, 7).0 + (y * n * 2) as u64,
    ]
}

/// 2026-10-14: Replays every launch over its grid Y and asserts it touches, in order, exactly
/// the addresses one unbounded launch would for slots 0..slots; returns the launch count.
fn assert_covers_like_one_launch(c: &W4a4SlotsCall) -> usize {
    let l = run(c);
    let one = MockLaunch {
        grid: [0, c.slots as u32, 1],
        ..l[0].clone()
    };
    let mut s = 0usize;
    for li in &l {
        assert!(li.grid[1] as usize <= CUDA_MAX_GRID_Y, "grid {:?}", li.grid);
        assert_eq!(li.grid[0], c.n.div_ceil(16) as u32);
        assert_eq!(li.block, [256, 1, 1]);
        for a in [4, 5, 6] {
            assert_eq!(ptr_arg(li, a), ptr_arg(&l[0], a), "weight table arg {a}");
        }
        let scalars: Vec<u32> = (8..12).map(|a| u32_arg(li, a)).collect();
        let want = [c.n, c.k, c.act_div, c.num_experts].map(|v| v as u32);
        assert_eq!(scalars, want);
        for y in 0..li.grid[1] as usize {
            assert_eq!(
                kernel_addrs(li, y, c.k.next_multiple_of(128)),
                kernel_addrs(&one, s, c.k.next_multiple_of(128)),
                "slot {s} (block row {y})"
            );
            s += 1;
        }
    }
    assert_eq!(s, c.slots, "slots launched");
    l.len()
}

/// 2026-10-14: Up to 65535 slots the call is the single launch moe_slots always issued: grid
/// (ceil(N/16), slots, 1) on the unmoved base pointers.
#[test]
fn up_to_the_grid_limit_it_is_one_launch_on_the_base_pointers() {
    for (n, k, slots, div) in [
        (704, 4096, 8, 8),
        (4096, 704, 16 * 8, 1),
        (704, 4096, 8191 * 8, 8),
        (4096, 4096, CUDA_MAX_GRID_Y, 1),
    ] {
        let c = call(n, k, slots, div);
        let l = run(&c);
        assert_eq!(l.len(), 1, "slots {slots}");
        assert_eq!(l[0].grid, [n.div_ceil(16) as u32, slots as u32, 1]);
        let ptrs: Vec<DevicePtr> = (0..8).map(|a| ptr_arg(&l[0], a)).collect();
        let want = [
            c.aq,
            c.as_,
            c.ag,
            c.ids,
            c.packed_ptrs,
            c.scale_ptrs,
            c.scale2_vals,
            c.out,
        ];
        assert_eq!(ptrs, want);
    }
}

/// 2026-10-14: 8208 rows x top 8 = 65664 slots, the prefill chunk that failed with grid
/// [44, 65664, 1]: gate (act_div 8, K 4096) and the k64 down (act_div 1, K 704 stored 768 wide) each
/// split in two launches that address every slot as one launch would.
#[test]
fn rows_8208_split_and_address_every_slot_like_one_launch() {
    let slots = 8208 * 8;
    assert!(slots > CUDA_MAX_GRID_Y, "the case exceeds the limit");
    assert_eq!(assert_covers_like_one_launch(&call(704, 4096, slots, 8)), 2);
    assert_eq!(assert_covers_like_one_launch(&call(4096, 704, slots, 1)), 2);
}

/// 2026-10-14: Pieces tile 0..slots in order, each at most the limit and starting on a row
/// boundary, for divisors that do and do not divide the limit and slot counts on both sides of
/// each multiple of it.
#[test]
fn pieces_tile_the_slots_on_row_boundaries() {
    for div in [1, 2, 7, 8, 9, 16, 4096, CUDA_MAX_GRID_Y] {
        for slots in [
            1,
            div,
            CUDA_MAX_GRID_Y,
            CUDA_MAX_GRID_Y + 1,
            3 * CUDA_MAX_GRID_Y + 5,
        ] {
            let p = slot_pieces(slots, div).unwrap();
            assert_eq!(
                p.len() == 1,
                slots <= CUDA_MAX_GRID_Y,
                "slots {slots} div {div}"
            );
            assert_eq!(p.first().unwrap().start, 0);
            assert_eq!(p.last().unwrap().end, slots);
            for w in p.windows(2) {
                assert_eq!(w[0].end, w[1].start);
            }
            for r in &p {
                assert!(!r.is_empty() && r.len() <= CUDA_MAX_GRID_Y);
                assert_eq!(r.start % div, 0, "slots {slots} div {div} piece {r:?}");
            }
        }
    }
}

/// 2026-10-14: A ragged k64 down (act_div 1) and a divisor that does not divide the limit, past
/// two splits, replay to one-launch addressing.
#[test]
fn ragged_and_odd_divisor_splits_address_like_one_launch() {
    assert_eq!(
        assert_covers_like_one_launch(&call(64, 704, 2 * CUDA_MAX_GRID_Y + 3, 1)),
        3
    );
    assert_eq!(
        assert_covers_like_one_launch(&call(64, 4096, 7 * 20_000, 7)),
        3
    );
}

/// 2026-10-14: No slots, a zero or over-limit divisor, and a K that is not a positive multiple
/// of 64 are refused before any launch.
#[test]
fn bad_calls_are_refused_before_launching() {
    for c in [
        call(704, 4096, 0, 8),
        call(704, 4096, 8, 0),
        call(704, 4096, 8, CUDA_MAX_GRID_Y + 1),
        call(4096, 0, 8, 1),
        call(4096, 700, 8, 1),
    ] {
        let gpu = MockGpuBackend::new();
        assert!(launch_w4a4_slots(&gpu, KERN, &c, 0).is_err(), "{c:?}");
        assert!(gpu.launches_snapshot().is_empty(), "{c:?}");
    }
}
