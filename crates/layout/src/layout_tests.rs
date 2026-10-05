// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-05: Each operation equals the function it stands for, refuses what has no layout, and
//! the swizzle and bank checks reproduce a shared-memory scheme gb10 kernels use today.
//!
//! Owner: metrale-layout tests.
//! Invariants: every test fails when the behaviour it names is removed (the mutation notes say
//! which change each one catches).

use crate::{Layout, Swizzle, bank_conflicts, coalesce, complement, compose, divide, product};

fn l(pairs: &[(u64, i64)]) -> Layout {
    Layout::new(pairs).unwrap()
}

// 2026-10-05: Mutation: making the last mode fastest, or skipping a digit, moves an offset.
#[test]
fn compact_layouts_and_coordinates() {
    let rm = Layout::compact(&[4, 8], true).unwrap();
    assert_eq!(rm, l(&[(4, 8), (8, 1)]));
    assert_eq!(rm.eval_coord(&[1, 2]), 10);
    assert_eq!(rm.cosize(), 32);
    assert!(rm.is_bijective());
    assert!(
        !l(&[(4, 1), (2, 2)]).is_injective(),
        "offsets 2 and 3 repeat"
    );
}

// 2026-10-05: Mutation: merging digits that do not continue each other, or dropping a size > 1
// digit, changes the function.
#[test]
fn coalesce_keeps_the_function() {
    for x in [l(&[(2, 1), (4, 2), (1, 9), (3, 8)]), l(&[(2, 4), (4, 1)])] {
        let c = coalesce(&x);
        assert!(c.same_function(&x), "{x:?} -> {c:?}");
    }
    assert_eq!(
        coalesce(&l(&[(2, 1), (4, 2)])).modes,
        [vec![crate::Digit { size: 8, stride: 1 }]]
    );
}

// 2026-10-05: Mutation: a wrong skip or take in `compose_digit` fails the evaluation check (it
// returns an error rather than a wrong layout); dropping the check would return one.
#[test]
fn compose_is_function_composition() {
    let a = Layout::compact(&[8, 8], true).unwrap();
    let b = l(&[(2, 1), (4, 16)]);
    let c = compose(&a, &b).unwrap();
    for i in 0..b.size() {
        assert_eq!(c.eval(i), a.eval(b.eval(i) as u64));
    }
    let id = Layout::compact(&[64], true).unwrap();
    assert!(compose(&a, &id).unwrap().same_function(&a));
    assert!(
        compose(&l(&[(3, 1), (5, 3)]), &l(&[(2, 2)])).is_err(),
        "2 steps across a 3"
    );
    assert!(compose(&a, &l(&[(2, 64)])).is_err(), "past the layout");
}

// 2026-10-05: Mutation: an off-by-one in the gap sizes leaves the pair short of a tiling.
#[test]
fn complement_fills_the_gaps() {
    let c = complement(&l(&[(4, 1)]), 16).unwrap();
    assert_eq!(c, l(&[(4, 4)]));
    let x = l(&[(2, 4)]);
    let c = complement(&x, 16).unwrap();
    assert!(
        Layout::from_modes(vec![x.modes[0].clone(), c.modes[0].clone()])
            .unwrap()
            .is_bijective()
    );
    assert!(
        complement(&l(&[(4, 3)]), 16).is_err(),
        "16 is not a multiple of the extent 12"
    );
    assert!(complement(&l(&[(4, -1)]), 16).is_err());
}

// 2026-10-05: Mutation: swapping the tile and tile-index modes, or taking the complement over
// the wrong size, breaks the tiling.
#[test]
fn divide_splits_into_tiles_and_product_repeats() {
    let m = Layout::compact(&[4, 8], false).unwrap(); // column-major 4 x 8
    let tile = l(&[(4, 1), (2, 4)]); // a 4 x 2 column block
    let d = divide(&m, &tile).unwrap();
    assert_eq!(d.mode_size(0), 8);
    assert_eq!(d.mode_size(1), 4);
    let first: Vec<i64> = (0..8).map(|i| d.eval(i)).collect();
    assert_eq!(first, [0, 1, 2, 3, 4, 5, 6, 7], "the first block");
    assert_eq!(d.eval(8), 8, "the second block starts after it");
    let p = product(&l(&[(2, 1), (2, 2)]), &l(&[(3, 1)])).unwrap();
    assert_eq!(p.offsets(), (0..12).collect::<Vec<_>>());
}

// 2026-10-05: gb10's e4m3 pipeline swizzles 16-byte chunks of 64-byte rows by
// `ch ^ ((row >> 1) & 3)` (kernels/gb10/common/e4m3_mma_pipe.cuh). Mutation: any other field
// placement disagrees with it somewhere.
#[test]
fn the_swizzle_reproduces_a_kernel_scheme_and_removes_its_conflicts() {
    let s = Swizzle::new(2, 4, 3).unwrap();
    for row in 0..64u64 {
        for ch in 0..4u64 {
            let o = row * 64 + ch * 16;
            assert_eq!(s.apply(o), row * 64 + (ch ^ ((row >> 1) & 3)) * 16);
            assert_eq!(s.apply(s.apply(o)), o, "an involution");
        }
    }
    let rows: Vec<u64> = (0..8).map(|r| r * 64).collect();
    assert_eq!(
        bank_conflicts(&rows, 16).ways,
        4,
        "unswizzled 64-byte rows: 4-way"
    );
    let swz: Vec<u64> = rows.iter().map(|&a| s.apply(a)).collect();
    assert_eq!(bank_conflicts(&swz, 16).ways, 1, "swizzled: conflict-free");
    assert!(crate::vectors_aligned(&swz, 16));
    assert!(Swizzle::new(3, 4, 2).is_err(), "overlapping fields");
}

// 2026-10-05: Composition is associative wherever both sides exist: checked over every small
// power-of-two layout pair and triple. Mutation: any closed form that is not the composite
// function breaks an equality here.
#[test]
fn composition_is_associative_on_small_layouts() {
    let sizes = [1u64, 2, 4];
    let strides = [0i64, 1, 2, 4, 8];
    let mut ones = Vec::new();
    for &s in &sizes {
        for &d in &strides {
            ones.push(l(&[(s, d)]));
        }
    }
    let mut twos = Vec::new();
    for x in &ones {
        for y in &ones {
            twos.push(Layout::from_modes(vec![x.modes[0].clone(), y.modes[0].clone()]).unwrap());
        }
    }
    let a = Layout::compact(&[4, 4, 4], false).unwrap();
    let mut checked = 0;
    for b in &twos {
        let Ok(ab) = compose(&a, b) else { continue };
        for c in &ones {
            let (Ok(bc), Ok(ab_c)) = (compose(b, c), compose(&ab, c)) else {
                continue;
            };
            let Ok(a_bc) = compose(&a, &bc) else { continue };
            assert!(ab_c.same_function(&a_bc), "{b:?} {c:?}");
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} triples composed");
}
