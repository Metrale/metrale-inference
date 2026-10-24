// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-24: Tests for embedding input validation and packing.
//!
//! Owner: model-engine (embedding model).
//! Invariants: none beyond the types.

use super::*;

fn code(e: &anyhow::Error) -> ErrorCode {
    e.downcast_ref::<metrale_error::Error>().unwrap().code()
}

/// 2026-10-24: Every input lands in exactly one pack, in order, and no pack exceeds the budget,
/// over many length mixes (including inputs exactly at the budget).
#[test]
fn packs_cover_in_order_within_budget() {
    let budget = 10;
    let mut seed = 7u64;
    for _ in 0..500 {
        let n = (seed % 40) as usize;
        let lens: Vec<usize> = (0..n)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                1 + (seed >> 33) as usize % budget
            })
            .collect();
        let packs = plan_packs(&lens, budget).unwrap();
        let mut next = 0;
        for p in &packs {
            assert_eq!(p.start, next, "packs must be contiguous");
            assert!(p.end > p.start, "no empty pack");
            let rows: usize = lens[p.clone()].iter().sum();
            assert!(rows <= budget, "{rows} > {budget}");
            next = p.end;
        }
        assert_eq!(next, lens.len());
        // 2026-10-24: Greedy: the next input never fits the pack that closed before it.
        for w in packs.windows(2) {
            let rows: usize = lens[w[0].clone()].iter().sum();
            assert!(rows + lens[w[1].start] > budget);
        }
    }
}

#[test]
fn an_input_over_the_budget_is_refused() {
    let e = plan_packs(&[3, 11, 2], 10).unwrap_err();
    assert_eq!(code(&e), ErrorCode::EmbedInputInvalid);
    assert!(e.to_string().contains("embedding input 1"), "{e}");
    assert!(plan_packs(&[], 10).unwrap().is_empty());
}

#[test]
fn validation_refuses_empty_long_and_out_of_vocab() {
    assert!(validate_inputs(&[vec![1, 2], vec![9]], 4, 10).is_ok());
    for (inputs, idx) in [
        (vec![vec![1], vec![]], 1),
        (vec![vec![1, 2, 3, 4, 5]], 0),
        (vec![vec![1], vec![2], vec![3, 10]], 2),
    ] {
        let e = validate_inputs(&inputs, 4, 10).unwrap_err();
        assert_eq!(code(&e), ErrorCode::EmbedInputInvalid);
        assert!(
            e.to_string().contains(&format!("embedding input {idx}")),
            "{e}"
        );
    }
}
