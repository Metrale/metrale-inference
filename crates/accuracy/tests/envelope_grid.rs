// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The envelope grid over the checked-out instances: every candidate is a contracted
//! entry point whose family implements the cell's formats, served cells carry today's routed
//! default, margin cells carry none, shards partition the grid, and cells the contracts do not
//! cover are reported.

mod common;

use std::collections::BTreeSet;

use common::{Tree, families};
use metrale_accuracy::contract::parse_contracts;
use metrale_accuracy::envelope::grid::{LADDER, grid, shard, uncovered};
use metrale_circuit::venn::Repo;

fn inputs() -> (
    metrale_accuracy::points::Sweep,
    metrale_accuracy::contract::Contracts,
    metrale_circuit::venn::Families,
) {
    let contracts =
        parse_contracts(&Tree.read("kernels/gb10/common/ACCURACY.toml").unwrap()).unwrap();
    let sweep = metrale_accuracy::points::sweep(&Tree, "gb10").unwrap();
    (sweep, contracts, families())
}

#[test]
fn candidates_are_contracted_kernels_of_families_that_run_the_formats() {
    let (sweep, contracts, fams) = inputs();
    let g = grid(&sweep, &contracts, &fams, &LADDER, false);
    assert!(!g.is_empty());
    for c in &g {
        assert!(!c.candidates.is_empty());
        for cand in &c.candidates {
            let k = &contracts.contracts[cand.contract];
            assert!(k.kernels.contains(&cand.kernel), "{cand:?}");
            assert_eq!(k.family, cand.family);
            assert_eq!(k.op, c.cell.op);
        }
        // 2026-10-10: A BF16 cell never offers an NVFP4 kernel, and the reverse.
        let nvfp4 = c.cell.weight.starts_with("nvfp4");
        for cand in &c.candidates {
            let is_w4 = cand.family.starts_with("w4a16") || cand.family == "tc_rows";
            assert!(!(is_w4 && !nvfp4), "{:?} offers {}", c.cell, cand.kernel);
        }
    }
    // 2026-10-10: Served cells carry today's routed entry. Where it is a candidate (a standalone
    // projection kernel with a contract) the sweep times it beside the others; where it is not
    // (a fused group such as the dual GEMV or a grouped MoE kernel, or a prefill-class GEMM with
    // no contract) the cell still sweeps its candidates and the default stays named.
    let served: Vec<_> = g.iter().filter(|c| c.default.is_some()).collect();
    let timed = served
        .iter()
        .filter(|c| {
            let d = c.default.as_ref().unwrap();
            c.candidates.iter().any(|x| &x.kernel == d)
        })
        .count();
    assert!(timed > 0, "no served cell times its default");
    assert!(
        g.iter()
            .any(|c| c.default.as_deref() == Some("dense_gemv_bf16_tc::dense_gemv_bf16_tc8")),
        "the BF16 tensor-core tier is no served default"
    );
}

#[test]
fn margin_cells_have_no_default_and_shards_partition_the_grid() {
    let (sweep, contracts, fams) = inputs();
    let g = grid(&sweep, &contracts, &fams, &LADDER, true);
    let margin: Vec<_> = g.iter().filter(|c| c.margin).collect();
    assert!(!margin.is_empty());
    assert!(
        margin
            .iter()
            .all(|c| c.default.is_none() && c.users.is_empty())
    );
    assert!(
        margin
            .iter()
            .all(|c| c.cell.k % 128 == 0 && c.cell.n % 64 == 0)
    );
    // 2026-10-10: Served cells come before margin cells.
    let first_margin = g.iter().position(|c| c.margin).unwrap();
    assert!(g[first_margin..].iter().all(|c| c.margin));
    let mut seen = BTreeSet::new();
    for i in 0..3 {
        for c in shard(&g, i, 3) {
            assert!(seen.insert(c.cell.clone()), "{:?} in two shards", c.cell);
        }
    }
    assert_eq!(seen.len(), g.len());
}

#[test]
fn shapes_no_contract_covers_are_reported() {
    let (sweep, contracts, fams) = inputs();
    let missing = uncovered(&sweep, &contracts, &fams);
    let g = grid(&sweep, &contracts, &fams, &LADDER, false);
    for (op, w, a, k, n) in &missing {
        assert!(
            !g.iter().any(|c| &c.cell.op == op
                && &c.cell.weight == w
                && &c.cell.activation == a
                && c.cell.k == *k
                && c.cell.n == *n),
            "{op} {w} {a} {k}x{n} is both uncovered and gridded"
        );
    }
}
