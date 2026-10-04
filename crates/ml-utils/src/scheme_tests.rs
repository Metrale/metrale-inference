// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: Every storage scheme of the three toy layouts is recognised with its companions,
//! and malformed companions are refused.
//!
//! Owner: metrale-ml-utils.
//! Invariants: none beyond the types.

use super::*;
use crate::testkit::{self, Store, linear};

fn one(store: Store, block: Option<(u64, u64)>) -> Result<Vec<QuantGroup>> {
    let mut v = Vec::new();
    linear(&mut v, "m", 256, 512, store);
    find_groups(&TensorIndex::from_entries(v).unwrap(), block)
}

#[test]
fn each_scheme_is_recognised() {
    let cases = [
        (Store::Fp8Block, Scheme::Fp8Block { bn: 128, bk: 128 }, 2),
        (Store::Fp8Tensor, Scheme::Fp8Tensor, 3),
        (Store::Fp8Channel, Scheme::Fp8Channel, 2),
        (
            Store::Nvfp4ModelOpt,
            Scheme::Nvfp4(Nvfp4Global::ModelOpt),
            4,
        ),
        (
            Store::Nvfp4Ct,
            Scheme::Nvfp4(Nvfp4Global::CompressedTensors),
            4,
        ),
    ];
    for (store, scheme, n) in cases {
        let g = one(store, Some((128, 128))).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].scheme, scheme);
        assert_eq!((g[0].rows, g[0].cols), (256, 512));
        assert_eq!(g[0].tensors().len(), n);
    }
    assert!(one(Store::Bf16, None).unwrap().is_empty());
}

#[test]
fn the_toy_checkpoints_group_every_scale() {
    for index in [
        testkit::moe_fp8().1,
        testkit::dense_ct().1,
        testkit::moe_nvfp4().2,
    ] {
        let groups = find_groups(&index, Some((128, 128))).unwrap();
        let covered: usize = groups.iter().map(|g| g.tensors().len()).sum();
        let scales = index.iter().filter(|e| is_scale_name(&e.name)).count();
        assert_eq!(covered, groups.len() + scales);
    }
}

#[test]
fn malformed_companions_are_refused() {
    assert!(
        one(Store::Fp8Block, None)
            .unwrap_err()
            .to_string()
            .contains("weight_block_size")
    );
    assert!(
        one(Store::Fp8Block, Some((64, 64)))
            .unwrap_err()
            .to_string()
            .contains("block grid")
    );
    let mut v = Vec::new();
    linear(&mut v, "m", 256, 512, Store::Nvfp4Ct);
    v.retain(|e| !e.name.ends_with("weight_global_scale"));
    let err = find_groups(&TensorIndex::from_entries(v).unwrap(), None).unwrap_err();
    assert!(err.to_string().contains("exactly one scalar"), "{err}");
    let orphan = vec![TensorEntry {
        name: "x.weight_scale".into(),
        dtype: Dtype::F32,
        shape: vec![],
        shard: "s".into(),
    }];
    let err = find_groups(&TensorIndex::from_entries(orphan).unwrap(), None).unwrap_err();
    assert!(err.to_string().contains("belongs to no"), "{err}");
}
