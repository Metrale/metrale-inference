// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: The sweep over the checked-out instances: every gb10 recipe contributes points,
//! the shapes are the models' own (read from their circuits, not restated here), shared kernels
//! are found across checkpoints, and the sweep is deterministic.

mod common;

use metrale_accuracy::points::{SweepError, sweep};

#[test]
fn every_gb10_instance_contributes_points_with_model_shapes() {
    let s = sweep(&common::Tree, "gb10").expect("sweep");
    assert!(s.recipes.len() >= 5, "recipes: {:?}", s.recipes);
    for r in &s.recipes {
        assert!(
            s.points.iter().any(|p| p.users.contains(r)),
            "{r} contributes no point"
        );
    }
    // 2026-10-09: The 27B's hidden width reaches a projection as K; the 35B's 2048 too.
    let k_of = |k: u64| {
        s.points
            .iter()
            .any(|p| p.shape.op.starts_with("linear") && p.shape.in_dim == k)
    };
    assert!(k_of(5120), "no linear with K=5120 (Qwen3.8-27B hidden)");
    assert!(k_of(2048), "no linear with K=2048 (Qwen3.6-35B-A3B hidden)");
    assert!(
        s.points.iter().any(|p| p.shared()),
        "no point is shared by two checkpoints"
    );
    assert!(
        s.points
            .iter()
            .all(|p| !p.users.is_empty() && !p.sites.is_empty())
    );
}

#[test]
fn the_sweep_is_deterministic_and_keys_are_unique() {
    let a = sweep(&common::Tree, "gb10").unwrap();
    let b = sweep(&common::Tree, "gb10").unwrap();
    let ka: Vec<String> = a.points.iter().map(|p| p.key()).collect();
    let kb: Vec<String> = b.points.iter().map(|p| p.key()).collect();
    assert_eq!(ka, kb);
    let mut sorted = ka.clone();
    sorted.sort();
    sorted.dedup();
    assert_eq!(sorted.len(), ka.len(), "two points share a key");
}

#[test]
fn a_hardware_without_instances_is_refused() {
    assert!(matches!(
        sweep(&common::Tree, "no-such-class"),
        Err(SweepError::NoInstance(_))
    ));
}
