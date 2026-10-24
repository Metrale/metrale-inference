// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The GLM lever map: off or on the A arm the kernels are unchanged; on, a lever
//! changes only its own kernel; an unregistered name is refused on every arm.
//! Owner: model-arch (GLM-5.3) tests.

use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::glm5next_dense_seg::DenseSeg;

fn kernels() -> Glm5NextMlpKernels {
    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextMlpKernels::resolve(&gpu, crate::glm5next_layer::GlmLevers::declared())
        .expect("mock resolves every kernel");
    k.gemv_batchm_few.f32 = KernelHandle(0x74);
    k.gemv_batchm_few.router_npb1 = KernelHandle(0x75);
    k
}

fn names(n: &[&str]) -> Vec<String> {
    n.iter().map(|s| s.to_string()).collect()
}

#[test]
fn the_a_arm_keeps_the_kernels_and_the_b_arm_changes_only_its_lever() {
    let k = kernels();
    let n = names(&["router_npb1"]);
    let a = apply(&k, &n, AbLevers::NONE).unwrap();
    assert_eq!(
        (a.gemv_batchm_few.f32.0, a.gemv_batchm_few.f32_npb),
        (0x74, crate::glm5next_layer::wide_gemv::FEW_NPB)
    );
    let b = apply(&k, &n, AbLevers::only(0)).unwrap();
    assert_eq!(
        (b.gemv_batchm_few.f32.0, b.gemv_batchm_few.f32_npb),
        (0x75, 1)
    );
    assert_eq!(b.gemv_batchm_few.bf16.0, k.gemv_batchm_few.bf16.0);
    assert_eq!(b.gemv_batchm_f32.0, k.gemv_batchm_f32.0);
}

#[test]
fn an_unregistered_name_is_refused_on_every_arm() {
    let k = kernels();
    for on in [AbLevers::NONE, AbLevers::first(2)] {
        let Err(e) = apply(&k, &names(&["router_npb1", "routr_npb1"]), on) else {
            panic!("an unregistered name must be refused");
        };
        assert!(e.to_string().contains("no lever `routr_npb1`"), "{e}");
    }
}

#[test]
fn swiglu_down_turns_on_the_fold_only_on_its_arm_and_needs_the_pipe() {
    let mut k = kernels();
    k.levers.dense.gemv_seg = true;
    k.levers.dense.gemv_pipe = true;
    let n = names(&["router_npb1", "swiglu_down"]);
    assert!(
        !apply(&k, &n, AbLevers::only(0))
            .unwrap()
            .levers
            .dense
            .swiglu_down
    );
    let b = apply(&k, &n, AbLevers::only(1)).unwrap();
    assert!(b.levers.dense.swiglu_down);
    assert_eq!(b.gemv_batchm_few.f32.0, 0x74, "the other lever stays off");
    k.levers.dense.gemv_pipe = false;
    let Err(e) = apply(&k, &n, AbLevers::NONE) else {
        panic!("without the pipe the arm would measure A against A");
    };
    assert!(e.to_string().contains("needs the W4A16 GEMV pipe"), "{e}");
}

#[test]
fn dense_seg_turns_on_the_segment_only_on_its_arm_and_needs_the_pipe() {
    let mut k = kernels();
    k.levers.dense.gemv_seg = true;
    k.levers.dense.gemv_pipe = true;
    let n = names(&["swiglu_down", "dense_seg"]);
    let only = names(&["dense_seg"]);
    let a = apply(&k, &n, AbLevers::only(0)).unwrap();
    assert!(a.levers.dense.dense_seg == k.levers.dense.dense_seg && a.levers.dense.swiglu_down);
    let b = apply(&k, &n, AbLevers::only(1)).unwrap();
    assert!(b.levers.dense.dense_seg == DenseSeg::Everywhere && !b.levers.dense.swiglu_down);
    k.levers.dense.gemv_seg = false;
    let Err(e) = apply(&k, &only, AbLevers::NONE) else {
        panic!("without the pipe the arm would measure A against A");
    };
    assert!(
        e.to_string()
            .contains("`dense_seg` needs the W4A16 GEMV pipe"),
        "{e}"
    );
}

#[test]
fn dense_seg_budget_sets_the_segment_and_its_sms_only_on_its_arm() {
    let mut k = kernels();
    k.levers.dense.gemv_seg = true;
    k.levers.dense.gemv_pipe = true;
    let n = names(&["swiglu_down", "dense_seg@24"]);
    let a = apply(&k, &n, AbLevers::only(0)).unwrap();
    assert!(a.levers.dense.dense_seg == k.levers.dense.dense_seg && a.levers.dense.dense_seg_sms.is_none());
    let b = apply(&k, &n, AbLevers::only(1)).unwrap();
    assert!(b.levers.dense.dense_seg == DenseSeg::Everywhere && !b.levers.dense.swiglu_down);
    assert_eq!(b.levers.dense.dense_seg_sms.map(|s| s.get()), Some(24));
    for bad in ["dense_seg@0", "dense_seg@", "dense_seg@x", "dense_seg@-3"] {
        let Err(e) = apply(&k, &names(&[bad]), AbLevers::NONE) else {
            panic!("`{bad}` must be refused on every arm");
        };
        assert!(e.to_string().contains("the SM budget"), "{bad}: {e}");
    }
    k.levers.dense.gemv_pipe = false;
    let Err(e) = apply(&k, &names(&["dense_seg@24"]), AbLevers::NONE) else {
        panic!("without the pipe the arm would measure A against A");
    };
    assert!(e.to_string().contains("needs the W4A16 GEMV pipe"), "{e}");
}

#[test]
fn kda_conv_one_changes_the_kda_levers_only_on_its_arm() {
    let mut base = crate::glm5next_layer::GlmLevers::declared();
    base.kda.verify_tokens = true;
    let n = names(&["swiglu_down", "kda_conv_one"]);
    assert_eq!(
        kda_apply(base, &n, AbLevers::NONE).unwrap(),
        None,
        "A runs as resolved"
    );
    assert_eq!(
        kda_apply(base, &n, AbLevers::only(0)).unwrap(),
        None,
        "an MLP lever"
    );
    let b = kda_apply(base, &n, AbLevers::only(1))
        .unwrap()
        .expect("B changes KDA");
    assert!(b.kda.conv_one && !base.kda.conv_one);
    assert_eq!(
        GlmLevers { kda: base.kda, ..b },
        base,
        "nothing else changes"
    );
    base.kda.verify_tokens = false;
    assert!(
        kda_apply(base, &n, AbLevers::NONE).is_err(),
        "B would run A"
    );
}

/// 2026-10-23: BF16-key DSA kernels: the walks at 0x81/0x82, the tensor-core scores at 0x83.
fn dsa_kernels() -> Glm5NextDsaKernels {
    let gpu = MockGpuBackend::new();
    let mut k = Glm5NextDsaKernels::resolve(&gpu).expect("mock resolves every kernel");
    k.rows.pool_keys_bf16 = true;
    k.rows.scores_heads_paged = KernelHandle(0x81);
    k.rows.scores_heads_window = KernelHandle(0x82);
    k.rows.scores_mma_paged = KernelHandle(0x83);
    k
}

/// 2026-10-23: `dsa_mma`: A keeps the walks, B launches the tensor-core scores as both
/// lane-per-head handles and changes nothing else.
#[test]
fn dsa_mma_swaps_only_the_lane_per_head_handles_on_the_b_arm() {
    let k = dsa_kernels();
    let n = names(&["swiglu_down", "dsa_mma"]);
    assert!(dsa_apply(&k, &n, AbLevers::NONE).unwrap().is_none());
    assert!(
        dsa_apply(&k, &n, AbLevers::only(0)).unwrap().is_none(),
        "another lever's arm"
    );
    let b = dsa_apply(&k, &n, AbLevers::only(1))
        .unwrap()
        .expect("B changes the kernels");
    assert_eq!(
        (b.rows.scores_heads_paged.0, b.rows.scores_heads_window.0),
        (0x83, 0x83)
    );
    assert_eq!(b.rows.pool_keys_store.0, k.rows.pool_keys_store.0);
    assert_eq!(b.rows.scores_warp_paged.0, k.rows.scores_warp_paged.0);
    assert_eq!(b.rows.topk_radix.0, k.rows.topk_radix.0);
}

/// 2026-10-23: An arm that could only run A is refused: FP32 keys, or the MMA already default.
#[test]
fn dsa_mma_is_refused_where_b_would_be_a() {
    let n = names(&["dsa_mma"]);
    let mut fp32 = dsa_kernels();
    fp32.rows.pool_keys_bf16 = false;
    assert!(dsa_apply(&fp32, &n, AbLevers::only(0)).is_err());
    assert!(
        dsa_apply(&fp32, &n, AbLevers::NONE).is_err(),
        "refused on every arm"
    );
    let mut on = dsa_kernels();
    on.rows = on.rows.with_scores_mma();
    assert!(dsa_apply(&on, &n, AbLevers::only(0)).is_err());
    assert!(
        apply(&kernels(), &n, AbLevers::only(0)).is_ok(),
        "the MLP map ignores it"
    );
}
