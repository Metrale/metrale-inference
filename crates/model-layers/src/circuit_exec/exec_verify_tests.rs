// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-29: The executor's MTP verify programs (K = 2, 3, 4 rows of one sequence), compiled
//! from the real dense circuit over synthetic bindings and run on the recording mock backend:
//! launch and copy counts, the rollback slots the recurrence and the conv snapshots write, the
//! per-row argmax, and the refusal of a state without rollback slots.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use metrale_circuit::Mode;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::Fusions;
use super::exec_fixture::*;
use super::program::{GdnState, StepEnv};

fn at(fusions: Fusions, k: u64) -> Fixture {
    build_at(fusions, Mode::Verify, k, |_| {}, |_| {})
        .unwrap_or_else(|e| panic!("verify K={k}, {fusions:?}: {e:#}"))
}

/// 2026-09-29: One sequence's states, the conv windows and their rollback slots allocated on
/// `gpu` (the snapshots copy them).
pub(super) fn states_on(gpu: &MockGpuBackend, f: &Fixture) -> Vec<Vec<GdnState>> {
    let d = |k: &str| f.circuit.dims[k] as usize;
    let window =
        (d("lin_k_heads") * d("lin_k_dim") * 2 + d("lin_v_heads") * d("lin_v_dim")) * 4 * 4;
    let alloc = || gpu.alloc(window).unwrap();
    states_rows(f, 0xD000_0000, 1)
        .into_iter()
        .map(|l| {
            l.into_iter()
                .map(|s| GdnState {
                    conv: alloc(),
                    conv_steps: [alloc(), alloc(), alloc()],
                    ..s
                })
                .collect()
        })
        .collect()
}

#[test]
fn each_verify_width_launches_and_copies_what_its_plan_counts() {
    for k in 2..=4u64 {
        for fusions in [Fusions::All, Fusions::ReferenceOnly] {
            let f = at(fusions, k);
            let gpu = MockGpuBackend::new();
            let gdn = states_on(&gpu, &f);
            let window = |p| {
                let mut b = vec![0u8; 64];
                gpu.copy_d2h(p, &mut b).unwrap();
                b
            };
            for (i, l) in gdn.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
                gpu.copy_h2d(&[i as u8 + 1; 64], l[0].conv).unwrap();
            }
            let launched = run_on(&gpu, &f, &gdn, 9);
            for (i, l) in gdn.iter().enumerate().filter(|(_, l)| !l.is_empty()) {
                for t in 0..3 {
                    let want = if (t as u64) < k - 1 { i as u8 + 1 } else { 0 };
                    assert_eq!(
                        window(l[0].conv_steps[t]),
                        vec![want; 64],
                        "K={k} layer {i} snapshot {t}"
                    );
                }
            }
            assert_eq!(launched.len() as u64, f.plan.launches(), "K={k}");
            let gdn_layers = gdn.iter().filter(|l| !l.is_empty()).count() as u64;
            assert_eq!(f.plan.copies(), gdn_layers * (k - 1), "K={k}");
            assert_eq!(gpu.d2d_count() as u64, f.plan.copies(), "K={k}");
            assert!(gpu.d2d_async_streams().iter().all(|&s| s == 7));
            assert_pointers_known(&f, &gdn, &launched);
        }
    }
}

#[test]
fn the_recurrence_writes_the_h_rollback_slots_of_its_width_and_argmax_every_row() {
    for k in 2..=4u64 {
        let f = at(Fusions::All, k);
        let gpu = MockGpuBackend::new();
        let gdn = states_on(&gpu, &f);
        let launched = run_on(&gpu, &f, &gdn, 9);
        let mut argmax_rows = Vec::new();
        for (j, l) in kernel_launches(&f).enumerate() {
            let args = &launched[j].args;
            if l.kernel.contains("gated_delta_rule_wy") {
                let layer = f.circuit.nodes[f.plan.groups[l.group].nodes[0]]
                    .layer
                    .unwrap();
                let st = gdn[layer][0];
                for t in 0..3 {
                    let reads = args.contains(&MockArg::Buffer(st.h_steps[t]));
                    assert_eq!(
                        reads,
                        (t as u64) < k - 1,
                        "K={k} layer {layer} slot {t}: {:#x?} {args:?}",
                        st.h_steps
                    );
                }
            }
            if l.kernel == "argmax::argmax_bf16" {
                let out = args.iter().find_map(|a| match a {
                    MockArg::Buffer(p)
                        if (f.fixed.tokens.0..f.fixed.tokens.0 + 64).contains(&p.0) =>
                    {
                        Some(p.0 - f.fixed.tokens.0)
                    }
                    _ => None,
                });
                argmax_rows.push(out.expect("argmax writes the token output"));
            }
        }
        assert_eq!(
            argmax_rows,
            (0..k).map(|i| 4 * i).collect::<Vec<_>>(),
            "K={k}"
        );
    }
}

#[test]
fn a_verify_state_without_rollback_slots_is_refused() {
    let f = at(Fusions::All, 3);
    let gpu = MockGpuBackend::new();
    let mut gdn = states_on(&gpu, &f);
    let first = gdn.iter().position(|l| !l.is_empty()).unwrap();
    gdn[first][0].h_steps[1] = metrale_gpu_runtime::gpu::DevicePtr::NULL;
    let err = f
        .program
        .run(&StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
        })
        .unwrap_err();
    assert!(format!("{err:#}").contains("rollback slots"), "{err:#}");
}
