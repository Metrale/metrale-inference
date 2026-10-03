// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The profiled decode run on the recording mock, whose timing events read the
//! launch count, so a group's "time" is exactly the launches between its two events; and the
//! attribution of the real dense decode plan's groups to legacy's report buckets.
//!
//! Owner: model-layers circuit executor (FEATURES workstream).
//! Invariants: none beyond the types.

use std::cell::RefCell;

use metrale_circuit::LayerKind;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::super::super::Fusions;
use super::super::super::exec_fixture::{Fixture, build};
use super::super::super::exec_fixture_run::{run, states};
use super::super::super::program::{LaunchKind, StepEnv};
use super::{Bucket, DecodeProfile, Part, attribute};

fn mock() -> MockGpuBackend {
    let gpu = MockGpuBackend::new();
    gpu.set_kernel_n_tile(KernelHandle(0xDEAD), 128);
    gpu
}

fn kernels_of(f: &Fixture, g: usize) -> usize {
    f.program
        .launches
        .iter()
        .filter(|l| l.group == g && l.kind == LaunchKind::Kernel)
        .count()
}

/// 2026-10-03: The member locals of group `g`.
fn locals(f: &Fixture, g: usize) -> Vec<&str> {
    f.plan.groups[g]
        .nodes
        .iter()
        .map(|&n| f.circuit.nodes[n].local.as_str())
        .collect()
}

#[test]
fn each_group_is_timed_over_exactly_its_own_launches_and_the_step_is_unchanged() {
    for fusions in [Fusions::All, Fusions::ReferenceOnly] {
        let f = build(fusions, |_| {}).unwrap();
        let gdn = states(&f, 0xD000_0000);
        let plain = run(&f, &gdn, 9);
        let gpu = mock();
        let p = DecodeProfile::new(&gpu, &f.circuit, &f.plan, &f.program).unwrap();
        let env = StepEnv {
            gpu: &gpu,
            stream: 7,
            gdn: &gdn,
            max_blocks_per_seq: 9,
            prefill: None,
        };
        let t = p.run(&f.program, &env, &mut |_| Ok(())).unwrap();
        assert_eq!(t.group_ms.len(), f.plan.groups.len());
        for (g, &ms) in t.group_ms.iter().enumerate() {
            assert_eq!(
                ms as usize,
                kernels_of(&f, g),
                "group {g} ({:?})",
                locals(&f, g)
            );
        }
        let profiled = gpu.launches_snapshot();
        assert_eq!(profiled.len(), plain.len());
        for (a, b) in profiled.iter().zip(&plain) {
            assert_eq!((a.grid, a.block, &a.args), (b.grid, b.block, &b.args));
        }
        p.free(&gpu).unwrap();
    }
}

#[test]
fn the_layer_hook_runs_once_per_layer_after_the_layers_last_launch() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let gdn = states(&f, 0xD000_0000);
    let gpu = mock();
    let p = DecodeProfile::new(&gpu, &f.circuit, &f.plan, &f.program).unwrap();
    let last_launch_of = |layer: usize| {
        f.program
            .launches
            .iter()
            .filter(|l| l.kind == LaunchKind::Kernel)
            .enumerate()
            .filter(|(_, l)| {
                matches!(p.buckets()[l.group], Bucket::Layer { layer: x, .. } if x == layer)
            })
            .map(|(i, _)| i + 1)
            .max()
            .unwrap()
    };
    let calls = RefCell::new(Vec::new());
    let env = StepEnv {
        gpu: &gpu,
        stream: 7,
        gdn: &gdn,
        max_blocks_per_seq: 9,
        prefill: None,
    };
    p.run(&f.program, &env, &mut |l| {
        calls.borrow_mut().push((l, gpu.launches_snapshot().len()));
        Ok(())
    })
    .unwrap();
    let layers = f.circuit.layer_kinds.len();
    let calls = calls.into_inner();
    assert_eq!(
        calls.iter().map(|c| c.0).collect::<Vec<_>>(),
        (0..layers).collect::<Vec<_>>()
    );
    for (l, launched) in calls {
        assert_eq!(launched, last_launch_of(l), "layer {l}");
    }
}

#[test]
fn the_dense_decode_plan_attributes_like_legacys_profile() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let b = attribute(&f.circuit, &f.plan).unwrap();
    for (layer, kind) in f.circuit.layer_kinds.iter().enumerate() {
        let parts: Vec<(usize, Part)> = b
            .iter()
            .enumerate()
            .filter_map(|(g, x)| match x {
                Bucket::Layer { layer: l, part } if *l == layer => Some((g, *part)),
                _ => None,
            })
            .collect();
        let count = |p: Part| parts.iter().filter(|(_, x)| *x == p).count();
        let gdn = *kind == LayerKind::LinearAttention;
        assert_eq!(count(Part::Qkvz), usize::from(gdn), "layer {layer}");
        assert_eq!(count(Part::BaGates) > 0, gdn, "layer {layer}");
        assert!(
            count(Part::Ffn) >= 2,
            "layer {layer}: post-norm and the projections"
        );
        for (g, part) in parts {
            let ls = locals(&f, g);
            let ffn_node = ls
                .iter()
                .any(|l| ["gate_up", "act", "down", "post_norm"].contains(l));
            assert_eq!(
                part == Part::Ffn,
                ffn_node,
                "layer {layer} group {g} {ls:?}"
            );
        }
    }
    let head: Vec<&str> = b
        .iter()
        .enumerate()
        .filter(|(_, x)| **x == Bucket::Head)
        .flat_map(|(g, _)| locals(&f, g))
        .collect();
    assert!(
        head.contains(&"final_norm") && head.contains(&"lm_head"),
        "{head:?}"
    );
    for (g, x) in b.iter().enumerate() {
        if *x == Bucket::Prologue {
            assert_eq!(locals(&f, g), ["embed"], "group {g}");
        }
    }
}

#[test]
fn a_group_fused_across_a_layer_boundary_counts_in_its_first_nodes_layer() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let b = attribute(&f.circuit, &f.plan).unwrap();
    let crossing: Vec<usize> = (0..f.plan.groups.len())
        .filter(|&g| {
            let ls: Vec<Option<usize>> = f.plan.groups[g]
                .nodes
                .iter()
                .map(|&n| f.circuit.nodes[n].layer)
                .collect();
            ls.windows(2).any(|w| w[0] != w[1])
        })
        .collect();
    assert_eq!(crossing.len(), f.circuit.layer_kinds.len() - 1);
    for g in crossing {
        let first = f.circuit.nodes[f.plan.groups[g].nodes[0]].layer.unwrap();
        assert_eq!(
            b[g],
            Bucket::Layer {
                layer: first,
                part: Part::Other
            },
            "{:?}",
            locals(&f, g)
        );
    }
}

#[test]
fn a_program_of_another_plan_is_refused() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let other = build(Fusions::ReferenceOnly, |_| {}).unwrap();
    let gpu = mock();
    let e = DecodeProfile::new(&gpu, &f.circuit, &f.plan, &other.program)
        .err()
        .unwrap();
    assert!(e.to_string().contains("not compiled from this plan"), "{e}");
}

#[test]
fn the_report_has_legacys_lines_with_each_bucket_in_its_own_line() {
    use super::{StepTimes, report};
    let l = |layer, part| Bucket::Layer { layer, part };
    let times = StepTimes {
        // 2026-10-03: Distinct powers of two, so a bucket summed into the wrong line shows.
        group_ms: vec![64.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        buckets: vec![
            Bucket::Prologue,
            l(0, Part::Other),
            l(0, Part::Qkvz),
            l(0, Part::BaGates),
            l(0, Part::Ffn),
            l(1, Part::Other),
            l(1, Part::Ffn),
            l(0, Part::Ffn),
            Bucket::Head,
        ],
    };
    let (layers, line) = report(
        &times,
        &[LayerKind::LinearAttention, LayerKind::FullAttention],
        41,
    );
    assert_eq!(
        layers,
        vec![
            vec![
                "    SSM qkvz: 500μs".to_string(),
                "    SSM ba_gates: 1000μs".to_string(),
                "  SSM-MoE: 18.0ms".to_string(),
            ],
            vec!["  Attn-MoE: 8.0ms".to_string()],
        ]
    );
    assert_eq!(
        line,
        "PROFILE tok=41: total=63.8ms attn=12.0ms(1) ssm=19.8ms(1) head=32.0ms"
    );
}
