// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The executor compiled from the real dense circuit's decode plan over synthetic
//! bindings, run on the recording mock backend: launch counts, the pointers every launch reads,
//! what a step may change, and the refusals.
//!
//! Owner: model-layers circuit executor.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use metrale_circuit::LinearRole;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};

use super::bindings::*;
use super::exec_fixture::*;
use super::kernels::KernelTable;
use super::{Fusions, compile, sources};
use crate::weight_map::DenseWeight;

#[test]
fn a_program_launches_exactly_what_its_plan_counts() {
    let full = build(Fusions::All, |_| {}).unwrap();
    let reference = build(Fusions::ReferenceOnly, |_| {}).unwrap();
    for f in [&full, &reference] {
        assert_eq!(f.program.launches.len() as u64, f.plan.launches());
        let launched = run(f, &states(f, 0xD000_0000), 9);
        assert_eq!(launched.len(), f.program.launches.len());
        assert!(launched.iter().all(|l| l.stream == 7));
    }
    let boundaries = full.circuit.layer_kinds.len() - 1;
    assert_eq!(
        reference.program.launches.len() - full.program.launches.len(),
        boundaries,
        "the cross-layer fusion saves one launch per layer boundary"
    );
}

#[test]
fn every_pointer_a_launch_reads_is_bound_placed_or_the_steps() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let gdn = states(&f, 0xD000_0000);
    assert_pointers_known(&f, &gdn);
}

#[test]
fn a_step_changes_only_the_state_and_width_arguments_that_read_them() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let a = run(&f, &states(&f, 0xD000_0000), 9);
    let b = run(&f, &states(&f, 0xE000_0000), 40);
    let mut changed = BTreeMap::<String, usize>::new();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            (x.grid, x.block, x.shared_mem),
            (y.grid, y.block, y.shared_mem)
        );
        if x.args != y.args {
            *changed
                .entry(f.program.launches[i].kernel.clone())
                .or_default() += 1;
        }
    }
    let gdn_layers = f
        .layers
        .iter()
        .filter(|l| matches!(l.mixer, MixerFacts::Gdn(_)))
        .count();
    let attn_layers = f.layers.len() - gdn_layers;
    assert_eq!(
        changed,
        BTreeMap::from([
            (
                "causal_conv1d::causal_conv1d_update_l2norm_f32".to_string(),
                gdn_layers
            ),
            (
                "gated_delta_rule::gated_delta_rule_decode_f32".to_string(),
                gdn_layers
            ),
            ("paged_decode::paged_decode_attn".to_string(), attn_layers),
        ])
    );
}

#[test]
fn the_cross_layer_norm_reads_the_next_layers_input_norm_weight() {
    let f = build(Fusions::All, |_| {}).unwrap();
    let launched = run(&f, &states(&f, 0xD000_0000), 9);
    let mut seen = 0;
    for (i, l) in f.program.launches.iter().enumerate() {
        if l.kernel != "residual_add_rms_norm_exact::residual_add_rms_norm_exact" {
            continue;
        }
        let group = &f.plan.groups[l.group];
        let next = f.circuit.nodes[group.nodes[1]].layer.unwrap();
        let want = tag(next, 1);
        assert!(
            launched[i].args.contains(&MockArg::Buffer(ptr(want))),
            "group {} does not read layer {next}'s input norm",
            l.group
        );
        seen += 1;
    }
    assert_eq!(seen, f.layers.len() - 1);
}

#[test]
fn a_binding_the_plan_does_not_describe_is_refused() {
    let err = |edit: &dyn Fn(&mut Vec<CircuitLayer>)| {
        build(Fusions::All, edit)
            .err()
            .map(|e| format!("{e:#}"))
            .unwrap_or_default()
    };
    let e = err(&|l| {
        l[0].mixer = MixerFacts::Gdn(GdnFacts {
            qkvz_deinterleaved: false,
        })
    });
    assert!(e.contains("deinterleaved"), "{e}");
    let e = err(&|l| {
        l[1].weights
            .insert(WeightSlot::Linear(LinearRole::GdnOut), dense(1));
    });
    assert!(e.contains("resolved") || e.contains("NVFP4"), "{e}");
    let e = err(&|l| {
        if let MixerFacts::Attention(a) = &mut l[3].mixer {
            a.paged_decode_plain_rows = 0;
        }
    });
    assert!(e.contains("plain paged kernel"), "{e}");
    let mut layers: Vec<Option<CircuitLayer>> = build(Fusions::All, |_| {})
        .unwrap()
        .layers
        .into_iter()
        .map(Some)
        .collect();
    let inst = sources::instance(RECIPE).unwrap();
    let circuit = metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .circuit;
    let head = HeadBinding {
        final_norm: DenseWeight { weight: ptr(1) },
        lm_head: dense(2),
        unmodelled: Vec::new(),
        batchm_max_rows: 8,
    };
    layers[5]
        .as_mut()
        .unwrap()
        .unmodelled
        .push("an out_proj LoRA adapter".into());
    let e = compile::check_bindings(&circuit, &layers, &head)
        .unwrap_err()
        .to_string();
    assert!(e.contains("layer 5: an out_proj LoRA adapter"), "{e}");
    layers[5] = None;
    let e = compile::check_bindings(&circuit, &layers, &head)
        .unwrap_err()
        .to_string();
    assert!(e.contains("layer 5 has no circuit binding"), "{e}");
}

#[test]
fn ptx_availability_reads_entry_points_not_names() {
    let ptx: &[u8] =
        b".visible .entry rope_forward(\n.param .u64 a\n)\n// rope_forward_strided is a comment\n";
    assert!(super::kernels::ptx_defines(ptx, "rope_forward"));
    assert!(!super::kernels::ptx_defines(ptx, "rope_forward_strided"));
    assert!(!super::kernels::ptx_defines(ptx, "rope"));
    let inst = sources::instance(RECIPE).unwrap();
    let rules = metrale_circuit::load(&inst, sources::sources(&inst).unwrap())
        .unwrap()
        .rules;
    let avail = super::kernels::available_in(&rules, &[("rope", ptx)]).unwrap();
    assert!(
        avail
            .kernels
            .iter()
            .all(|k| k.module == "rope" && k.func == "rope_forward")
    );
    let gpu = MockGpuBackend::new();
    let table = KernelTable::resolve(&gpu, &avail);
    assert!(
        gpu.kernel_lookups_snapshot().is_empty(),
        "no lookup is issued for a kernel the target does not define"
    );
    assert!(
        table
            .handle(&metrale_circuit::KernelId {
                module: "norm".into(),
                func: "rms_norm".into()
            })
            .is_err()
    );
}
