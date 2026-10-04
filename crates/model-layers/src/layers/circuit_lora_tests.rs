// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: A layer's installed adapters as the circuit binds them, and the refusals.
//!
//! Owner: model-layers (FEATURES workstream).
//! Invariants: none beyond the types.

use metrale_circuit::LinearRole;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::{Installed, bind};
use crate::layers::ops::lora_delta::{
    LoraAttnWeights, LoraFfnWeights, LoraKernels, LoraPair, LoraRoute,
};
use crate::weight_map::DenseWeight;

fn kernels() -> LoraKernels {
    LoraKernels::new(&MockGpuBackend::new()).unwrap()
}

fn pair(tag: u64) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(tag),
        },
        b: DenseWeight {
            weight: DevicePtr(tag + 1),
        },
        rank: 8,
        k_in: 64,
        n_out: 32,
        scale: 1.0,
        max_rank: 8,
    }
}

fn route(tag: u64) -> LoraRoute {
    LoraRoute {
        a_table: DevicePtr(tag),
        b_table: DevicePtr(tag + 1),
        scale_table: DevicePtr(tag + 2),
        k_in: 64,
        n_out: 32,
        max_rank: 8,
    }
}

fn attn(q: (bool, bool)) -> LoraAttnWeights {
    LoraAttnWeights {
        layer_idx: 3,
        q: q.0.then(|| pair(0x100)),
        k: None,
        v: Some(pair(0x300)),
        o: None,
        kernels: kernels(),
        q_route: q.1.then(|| route(0x1000)),
        k_route: None,
        v_route: Some(route(0x3000)),
        o_route: Some(route(0x4000)),
    }
}

fn ffn(gate: bool, up: bool, down: bool) -> LoraFfnWeights {
    LoraFfnWeights {
        gate: gate.then(|| pair(0x500)),
        up: up.then(|| pair(0x600)),
        down: down.then(|| pair(0x700)),
        kernels: kernels(),
    }
}

#[test]
fn attention_binds_every_routed_projection_and_the_ffn_its_active_pairs() {
    let a = attn((true, true));
    let f = ffn(true, true, true);
    let l = bind(Installed {
        attn: Some(&a),
        ffn: Some(&f),
        gdn_out: None,
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        l.routes.keys().copied().collect::<Vec<_>>(),
        [LinearRole::Q, LinearRole::V, LinearRole::O]
    );
    assert_eq!(l.routes[&LinearRole::O].a_table, DevicePtr(0x4000));
    let gu: Vec<u64> = l.pairs[&LinearRole::GateUp]
        .iter()
        .map(|p| p.a.weight.0)
        .collect();
    assert_eq!(gu, [0x500, 0x600], "gate, then up");
    assert_eq!(l.pairs[&LinearRole::Down][0].a.weight.0, 0x700);
    assert!(!l.pairs.contains_key(&LinearRole::GdnOut));
}

#[test]
fn a_gdn_layer_binds_its_out_proj_pair() {
    let out = (pair(0x900), kernels());
    let l = bind(Installed {
        attn: None,
        ffn: None,
        gdn_out: Some(&out),
    })
    .unwrap()
    .unwrap();
    assert_eq!(l.pairs[&LinearRole::GdnOut][0].a.weight.0, 0x900);
    assert!(l.routes.is_empty());
}

#[test]
fn a_layer_without_adapters_binds_none() {
    let empty = ffn(false, false, false);
    let l = bind(Installed {
        attn: None,
        ffn: Some(&empty),
        gdn_out: None,
    })
    .unwrap();
    assert!(l.is_none());
}

#[test]
fn an_unroutable_attention_adapter_and_a_half_gate_up_are_refused() {
    let a = attn((true, false));
    let e = bind(Installed {
        attn: Some(&a),
        ffn: None,
        gdn_out: None,
    })
    .err()
    .unwrap();
    assert!(e.contains("`q` adapter without its routing table"), "{e}");
    for (g, u) in [(true, false), (false, true)] {
        let f = ffn(g, u, true);
        let e = bind(Installed {
            attn: None,
            ffn: Some(&f),
            gdn_out: None,
        })
        .err()
        .unwrap();
        assert!(e.contains("one of gate and up"), "{e}");
    }
}
