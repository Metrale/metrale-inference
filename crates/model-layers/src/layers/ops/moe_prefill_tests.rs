// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: The token-major gate+up and silu+down launches with a null shared expert
//! (a BF16 shared expert its caller runs itself): they must launch no shared rows, so the
//! shared outputs stay byte-for-byte untouched, while every routed row stays
//! byte-identical to the launch with an NVFP4 shared expert. A partly null shared expert
//! is refused before any launch.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release moe_prefill_tests -- --ignored --nocapture
//! ```
//! on a host whose build has the `qwen3.8-flash-next/nvfp4` kernel set.

use super::*;

const HIDDEN: usize = 64;
const INTER: usize = 32;
const EXPERTS: usize = 4;
const TOP_K: usize = 2;
const TOKENS: usize = 3;
const SENTINEL: u8 = 0xAB;

/// 2026-10-13: A placeholder NVFP4 weight whose only meaningful field is a non-null
/// pointer, for the host-side row rule.
fn non_null() -> QuantizedWeight {
    QuantizedWeight {
        weight: DevicePtr(0x1000),
        ..QuantizedWeight::null()
    }
}

#[test]
fn shared_rows_are_one_for_nvfp4_zero_for_null_and_refused_when_partly_null() {
    let (w, z) = (non_null(), QuantizedWeight::null());
    assert_eq!(shared_expert_rows(&[&w, &w]).unwrap(), 1);
    assert_eq!(shared_expert_rows(&[&w]).unwrap(), 1);
    assert_eq!(shared_expert_rows(&[&z, &z]).unwrap(), 0);
    assert_eq!(shared_expert_rows(&[&z]).unwrap(), 0);
    for partial in [[&w, &z], [&z, &w]] {
        let err = shared_expert_rows(&partial).unwrap_err();
        assert!(format!("{err:#}").contains("1 of 2"), "{err:#}");
    }
}

/// 2026-10-13: Deterministic bytes; scale bytes stay in E4M3's finite range.
fn bytes(n: usize, seed: u32, scale: bool) -> Vec<u8> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            if scale {
                0x28 + (x % 16) as u8
            } else {
                x as u8
            }
        })
        .collect()
}

/// 2026-10-13: BF16 activations in [-1, 1].
fn bf16s(n: usize, seed: u32) -> Vec<u8> {
    bytes(n, seed, false)
        .into_iter()
        .flat_map(|b| {
            let v = b as f32 / 127.5 - 1.0;
            ((v.to_bits() >> 16) as u16).to_le_bytes()
        })
        .collect()
}

fn upload(g: &dyn GpuBackend, data: &[u8]) -> DevicePtr {
    let p = g.alloc(data.len()).unwrap();
    g.copy_h2d(data, p).unwrap();
    p
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    g.copy_d2h(p, &mut out).unwrap();
    out
}

/// 2026-10-13: An `[n, k]` NVFP4 weight on the device.
fn nvfp4(g: &dyn GpuBackend, n: usize, k: usize, seed: u32) -> QuantizedWeight {
    QuantizedWeight {
        weight: upload(g, &bytes(n * k / 2, seed, false)),
        weight_scale: upload(g, &bytes(n * k / 16, seed ^ 0x5bd1_e995, true)),
        weight_scale_2: 0.75,
        ..QuantizedWeight::null()
    }
}

/// 2026-10-13: Per-expert pointer tables (packed, scale, scale2) of `[n, k]` experts.
fn experts(g: &dyn GpuBackend, n: usize, k: usize, seed: u32) -> [DevicePtr; 3] {
    let ws: Vec<_> = (0..EXPERTS)
        .map(|e| nvfp4(g, n, k, seed + e as u32))
        .collect();
    let table = |f: &dyn Fn(&QuantizedWeight) -> u64| -> Vec<u8> {
        ws.iter().flat_map(|w| f(w).to_le_bytes()).collect()
    };
    let s2: Vec<u8> = (0..EXPERTS)
        .flat_map(|e| (0.5 + e as f32 / 8.0).to_le_bytes())
        .collect();
    [
        upload(g, &table(&|w| w.weight.0)),
        upload(g, &table(&|w| w.weight_scale.0)),
        upload(g, &s2),
    ]
}

/// 2026-10-13: A buffer of `n` bytes filled with [`SENTINEL`].
fn sentinel(g: &dyn GpuBackend, n: usize) -> DevicePtr {
    let p = g.alloc(n).unwrap();
    g.memset(p, SENTINEL, n).unwrap();
    p
}

#[test]
#[ignore]
fn a_null_shared_expert_launches_no_shared_rows_and_leaves_routed_rows_identical() {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    let gpu = metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend");
    let g: &dyn GpuBackend = &gpu;
    let gate_up_k = g
        .kernel("moe_prefill", "moe_expert_gate_up_shared_prefill")
        .unwrap();
    let silu_down_k = g
        .kernel("moe_prefill", "moe_expert_silu_down_shared_prefill")
        .unwrap();
    let stream = g.create_stream().unwrap();
    let input = upload(g, &bf16s(TOKENS * HIDDEN, 0x2545_F491));
    let indices: Vec<u8> = [3u32, 0, 1, 3, 2, 1]
        .iter()
        .flat_map(|e| e.to_le_bytes())
        .collect();
    let indices = upload(g, &indices);
    let [gp, gs, g2] = experts(g, INTER, HIDDEN, 11);
    let [upp, ups, up2] = experts(g, INTER, HIDDEN, 23);
    let [dp, ds, d2] = experts(g, HIDDEN, INTER, 37);
    let (sh_gate, sh_up, sh_down) = (
        nvfp4(g, INTER, HIDDEN, 41),
        nvfp4(g, INTER, HIDDEN, 43),
        nvfp4(g, HIDDEN, INTER, 47),
    );
    let routed_inter = TOKENS * TOP_K * INTER * 2;
    let routed_hidden = TOKENS * TOP_K * HIDDEN * 2;
    let (sh_inter, sh_hidden) = (TOKENS * INTER * 2, TOKENS * HIDDEN * 2);

    // 2026-10-13: Returns (routed gate, up, down) and (shared gate, up, down) bytes.
    let run = |shared: [&QuantizedWeight; 3]| {
        let (gate_out, up_out) = (sentinel(g, routed_inter), sentinel(g, routed_inter));
        let down_out = sentinel(g, routed_hidden);
        let (sg, su, sd) = (
            sentinel(g, sh_inter),
            sentinel(g, sh_inter),
            sentinel(g, sh_hidden),
        );
        let (h, i, k, t) = (HIDDEN as u32, INTER as u32, TOP_K as u32, TOKENS as u32);
        moe_expert_gate_up_shared_prefill(
            g, gate_up_k, input, gp, gs, g2, gate_out, upp, ups, up2, up_out, indices, shared[0],
            sg, shared[1], su, i, h, k, t, stream,
        )
        .unwrap();
        moe_expert_silu_down_shared_prefill(
            g,
            silu_down_k,
            gate_out,
            up_out,
            dp,
            ds,
            d2,
            down_out,
            indices,
            sg,
            su,
            shared[2],
            sd,
            h,
            i,
            k,
            t,
            stream,
        )
        .unwrap();
        g.synchronize(stream).unwrap();
        (
            [
                download(g, gate_out, routed_inter),
                download(g, up_out, routed_inter),
                download(g, down_out, routed_hidden),
            ],
            [
                download(g, sg, sh_inter),
                download(g, su, sh_inter),
                download(g, sd, sh_hidden),
            ],
        )
    };
    let (routed_nvfp4, shared_nvfp4) = run([&sh_gate, &sh_up, &sh_down]);
    let null = QuantizedWeight::null();
    let (routed_null, shared_null) = run([&null, &null, &null]);

    for (name, out) in ["gate", "up", "down"].iter().zip(&routed_nvfp4) {
        assert!(
            out.iter().any(|&b| b != SENTINEL),
            "control: routed {name} was written"
        );
    }
    for (name, out) in ["gate", "up", "down"].iter().zip(&shared_nvfp4) {
        assert!(
            out.iter().any(|&b| b != SENTINEL),
            "control: the NVFP4 shared {name} rows ran"
        );
    }
    assert_eq!(routed_null, routed_nvfp4, "routed rows are byte-identical");
    for (name, out) in ["gate", "up", "down"].iter().zip(&shared_null) {
        assert!(
            out.iter().all(|&b| b == SENTINEL),
            "a null shared expert launched {name} rows"
        );
    }
}
