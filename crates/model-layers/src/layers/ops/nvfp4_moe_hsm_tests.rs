// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The Hopper staged NVFP4 MoE decode pair (kernels/hopper/common/
//! moe_nvfp4_decode_hopper.cu) writes the split pair's bytes:
//! `moe_expert_gate_up_act_nvfp4_grouped_hsm` against `_tc_split` (act and shared act) and
//! `moe_expert_down_act_nvfp4_grouped_hsm` against `_tc_mt1` on the same act (output and shared
//! output), on the split gate's fixture (`nvfp4_moe_tc_split_tests.rs`: Flash-Next experts, three
//! hot experts, a null gate and a null down expert) at {1, 2, 3, 8, 13, 16, 32, 64, 128} tokens,
//! once with the shared scales at 4- and 2-byte alignment (the staging's 4-byte and byte copies;
//! the split pair reads scales as 2-byte words, so 2 is the narrowest alignment it admits); a
//! row's bytes alone equal its bytes in a batch; and both pairs' time per launch with every
//! expert's weights distinct and cold in L2, as microseconds and TB/s of weight bytes.
//!
//! GPU test: run with
//! ```text
//! cargo test -p metrale-model-layers --release nvfp4_moe_hsm_tests -- --ignored --nocapture --test-threads 1
//! ```
//! on a Hopper host whose build has the `qwen3.8-flash-next/nvfp4` kernel set
//! (`METRALE_TARGET_HW=hopper`).

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::nvfp4_moe_tc_split_tests::{
    EXPERTS, HIDDEN, INTER, Kern, NULL_DOWN, NULL_GATE, Routing, TOP_K, Weights, all_rows_written,
    bf16s, down, download, gate_up, picks, projection, routing, routing_of, time_us, upload,
};
use super::{
    NVFP4_GROUPED_DOWN_HSM, NVFP4_GROUPED_DOWN_TC_MT1, NVFP4_GROUPED_GATE_UP_HSM,
    NVFP4_GROUPED_GATE_UP_TC_SPLIT, Nvfp4ExpertTables, moe_expert_down_act_nvfp4_grouped,
    moe_expert_gate_up_act_nvfp4_grouped, nvfp4_grouped_hsm_down_smem,
    nvfp4_grouped_hsm_gate_up_smem, nvfp4_grouped_hsm_pair_shape_ok,
};

const WIDTHS: [u32; 9] = [1, 2, 3, 8, 13, 16, 32, 64, 128];
const INVARIANT: [u32; 5] = [2, 3, 8, 13, 16];
const TIMED: [u32; 5] = [1, 2, 4, 8, 16];
/// 2026-10-14: Routings the cold timing rotates through; routing q's experts start at
/// q * COLD_STRIDE, so consecutive launches share no expert at up to 16 tokens.
const COLD_ROUTINGS: u32 = 16;
const COLD_STRIDE: u32 = 160;
/// 2026-10-14: Weight bytes of one expert's gate+up (two [INTER, HIDDEN]) and down projection:
/// packed E2M1 (K / 2 per row) plus E4M3 scales (K / 16 per row).
const GATE_UP_BYTES: f64 = 2.0 * (INTER * HIDDEN) as f64 * 9.0 / 16.0;
const DOWN_BYTES: f64 = (HIDDEN * INTER) as f64 * 9.0 / 16.0;

/// 2026-10-14: The (split gate+up, split down) pair and the (hsm gate+up, hsm down) pair.
fn kernels(g: &dyn GpuBackend) -> ([Kern; 2], [Kern; 2]) {
    let tc = |name: &str| g.kernel("moe_nvfp4_grouped_tc", name).unwrap();
    let hsm = |name: &str| g.kernel("moe_nvfp4_decode_hopper", name).unwrap();
    assert!(nvfp4_grouped_hsm_pair_shape_ok(HIDDEN, INTER));
    (
        [
            (
                tc("moe_expert_gate_up_act_nvfp4_grouped_tc_split"),
                NVFP4_GROUPED_GATE_UP_TC_SPLIT,
                0,
            ),
            (
                tc("moe_expert_down_act_nvfp4_grouped_tc_mt1"),
                NVFP4_GROUPED_DOWN_TC_MT1,
                0,
            ),
        ],
        [
            (
                hsm("moe_expert_gate_up_act_nvfp4_grouped_hsm"),
                NVFP4_GROUPED_GATE_UP_HSM,
                nvfp4_grouped_hsm_gate_up_smem(HIDDEN),
            ),
            (
                hsm("moe_expert_down_act_nvfp4_grouped_hsm"),
                NVFP4_GROUPED_DOWN_HSM,
                nvfp4_grouped_hsm_down_smem(INTER),
            ),
        ],
    )
}

fn backend() -> metrale_gpu_runtime::cuda_backend::MetraleCudaBackend {
    let set = metrale_kernels::ptx_for_exact_target("qwen3.8-flash-next", "nvfp4")
        .expect("qwen3.8-flash-next/nvfp4 not in this build");
    metrale_gpu_runtime::cuda_backend::MetraleCudaBackend::new(0, &set.modules)
        .expect("CUDA backend")
}

fn fixture(g: &dyn GpuBackend) -> Weights {
    Weights {
        gate: projection(g, INTER, HIDDEN, 11, 0.02, Some(NULL_GATE)),
        up: projection(g, INTER, HIDDEN, 13, 0.02, None),
        down: projection(g, HIDDEN, INTER, 17, 0.01, Some(NULL_DOWN)),
    }
}

/// 2026-10-14: A copy of the `n`-byte buffer `p` at `p' + offset` (`p'` 256-byte aligned).
fn shifted(g: &dyn GpuBackend, p: DevicePtr, n: usize, offset: usize) -> DevicePtr {
    let mut bytes = vec![0u8; offset];
    bytes.extend(download(g, p, n));
    DevicePtr(upload(g, &bytes).0 + offset as u64)
}

/// 2026-10-14: (act, shared act, down, shared down) of one pair on `m` tokens of `a`.
fn pair(
    g: &dyn GpuBackend,
    [gu, dn]: [Kern; 2],
    w: &Weights,
    a: DevicePtr,
    r: &Routing,
    m: u32,
) -> [Vec<u8>; 4] {
    let act = gate_up(g, gu, w, a, r, m);
    let dev = (upload(g, &act.0), upload(g, &act.1));
    let out = down(g, dn, w, dev, r, m);
    g.free(dev.0).unwrap();
    g.free(dev.1).unwrap();
    [act.0, act.1, out.0, out.1]
}

fn free_routing(g: &dyn GpuBackend, r: &Routing) {
    for p in [r.offsets, r.sorted, r.active, r.count] {
        g.free(p).unwrap();
    }
}

#[test]
#[ignore]
fn nvfp4_hsm_pair_writes_the_split_pairs_bytes() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let (split, hsm) = kernels(g);
    let mut w = fixture(g);
    for aligned in [true, false] {
        if !aligned {
            let (gs, us, ds) = (
                (INTER * HIDDEN / 16) as usize,
                (INTER * HIDDEN / 16) as usize,
                (HIDDEN * INTER / 16) as usize,
            );
            w.gate.1.weight_scale = shifted(g, w.gate.1.weight_scale, gs, 4);
            w.up.1.weight_scale = shifted(g, w.up.1.weight_scale, us, 2);
            w.down.1.weight_scale = shifted(g, w.down.1.weight_scale, ds, 2);
        }
        for m in WIDTHS {
            let r = routing(g, m);
            let a = upload(g, &bf16s((m * HIDDEN) as usize, 23 + m));
            let label = format!(
                "aligned={aligned} m={m} positions={} active={}",
                r.positions, r.cap
            );
            let want = pair(g, split, &w, a, &r, m);
            let got = pair(g, hsm, &w, a, &r, m);
            for (i, name) in ["act", "shared act", "down", "shared down"]
                .iter()
                .enumerate()
            {
                let row = if i < 2 { INTER * 4 } else { HIDDEN * 2 };
                all_rows_written(&want[i], row as usize, &format!("{label} split {name}"));
                assert_eq!(got[i], want[i], "{label}: {name}");
            }
            g.free(a).unwrap();
            free_routing(g, &r);
        }
    }
}

#[test]
#[ignore]
fn nvfp4_hsm_pair_rows_are_batch_invariant() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let (_, hsm) = kernels(g);
    let w = fixture(g);
    for m in INVARIANT {
        let r = routing(g, m);
        let rows = bf16s((m * HIDDEN) as usize, 29 + m);
        let a = upload(g, &rows);
        let batch = pair(g, hsm, &w, a, &r, m);
        for t in 0..m {
            let one = routing_of(g, &[picks(t)]);
            let row = (t * HIDDEN * 2) as usize..((t + 1) * HIDDEN * 2) as usize;
            let a1 = upload(g, &rows[row]);
            let alone = pair(g, hsm, &w, a1, &one, 1);
            let label = format!("m={m} token={t}");
            for (j, (&p, &p1)) in r.pos[t as usize].iter().zip(&one.pos[0]).enumerate() {
                for (i, width) in [(0, INTER * 4), (2, HIDDEN * 2)] {
                    let (p, p1, width) = (p as usize, p1 as usize, width as usize);
                    assert_eq!(
                        alone[i][p1 * width..(p1 + 1) * width],
                        batch[i][p * width..(p + 1) * width],
                        "{label} pick {j} output {i}"
                    );
                }
            }
            for (i, width) in [(1, INTER * 4), (3, HIDDEN * 2)] {
                let (t, width) = (t as usize, width as usize);
                assert_eq!(
                    alone[i][..width],
                    batch[i][t * width..(t + 1) * width],
                    "{label} shared output {i}"
                );
            }
            g.free(a1).unwrap();
            free_routing(g, &one);
        }
        g.free(a).unwrap();
        free_routing(g, &r);
    }
}

/// 2026-10-14: One projection with a distinct `[n, k]` NVFP4 matrix per expert in two
/// contiguous device buffers (packed, scales), filled with constant bytes (finite E4M3 scales).
fn cold_tables(g: &dyn GpuBackend, n: u32, k: u32) -> (Nvfp4ExpertTables, [DevicePtr; 5]) {
    let (packed, scales) = ((n * k / 2) as usize, (n * k / 16) as usize);
    let wp = g.alloc(packed * EXPERTS as usize).unwrap();
    let sp = g.alloc(scales * EXPERTS as usize).unwrap();
    g.memset(wp, 0x5A, packed * EXPERTS as usize).unwrap();
    g.memset(sp, 0x30, scales * EXPERTS as usize).unwrap();
    let ptrs = |base: DevicePtr, size: usize| -> Vec<u8> {
        (0..EXPERTS as u64)
            .flat_map(|e| (base.0 + e * size as u64).to_le_bytes())
            .collect()
    };
    let s2: Vec<u8> = (0..EXPERTS).flat_map(|_| 0.02f32.to_le_bytes()).collect();
    let t = Nvfp4ExpertTables {
        packed_ptrs: upload(g, &ptrs(wp, packed)),
        scale_ptrs: upload(g, &ptrs(sp, scales)),
        scale2_vals: upload(g, &s2),
    };
    (t, [wp, sp, t.packed_ptrs, t.scale_ptrs, t.scale2_vals])
}

#[test]
#[ignore]
fn nvfp4_hsm_pair_cold_timing() {
    let gpu = backend();
    let g: &dyn GpuBackend = &gpu;
    let (split, hsm) = kernels(g);
    let (gate, gb) = cold_tables(g, INTER, HIDDEN);
    let (up, ub) = cold_tables(g, INTER, HIDDEN);
    let (dn, db) = cold_tables(g, HIDDEN, INTER);
    let none = crate::weight_map::QuantizedWeight::null();
    for m in TIMED {
        let rs: Vec<Routing> = (0..COLD_ROUTINGS)
            .map(|q| {
                let picks: Vec<Vec<u32>> = (0..m)
                    .map(|t| {
                        (0..TOP_K)
                            .map(|j| (q * COLD_STRIDE + t * TOP_K + j) % EXPERTS)
                            .collect()
                    })
                    .collect();
                routing_of(g, &picks)
            })
            .collect();
        let active = rs[0].cap as f64;
        let a = upload(g, &bf16s((m * HIDDEN) as usize, 31 + m));
        let positions = (m * TOP_K) as usize;
        let act = g.alloc(positions * (INTER * 4) as usize).unwrap();
        g.memset(act, 0, positions * (INTER * 4) as usize).unwrap();
        let out = g.alloc(positions * (HIDDEN * 2) as usize).unwrap();
        let gu_us = |(kernel, geometry, smem): Kern| {
            let mut q = 0;
            time_us(g, || {
                let r = &rs[q % rs.len()];
                q += 1;
                moe_expert_gate_up_act_nvfp4_grouped(
                    g, kernel, geometry, smem, a, gate, up, act, r.offsets, r.sorted, r.active,
                    r.count, &none, &none, DevicePtr(0), INTER, HIDDEN, r.cap, 0, 0,
                )
                .unwrap()
            })
        };
        let dn_us = |(kernel, geometry, smem): Kern| {
            let mut q = 0;
            time_us(g, || {
                let r = &rs[q % rs.len()];
                q += 1;
                moe_expert_down_act_nvfp4_grouped(
                    g, kernel, geometry, smem, act, dn, out, r.offsets, r.active, r.count, DevicePtr(0),
                    &none, DevicePtr(0), HIDDEN, INTER, r.cap, 0, 0,
                )
                .unwrap()
            })
        };
        let tbs = |bytes: f64, us: f64| active * bytes / us / 1e6;
        let (gs, gh, ds, dh) = (
            gu_us(split[0]),
            gu_us(hsm[0]),
            dn_us(split[1]),
            dn_us(hsm[1]),
        );
        println!(
            "HSM_COLD_US m={m} active={active} gate_up split={gs:.2}us ({:.2} TB/s) \
             hsm={gh:.2}us ({:.2} TB/s) down mt1={ds:.2}us ({:.2} TB/s) hsm={dh:.2}us ({:.2} TB/s)",
            tbs(GATE_UP_BYTES, gs),
            tbs(GATE_UP_BYTES, gh),
            tbs(DOWN_BYTES, ds),
            tbs(DOWN_BYTES, dh),
        );
        for p in [a, act, out] {
            g.free(p).unwrap();
        }
        for r in &rs {
            free_routing(g, r);
        }
    }
    for p in gb.into_iter().chain(ub).chain(db) {
        g.free(p).unwrap();
    }
}
