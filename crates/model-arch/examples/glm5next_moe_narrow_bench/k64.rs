// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: The one-row byte gate of `glm5next_moe_narrow_bench` at a 704-wide expert slice
//! (the 64-unit `tp` expert layout, K % 128 == 64 on the down projection): gate and up on the
//! plain own-slots sweep (N 704, K 4096) against the plain slot GEMV, and down on the `_k64`
//! own-slots sweep (`w4a4_gemv_mx8_moe_slots_sweep_k64`, K 704 over activations the `_k64`
//! quantizer writes at the padded 768) against the `_k64` slot GEMV, every output byte, on the
//! same rows as the EP-shape gate; plus a swapped-expert negative control on the `_k64` down.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::device::*;
use crate::{Setup, own_sweep, route_of, row_with_local, slots_on};

/// 2026-10-10: The slice width and its 128-padded activation width.
const W: usize = 704;
const WP: usize = 768;

struct K64 {
    quant: KernelHandle,
    slots: KernelHandle,
    own: KernelHandle,
}

pub fn run(gpu: &dyn GpuBackend, s: &Setup, rng: &mut Rng, rows: &[Vec<i32>]) -> Result<()> {
    let m = "w4a4_gemv_mx_moe";
    let kk = K64 {
        quant: gpu.kernel(m, "w4a4_quant_rows_static_k64")?,
        slots: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots_k64")?,
        own: gpu.kernel(m, "w4a4_gemv_mx8_moe_slots_sweep_k64")?,
    };
    let (gate, upt) = (
        table(gpu, rng, W, HIDDEN, GS_GATE_UP)?,
        table(gpu, rng, W, HIDDEN, GS_GATE_UP)?,
    );
    let down = table(gpu, rng, HIDDEN, W, GS_DOWN)?;
    // 2026-10-10: The down input: one 704-wide row per slot, written 768 wide.
    let act: Vec<f32> = (0..TOP_K * W).map(|_| rng.unit() * 10.0).collect();
    let ad = up_bf16(gpu, &act)?;
    let d_act = Act {
        aq: gpu.alloc(TOP_K * WP / 2)?,
        as_: gpu.alloc(TOP_K * WP / 16)?,
        ag: gpu.alloc(TOP_K * 4)?,
    };
    KernelLaunch::new(gpu, kk.quant)
        .grid([TOP_K as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ad)
        .arg_ptr(d_act.aq)
        .arg_ptr(d_act.as_)
        .arg_ptr(d_act.ag)
        .arg_u32(W as u32)
        .arg_f32(GS_DOWN)
        .launch(0)?;
    let (gu, dn) = ((W, HIDDEN, TOP_K), (HIDDEN, W, 1));
    let [o0, o1, o2] = s.out;
    let (ng, nd) = (TOP_K * W * 2, TOP_K * HIDDEN * 2);
    let ctas = s.kn.ctas;
    for (i, host) in rows.iter().enumerate() {
        let local = host
            .iter()
            .filter(|&&e| e >= 0 && (e as usize) < LOCAL)
            .count();
        let r = route_of(gpu, host.clone())?;
        for o in s.out {
            gpu.memset(o, 0, nd)?;
        }
        own_sweep(
            gpu,
            s.own,
            ctas,
            &s.x_act,
            &r,
            &[&gate, &upt],
            &[o0, o1],
            gu,
            0,
        )?;
        let (g, u) = (read(gpu, o0, ng)?, read(gpu, o1, ng)?);
        for (name, t, got) in [("gate", &gate, g), ("up", &upt, u)] {
            gpu.memset(o2, 0, nd)?;
            slots_on(gpu, s.kn.slots, &s.x_act, &r, t, o2, gu, 0)?;
            ensure!(
                read(gpu, o2, ng)? == got,
                "width {W} row {i} {name}: own-slots sweep differs from the slot GEMV ({host:?})"
            );
        }
        gpu.memset(o0, 0, nd)?;
        gpu.memset(o2, 0, nd)?;
        own_sweep(gpu, kk.own, ctas, &d_act, &r, &[&down], &[o0], dn, 0)?;
        slots_on(gpu, kk.slots, &d_act, &r, &down, o2, dn, 0)?;
        let d = read(gpu, o0, nd)?;
        ensure!(
            d == read(gpu, o2, nd)?,
            "width {W} row {i} down: k64 own-slots sweep differs from the k64 slot GEMV ({host:?})"
        );
        ensure!(
            d.iter().any(|&b| b != 0) == (local > 0),
            "width {W} row {i}: down wrote {} with {local} local slots",
            d.iter().any(|&b| b != 0)
        );
    }
    println!(
        "byte gate 1 row at width {W}: {} rows: gate, up (plain own sweep) and down (k64 own \
         sweep) identical to the slot GEMVs",
        rows.len()
    );
    // 2026-10-10: Negative control: a swapped local expert must change the k64 down bytes.
    let host = row_with_local(rng, 3);
    let mut moved = host.clone();
    while host.contains(&moved[0]) {
        moved[0] = (moved[0] + 1) % LOCAL as i32;
    }
    let (r, mv) = (route_of(gpu, host)?, route_of(gpu, moved)?);
    gpu.memset(o0, 0, nd)?;
    gpu.memset(o2, 0, nd)?;
    own_sweep(gpu, kk.own, ctas, &d_act, &mv, &[&down], &[o0], dn, 0)?;
    slots_on(gpu, kk.slots, &d_act, &r, &down, o2, dn, 0)?;
    ensure!(
        read(gpu, o0, nd)? != read(gpu, o2, nd)?,
        "width {W} negative control: a swapped expert left the k64 down bytes equal"
    );
    println!("width {W} negative control: a swapped expert changes the k64 down bytes");
    for t in [gate, upt, down] {
        for p in t.projs {
            gpu.free(p.packed).context("free")?;
            gpu.free(p.scale)?;
        }
        gpu.free(t.packed)?;
        gpu.free(t.scale)?;
        gpu.free(t.scale2)?;
    }
    Ok(())
}
