// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The tensor-core grouped FP8 MoE decode, W8A16 (`moe_fp8_grouped_tc.cu`) and
//! the opt-in W8A8 twin (`moe_fp8_grouped_tc_w8a8.cu`), at Qwen3.6-35B-A3B shapes: row
//! invariance and closeness to the scalar grouped kernels.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error unless, for every M and both tensor-core legs, each row of the M-row
//!   dispatch equals, byte for byte, the same row dispatched alone (M = 1), and the whole
//!   output is within the leg's bound (`MAX_REL_L2`, `MAX_REL_L2_W8A8`; relative L2) of the
//!   scalar grouped kernels' output on the same routing.
//! - Before the sweep, two known-bad outputs (one flipped bit, one zeroed row) must be
//!   refused by those two checks.
//!
//! Both legs run sort, gate+up, down and the grouped blend with the same routing. Shapes
//! follow the 35B MODEL.toml (hidden 2048, expert and shared-expert intermediate 512, top-8)
//! with 256 routed experts by default. Each leg's mean time over 20 iterations is printed.
//!
//! Run (GB10), optionally with the routed-expert count and a Zipf routing exponent:
//!   cargo run --release -p metrale-model-arch --features cuda,gpu-examples \
//!     --example fp8_moe_grouped_tc_microtest -- [experts] [zipf_alpha]

use anyhow::{Result, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::Fp8Weight;

#[path = "common/fp8_moe_grouped_fixture.rs"]
#[allow(dead_code)]
mod fixture;
use fixture::{
    Rng, bf16_bytes, experts, fp8_bytes, fp8w, scale_bytes, upload, zipf_alpha, zipf_routing,
};

const H: usize = 2048;
const INTER: usize = 512;
const TOP_K: usize = 8;
const MAX_M: usize = 64;
const GUARD: usize = 64;
/// 2026-09-28: The tensor-core leg sums in another order (its FP32 SiLU product enters the
/// down MMAs as two BF16 terms); both legs round their outputs to BF16, so 1e-2 from the
/// scalar leg.
const MAX_REL_L2: f64 = 1e-2;
/// 2026-09-28: W8A8 rounds the input and the SiLU product to E4M3 (3 mantissa bits) per
/// 128 group; on these uniform random activations that is 4.2e-2 against FP64, so 8e-2.
const MAX_REL_L2_W8A8: f64 = 8e-2;

/// 2026-09-28: The expert kernels a dispatch runs.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Leg {
    Scalar,
    Tc,
    TcW8a8,
}

struct Kernels {
    sort: KernelHandle,
    blend: KernelHandle,
    gate_up: [KernelHandle; 2],
    down: [KernelHandle; 2],
    quant_w8a8: KernelHandle,
    gate_up_w8a8: KernelHandle,
    down_w8a8: KernelHandle,
}

struct Experts {
    gate: (DevicePtr, DevicePtr),
    up: (DevicePtr, DevicePtr),
    down: (DevicePtr, DevicePtr),
    sh_gate: Fp8Weight,
    sh_up: Fp8Weight,
    sh_down: Fp8Weight,
    sh_gate_vec: DevicePtr,
}

struct Scratch {
    e: usize,
    sort: DevicePtr,
    act: DevicePtr,
    sh_act: DevicePtr,
    down_out: DevicePtr,
    sh_down_out: DevicePtr,
}

/// 2026-09-28: One grouped dispatch of `m` rows from `input` / `indices` / `weights` into
/// `out` through the expert kernels of `leg`.
#[allow(clippy::too_many_arguments)]
fn dispatch(
    gpu: &dyn GpuBackend,
    k: &Kernels,
    x: &Experts,
    s: &Scratch,
    input: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    out: DevicePtr,
    m: usize,
    leg: Leg,
) -> Result<()> {
    let te = m * TOP_K;
    let e = s.e;
    let sort = ops::Fp8GroupedSortOut {
        sorted_token_ids: s.sort,
        sorted_expert_ids: s.sort.offset(te * 4),
        expert_offsets: s.sort.offset(te * 8),
        token_to_perm: s.sort.offset(te * 8 + (e + 1) * 4),
        active_experts: s.sort.offset(te * 12 + (e + 1) * 4),
        active_count: s.sort.offset(
            te * 12
                + (e + 1) * 4
                + ops::fp8_grouped_active_cap(m as u32, TOP_K as u32, e as u32) as usize * 4,
        ),
    };
    let cap = ops::fp8_grouped_active_cap(m as u32, TOP_K as u32, e as u32);
    let tc = leg == Leg::Tc;
    let (gu_geom, down_geom) = if tc {
        (ops::FP8_GROUPED_GATE_UP_TC, ops::FP8_GROUPED_DOWN_TC)
    } else {
        (
            ops::FP8_GROUPED_GATE_UP_SCALAR,
            ops::FP8_GROUPED_DOWN_SCALAR,
        )
    };
    let (offsets, sorted_ids, active, count, perm) = (
        sort.expert_offsets,
        sort.sorted_token_ids,
        sort.active_experts,
        sort.active_count,
        sort.token_to_perm,
    );
    ops::moe_fp8_grouped_sort(
        gpu,
        k.sort,
        sort,
        indices,
        te as u32,
        e as u32,
        TOP_K as u32,
        0,
    )?;
    if leg == Leg::TcW8a8 {
        let lay =
            ops::Fp8GroupedW8a8Layout::new(m, TOP_K, H, INTER, te * INTER * 4, m * INTER * 4)?;
        let (xq, xs) = (s.act.offset(lay.xq), s.act.offset(lay.xs));
        let act = (s.act, s.act.offset(lay.act_s));
        let sh = (s.sh_act, s.sh_act.offset(lay.sh_s));
        let rows = ops::Fp8GroupedW8a8Rows {
            expert_offsets: offsets,
            sorted_token_ids: sorted_ids,
            active_experts: active,
            active_count: count,
            cap,
            num_tokens: m as u32,
        };
        ops::moe_act_quant_e4m3(gpu, k.quant_w8a8, input, xq, xs, m as u32, H as u32, 0)?;
        ops::moe_expert_gate_up_act_fp8_grouped_tc_w8a8(
            gpu,
            k.gate_up_w8a8,
            xq,
            xs,
            x.gate,
            x.up,
            act,
            &rows,
            &x.sh_gate,
            &x.sh_up,
            sh,
            INTER as u32,
            H as u32,
            0,
        )?;
        ops::moe_expert_down_act_fp8_grouped_tc_w8a8(
            gpu,
            k.down_w8a8,
            act,
            x.down,
            s.down_out,
            &rows,
            sh,
            &x.sh_down,
            s.sh_down_out,
            H as u32,
            INTER as u32,
            0,
        )?;
    } else {
        ops::moe_expert_gate_up_act_fp8_grouped(
            gpu,
            k.gate_up[tc as usize],
            gu_geom,
            input,
            x.gate.0,
            x.gate.1,
            x.up.0,
            x.up.1,
            s.act,
            offsets,
            sorted_ids,
            active,
            count,
            &x.sh_gate,
            &x.sh_up,
            s.sh_act,
            INTER as u32,
            H as u32,
            cap,
            m as u32,
            0,
        )?;
        ops::moe_expert_down_act_fp8_grouped(
            gpu,
            k.down[tc as usize],
            down_geom,
            s.act,
            x.down.0,
            x.down.1,
            s.down_out,
            offsets,
            active,
            count,
            s.sh_act,
            &x.sh_down,
            s.sh_down_out,
            H as u32,
            INTER as u32,
            cap,
            m as u32,
            0,
        )?;
    }
    ops::moe_weighted_sum_blend_fp8_grouped(
        gpu,
        k.blend,
        out,
        s.down_out,
        weights,
        perm,
        s.sh_down_out,
        input,
        x.sh_gate_vec,
        H as u32,
        TOP_K as u32,
        H as u32,
        m as u32,
        0,
    )
}

fn to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
}

/// 2026-09-28: Row `r` of the batched output must equal the row dispatched alone.
fn check_rows(batched: &[u8], alone: &[Vec<u8>], m: usize) -> Result<()> {
    for (r, want) in alone.iter().enumerate().take(m) {
        let got = &batched[r * H * 2..(r + 1) * H * 2];
        ensure!(
            got == &want[..],
            "M={m}: row {r} differs from the same row alone"
        );
    }
    Ok(())
}

/// 2026-09-28: Relative L2 distance of `got` from `want`, refused above `bound` or on a
/// non-finite value.
fn check_close(got: &[u8], want: &[u8], bound: f64) -> Result<f64> {
    let (g, w) = (to_f32(got), to_f32(want));
    ensure!(
        g.iter().all(|v| v.is_finite()),
        "non-finite tensor-core output"
    );
    let num: f64 = g
        .iter()
        .zip(&w)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = w.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel = (num / den.max(f64::MIN_POSITIVE)).sqrt();
    ensure!(
        rel <= bound,
        "relative L2 {rel:.3e} from the scalar kernels exceeds {bound:.0e}"
    );
    Ok(rel)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    const SCALAR: &str = "moe_shared_expert_fused_fp8_grouped";
    const TC: &str = "moe_fp8_grouped_tc";
    const W8A8: &str = "moe_fp8_grouped_tc_w8a8";
    let k = Kernels {
        sort: gpu.kernel("moe_fp8_grouped_sort", "moe_fp8_grouped_sort")?,
        blend: gpu.kernel(
            "moe_fp8_grouped_blend",
            "moe_weighted_sum_blend_fp8_grouped",
        )?,
        gate_up: [
            gpu.kernel(SCALAR, "moe_expert_gate_up_act_fp8_grouped")?,
            gpu.kernel(TC, "moe_expert_gate_up_act_fp8_grouped_tc")?,
        ],
        down: [
            gpu.kernel(SCALAR, "moe_expert_down_act_fp8_grouped")?,
            gpu.kernel(TC, "moe_expert_down_act_fp8_grouped_tc")?,
        ],
        quant_w8a8: gpu.kernel(W8A8, "moe_act_quant_e4m3")?,
        gate_up_w8a8: gpu.kernel(W8A8, "moe_expert_gate_up_act_fp8_grouped_tc_w8a8")?,
        down_w8a8: gpu.kernel(W8A8, "moe_expert_down_act_fp8_grouped_tc_w8a8")?,
    };
    let e_count = if std::env::args().nth(1).is_some() {
        experts()
    } else {
        256
    };
    let mut rng = Rng(0x7463_2d6d_6f65_2028);
    let mut table = |n: usize, kk: usize| -> Result<(DevicePtr, DevicePtr)> {
        let mut wp = Vec::with_capacity(e_count * 8);
        let mut sp = Vec::with_capacity(e_count * 8);
        for _ in 0..e_count {
            wp.extend_from_slice(&(upload(&gpu, &fp8_bytes(&mut rng, n * kk))?.0).to_le_bytes());
            sp.extend_from_slice(&(upload(&gpu, &scale_bytes(&mut rng, n, kk))?.0).to_le_bytes());
        }
        Ok((upload(&gpu, &wp)?, upload(&gpu, &sp)?))
    };
    let (gate, up, down) = (table(INTER, H)?, table(INTER, H)?, table(H, INTER)?);
    let mut shared = |n: usize, kk: usize| -> Result<Fp8Weight> {
        let w = upload(&gpu, &fp8_bytes(&mut rng, n * kk))?;
        Ok(fp8w(w, upload(&gpu, &scale_bytes(&mut rng, n, kk))?, n, kk))
    };
    let (sh_gate, sh_up, sh_down) = (shared(INTER, H)?, shared(INTER, H)?, shared(H, INTER)?);
    let x = Experts {
        gate,
        up,
        down,
        sh_gate,
        sh_up,
        sh_down,
        sh_gate_vec: upload(&gpu, &bf16_bytes(&mut rng, H, 0.05))?,
    };
    let input = upload(&gpu, &bf16_bytes(&mut rng, MAX_M * H, 1.0))?;
    let idx = zipf_routing(&mut rng, e_count, zipf_alpha(), MAX_M, TOP_K);
    let indices = upload(
        &gpu,
        &idx.iter()
            .flat_map(|e| e.to_le_bytes())
            .collect::<Vec<u8>>(),
    )?;
    let w_bytes: Vec<u8> = (0..MAX_M * TOP_K)
        .flat_map(|_| ((rng.next() % 1000) as f32 / 1000.0).to_le_bytes())
        .collect();
    let weights = upload(&gpu, &w_bytes)?;
    let te = MAX_M * TOP_K;
    let cap = ops::fp8_grouped_active_cap(MAX_M as u32, TOP_K as u32, e_count as u32) as usize;
    let s = Scratch {
        e: e_count,
        sort: upload(
            &gpu,
            &vec![0u8; te * 12 + (e_count + 1) * 4 + (cap + 1) * 4],
        )?,
        act: upload(&gpu, &vec![0u8; te * INTER * 4])?,
        sh_act: upload(&gpu, &vec![0u8; MAX_M * INTER * 4])?,
        down_out: upload(&gpu, &vec![0u8; te * H * 2])?,
        sh_down_out: upload(&gpu, &vec![0u8; MAX_M * H * 2])?,
    };
    let out = upload(&gpu, &vec![0u8; MAX_M * H * 2 + GUARD])?;
    let read = |m: usize| -> Result<Vec<u8>> {
        gpu.synchronize(0)?;
        let mut b = vec![0u8; m * H * 2];
        gpu.copy_d2h(out, &mut b)?;
        Ok(b)
    };

    // 2026-09-28: Every row dispatched alone through each tensor-core leg.
    let mut alone: Vec<Vec<Vec<u8>>> = Vec::new();
    for leg in [Leg::Tc, Leg::TcW8a8] {
        let mut rows = Vec::with_capacity(MAX_M);
        for r in 0..MAX_M {
            let (inp, ind, wts) = (
                input.offset(r * H * 2),
                indices.offset(r * TOP_K * 4),
                weights.offset(r * TOP_K * 4),
            );
            dispatch(&gpu, &k, &x, &s, inp, ind, wts, out, 1, leg)?;
            rows.push(read(1)?);
        }
        alone.push(rows);
    }

    let mut first = true;
    for m in [1usize, 2, 3, 4, 7, 8, 9, 16, 17, 32, 48, 64] {
        dispatch(
            &gpu,
            &k,
            &x,
            &s,
            input,
            indices,
            weights,
            out,
            m,
            Leg::Scalar,
        )?;
        let scalar = read(m)?;
        let mut line = format!("M={m:2}");
        for (i, (leg, bound)) in [(Leg::Tc, MAX_REL_L2), (Leg::TcW8a8, MAX_REL_L2_W8A8)]
            .into_iter()
            .enumerate()
        {
            dispatch(&gpu, &k, &x, &s, input, indices, weights, out, m, leg)?;
            let got = read(m)?;
            if first {
                let mut bad = got.clone();
                bad[3] ^= 1;
                ensure!(
                    check_rows(&bad, &alone[i], m).is_err(),
                    "a flipped bit was admitted"
                );
                println!("KNOWN_BAD flipped-bit: refused");
                let zero = vec![0u8; got.len()];
                ensure!(
                    check_close(&zero, &scalar, bound).is_err(),
                    "a zeroed row was admitted"
                );
                println!("KNOWN_BAD zeroed-row: refused");
                first = false;
            }
            check_rows(&got, &alone[i], m)?;
            let rel = check_close(&got, &scalar, bound)?;
            line += &format!(" | leg {i}: rows == alone, rel_l2 vs scalar {rel:.2e}");
        }
        let time = |leg: Leg| -> Result<f64> {
            gpu.synchronize(0)?;
            let t = std::time::Instant::now();
            for _ in 0..20 {
                dispatch(&gpu, &k, &x, &s, input, indices, weights, out, m, leg)?;
            }
            gpu.synchronize(0)?;
            Ok(t.elapsed().as_secs_f64() * 1e6 / 20.0)
        };
        let (us_s, us_tc, us_w8a8) = (time(Leg::Scalar)?, time(Leg::Tc)?, time(Leg::TcW8a8)?);
        println!("{line} | scalar {us_s:7.1}us tc {us_tc:7.1}us w8a8 {us_w8a8:7.1}us  PASS");
    }
    println!(
        "ALL PASS: tensor-core grouped FP8 MoE decode (W8A16, W8A8) is row-invariant and within \
         {MAX_REL_L2:.0e} / {MAX_REL_L2_W8A8:.0e} of the scalar kernels, M=1..64"
    );
    Ok(())
}
