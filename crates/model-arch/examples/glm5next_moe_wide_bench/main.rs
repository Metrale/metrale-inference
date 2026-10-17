// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Microbench and byte gate of GLM-5.3's routed-expert sweep at wide decode rows on
//! the declared W4A4 path, at the per-rank shapes of the TP=3 / EP=3 serve (hidden 4096, expert
//! width 2048, 288 experts of which this rank holds 0..96, top 8).
//!
//! 1. Byte gate, at 1, 4, 8 and 16 rows of random routing: the union tables of
//!    `glm5next_moe_row_union` against a host port of the serial builder, and every (row, slot)
//!    of the union GEMV (gate, up, down) against the slot GEMV `w4a4_gemv_mx8_moe_slots`.
//! 2. Timing (CUDA-graph replay, median of `REPS` replays of `ITERS` launches, with min and max):
//!    the union build, each union GEMV with its effective GB/s over the local union experts'
//!    bytes, a launch whose entries are all empty, and a device copy as the DRAM reference.
//! 3. `forward_moe` at W4A4, FNV-1a of the output bytes at 1, 4, 8 and 16 rows: equal digests
//!    on two builds mean equal outputs for every token.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on the first byte mismatch.
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! GLM_BENCH_GPU_ORDINAL=0 cargo run -p metrale-model-arch --release \
//!     --example glm5next_moe_wide_bench --features cuda,gpu-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_mlp::build::build_moe;
use metrale_model_arch::glm5next_mlp::forward::{Glm5NextMlpWorkspace, forward_moe};
use metrale_model_arch::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use metrale_model_arch::glm5next_mlp::weights::Glm5NextExpertWeights;
use metrale_model_arch::glm5next_mlp::{
    Glm5NextMlpConfig, Glm5NextMlpKernels, W4A4_SWEEP_CTAS_PER_SM,
};

mod device;
use device::*;

struct Setup {
    kn: Kern,
    gate: Table,
    upt: Table,
    down: Table,
    x_act: Act,
    d_act: Act,
    out: DevicePtr,
    out_ref: DevicePtr,
}

/// 2026-10-09: Part 1: tables and every (row, slot) of the three union GEMVs, byte for byte.
fn byte_gate(gpu: &dyn GpuBackend, s: &Setup, rng: &mut Rng) -> Result<()> {
    let out_bytes = MAX_ROWS * TOP_K * HIDDEN * 2;
    for rows in [1usize, 4, 8, 16] {
        let r = route(gpu, rng, rows)?;
        row_union(gpu, &s.kn, &r, 0)?;
        let (want_eid, want_slot) = host_union(&r.host, rows);
        ensure!(
            read_i32(gpu, r.u_eid, rows * TOP_K)? == want_eid
                && read_i32(gpu, r.u_slot, rows * TOP_K * rows)? == want_slot,
            "rows={rows}: union tables differ from the serial builder"
        );
        for (name, t, nka, a) in [
            ("gate", &s.gate, (MI, HIDDEN, TOP_K), &s.x_act),
            ("up", &s.upt, (MI, HIDDEN, TOP_K), &s.x_act),
            ("down", &s.down, (HIDDEN, MI, 1), &s.d_act),
        ] {
            gpu.memset(s.out, 0, out_bytes)?;
            gpu.memset(s.out_ref, 0, out_bytes)?;
            union_gemv(gpu, union_kernel(&s.kn, rows), a, &r, t, s.out, nka, 0)?;
            slots_gemv(gpu, &s.kn, a, &r, t, s.out_ref, nka)?;
            let n = rows * TOP_K * nka.0 * 2;
            let (got, want) = (read(gpu, s.out, n)?, read(gpu, s.out_ref, n)?);
            ensure!(
                got.iter().any(|&b| b != 0),
                "rows={rows} {name}: union wrote nothing"
            );
            ensure!(
                got == want,
                "rows={rows} {name}: union differs from the slot GEMV"
            );
            gpu.memset(s.out, 0, out_bytes)?;
            sweep_gemv(gpu, &s.kn, a, &r, &[t], &[s.out], nka, 0)?;
            ensure!(
                read(gpu, s.out, n)? == want,
                "rows={rows} {name}: sweep differs"
            );
        }
        // 2026-10-09: Gate and up in one sweep, each against its slot GEMV output.
        let n = rows * TOP_K * MI * 2;
        let gu = (MI, HIDDEN, TOP_K);
        gpu.memset(s.out, 0, out_bytes)?;
        gpu.memset(s.out_ref, 0, out_bytes)?;
        sweep_gemv(
            gpu,
            &s.kn,
            &s.x_act,
            &r,
            &[&s.gate, &s.upt],
            &[s.out, s.out_ref],
            gu,
            0,
        )?;
        let (g2, u2) = (read(gpu, s.out, n)?, read(gpu, s.out_ref, n)?);
        for (name, t, got) in [("gate", &s.gate, g2), ("up", &s.upt, u2)] {
            gpu.memset(s.out, 0, out_bytes)?;
            slots_gemv(gpu, &s.kn, &s.x_act, &r, t, s.out, gu)?;
            ensure!(
                read(gpu, s.out, n)? == got,
                "rows={rows} fused {name}: differs"
            );
        }
        println!("byte gate rows={rows:2}: tables; gate, up, down (grid, sweep, fused) identical");
    }
    Ok(())
}

/// 2026-10-09: Part 2: timings at 4, 8 and 16 rows.
fn timing(gpu: &dyn GpuBackend, s: &Setup, rng: &mut Rng, stream: u64) -> Result<()> {
    let big = 256usize << 20;
    let (src, dst) = (gpu.alloc(big)?, gpu.alloc(big)?);
    let cp = time(gpu, stream, &|| gpu.copy_d2d_async(src, dst, big, stream))?;
    println!(
        "device copy 256 MiB: {}  ({:.0} GB/s read + write)",
        fmt(cp),
        2.0 * big as f64 / cp[0] / 1e3
    );
    for rows in [4usize, 8, 16] {
        time_route(gpu, s, &route(gpu, rng, rows)?, stream)?;
    }
    Ok(())
}

/// 2026-10-09: Every candidate on one routing, interleaved.
fn time_route(gpu: &dyn GpuBackend, s: &Setup, r: &Route, stream: u64) -> Result<()> {
    let (rows, nloc) = (r.rows, local_union(r));
    println!(
        "rows={rows:2}: {} ids, {nloc} local union experts",
        rows * TOP_K
    );
    row_union(gpu, &s.kn, r, stream)?;
    let kern = union_kernel(&s.kn, rows);
    let empty = Route {
        rows,
        ids: r.ids,
        u_eid: gpu.alloc(rows * TOP_K * 4)?,
        u_slot: r.u_slot,
        host: Vec::new(),
    };
    gpu.memset(empty.u_eid, 0xFF, rows * TOP_K * 4)?;
    let (gu, dn) = ((MI, HIDDEN, TOP_K), (HIDDEN, MI, 1));
    let (xa, da, o2) = (&s.x_act, &s.d_act, [s.out, s.out_ref]);
    let t = time_set(
        gpu,
        stream,
        &[
            &|| row_union(gpu, &s.kn, r, stream),
            &|| union_gemv(gpu, kern, xa, r, &s.gate, s.out, gu, stream),
            &|| union_gemv(gpu, kern, xa, r, &s.upt, s.out, gu, stream),
            &|| union_gemv(gpu, kern, da, r, &s.down, s.out, dn, stream),
            &|| sweep_gemv(gpu, &s.kn, xa, r, &[&s.gate], &o2[..1], gu, stream),
            &|| sweep_gemv(gpu, &s.kn, xa, r, &[&s.gate, &s.upt], &o2, gu, stream),
            &|| sweep_gemv(gpu, &s.kn, da, r, &[&s.down], &o2[..1], dn, stream),
            &|| union_gemv(gpu, kern, xa, &empty, &s.gate, s.out, gu, stream),
        ],
    )?;
    let b1 = nloc * s.gate.bytes_per_expert;
    for (i, name, bytes) in [
        (0, "row_union", 0),
        (1, "grid gate", b1),
        (2, "grid up", b1),
        (3, "grid down", b1),
        (4, "sweep gate", b1),
        (5, "sweep gate+up", 2 * b1),
        (6, "sweep down", b1),
        (7, "grid gate, all empty", 0),
    ] {
        let gbs = bytes as f64 / t[i][0] / 1e3;
        println!("  {name:21} {}  {gbs:6.1} GB/s", fmt(t[i]));
    }
    println!(
        "  old gate+up+down {:.1} us, sweep gate+up + down {:.1} us (medians)",
        t[1][0] + t[2][0] + t[3][0],
        t[5][0] + t[6][0]
    );
    Ok(())
}

/// 2026-10-09: Part 3: `forward_moe` digests at W4A4.
fn digests(gpu: &MetraleCudaBackend, s: &Setup, rng: &mut Rng, stream: u64) -> Result<()> {
    let c = Glm5NextMlpConfig {
        hidden: HIDDEN,
        local_dense_intermediate: 4096,
        dense_start: 0,
        moe_intermediate: MI,
        local_shared_intermediate: 688,
        shared_start: 0,
        num_experts: EXPERTS,
        local_experts: LOCAL,
        ep_rank: 0,
        top_k: TOP_K,
        routed_scale: 2.5,
        renormalize: true,
        swiglu_limit: 10.0,
        router_bf16_ladder: false,
        tp_world_size: 3,
        ep_world_size: 3,
        expert_shard: metrale_model_arch::glm5next_mlp::ExpertShard::Whole,
    };
    let k = Glm5NextMlpKernels::resolve(gpu)?;
    let router: Vec<f32> = (0..EXPERTS * HIDDEN).map(|_| rng.unit() * 0.05).collect();
    let shared: Vec<f32> = (0..HIDDEN * 688 * 3).map(|_| rng.unit() * 0.02).collect();
    let load = |n: &str| -> Result<Vec<f32>> {
        Ok(match n {
            "mlp.gate.weight" => router.clone(),
            "mlp.gate.e_score_correction_bias" => vec![0.0; EXPERTS],
            _ => shared.clone(),
        })
    };
    let expert = |id: usize| -> Result<Glm5NextExpertWeights> {
        Ok(Glm5NextExpertWeights {
            gate_proj: s.gate.projs[id],
            up_proj: s.upt.projs[id],
            down_proj: s.down.projs[id],
        })
    };
    let w = build_moe(
        gpu,
        &c,
        688 * 3,
        &load,
        &expert,
        &|has| {
            GroupPrecision::resolve(
                MlpGroup::RoutedExperts,
                ActivationQuantization::default()
                    .ladder(ProjFamily::Moe)
                    .clone(),
                Nvfp4Act::A4,
                k.w4a4_expert_rows(),
                has,
            )
        },
        MAX_ROWS,
    )?;
    let ws = Glm5NextMlpWorkspace::new(gpu, &c, MAX_ROWS)?;
    let x: Vec<f32> = (0..MAX_ROWS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let xd = up_bf16(gpu, &x)?;
    for rows in [1usize, 4, 8, 16] {
        forward_moe(gpu, &k, &c, &w, xd, s.out, rows, &ws, false, stream)?;
        gpu.synchronize(stream)?;
        let b = read(gpu, s.out, rows * HIDDEN * 2)?;
        println!("forward_moe W4A4 rows={rows:2}: fnv1a {:016x}", fnv1a(&b));
        let t = time(gpu, stream, &|| {
            forward_moe(gpu, &k, &c, &w, xd, s.out, rows, &ws, true, stream)
        })?;
        println!("  forward_moe          {}", fmt(t));
    }
    Ok(())
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let stream = gpu.create_stream()?;
    let g: &dyn GpuBackend = &gpu;
    let kn = Kern {
        quant: g.kernel("w4a4_gemv_mx_moe", "w4a4_quant_rows_static")?,
        row_union: g.kernel("w4a16_gemv", "glm5next_moe_row_union")?,
        slots: g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_slots")?,
        union: [
            g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_union")?,
            g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx16_moe_union")?,
        ],
        sweep: [
            g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx8_moe_union_sweep")?,
            g.kernel("w4a4_gemv_mx_moe", "w4a4_gemv_mx16_moe_union_sweep")?,
        ],
        ctas: W4A4_SWEEP_CTAS_PER_SM * g.sm_count()?,
    };
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let (gate, upt) = (
        table(g, &mut rng, MI, HIDDEN, GS_GATE_UP)?,
        table(g, &mut rng, MI, HIDDEN, GS_GATE_UP)?,
    );
    let down = table(g, &mut rng, HIDDEN, MI, GS_DOWN)?;
    let x: Vec<f32> = (0..MAX_ROWS * HIDDEN).map(|_| rng.unit() * 3.0).collect();
    let act: Vec<f32> = (0..MAX_ROWS * TOP_K * MI)
        .map(|_| rng.unit() * 10.0)
        .collect();
    let (xd, ad) = (up_bf16(g, &x)?, up_bf16(g, &act)?);
    let x_act = quantize(g, &kn, xd, MAX_ROWS, HIDDEN, GS_GATE_UP)?;
    let d_act = quantize(g, &kn, ad, MAX_ROWS * TOP_K, MI, GS_DOWN)?;
    let out_bytes = MAX_ROWS * TOP_K * HIDDEN * 2;
    let s = Setup {
        kn,
        gate,
        upt,
        down,
        x_act,
        d_act,
        out: g.alloc(out_bytes)?,
        out_ref: g.alloc(out_bytes)?,
    };
    byte_gate(g, &s, &mut rng)?;
    timing(g, &s, &mut rng, stream)?;
    digests(&gpu, &s, &mut rng, stream)
}
