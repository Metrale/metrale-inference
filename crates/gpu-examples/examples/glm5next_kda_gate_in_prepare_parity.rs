// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of `METRALE_GLM_KDA_GATE_IN_PREPARE`: the 64-token
//! chunked scan whose prepare computes the KDA forget gate from f_b's BF16 rows
//! (`kda_chunk64_prepare_graw`, `kda_chunk64_prepare_conv_graw`) against the path it replaces
//! (`kda_gate_bf16` writing the FP32 gate rows, then `kda_chunk64_prepare` /
//! `kda_chunk64_prepare_conv` reading them), at GLM-5.3's TP=3 per-rank shape (22 heads of 128,
//! conv width 4, lower bound -5), for both input layouts and both prepares (plain and with the
//! conv inside, `METRALE_GLM_KDA_CHUNK64_CONV`). The scan is `kda_chunk64_scan` in every arm.
//!
//! Arms per case and layout: KERNEL (the gate launch, then the plain path), KERNEL+CONV (the gate
//! launch, then the conv-fused path), PREPARE and PREPARE+CONV (the `_graw` prepares).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte of KERNEL+CONV, PREPARE or PREPARE+CONV that differs
//!   from KERNEL: the core output, the recurrent state, the advanced conv state, or any slab's
//!   six factor buffers. f_b's rows are noise of scale 3 with every 1009th element +-40 (both
//!   ends of the sigmoid); `dt_bias` and `A_log` are noise, so a missing or misplaced term moves
//!   the bytes.
//! - Exits with an error unless KERNEL's core output is finite, and unless the negative control
//!   (one f_b element of row T - 2, head 5, channel 3, moved 4 across zero, on each `_graw`
//!   prepare) changes the core output and the recurrent state.
//! - Times the gate launch plus each prepare path against the `_graw` path as captured CUDA
//!   graphs, median [min..max] of 7 samples (`common/graph_timing.rs`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_kda_gate_in_prepare_parity \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_kda_chunk64_conv_arms.rs"]
mod arms;
#[path = "common/glm5next_kda_chunk64_case.rs"]
mod cases;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result, ensure};
use arms::{Arm, Gate, GateRaw, H, In, Kernels, Layout, QKV, fused, unfused};
use cases::{bumped_copy, case_rows, compare, ensure_finite, free_inputs, inputs, run_arm};
use fixture::{Lcg, up_bf16, up_f32};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::chunk64::CHUNK64_D as D;

/// 2026-10-14: `linear_attn_config.gate_lower_bound` of GLM-5.3-Flash.
const LOWER_BOUND: f32 = -5.0;

/// 2026-10-14: The gate kernel and the two `_graw` prepares.
struct GateKernels {
    gate: KernelHandle,
    prep: KernelHandle,
    prep_conv: KernelHandle,
}

/// 2026-10-14: f_b's BF16 rows `[t, QKV]`, `dt_bias` `[QKV]` and `A_log` `[H]` on the device.
fn gate_inputs(g: &dyn GpuBackend, rng: &mut Lcg, t: usize) -> Result<GateRaw> {
    let mut raw = rng.vec(t * QKV, 3.0);
    for (i, x) in raw.iter_mut().enumerate() {
        if i % 1009 == 0 {
            *x = if i % 2 == 0 { 40.0 } else { -40.0 };
        }
    }
    Ok(GateRaw {
        g_raw: up_bf16(g, &raw)?,
        dt_bias: up_f32(g, &rng.vec(QKV, 1.0))?,
        a_log: up_f32(g, &rng.vec(H, 1.0))?,
        lower_bound: LOWER_BOUND,
    })
}

/// 2026-10-14: `kda_gate_bf16` into `x.gate`, as the front end launches it.
fn gate_launch(g: &dyn GpuBackend, gk: &GateKernels, r: &GateRaw, x: &In, s: u64) -> Result<()> {
    KernelLaunch::new(g, gk.gate)
        .grid([(x.t * H) as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(r.g_raw)
        .arg_ptr(r.dt_bias)
        .arg_ptr(r.a_log)
        .arg_ptr(x.gate)
        .arg_u32(x.t as u32)
        .arg_u32(H as u32)
        .arg_u32(D as u32)
        .arg_f32(r.lower_bound)
        .launch(s)
}

/// 2026-10-14: The four paths of a case.
#[derive(Clone, Copy, Debug)]
enum Path {
    Kernel,
    KernelConv,
    Prepare,
    PrepareConv,
}

/// 2026-10-14: One case and layout: what every path launches on.
struct Ctx<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Kernels,
    gk: &'a GateKernels,
    x: &'a In,
    a: &'a Arm,
    layout: Layout,
}

impl Ctx<'_> {
    /// 2026-10-14: `path` on f_b's rows `r`, each slab's factors pushed to `rec` when given.
    fn go(
        &self,
        path: Path,
        r: &GateRaw,
        s: u64,
        rec: Option<&mut Vec<[Vec<u8>; 6]>>,
    ) -> Result<()> {
        let (g, k, gk, x, a, layout) = (self.g, self.k, self.gk, self.x, self.a, self.layout);
        let src = x.src(layout);
        match path {
            Path::Kernel => {
                gate_launch(g, gk, r, x, s)?;
                unfused(g, k, k.prep, Gate::Rows, x, a, layout, s, rec)
            }
            Path::KernelConv => {
                gate_launch(g, gk, r, x, s)?;
                fused(g, k, k.prep_conv, Gate::Rows, x, a, layout, src, s, rec)
            }
            Path::Prepare => unfused(g, k, gk.prep, Gate::Raw(*r), x, a, layout, s, rec),
            Path::PrepareConv => {
                fused(g, k, gk.prep_conv, Gate::Raw(*r), x, a, layout, src, s, rec)
            }
        }
    }
}

fn case(
    g: &dyn GpuBackend,
    s: u64,
    k: &Kernels,
    gk: &GateKernels,
    rng: &mut Lcg,
    t: usize,
) -> Result<()> {
    let x = inputs(g, rng, t)?;
    let r = gate_inputs(g, rng, t)?;
    let a = Arm::new(g, &x)?;
    for layout in [Layout::Packed, Layout::Parts] {
        let what = format!("T={t} {layout:?}");
        let c = Ctx {
            g,
            k,
            gk,
            x: &x,
            a: &a,
            layout,
        };
        let run = |path, r: &GateRaw| run_arm(g, s, &x, &a, &mut |rec| c.go(path, r, s, rec));
        let base = run(Path::Kernel, &r)?;
        ensure_finite(&what, &base)?;
        for path in [Path::KernelConv, Path::Prepare, Path::PrepareConv] {
            compare(
                &format!("{what} {path:?} vs Kernel"),
                &base,
                &run(path, &r)?,
            )?;
        }
        // 2026-10-14: Negative control: one f_b element of row T - 2 (head 5, channel 3)
        // moved 4 across zero, on each _graw prepare; the core and the recurrent state must move.
        let at = ((t - 2) * QKV + 5 * D + 3) * 2;
        let bad = GateRaw {
            g_raw: bumped_copy(g, r.g_raw, t * QKV * 2, at)?,
            ..r
        };
        let mut caught = true;
        for path in [Path::Prepare, Path::PrepareConv] {
            let ctl = run(path, &bad)?;
            let cmp = compare(&format!("{what} CONTROL {path:?} vs Kernel"), &base, &ctl);
            caught &= cmp.is_err() && ctl.core != base.core && ctl.rec != base.rec;
        }
        g.free(bad.g_raw)?;
        ensure!(
            caught,
            "{what}: the negative control was not caught (compare, core and recurrent state must differ on both prepares)"
        );
        println!(
            "{what}: Kernel = KernelConv = Prepare = PrepareConv byte for byte ({} slabs); NEG CONTROL: differs (core, recurrent state) on both prepares - caught",
            base.slabs.len()
        );
        let copies = if t <= 300 { 8 } else { 2 };
        for (old, new) in [
            (Path::Kernel, Path::Prepare),
            (Path::KernelConv, Path::PrepareConv),
        ] {
            let t_old = timing::time_graph(g, s, copies, &mut |s| c.go(old, &r, s, None))?;
            let t_new = timing::time_graph(g, s, copies, &mut |s| c.go(new, &r, s, None))?;
            timing::report(&format!("{new:?} {layout:?}"), t, t_old, t_new);
        }
    }
    a.free(g)?;
    for p in [r.g_raw, r.dt_bias, r.a_log] {
        g.free(p)?;
    }
    free_inputs(g, x)
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let k = Kernels::load(g)?;
    let gk = GateKernels {
        gate: g.kernel("kda_gate", "kda_gate_bf16")?,
        prep: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_graw")?,
        prep_conv: g.kernel("kda_chunk64_prepare", "kda_chunk64_prepare_conv_graw")?,
    };
    let mut rng = Lcg(0x6761_7465_5f70_7265);
    for t in case_rows()? {
        case(g, s, &k, &gk, &mut rng, t)?;
    }
    println!(
        "PASS: the gate inside the chunk64 prepare is byte-identical to the gate launch then the prepare"
    );
    Ok(())
}
