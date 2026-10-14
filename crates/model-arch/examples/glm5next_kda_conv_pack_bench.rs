// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: Byte gate and microbench of `METRALE_GLM_KDA_CONV_PACK`: the KDA q/k/v pack
//! (`kda_pack_qkv_bf16`) then the per-sequence conv (`causal_conv1d_update_l2norm_tokens`) and
//! window advance (old) against `causal_conv1d_update_l2norm_tokens_parts` and the same advance
//! (new), at GLM-5.3's TP=3 per-rank shape (22 heads of 128, conv width 4), at 1, 12, 2736 and
//! 8208 rows of one sequence, the new kernel at every row-span count from two to six blocks
//! per SM.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch of the packed rows, the conv output or the
//!   advanced conv state; the inputs carry spikes (+-6e4, zeros) next to the random rows, so the
//!   SiLU's overflow and the L2 of a zero head are compared too. Width 2 and 3 are gated at 12
//!   rows (the kernel's other template arms).
//! - Times each arm as a captured CUDA graph, median [min..max] of 7 samples
//!   (`common/graph_timing.rs`).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_conv_pack_bench \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Context, Result};
use fixture::{Lcg, same, up_bf16, up_f32};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-10-13: Per-rank KDA heads at TP=3 (64 heads split 22/21/21), head_dim 128.
const H: usize = 22;
const D: usize = 128;
/// 2026-10-13: One row, a K=12 verify, and two prefill sub-chunks (2736 = 8208 / 3).
const ROWS: [usize; 4] = [1, 12, 2736, 8208];
/// 2026-10-13: Copies per captured graph: the 34 KDA layers at decode widths, fewer for the
/// wide sub-chunks (each copy runs for milliseconds).
fn copies(rows: usize) -> usize {
    if rows <= 16 { 34 } else { 4 }
}

struct Kernels {
    pack: KernelHandle,
    conv: KernelHandle,
    window: KernelHandle,
    parts: KernelHandle,
}

/// 2026-10-13: One arm's buffers: its conv state, packed rows and conv output.
#[derive(Clone, Copy)]
struct Arm {
    state: DevicePtr,
    packed: DevicePtr,
    out: DevicePtr,
}

struct Case {
    rows: usize,
    kc: usize,
    parts: DevicePtr,
    weight: DevicePtr,
}

fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}

fn read(g: &dyn GpuBackend, s: u64, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(s)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn window(g: &dyn GpuBackend, k: &Kernels, c: &Case, a: Arm, s: u64) -> Result<()> {
    let cd = 3 * H * D;
    KernelLaunch::new(g, k.window)
        .grid([cd.div_ceil(256) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a.state)
        .arg_ptr(a.packed)
        .arg_u32(cd as u32)
        .arg_u32(c.kc as u32)
        .arg_u32(c.rows as u32)
        .launch(s)
}

fn old(g: &dyn GpuBackend, k: &Kernels, c: &Case, a: Arm, s: u64) -> Result<()> {
    let (qkv, t) = (H * D, c.rows);
    let cd = 3 * qkv;
    KernelLaunch::new(g, k.pack)
        .grid([qkv.div_ceil(256) as u32, t as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(c.parts)
        .arg_ptr(c.parts.offset(t * qkv * 2))
        .arg_ptr(c.parts.offset(2 * t * qkv * 2))
        .arg_ptr(a.packed)
        .arg_u32(t as u32)
        .arg_u32(qkv as u32)
        .launch(s)?;
    KernelLaunch::new(g, k.conv)
        .grid([cd.div_ceil(256) as u32, t as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a.state)
        .arg_ptr(a.packed)
        .arg_ptr(c.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(a.out)
        .arg_u32(cd as u32)
        .arg_u32(c.kc as u32)
        .arg_u32((2 * qkv) as u32)
        .arg_u32(D as u32)
        .arg_f32(1e-6)
        .launch(s)?;
    window(g, k, c, a, s)
}

fn new(g: &dyn GpuBackend, k: &Kernels, c: &Case, a: Arm, spans: u32, s: u64) -> Result<()> {
    let (qkv, t) = (H * D, c.rows);
    let cd = 3 * qkv;
    KernelLaunch::new(g, k.parts)
        .grid([cd.div_ceil(256) as u32, spans, 1])
        .block([256, 1, 1])
        .arg_ptr(a.state)
        .arg_ptr(c.parts)
        .arg_ptr(c.parts.offset(t * qkv * 2))
        .arg_ptr(c.parts.offset(2 * t * qkv * 2))
        .arg_ptr(c.weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(a.packed)
        .arg_ptr(a.out)
        .arg_u32(cd as u32)
        .arg_u32(qkv as u32)
        .arg_u32(t as u32)
        .arg_u32(c.kc as u32)
        .arg_u32((2 * qkv) as u32)
        .arg_u32(D as u32)
        .arg_f32(1e-6)
        .launch(s)?;
    window(g, k, c, a, s)
}

/// 2026-10-13: `n` random values with spikes: every 997th +-6e4, every 1009th zero, and one
/// whole head-sized run of zeros at the start of each row (a zero L2 head).
fn inputs(rng: &mut Lcg, rows: usize, part: usize) -> Vec<f32> {
    let mut v = rng.vec(3 * rows * part, 2.0);
    for (i, x) in v.iter_mut().enumerate() {
        if i % 997 == 0 {
            *x = if i % 2 == 0 { 6e4 } else { -6e4 };
        } else if i % 1009 == 0 || i % part < D {
            *x = 0.0;
        }
    }
    v
}

/// 2026-10-13: Gate every span count, then (at width 4) time old against each.
fn case(
    g: &dyn GpuBackend,
    s: u64,
    k: &Kernels,
    rng: &mut Lcg,
    rows: usize,
    kc: usize,
) -> Result<()> {
    let (qkv, sms) = (H * D, g.sm_count()?);
    let cd = 3 * qkv;
    let c = Case {
        rows,
        kc,
        parts: up_bf16(g, &inputs(rng, rows, qkv))?,
        weight: up_bf16(g, &rng.vec(cd * kc, 0.5))?,
    };
    let state0 = rng.vec(cd * kc, 1.0);
    let arm = || -> Result<Arm> {
        Ok(Arm {
            state: up_f32(g, &state0)?,
            packed: zeros(g, rows * cd * 2)?,
            out: zeros(g, rows * cd * 2)?,
        })
    };
    let a = arm()?;
    old(g, k, &c, a, s)?;
    let want = [
        read(g, s, a.packed, rows * cd * 2)?,
        read(g, s, a.out, rows * cd * 2)?,
        read(g, s, a.state, cd * kc * 4)?,
    ];
    let grid_x = cd.div_ceil(256) as u32;
    let mut arms = Vec::new();
    for bps in 2..=6u32 {
        let spans = (sms * bps).div_ceil(grid_x);
        let b = arm()?;
        new(g, k, &c, b, spans, s)?;
        for (i, (what, p, n)) in [
            ("packed", b.packed, rows * cd * 2),
            ("conv out", b.out, rows * cd * 2),
            ("conv state", b.state, cd * kc * 4),
        ]
        .into_iter()
        .enumerate()
        {
            same(
                &format!("conv+pack rows={rows} width={kc} spans={spans} {what}"),
                &want[i],
                &read(g, s, p, n)?,
            )?;
        }
        arms.push((bps, spans, b));
    }
    if kc == 4 {
        let n = copies(rows);
        let t_old = timing::time_graph(g, s, n, &mut |s| old(g, k, &c, a, s))?;
        for &(bps, spans, b) in &arms {
            let t_new = timing::time_graph(g, s, n, &mut |s| new(g, k, &c, b, spans, s))?;
            timing::report(&format!("pack+conv {bps}/SM G={spans}"), rows, t_old, t_new);
        }
    }
    for p in [c.parts, c.weight, a.state, a.packed, a.out]
        .into_iter()
        .chain(arms.iter().flat_map(|(_, _, b)| [b.state, b.packed, b.out]))
    {
        g.free(p)?;
    }
    Ok(())
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
    let k = Kernels {
        pack: g.kernel("kda_layer_ops", "kda_pack_qkv_bf16")?,
        conv: g.kernel("kda_conv_tokens", "causal_conv1d_update_l2norm_tokens")?,
        window: g.kernel("kda_conv_tokens", "causal_conv1d_window_advance")?,
        parts: g.kernel(
            "kda_conv_tokens",
            "causal_conv1d_update_l2norm_tokens_parts",
        )?,
    };
    let mut rng = Lcg(0x6b64_6170_6b01);
    for rows in ROWS {
        case(g, s, &k, &mut rng, rows, 4)?;
    }
    for kc in [2, 3] {
        case(g, s, &k, &mut rng, 12, kc)?;
    }
    println!("PASS: the fused pack + conv is byte-identical to the pack then the conv");
    Ok(())
}
