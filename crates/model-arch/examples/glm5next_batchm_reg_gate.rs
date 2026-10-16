// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: Byte gate and microbench for the register-resident batched BF16 GEMV
//! (`dense_gemv_bf16_batchm_reg8` and `_reg8_fp32out` at 1..=8 rows, `dense_gemv_bf16_batchm_wide`
//! and `_wide_fp32out` at 9..=16), against the
//! runtime-M entries (`dense_gemv_bf16_batchm`, `_fp32out`) whose bits decode produces today,
//! and for BF16 output against the M = 1 GEMV of each row. GLM-5.3 per-rank shapes: router
//! logits 288 x 4096 and DSA indexer wq_b 4096 x 1536 (FP32 out), the vocab-TP=3 LM head shard
//! 51627 x 4096 and KDA q/k/v 2816 x 4096 (BF16 out), plus 301 x 2816 (N % 4 = 1; the head
//! shard has N % 4 = 3), so partial last blocks are covered. K stays a multiple of 8, as every
//! caller requires (the A rows are read as 16-byte vectors).
//!
//! Negative control: one input element of the last row is changed and the register result of
//! that row must then differ from the reference (so the comparison can fail), while every
//! other row must still match (rows do not interact); at M = 8 (reg8) and M = 16 (wide).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch or a control that does not trip; prints
//!   microseconds per call otherwise.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_batchm_reg_gate \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const ITERS: usize = 50;
const MAX_M: usize = 16;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bf16_bytes(rng: &mut Lcg, n: usize, s: f32) -> Vec<u8> {
    (0..n)
        .flat_map(|_| bf16::from_f32(rng.f() * s).to_le_bytes())
        .collect()
}
fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn zeros(g: &dyn GpuBackend, n: usize) -> Result<DevicePtr> {
    let p = g.alloc(n.max(1))?;
    g.memset(p, 0, n)?;
    Ok(p)
}
fn read(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    g.synchronize(0)?;
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn time(g: &dyn GpuBackend, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..3 {
        f()?;
    }
    g.synchronize(0)?;
    let t = std::time::Instant::now();
    for _ in 0..ITERS {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t.elapsed().as_secs_f64() * 1e6 / ITERS as f64)
}
fn same(what: &str, a: &[u8], b: &[u8]) -> Result<()> {
    if a != b {
        let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap_or(0);
        bail!("{what}: bytes differ at {i}");
    }
    Ok(())
}

/// 2026-10-16: The batched launch every caller uses: grid (ceil(N / 4), 1, 1), out_stride N.
#[allow(clippy::too_many_arguments)]
fn batched(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(kk as u32)
        .arg_u32(n as u32)
        .launch(0)
}

struct Entries {
    gemv: KernelHandle,
    narrow: KernelHandle,
    reg8: KernelHandle,
    wide: KernelHandle,
}

impl Entries {
    /// 2026-10-16: The register entry for `m` rows (`ops::DenseGemvBatchmPair::for_rows`'s bands).
    fn reg(&self, m: usize) -> KernelHandle {
        if m <= 8 { self.reg8 } else { self.wide }
    }
}

/// 2026-10-16: One shape: every M in 1..=16 byte-compared, then the negative controls.
fn shape(
    g: &dyn GpuBackend,
    rng: &mut Lcg,
    e: &Entries,
    (n, kk): (usize, usize),
    out: usize,
) -> Result<()> {
    let b = up(g, &bf16_bytes(rng, n * kk, 0.03))?;
    let a_host = bf16_bytes(rng, MAX_M * kk, 1.0);
    let a = up(g, &a_host)?;
    let row = n * out;
    let mut ref1 = Vec::new();
    if out == 2 {
        let c1 = zeros(g, MAX_M * row)?;
        for r in 0..MAX_M {
            KernelLaunch::new(g, e.gemv)
                .grid([n.div_ceil(4) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a.offset(r * kk * 2))
                .arg_ptr(b)
                .arg_ptr(c1.offset(r * row))
                .arg_u32(n as u32)
                .arg_u32(kk as u32)
                .launch(0)?;
        }
        ref1 = read(g, c1, MAX_M * row)?;
    }
    print!("  {n:>5}x{kk:<5} out={}B", out);
    for m in 1..=MAX_M {
        let (cn, cr) = (zeros(g, m * row)?, zeros(g, m * row)?);
        let tn = time(g, || batched(g, e.narrow, a, b, cn, m, n, kk))?;
        let tr = time(g, || batched(g, e.reg(m), a, b, cr, m, n, kk))?;
        let (vn, vr) = (read(g, cn, m * row)?, read(g, cr, m * row)?);
        same(&format!("reg vs runtime-M, M={m} {n}x{kk}"), &vr, &vn)?;
        if out == 2 {
            same(
                &format!("reg vs M=1 GEMV, M={m} {n}x{kk}"),
                &vr,
                &ref1[..m * row],
            )?;
        }
        print!(" | M={m} {tn:.1}->{tr:.1}");
    }
    println!();
    // 2026-10-16: Negative controls, at the widest M of each register entry: the last row's
    // first element becomes 8.0 (4.0 if it was 8.0 already), so at least one of its N outputs
    // moves, while the rows above it must not.
    for mc in [8, MAX_M] {
        let mut a_bad = a_host.clone();
        let at = (mc - 1) * kk * 2;
        let eight = bf16::from_f32(8.0).to_le_bytes();
        let new = if a_bad[at..at + 2] == eight {
            bf16::from_f32(4.0).to_le_bytes()
        } else {
            eight
        };
        a_bad[at..at + 2].copy_from_slice(&new);
        let a2 = up(g, &a_bad)?;
        let (cn, cr) = (zeros(g, mc * row)?, zeros(g, mc * row)?);
        batched(g, e.narrow, a, b, cn, mc, n, kk)?;
        batched(g, e.reg(mc), a2, b, cr, mc, n, kk)?;
        let (vn, vr) = (read(g, cn, mc * row)?, read(g, cr, mc * row)?);
        let last = (mc - 1) * row;
        same(
            &format!("control M={mc}: untouched rows {n}x{kk}"),
            &vr[..last],
            &vn[..last],
        )?;
        ensure!(
            vr[last..] != vn[last..],
            "control M={mc}: a changed input row compared equal ({n}x{kk}) -- the comparison cannot fail"
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let g = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut rng = Lcg(0x6c6d_ba7c_0016);
    let m = "dense_gemv_bf16_batchm";
    let bf = Entries {
        gemv: g.kernel("gemv", "dense_gemv_bf16")?,
        narrow: g.kernel(m, "dense_gemv_bf16_batchm")?,
        reg8: g.kernel(m, "dense_gemv_bf16_batchm_reg8")?,
        wide: g.kernel(m, "dense_gemv_bf16_batchm_wide")?,
    };
    let f32o = Entries {
        gemv: KernelHandle(0),
        narrow: g.kernel(m, "dense_gemv_bf16_batchm_fp32out")?,
        reg8: g.kernel(m, "dense_gemv_bf16_batchm_reg8_fp32out")?,
        wide: g.kernel(m, "dense_gemv_bf16_batchm_wide_fp32out")?,
    };
    println!("us/call runtime-M -> register entry");
    for s in [(288, 4096), (4096, 1536), (301, 2816)] {
        shape(&g, &mut rng, &f32o, s, 4)?;
    }
    for s in [(51627, 4096), (2816, 4096), (301, 2816)] {
        shape(&g, &mut rng, &bf, s, 2)?;
    }
    println!("PASS: register entry byte-identical at M = 1..=16; negative controls tripped");
    Ok(())
}
