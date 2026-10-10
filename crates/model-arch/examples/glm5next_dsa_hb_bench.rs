// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Gate and microbench for the head-batched DSA decode
//! (`glm5next_dsa_mla_decode_hb`) against the per-head `glm5next_dsa_mla_decode_fp8`, at the
//! per-rank shapes of the TP=3 layout (22 and 21 heads) plus 32 and 64 heads (head groups).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error when, for any case: the head-batched output is further from an f64
//!   reference than [`TOL_ULP`] BF16 ulps of the reference's per-head scale; a row of a
//!   multi-row launch differs in any byte from the same row launched alone (batch
//!   invariance); an all -1 row or a row whose entries are all past `seq_len` is not exactly
//!   zero; or a `seq_len` 0 row is written.
//! - Prints old vs new microseconds per launch (median, p10-p90 of [`SAMPLES`] samples of
//!   [`PER_SAMPLE`] launches). Launches rotate over [`fixture::VARIANTS`] block tables and selections
//!   into a [`fixture::POOL_BLOCKS`]-block (128 MB) latent pool, so the selected tokens are not
//!   L2-resident from the previous launch, as in a serve where the other layers run between.
//!   `HB_SPLITS=a,b,..` picks the split counts; `HB_PTX=<file>` swaps in a locally compiled
//!   `glm5next_dsa_mla_decode_hb` PTX to measure a variant without rebuilding the kernels.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_hb_bench \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_hb_fixture.rs"]
mod fixture;

use anyhow::{Context, Result, bail};
use fixture::{
    BLOCK, BLOCKS_PER_SEQ, Case, KVL, Lcg, Pool, VARIANTS, View, WIDTH, bf16s, make_case,
    make_pool, read, reference, up_i32,
};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const SAMPLES: usize = 41;
const PER_SAMPLE: usize = 5;
/// 2026-10-09: Heads per CTA of the head-batched entry.
const HB_HEADS: usize = 24;
/// 2026-10-09: Allowed distance from the f64 reference, in BF16 ulps (2^-8 relative) of the
/// head's largest |O|. The output itself rounds to half an ulp of each value; the rest is the
/// fp32 summation and the hi + lo P split. The per-head kernel measures the same (printed).
const TOL_ULP: f64 = 2.0;

struct Kernels {
    old: KernelHandle,
    hb: KernelHandle,
    merge: KernelHandle,
}

fn launch_old(
    g: &dyn GpuBackend,
    k: &Kernels,
    cache: DevicePtr,
    v: View,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k.old)
        .grid([v.heads as u32, v.rows as u32, 1])
        .block([512, 1, 1])
        .arg_ptr(v.q)
        .arg_ptr(cache)
        .arg_ptr(cache)
        .arg_ptr(out)
        .arg_ptr(v.bt)
        .arg_ptr(v.sl)
        .arg_ptr(v.sel)
        .arg_u32(WIDTH as u32)
        .arg_u32(BLOCKS_PER_SEQ as u32)
        .arg_u32(v.heads as u32)
        .arg_u32(1)
        .arg_u32(KVL as u32)
        .arg_u32(BLOCK as u32)
        .arg_f32((KVL as f32).powf(-0.5))
        .arg_f32(1.0)
        .arg_f32(1.0)
        .arg_u64((BLOCK * KVL) as u64)
        .launch(0)
}

fn launch_hb(
    g: &dyn GpuBackend,
    k: &Kernels,
    cache: DevicePtr,
    v: View,
    out: DevicePtr,
    ws: (DevicePtr, DevicePtr),
    splits: usize,
) -> Result<()> {
    KernelLaunch::new(g, k.hb)
        .grid([
            splits as u32,
            v.rows as u32,
            v.heads.div_ceil(HB_HEADS) as u32,
        ])
        .block([256, 1, 1])
        .arg_ptr(v.q)
        .arg_ptr(cache)
        .arg_ptr(out)
        .arg_ptr(v.bt)
        .arg_ptr(v.sl)
        .arg_ptr(v.sel)
        .arg_ptr(ws.0)
        .arg_ptr(ws.1)
        .arg_u32(WIDTH as u32)
        .arg_u32(BLOCKS_PER_SEQ as u32)
        .arg_u32(v.heads as u32)
        .arg_u32(BLOCK as u32)
        .arg_f32((KVL as f32).powf(-0.5))
        .arg_f32(1.0)
        .arg_f32(1.0)
        .arg_u64((BLOCK * KVL) as u64)
        .arg_u32(splits as u32)
        .launch(0)?;
    if splits > 1 {
        KernelLaunch::new(g, k.merge)
            .grid([v.heads as u32, v.rows as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(ws.0)
            .arg_ptr(ws.1)
            .arg_ptr(v.sl)
            .arg_ptr(out)
            .arg_u32(v.heads as u32)
            .arg_u32(splits as u32)
            .arg_f32(1.0)
            .launch(0)?;
    }
    Ok(())
}

/// 2026-10-09: Worst distance from the reference over the first `rows_checked` rows, in BF16
/// ulps of each head's largest |O|.
fn worst_ulp(c: &Case, pool: &Pool, out: &[f32], rows_checked: usize) -> f64 {
    let mut worst = 0f64;
    for r in 0..rows_checked {
        for h in 0..c.heads {
            let o = reference(c, pool, r, h);
            let ulp = o.iter().fold(0f64, |a, x| a.max(x.abs())).max(1e-30) / 256.0;
            for (i, x) in o.iter().enumerate() {
                worst = worst.max((out[(r * c.heads + h) * KVL + i] as f64 - x).abs() / ulp);
            }
        }
    }
    worst
}

/// 2026-10-09: Fraction of outputs bit-equal to the per-head kernel's.
fn same_as_old(new: &[u8], old: &[u8]) -> f64 {
    let n = new
        .chunks(2)
        .zip(old.chunks(2))
        .filter(|(a, b)| a == b)
        .count();
    n as f64 / (new.len() / 2) as f64
}

/// 2026-10-09: Microseconds per launch, (median, p10, p90) over the samples; `f` gets the
/// launch index, so callers rotate variants.
fn time(g: &dyn GpuBackend, mut f: impl FnMut(usize) -> Result<()>) -> Result<(f64, f64, f64)> {
    let mut it = 0usize;
    for _ in 0..3 {
        f(it)?;
        it += 1;
    }
    let mut v = Vec::new();
    for _ in 0..SAMPLES {
        g.synchronize(0)?;
        let t = std::time::Instant::now();
        for _ in 0..PER_SAMPLE {
            f(it)?;
            it += 1;
        }
        g.synchronize(0)?;
        v.push(t.elapsed().as_secs_f64() * 1e6 / PER_SAMPLE as f64);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((
        v[SAMPLES / 2],
        v[SAMPLES / 10],
        v[SAMPLES - 1 - SAMPLES / 10],
    ))
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let mut modules = glm.modules.clone();
    if let Ok(path) = std::env::var("HB_PTX") {
        let text: &'static [u8] =
            Box::leak(std::fs::read(&path).context("HB_PTX")?.into_boxed_slice());
        let slot = modules
            .iter_mut()
            .find(|m| m.0 == "glm5next_dsa_mla_decode_hb")
            .context("no glm5next_dsa_mla_decode_hb module to replace")?;
        slot.1 = text;
        println!("glm5next_dsa_mla_decode_hb from {path}");
    }
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let k = Kernels {
        old: g.kernel("glm5next_dsa_mla_decode", "glm5next_dsa_mla_decode_fp8")?,
        hb: g.kernel(
            "glm5next_dsa_mla_decode_hb",
            "glm5next_dsa_mla_decode_hb_fp8",
        )?,
        merge: g.kernel("glm5next_dsa_mla_decode_hb", "glm5next_dsa_mla_merge")?,
    };
    let splits_list: Vec<usize> = std::env::var("HB_SPLITS")
        .unwrap_or_else(|_| "1,4,8,16,32".into())
        .split(',')
        .map(|s| s.parse().context("HB_SPLITS"))
        .collect::<Result<_>>()?;
    let mut rng = Lcg(0x0D5A_4B42);
    // 2026-10-09: `HB_FP8_EXP_MAX` (default 8, max 15) widens the latent's E4M3 exponent range.
    let exp_max: u64 = std::env::var("HB_FP8_EXP_MAX").map_or(Ok(8), |v| v.parse())?;
    let pool = make_pool(g, &mut rng, exp_max.min(15))?;
    // 2026-10-09: `HB_Q_SCALE` (default 4) bounds |Q|.
    let q_scale: f32 = std::env::var("HB_Q_SCALE").map_or(Ok(4.0), |v| v.parse())?;
    let mut failed = false;
    println!(
        "us/launch median (p10-p90); err = worst distance from f64 in head-scale bf16 ulps; \
         eq = outputs bit-equal to the per-head kernel"
    );
    for (heads, rows, valid) in [
        (22usize, 1usize, 1000usize),
        (22, 1, 2051),
        (22, 16, 1000),
        (22, 16, 2051),
        (21, 16, 2051),
        (32, 4, 1500),
        (64, 2, 2051),
    ] {
        let c = make_case(g, &mut rng, heads, rows, valid, q_scale)?;
        let n_out = rows * heads * KVL * 2;
        let out_old = g.alloc(n_out)?;
        let t_old = time(g, |i| {
            launch_old(g, &k, pool.dev, c.view(i % VARIANTS, 0, rows), out_old)
        })?;
        launch_old(g, &k, pool.dev, c.view(0, 0, rows), out_old)?;
        let old_bytes = read(g, out_old, n_out)?;
        let checked = rows.min(2);
        let e_old = worst_ulp(&c, &pool, &bf16s(&old_bytes), checked);
        println!(
            "heads {heads} rows {rows} valid {valid}: old {:7.1} ({:.1}-{:.1}) err {e_old:.2}",
            t_old.0, t_old.1, t_old.2
        );
        for &splits in &splits_list {
            let ws = (
                g.alloc((rows * splits * heads * KVL * 4).max(4))?,
                g.alloc((rows * splits * heads * 8).max(4))?,
            );
            let out = g.alloc(n_out)?;
            let t = time(g, |i| {
                launch_hb(
                    g,
                    &k,
                    pool.dev,
                    c.view(i % VARIANTS, 0, rows),
                    out,
                    ws,
                    splits,
                )
            })?;
            launch_hb(g, &k, pool.dev, c.view(0, 0, rows), out, ws, splits)?;
            let all = read(g, out, n_out)?;
            let e = worst_ulp(&c, &pool, &bf16s(&all), checked);
            println!(
                "    splits {splits:2}: {:7.1} ({:.1}-{:.1}) err {e:.2} eq {:.1}%  x{:.2}",
                t.0,
                t.1,
                t.2,
                same_as_old(&all, &old_bytes) * 100.0,
                t_old.0 / t.0
            );
            if e > TOL_ULP {
                println!("    FAIL: {e:.2} ulp from the reference (per-head kernel {e_old:.2})");
                failed = true;
            }
            // 2026-10-09: Batch invariance: each row alone must give the multi-row bytes.
            let one = g.alloc(heads * KVL * 2)?;
            for r in 0..rows {
                launch_hb(g, &k, pool.dev, c.view(0, r, 1), one, ws, splits)?;
                if read(g, one, heads * KVL * 2)? != all[r * heads * KVL * 2..][..heads * KVL * 2] {
                    println!("    FAIL: row {r} alone differs from the batched launch");
                    failed = true;
                }
            }
        }
    }
    failed |= edge_cases(g, &k, &pool, &mut rng)?;
    if failed {
        bail!("head-batched DSA decode gate FAILED");
    }
    println!("GATE PASS");
    Ok(())
}

/// 2026-10-09: Row 0 all -1 (exact zeros), row 1 `seq_len` 0 (untouched), row 2 every entry
/// at or past `seq_len` (zeros), row 3 duplicates and out-of-range entries mixed in (vs the
/// reference, which attends a duplicate twice). Returns whether any failed.
fn edge_cases(g: &dyn GpuBackend, k: &Kernels, pool: &Pool, rng: &mut Lcg) -> Result<bool> {
    let mut c = make_case(g, rng, 22, 4, 300, 4.0)?;
    let row = |r: usize| r * WIDTH..(r + 1) * WIDTH;
    c.sel[0][row(0)].fill(-1);
    c.sl[1] = 0;
    let sl2 = c.sl[2];
    c.sel[0][row(2)]
        .iter_mut()
        .filter(|x| **x >= 0)
        .for_each(|x| *x = sl2 + 3);
    for j in 0..200 {
        match j % 5 {
            0 => c.sel[0][3 * WIDTH + j] = 17,
            1 => c.sel[0][3 * WIDTH + j] = c.sl[3] + j as i32,
            _ => {}
        }
    }
    c.sel_d[0] = up_i32(g, &c.sel[0])?;
    c.sl_d = up_i32(g, &c.sl)?;
    let per = 22 * KVL * 2;
    let mut failed = false;
    for splits in [1usize, 8] {
        let out = g.alloc(4 * per)?;
        g.memset(out, 0xAB, 4 * per)?;
        let ws = (
            g.alloc(4 * splits * 22 * KVL * 4)?,
            g.alloc(4 * splits * 22 * 8)?,
        );
        launch_hb(g, k, pool.dev, c.view(0, 0, 4), out, ws, splits)?;
        let b = read(g, out, 4 * per)?;
        let zero = |r: usize| b[r * per..][..per].iter().all(|x| *x == 0);
        let untouched = b[per..2 * per].iter().all(|x| *x == 0xAB);
        let f = bf16s(&b);
        let o3 = reference(&c, pool, 3, 0);
        let ulp = o3.iter().fold(0f64, |a, x| a.max(x.abs())) / 256.0;
        let e3 = (0..KVL).fold(0f64, |a, i| {
            a.max((f[3 * 22 * KVL + i] as f64 - o3[i]).abs() / ulp)
        });
        let ok = zero(0) && untouched && zero(2) && e3 <= TOL_ULP;
        println!(
            "edge splits {splits}: all -1 zero {} | seq_len 0 untouched {untouched} | past seq_len zero {} | dup + out-of-range {e3:.2} ulp -> {}",
            zero(0),
            zero(2),
            if ok { "PASS" } else { "FAIL" }
        );
        failed |= !ok;
    }
    Ok(failed)
}
