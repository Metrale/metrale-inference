// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-09: Microbench and byte gate for the three kernels that dominate GLM-5.3's C16
//! decode graph, at the per-rank shapes of the three-way TP=3/EP=3 layout:
//!
//! 1. BF16 batched GEMV: `dense_gemv_bf16` (M = 1), the runtime-M `dense_gemv_bf16_batchm`
//!    and the register-resident `dense_gemv_bf16_batchm_wide` at M = 8, 9, 12, 16. Every row of
//!    every batched result must equal the M = 1 GEMV of that row, byte for byte.
//! 2. Routed experts: sixteen rows as two 8-row union sweeps (`_m8`, the old decode shape)
//!    against one 16-row sweep (`_m16`), union build included, for the gate projection. The
//!    two outputs must be byte-identical.
//! 3. KDA recurrence: sixteen `kda_recurrent_decode_bf16_smem` launches against one
//!    `kda_recurrent_decode_bf16_smem_rows` and one `kda_recurrent_decode_bf16_rows_reg`.
//!    Outputs and states must be byte-identical.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte mismatch; prints microseconds per call otherwise.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_c16_kernel_bench \
//!       --features cuda,gpu-examples

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const ITERS: usize = 50;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn byte(&mut self) -> u8 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u8
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, rng: &mut Lcg, n: usize, s: f32) -> Result<DevicePtr> {
    let v: Vec<u8> = (0..n)
        .flat_map(|_| bf16::from_f32(rng.f() * s).to_le_bytes())
        .collect();
    up(g, &v)
}
fn up_f32(g: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    up(
        g,
        &v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )
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

/// 2026-10-09: Microseconds per call of `f`, after three warm-up calls.
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

#[allow(clippy::too_many_arguments)]
fn gemv_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    batched: bool,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: usize,
    n: usize,
    kk: usize,
) -> Result<()> {
    let l = KernelLaunch::new(g, k)
        .grid([n.div_ceil(4) as u32, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c);
    if batched {
        l.arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(kk as u32)
            .arg_u32(n as u32)
            .launch(0)
    } else {
        l.arg_u32(n as u32).arg_u32(kk as u32).launch(0)
    }
}

/// 2026-10-09: Part 1. Shapes: KDA q/k/v (2816 x 4096), KDA o_proj (4096 x 2816), DSA q_a
/// (1536 x 4096), a square 4096.
fn dense(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    let gemv = g.kernel("gemv", "dense_gemv_bf16")?;
    let narrow = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    let wide = g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_wide")?;
    println!("BF16 GEMV (us/call; GB/s of weight)");
    for (n, kk) in [
        (2816usize, 4096usize),
        (4096, 2816),
        (1536, 4096),
        (4096, 4096),
    ] {
        let b = up_bf16(g, rng, n * kk, 0.03)?;
        let a = up_bf16(g, rng, 16 * kk, 1.0)?;
        let c1 = zeros(g, 16 * n * 2)?;
        for r in 0..16 {
            gemv_launch(
                g,
                gemv,
                false,
                a.offset(r * kk * 2),
                b,
                c1.offset(r * n * 2),
                1,
                n,
                kk,
            )?;
        }
        let ref_rows = read(g, c1, 16 * n * 2)?;
        let wbytes = (n * kk * 2) as f64;
        let t1 = time(g, || gemv_launch(g, gemv, false, a, b, c1, 1, n, kk))?;
        print!("  {n:>5}x{kk:<5} M=1 {t1:7.1} ({:.0})", wbytes / t1 / 1e3);
        for m in [8usize, 9, 12, 16] {
            let cn = zeros(g, m * n * 2)?;
            let tn = time(g, || gemv_launch(g, narrow, true, a, b, cn, m, n, kk))?;
            same(
                &format!("batchm M={m} {n}x{kk}"),
                &read(g, cn, m * n * 2)?,
                &ref_rows[..m * n * 2],
            )?;
            print!(" | M={m} batchm {tn:7.1}");
            if m >= 9 {
                let cw = zeros(g, m * n * 2)?;
                let tw = time(g, || gemv_launch(g, wide, true, a, b, cw, m, n, kk))?;
                same(
                    &format!("wide M={m} {n}x{kk}"),
                    &read(g, cw, m * n * 2)?,
                    &ref_rows[..m * n * 2],
                )?;
                print!(" wide {tw:7.1}");
            }
        }
        println!();
    }
    Ok(())
}

/// 2026-10-09: Part 2. The gate projection of one MoE layer: 288 experts, 96 owned by this
/// rank (the rest null, as EP=3 leaves them), moe_intermediate 2048, hidden 4096, top-8.
fn moe(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    const E: usize = 288;
    const LOCAL: usize = 96;
    const N: usize = 2048;
    const K: usize = 4096;
    const TOP_K: usize = 8;
    const ROWS: usize = 16;
    let m8 = g.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m8")?;
    let m16 = g.kernel("w4a16_gemv", "w4a16_gemv_sw_moe_batchm_m16")?;
    let union = g.kernel("w4a16_gemv", "glm5next_moe_row_union")?;
    let (mut packed, mut scale, mut scale2) = (vec![0u64; E], vec![0u64; E], vec![0f32; E]);
    for e in 0..LOCAL {
        let w: Vec<u8> = (0..N * K / 2).map(|_| rng.byte()).collect();
        let s: Vec<u8> = (0..N * K / 16).map(|_| 0x38 | (rng.byte() & 7)).collect();
        packed[e * 3] = up(g, &w)?.0;
        scale[e * 3] = up(g, &s)?.0;
        scale2[e * 3] = 1.0;
    }
    let tp = up(
        g,
        &packed
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let ts = up(
        g,
        &scale
            .iter()
            .flat_map(|p| p.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let t2 = up_f32(g, &scale2)?;
    let mut ids = Vec::with_capacity(ROWS * TOP_K);
    for _ in 0..ROWS {
        let mut row: Vec<i32> = Vec::new();
        while row.len() < TOP_K {
            let e = (rng.byte() as usize * 256 + rng.byte() as usize) % E;
            if !row.contains(&(e as i32)) {
                row.push(e as i32);
            }
        }
        ids.extend(row);
    }
    let d_ids = up(
        g,
        &ids.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>(),
    )?;
    let x = up_bf16(g, rng, ROWS * K, 1.0)?;
    let u_eid = zeros(g, ROWS * TOP_K * 4)?;
    let u_slot = zeros(g, ROWS * TOP_K * ROWS * 4)?;
    let out_bytes = ROWS * TOP_K * N * 2;
    let sweep = |k: KernelHandle, r0: usize, rows: usize, c: DevicePtr| -> Result<()> {
        KernelLaunch::new(g, union)
            .grid([1, 1, 1])
            .block([(rows * TOP_K) as u32, 1, 1])
            .arg_ptr(d_ids.offset(r0 * TOP_K * 4))
            .arg_ptr(u_eid)
            .arg_ptr(u_slot)
            .arg_u32(rows as u32)
            .arg_u32(TOP_K as u32)
            .launch(0)?;
        KernelLaunch::new(g, k)
            .grid([N.div_ceil(8) as u32, (rows * TOP_K) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(x.offset(r0 * K * 2))
            .arg_ptr(tp)
            .arg_ptr(ts)
            .arg_ptr(t2)
            .arg_ptr(c.offset(r0 * TOP_K * N * 2))
            .arg_ptr(u_eid)
            .arg_ptr(u_slot)
            .arg_u32(N as u32)
            .arg_u32(K as u32)
            .arg_u32(E as u32)
            .arg_u32(K as u32)
            .arg_u32(0)
            .arg_u32((TOP_K * N) as u32)
            .launch(0)
    };
    let c_halves = zeros(g, out_bytes)?;
    let c_one = zeros(g, out_bytes)?;
    let halves = || -> Result<()> {
        sweep(m8, 0, 8, c_halves)?;
        sweep(m8, 8, 8, c_halves)
    };
    let one = || sweep(m16, 0, ROWS, c_one);
    let th = time(g, halves)?;
    let to = time(g, one)?;
    same(
        "MoE m8x2 vs m16",
        &read(g, c_halves, out_bytes)?,
        &read(g, c_one, out_bytes)?,
    )?;
    let local: std::collections::BTreeSet<i32> = ids
        .iter()
        .copied()
        .filter(|e| (*e as usize).is_multiple_of(3))
        .collect();
    println!(
        "MoE gate, 16 rows, {} local experts in the union: two m8 sweeps {th:.1} us, one m16 \
         sweep {to:.1} us",
        local.len()
    );
    Ok(())
}

/// 2026-10-09: Part 3. 22 heads (64 at TP=3, the widest rank), head_dim 128, 16 rows.
fn kda(g: &dyn GpuBackend, rng: &mut Lcg) -> Result<()> {
    const H: usize = 22;
    const D: usize = 128;
    const ROWS: usize = 16;
    const VPB: usize = 32;
    let qkv = H * D;
    let cd = 3 * qkv;
    let single = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem")?;
    let rows_k = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_smem_rows")?;
    let reg_k = g.kernel("kda_recurrent", "kda_recurrent_decode_bf16_rows_reg")?;
    let smem = ((3 * D + VPB * (D + 1)) * 4) as u32;
    let conv = up_bf16(g, rng, ROWS * cd, 0.1)?;
    let gate = up_f32(
        g,
        &(0..ROWS * qkv)
            .map(|_| -0.05 - rng.f().abs() * 0.1)
            .collect::<Vec<_>>(),
    )?;
    let beta = up_f32(
        g,
        &(0..ROWS * H)
            .map(|_| 0.5 + rng.f() * 0.2)
            .collect::<Vec<_>>(),
    )?;
    let sb = H * D * D * 4;
    let init: Vec<Vec<f32>> = (0..ROWS)
        .map(|_| (0..H * D * D).map(|_| rng.f() * 0.1).collect())
        .collect();
    let mk =
        |v: &Vec<Vec<f32>>| -> Result<Vec<DevicePtr>> { v.iter().map(|s| up_f32(g, s)).collect() };
    let (sa, sbat, sreg) = (mk(&init)?, mk(&init)?, mk(&init)?);
    let oa = zeros(g, ROWS * qkv * 4)?;
    let ob = zeros(g, ROWS * qkv * 4)?;
    let oreg = zeros(g, ROWS * qkv * 4)?;
    let scale = 1.0 / (D as f32).sqrt();
    let per_row = |states: &[DevicePtr], out: DevicePtr| -> Result<()> {
        for (r, s) in states.iter().enumerate() {
            let q = conv.offset(r * cd * 2);
            KernelLaunch::new(g, single)
                .grid([H as u32, (D / VPB) as u32, 1])
                .block([VPB as u32, 1, 1])
                .shared_mem(smem)
                .arg_ptr(q)
                .arg_ptr(q.offset(qkv * 2))
                .arg_ptr(q.offset(qkv * 4))
                .arg_ptr(gate.offset(r * qkv * 4))
                .arg_ptr(beta.offset(r * H * 4))
                .arg_ptr(*s)
                .arg_ptr(out.offset(r * qkv * 4))
                .arg_u32(H as u32)
                .arg_u32(D as u32)
                .arg_f32(scale)
                .arg_u32(VPB as u32)
                .launch(0)?;
        }
        Ok(())
    };
    let batched = |states: &[DevicePtr], out: DevicePtr| -> Result<()> {
        let mut l = KernelLaunch::new(g, rows_k)
            .grid([H as u32, (D / VPB) as u32, ROWS as u32])
            .block([VPB as u32, 1, 1])
            .shared_mem(smem)
            .arg_ptr(conv)
            .arg_ptr(conv.offset(qkv * 2))
            .arg_ptr(conv.offset(qkv * 4))
            .arg_ptr(gate)
            .arg_ptr(beta)
            .arg_ptr(out)
            .arg_u32(H as u32)
            .arg_u32(D as u32)
            .arg_f32(scale)
            .arg_u32(VPB as u32)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(H as u32)
            .arg_u32(qkv as u32);
        for r in 0..16 {
            l = l.arg_u64(states.get(r).map_or(0, |s| s.0));
        }
        l.launch(0)
    };
    let reg = |states: &[DevicePtr], out: DevicePtr| -> Result<()> {
        let mut l = KernelLaunch::new(g, reg_k)
            .grid([H as u32, 1, ROWS as u32])
            .block([D as u32, 1, 1])
            .shared_mem((3 * D * 4) as u32)
            .arg_ptr(conv)
            .arg_ptr(conv.offset(qkv * 2))
            .arg_ptr(conv.offset(qkv * 4))
            .arg_ptr(gate)
            .arg_ptr(beta)
            .arg_ptr(out)
            .arg_u32(H as u32)
            .arg_f32(scale)
            .arg_u32(cd as u32)
            .arg_u32(qkv as u32)
            .arg_u32(H as u32)
            .arg_u32(qkv as u32);
        for r in 0..16 {
            l = l.arg_u64(states.get(r).map_or(0, |s| s.0));
        }
        l.launch(0)
    };
    // 2026-10-09: One step each from the same initial states, compared; then timing (the
    // states keep advancing, which does not change the work).
    per_row(&sa, oa)?;
    batched(&sbat, ob)?;
    reg(&sreg, oreg)?;
    let want = read(g, oa, ROWS * qkv * 4)?;
    same("KDA out (rows)", &want, &read(g, ob, ROWS * qkv * 4)?)?;
    same("KDA out (rows_reg)", &want, &read(g, oreg, ROWS * qkv * 4)?)?;
    for r in 0..ROWS {
        let st = read(g, sa[r], sb)?;
        same(
            &format!("KDA state {r} (rows)"),
            &st,
            &read(g, sbat[r], sb)?,
        )?;
        same(
            &format!("KDA state {r} (rows_reg)"),
            &st,
            &read(g, sreg[r], sb)?,
        )?;
    }
    let tp = time(g, || per_row(&sa, oa))?;
    let tb = time(g, || batched(&sbat, ob))?;
    let tr = time(g, || reg(&sreg, oreg))?;
    let mb = (2 * ROWS * sb) as f64 / 1e6;
    println!(
        "KDA recurrence, 16 rows x {H} heads ({mb:.1} MB state read+write): 16 single-row \
         launches {tp:.1} us, one smem rows launch {tb:.1} us ({:.0} GB/s), one register \
         rows launch {tr:.1} us ({:.0} GB/s)",
        mb * 1e3 / tb,
        mb * 1e3 / tr
    );
    Ok(())
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let mut rng = Lcg(0x6c6d_c16b_0001);
    dense(&gpu, &mut rng)?;
    moe(&gpu, &mut rng)?;
    kda(&gpu, &mut rng)?;
    println!("PASS: every batched result is byte-identical to its reference");
    Ok(())
}
