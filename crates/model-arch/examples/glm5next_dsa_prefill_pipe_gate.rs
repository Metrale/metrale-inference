// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Gate for the pipelined prefill DSA attend `glm5next_dsa_mla_prefill_pipe_phi`
//! (`METRALE_GLM_DSA_HB_PREFILL_PIPE`) against the kernel it replaces,
//! `glm5next_dsa_mla_decode_hb_fp8_phi`.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error when any of these fails:
//! - (i) byte equality: every output byte of the pipe equals the phi kernel's, over the decode
//!   bench shapes (22, 21, 32, 64 heads; 1 and 8 splits), the edge rows, and a 32k-token
//!   prefill sub-chunk of [`ROWS32`] rows with correlated selections;
//! - (ii) row invariance: each prefill row's bytes are the same launched in the full sub-chunk,
//!   in sub-chunks of 1, 7, 512, 2052 and 4104 rows and an uneven split, among permuted
//!   neighbours, and through a per-row block table of a larger stride (another max-seq-len);
//! - (iii) the pipe is within [`fixture::TOL_ULP`] of an f64 reference on sampled rows, and the
//!   edge rows are exact (all -1 and past-`seq_len` rows zero, a `seq_len` 0 row untouched);
//! - a negative control: the hi + lo entry, a different numerics point, must differ from the
//!   phi kernel somewhere, or the byte comparison is not able to see a change;
//! - (iv) timing: over [`ROWS32`] rows at one split and 22 heads, the pipe's median must be
//!   below the phi kernel's (interleaved samples, two block tables rotating so the latent is
//!   not L2-resident from the previous launch).
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_prefill_pipe_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_hb_fixture.rs"]
mod fixture;
#[path = "common/glm5next_dsa_prefill_fixture.rs"]
mod prefill;

use anyhow::{Context, Result, bail};
use fixture::{
    Case, KVL, Lcg, Pool, TOL_ULP, View, WIDTH, bf16s, launch_hb_entry, make_case, make_edge_case,
    make_pool, read, upload_case, worst_ulp,
};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_dsa::attend::{
    DSA_DECODE_HB_MODULE, DSA_PREFILL_PIPE_MODULE, PIPE_SMEM_BYTES,
};

/// 2026-10-14: Rows of the timed prefill sub-chunk.
const ROWS32: usize = 8208;
const HEADS32: usize = 22;
const SAMPLES: usize = 15;

struct Gate<'a> {
    g: &'a dyn GpuBackend,
    phi: KernelHandle,
    pipe: KernelHandle,
    hilo: KernelHandle,
    merge: KernelHandle,
    pool: Pool,
    failed: bool,
}

impl Gate<'_> {
    /// 2026-10-14: `entry` over `v` into `out`; the pipe gets its dynamic shared memory.
    fn run(&self, entry: KernelHandle, v: View, out: DevicePtr, splits: usize) -> Result<()> {
        let smem = if entry.0 == self.pipe.0 {
            PIPE_SMEM_BYTES
        } else {
            0
        };
        let ws = if splits > 1 {
            (
                self.g.alloc(v.rows * splits * v.heads * KVL * 4)?,
                self.g.alloc(v.rows * splits * v.heads * 8)?,
            )
        } else {
            (DevicePtr(0), DevicePtr(0))
        };
        launch_hb_entry(
            self.g,
            entry,
            smem,
            self.merge,
            self.pool.dev,
            v,
            out,
            ws,
            splits,
        )?;
        if splits > 1 {
            self.g.synchronize(0)?;
            self.g.free(ws.0)?;
            self.g.free(ws.1)?;
        }
        Ok(())
    }

    /// 2026-10-14: The output bytes of `entry` over `v`.
    fn bytes(&self, entry: KernelHandle, v: View, splits: usize) -> Result<Vec<u8>> {
        let n = v.rows * v.heads * KVL * 2;
        let out = self.g.alloc(n)?;
        self.run(entry, v, out, splits)?;
        let b = read(self.g, out, n)?;
        self.g.free(out)?;
        Ok(b)
    }

    fn check(&mut self, ok: bool, what: &str) {
        println!("  {} {what}", if ok { "pass" } else { "FAIL" });
        self.failed |= !ok;
    }
}

/// 2026-10-14: Rows of `a` and `b` (`per` bytes each) that differ in any byte.
fn rows_differing(a: &[u8], b: &[u8], per: usize) -> usize {
    a.chunks(per)
        .zip(b.chunks(per))
        .filter(|(x, y)| x != y)
        .count()
}

fn main() -> Result<()> {
    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let mut rng = Lcg(0x0D5A_9192);
    let mut gate = Gate {
        g,
        phi: g.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8_phi")?,
        pipe: g.kernel(DSA_PREFILL_PIPE_MODULE, "glm5next_dsa_mla_prefill_pipe_phi")?,
        hilo: g.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8")?,
        merge: g.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_merge")?,
        pool: make_pool(g, &mut rng, 8)?,
        failed: false,
    };
    decode_shapes(&mut gate, &mut rng)?;
    edge_rows(&mut gate, &mut rng)?;
    prefill_32k(&mut gate, &mut rng)?;
    if gate.failed {
        bail!("prefill pipe DSA attend gate FAILED");
    }
    println!("GATE PASS");
    Ok(())
}

/// 2026-10-14: (i) and (iii) over the decode bench's shapes, plus the negative control.
fn decode_shapes(gate: &mut Gate, rng: &mut Lcg) -> Result<()> {
    for (heads, rows, valid) in [
        (22usize, 1usize, 1000usize),
        (22, 16, 2051),
        (21, 16, 2051),
        (32, 4, 1500),
        (64, 2, 2051),
        (22, 256, 2051),
    ] {
        let c = make_case(gate.g, rng, heads, rows, valid, 4.0)?;
        let per = heads * KVL * 2;
        for splits in [1usize, 8] {
            if rows > 16 && splits > 1 {
                continue;
            }
            println!("heads {heads} rows {rows} valid {valid} splits {splits}");
            let v = c.view(0, 0, rows);
            let phi = gate.bytes(gate.phi, v, splits)?;
            let pipe = gate.bytes(gate.pipe, v, splits)?;
            let d = rows_differing(&pipe, &phi, per);
            gate.check(d == 0, &format!("(i) {d} of {rows} rows differ from phi"));
            let e = worst_ulp(&c, &gate.pool, &bf16s(&pipe), 0..rows.min(2));
            gate.check(e <= TOL_ULP, &format!("(iii) {e:.2} ulp from f64"));
            let mut alone = 0;
            for r in 0..rows.min(16) {
                alone += usize::from(
                    gate.bytes(gate.pipe, c.view(0, r, 1), splits)? != pipe[r * per..][..per],
                );
            }
            gate.check(
                alone == 0,
                &format!("(ii) {alone} rows differ launched alone"),
            );
            if rows == 256 {
                let hilo = gate.bytes(gate.hilo, v, splits)?;
                let d = rows_differing(&hilo, &phi, per);
                gate.check(
                    d > 0,
                    &format!("control: hi + lo differs from phi in {d} rows"),
                );
            }
        }
    }
    Ok(())
}

/// 2026-10-14: (i) and (iii) on the edge rows of [`make_edge_case`].
fn edge_rows(gate: &mut Gate, rng: &mut Lcg) -> Result<()> {
    let c = make_edge_case(gate.g, rng)?;
    let per = c.heads * KVL * 2;
    for splits in [1usize, 8] {
        println!("edge rows splits {splits}");
        let mut outs = Vec::new();
        for entry in [gate.phi, gate.pipe] {
            let out = gate.g.alloc(4 * per)?;
            gate.g.memset(out, 0xAB, 4 * per)?;
            gate.run(entry, c.view(0, 0, 4), out, splits)?;
            outs.push(read(gate.g, out, 4 * per)?);
            gate.g.free(out)?;
        }
        let b = &outs[1];
        gate.check(
            b == &outs[0],
            "(i) bytes equal to phi, untouched row included",
        );
        let zero = |r: usize| b[r * per..][..per].iter().all(|x| *x == 0);
        gate.check(
            zero(0) && zero(2),
            "(iii) all -1 and past-seq_len rows are zero",
        );
        gate.check(
            b[per..2 * per].iter().all(|x| *x == 0xAB),
            "(iii) seq_len 0 row untouched",
        );
        let e = worst_ulp(&c, &gate.pool, &bf16s(b), [3]);
        gate.check(
            e <= TOL_ULP,
            &format!("(iii) duplicates row {e:.2} ulp from f64"),
        );
    }
    Ok(())
}

/// 2026-10-14: The 32k prefill sub-chunk: (i), (ii), (iii) and (iv).
fn prefill_32k(gate: &mut Gate, rng: &mut Lcg) -> Result<()> {
    let c = prefill::make_prefill_case(gate.g, rng, HEADS32, ROWS32, 2, 1.0)?;
    let (adjacent, union) = prefill::overlap(&c, 64);
    println!(
        "prefill {ROWS32} rows of a {}-token sequence, {HEADS32} heads: a row shares {:.0}% of \
         its pools with the previous row; 64 rows select {union:.2}x 512 pools between them",
        prefill::SEQ32,
        adjacent * 100.0
    );
    let per = HEADS32 * KVL * 2;
    let full = c.view(0, 0, ROWS32);
    let phi = gate.bytes(gate.phi, full, 1)?;
    let pipe = gate.bytes(gate.pipe, full, 1)?;
    let d = rows_differing(&pipe, &phi, per);
    gate.check(d == 0, &format!("(i) {d} of {ROWS32} rows differ from phi"));
    let e = worst_ulp(
        &c,
        &gate.pool,
        &bf16s(&pipe),
        [0, 1, ROWS32 / 2, ROWS32 - 1],
    );
    gate.check(
        e <= TOL_ULP,
        &format!("(iii) {e:.2} ulp from f64 on 4 rows"),
    );

    let out = gate.g.alloc(ROWS32 * per)?;
    let mut plans: Vec<(String, Vec<usize>)> = [1usize, 7, 512, 2052, 4104]
        .iter()
        .map(|n| (format!("sub-chunks of {n}"), vec![*n]))
        .collect();
    plans.push((
        "uneven sub-chunks 333, 1000, 4096, ...".into(),
        vec![333, 1000, 4096],
    ));
    for (name, sizes) in plans {
        gate.g.memset(out, 0xAB, ROWS32 * per)?;
        let (mut r0, mut i) = (0, 0);
        while r0 < ROWS32 {
            let n = sizes[i.min(sizes.len() - 1)].min(ROWS32 - r0);
            gate.run(gate.pipe, c.view(0, r0, n), out.offset(r0 * per), 1)?;
            (r0, i) = (r0 + n, i + 1);
        }
        let d = rows_differing(&read(gate.g, out, ROWS32 * per)?, &pipe, per);
        gate.check(d == 0, &format!("(ii) {name}: {d} rows differ"));
    }
    gate.g.free(out)?;

    neighbours_and_stride(gate, &c, &pipe)?;

    // 2026-10-14: The first 16 rows at 8 splits (+ merge): another split count, same bytes.
    let v16 = c.view(0, 0, 16);
    let d = rows_differing(
        &gate.bytes(gate.pipe, v16, 8)?,
        &gate.bytes(gate.phi, v16, 8)?,
        per,
    );
    gate.check(
        d == 0,
        &format!("(i) 8 splits: {d} of 16 rows differ from phi"),
    );

    timing(gate, &c)
}

/// 2026-10-14: (ii) rows 0..1024 permuted among themselves, and rows 0..256 through a per-row
/// block table at stride 640 (a larger max-seq-len) holding the shared table then other blocks.
fn neighbours_and_stride(gate: &mut Gate, c: &Case, pipe: &[u8]) -> Result<()> {
    let per = c.heads * KVL * 2;
    let qrow = c.heads * KVL;
    let (n, mult) = (1024usize, 389usize);
    let src = |j: usize| (j * mult) % n;
    let q: Vec<f32> = (0..n)
        .flat_map(|j| c.q[src(j) * qrow..][..qrow].to_vec())
        .collect();
    let sl: Vec<i32> = (0..n).map(|j| c.sl[src(j)]).collect();
    let sel: Vec<i32> = (0..n)
        .flat_map(|j| c.sel[0][src(j) * WIDTH..][..WIDTH].to_vec())
        .collect();
    let p = upload_case(gate.g, c.heads, 0, q, sl, vec![c.bt[0].clone()], vec![sel])?;
    let b = gate.bytes(gate.pipe, p.view(0, 0, n), 1)?;
    let d = (0..n)
        .filter(|j| b[j * per..][..per] != pipe[src(*j) * per..][..per])
        .count();
    gate.check(
        d == 0,
        &format!("(ii) {d} of {n} rows differ among permuted neighbours"),
    );
    free_case(gate.g, &p)?;

    let (rows, bps) = (256usize, 640usize);
    let tail: Vec<i32> = (0..(bps - c.bt[0].len()) as i32).collect();
    let bt: Vec<i32> = (0..rows)
        .flat_map(|_| c.bt[0].iter().chain(&tail).copied())
        .collect();
    let s = upload_case(
        gate.g,
        c.heads,
        bps,
        c.q[..rows * qrow].to_vec(),
        c.sl[..rows].to_vec(),
        vec![bt],
        vec![c.sel[0][..rows * WIDTH].to_vec()],
    )?;
    let b = gate.bytes(gate.pipe, s.view(0, 0, rows), 1)?;
    let d = rows_differing(&b, &pipe[..rows * per], per);
    gate.check(
        d == 0,
        &format!("(ii) {d} of {rows} rows differ at block-table stride {bps}"),
    );
    free_case(gate.g, &s)
}

fn free_case(g: &dyn GpuBackend, c: &Case) -> Result<()> {
    for p in [c.q_d, c.sl_d].iter().chain(&c.bt_d).chain(&c.sel_d) {
        g.free(*p)?;
    }
    Ok(())
}

/// 2026-10-14: (iv) phi vs pipe over the whole sub-chunk at one split, [`SAMPLES`] interleaved
/// samples (order alternating), variants rotating; milliseconds per launch and effective
/// TFLOP/s over the valid entries (QK and PV, 2 flops per MAC).
fn timing(gate: &mut Gate, c: &Case) -> Result<()> {
    let per = HEADS32 * KVL * 2;
    let out = gate.g.alloc(ROWS32 * per)?;
    let mut t = [Vec::new(), Vec::new()];
    let entries = [gate.phi, gate.pipe];
    for (e, entry) in entries.iter().enumerate() {
        gate.run(*entry, c.view(e % c.sel.len(), 0, ROWS32), out, 1)?;
    }
    for s in 0..SAMPLES {
        for k in 0..2 {
            let e = (s + k) % 2;
            gate.g.synchronize(0)?;
            let t0 = std::time::Instant::now();
            gate.run(entries[e], c.view((s + k) % c.sel.len(), 0, ROWS32), out, 1)?;
            gate.g.synchronize(0)?;
            t[e].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    gate.g.free(out)?;
    let valid: usize = (0..ROWS32)
        .map(|r| {
            c.sel[0][r * WIDTH..][..WIDTH]
                .iter()
                .filter(|x| **x >= 0 && **x < c.sl[r])
                .count()
        })
        .sum();
    let flop = (valid * HEADS32 * KVL * 4) as f64;
    let med = |v: &mut Vec<f64>| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    };
    let (p, q) = (med(&mut t[0].clone()), med(&mut t[1].clone()));
    for (name, m) in [("phi ", p), ("pipe", q)] {
        println!(
            "  {name} {:8.3} ms (min {:.3}, max {:.3})  {:5.1} TFLOP/s",
            m.0,
            m.1,
            m.2,
            flop / m.0 / 1e9
        );
    }
    gate.check(
        q.0 < p.0,
        &format!(
            "(iv) pipe {:.3} ms vs phi {:.3} ms: x{:.3}",
            q.0,
            p.0,
            p.0 / q.0
        ),
    );
    Ok(())
}
