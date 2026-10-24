// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Gate for `glm5next_dsa_mla_prefill_tc_phi_nanor` (`METRALE_GLM_DSA_TC_NANOR`), the
//! tc prefill attend with its E4M3 NaN-byte test made once per producer share of a tile instead
//! of once per 32-bit word:
//! - every check of common/glm5next_dsa_twin_gate.rs against the phi kernel (byte equality, row
//!   invariance, the f64 bound, its controls and timing);
//! - byte equality with the tc entry it replaces at the serve rows (8192, 8208 of a 32k-token
//!   sequence) and the c1k rows (1040, 1056 of a 1040- and a 1056-token one) over three latent
//!   pools: every one of the 256 byte values at every byte position of a token (NaN 0x7f / 0xff,
//!   the ±448 max-finite codes 0x7e / 0xfe just below them, the denormals 0x01..0x07 and
//!   0x81..0x87, ±0), checked on the host over the tokens each case selects; the 254 finite
//!   values only (every share takes the finite converts); and NaN-dense tokens (one token in 97
//!   all NaN, one byte in 4099 NaN) in the random pool;
//! - negative control: the tc entry over the all-values pool with its NaN bytes remapped to the
//!   max-finite code of the same sign (a wrong NaN mapping) must differ from the twin over the
//!   pool itself, or the byte comparison could not see one;
//! - timing: interleaved medians of phi, tc and the twin at each of those shapes, the twin's
//!   median below the tc entry's at the serve rows.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error when any check fails.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-gpu-examples --release --example glm5next_dsa_prefill_tc_nanor_gate \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_dsa_hb_fixture.rs"]
mod fixture;
#[path = "common/glm5next_dsa_prefill_fixture.rs"]
mod prefill;
#[path = "common/glm5next_dsa_twin_gate.rs"]
mod twin;

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_dsa::attend::{
    DSA_DECODE_HB_MODULE, DSA_PREFILL_TC_MODULE, TC_BLOCK, TC_SMEM_BYTES,
};

use fixture::{BLOCK, Case, KVL, Lcg, Pool, WIDTH, launch_hb_entry, make_pool, read, up};

const NANOR: &str = "glm5next_dsa_mla_prefill_tc_phi_nanor";
const HEADS: usize = 22;
const SAMPLES: usize = 9;
/// 2026-10-15: (rows, sequence length): the serve grid rows of a 32k prefill, then the c1k rows
/// at a 32k sequence (the 32k selection width) and at their own 1k sequence (the c1k leg).
const SHAPES: [(usize, usize); 6] = [
    (8192, 32768),
    (8208, 32768),
    (1040, 32768),
    (1056, 32768),
    (1040, 1040),
    (1056, 1056),
];

struct Entries {
    phi: KernelHandle,
    tc: KernelHandle,
    nanor: KernelHandle,
    merge: KernelHandle,
}

struct Gate<'a> {
    g: &'a dyn GpuBackend,
    k: Entries,
    failed: bool,
}

impl Gate<'_> {
    /// 2026-10-15: `entry` over `v` of the pool at `cache` into `out`, one split.
    fn run(
        &self,
        entry: KernelHandle,
        cache: DevicePtr,
        v: fixture::View,
        out: DevicePtr,
    ) -> Result<()> {
        let (block, smem) = if entry.0 == self.k.phi.0 {
            (256, 0)
        } else {
            (TC_BLOCK, TC_SMEM_BYTES)
        };
        let none = (DevicePtr(0), DevicePtr(0));
        launch_hb_entry(
            self.g,
            entry,
            block,
            smem,
            self.k.merge,
            cache,
            v,
            out,
            none,
            1,
        )
    }

    fn bytes(&self, entry: KernelHandle, cache: DevicePtr, v: fixture::View) -> Result<Vec<u8>> {
        let n = v.rows * v.heads * KVL * 2;
        let out = self.g.alloc(n)?;
        self.run(entry, cache, v, out)?;
        let b = read(self.g, out, n)?;
        self.g.free(out)?;
        Ok(b)
    }

    fn check(&mut self, ok: bool, what: &str) {
        println!("  {} {what}", if ok { "pass" } else { "FAIL" });
        self.failed |= !ok;
    }
}

fn rows_differing(a: &[u8], b: &[u8], per: usize) -> usize {
    a.chunks(per)
        .zip(b.chunks(per))
        .filter(|(x, y)| x != y)
        .count()
}

fn is_nan(b: u8) -> bool {
    b & 0x7f == 0x7f
}

/// 2026-10-15: The pool byte at `i`: token `i / KVL`, position `i % KVL`; position j of token t
/// holds `(j + t) mod 256`, so 256 consecutive tokens put every value at every position.
fn all_values(i: usize) -> u8 {
    ((i % KVL + i / KVL) % 256) as u8
}

/// 2026-10-15: As [`all_values`] over the 254 finite codes, skipping 0x7f and 0xff.
fn finite_values(i: usize) -> u8 {
    let v = ((i % KVL + i / KVL) % 254) as u8;
    if v >= 0x7f { v + 1 } else { v }
}

fn pool_from(g: &dyn GpuBackend, host: Vec<u8>) -> Result<Pool> {
    Ok(Pool {
        dev: up(g, &host)?,
        host,
    })
}

/// 2026-10-15: The distinct pool tokens `c` selects through variant 0, as pool token indices.
fn selected_tokens(c: &Case) -> Vec<usize> {
    let mut seen = vec![false; fixture::POOL_BLOCKS * BLOCK];
    for (r, row) in c.sel[0].chunks(WIDTH).enumerate() {
        for t in row.iter().filter(|t| **t >= 0 && **t < c.sl[r]) {
            let t = *t as usize;
            seen[c.bt[0][t / BLOCK] as usize * BLOCK + t % BLOCK] = true;
        }
    }
    (0..seen.len()).filter(|i| seen[*i]).collect()
}

/// 2026-10-15: How many of the 256 x [`KVL`] (value, position) pairs the selected tokens of `c`
/// hold in `pool`.
fn lane_coverage(c: &Case, pool: &[u8]) -> usize {
    let mut seen = vec![false; 256 * KVL];
    for t in selected_tokens(c) {
        for (j, b) in pool[t * KVL..][..KVL].iter().enumerate() {
            seen[*b as usize * KVL + j] = true;
        }
    }
    seen.iter().filter(|x| **x).count()
}

fn free_case(g: &dyn GpuBackend, c: &Case) -> Result<()> {
    for p in [c.q_d, c.sl_d].iter().chain(&c.bt_d).chain(&c.sel_d) {
        g.free(*p)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    twin::run_gate(&twin::Twin {
        label: "tc_nanor",
        module: DSA_PREFILL_TC_MODULE,
        entry: NANOR,
        block: TC_BLOCK,
        smem: TC_SMEM_BYTES,
        splits: &[1],
    })?;

    let sets = metrale_kernels::all_ptx_sets();
    let glm = sets
        .iter()
        .find(|s| s.target.model == "glm-5.3-flash")
        .context("glm-5.3-flash kernel target not built")?;
    let gpu = MetraleCudaBackend::new(0, &glm.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let mut gate = Gate {
        g,
        k: Entries {
            phi: g.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_decode_hb_fp8_phi")?,
            tc: g.kernel(DSA_PREFILL_TC_MODULE, "glm5next_dsa_mla_prefill_tc_phi")?,
            nanor: g.kernel(DSA_PREFILL_TC_MODULE, NANOR)?,
            merge: g.kernel(DSA_DECODE_HB_MODULE, "glm5next_dsa_mla_merge")?,
        },
        failed: false,
    };
    let mut rng = Lcg(0x7A70_0E15);
    let random = make_pool(g, &mut rng, 8)?;
    let n = random.host.len();
    let mut dense = random.host.clone();
    for t in (0..n / KVL).step_by(97) {
        for (j, b) in dense[t * KVL..][..KVL].iter_mut().enumerate() {
            *b = if j % 2 == 0 { 0x7f } else { 0xff };
        }
    }
    for i in (0..n).step_by(4099) {
        dense[i] |= 0x7f;
    }
    let all: Vec<u8> = (0..n).map(all_values).collect();
    let wrong: Vec<u8> = all
        .iter()
        .map(|b| if is_nan(*b) { b - 1 } else { *b })
        .collect();
    let pools = [
        ("all 256 values", pool_from(g, all)?),
        (
            "254 finite values",
            pool_from(g, (0..n).map(finite_values).collect())?,
        ),
        ("NaN-dense tokens", pool_from(g, dense)?),
    ];
    let wrong = pool_from(g, wrong)?;

    let mut report = Vec::new();
    for (rows, seq) in SHAPES {
        let c = prefill::make_prefill_case_at(g, &mut rng, seq, HEADS, rows, 2, 1.0)?;
        let per = HEADS * KVL * 2;
        let v = c.view(0, 0, rows);
        println!("{rows} rows of a {seq}-token sequence, {HEADS} heads");
        for (name, p) in &pools {
            let tc = gate.bytes(gate.k.tc, p.dev, v)?;
            let nr = gate.bytes(gate.k.nanor, p.dev, v)?;
            let d = rows_differing(&nr, &tc, per);
            gate.check(
                d == 0,
                &format!("{name}: {d} of {rows} rows differ from tc"),
            );
            if name.starts_with("all") {
                let cov = lane_coverage(&c, &p.host);
                gate.check(
                    seq < 32768 || cov == 256 * KVL,
                    &format!(
                        "{name}: selected tokens hold {cov} of {} (value, position) pairs",
                        256 * KVL
                    ),
                );
                let w = gate.bytes(gate.k.tc, wrong.dev, v)?;
                let d = rows_differing(&nr, &w, per);
                gate.check(
                    d > 0,
                    &format!("control: NaN read as max-finite differs in {d} of {rows} rows"),
                );
            }
        }
        report.push((rows, seq, timing(&gate, &random, &c)?));
        free_case(g, &c)?;
    }
    println!("rows   seq     phi ms   tc ms    nanor ms (min..max)   nanor/tc");
    for (rows, seq, [p, t, q]) in &report {
        println!(
            "{rows:5} {seq:6} {:8.3} {:8.3} {:8.3} ({:.3}..{:.3})  x{:.3}",
            p.0,
            t.0,
            q.0,
            q.1,
            q.2,
            q.0 / t.0
        );
        if *seq == 32768 && *rows > 4096 {
            gate.check(
                q.0 < t.0,
                &format!("nanor {:.3} ms below tc {:.3} ms at {rows} rows", q.0, t.0),
            );
        }
    }
    for p in pools.iter().map(|(_, p)| p).chain([&wrong, &random]) {
        g.free(p.dev)?;
    }
    if gate.failed {
        bail!("tc_nanor DSA attend gate FAILED");
    }
    println!("NANOR GATE PASS");
    Ok(())
}

/// 2026-10-15: phi, tc and the twin over `c` and the random pool, [`SAMPLES`] interleaved samples
/// each (order rotating, the two block-table variants alternating so the latent is not L2-resident
/// from the previous launch): (median, min, max) milliseconds per launch.
fn timing(gate: &Gate, pool: &Pool, c: &Case) -> Result<[(f64, f64, f64); 3]> {
    let rows = c.sl.len();
    let out = gate.g.alloc(rows * HEADS * KVL * 2)?;
    let entries = [gate.k.phi, gate.k.tc, gate.k.nanor];
    let mut t = [Vec::new(), Vec::new(), Vec::new()];
    for e in entries {
        gate.run(e, pool.dev, c.view(0, 0, rows), out)?;
    }
    for s in 0..SAMPLES {
        for k in 0..3 {
            let e = (s + k) % 3;
            gate.g.synchronize(0)?;
            let t0 = std::time::Instant::now();
            gate.run(
                entries[e],
                pool.dev,
                c.view((s + k) % c.sel.len(), 0, rows),
                out,
            )?;
            gate.g.synchronize(0)?;
            t[e].push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    gate.g.free(out)?;
    Ok(t.map(|mut v| {
        v.sort_by(f64::total_cmp);
        (v[v.len() / 2], v[0], v[v.len() - 1])
    }))
}
