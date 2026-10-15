// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Gate and microbench for `METRALE_GLM_DSA_HOST_ROWS_PINNED` (the DSA host
//! path's uploads from page-locked staging, `glm5next_dsa::layer::HostRowsStaging`) on the
//! CUDA backend:
//! - Bytes: for an 8,208-row first sub-chunk, an 8,192-row continuation, a 335-row tail and a
//!   prefix-hit continuation at an unaligned position, the four device arrays (slots,
//!   positions, block table, lengths) are equal byte for byte staged and pageable.
//! - Ring: three staged calls of different bytes enqueued behind ~50 ms of device work (so the
//!   third rewrites the first's region while its copies may still be pending) each land their
//!   own bytes.
//! - Negative control: the rows one position on, staged, differ from the reference.
//! - Timing: per sub-chunk, eleven calls of one sub-chunk's bytes (one per DSA layer), each
//!   behind a few ms of device work, staged vs pageable: the host time per call (median, p10,
//!   p90 over every call of every run) and per sub-chunk.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error when any check fails; the last line is `RESULT: PASS`.
//!
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_dsa_host_rows_gate \
//!       --features cuda,gpu-examples

use std::time::Instant;

use anyhow::{Result, ensure};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_arch::glm5next_dsa::layer::{HostRowBytes, HostRowsStaging, upload_rows};

/// 2026-10-15: The KV and table block size of the cases (any value exercises the bytes).
const BLOCK: usize = 64;
/// 2026-10-15: Table entries (a 64k-token context) and the staging's row capacity (the
/// production sub-chunk height).
const BT_CAP: usize = 1024;
const MAX_ROWS: usize = 8208;
/// 2026-10-15: Busy work: D2D copies of this many bytes (about 2 ms each on GB10).
const BUSY_BYTES: usize = 256 << 20;
const RUNS: usize = 7;
const LAYERS: usize = 11;

/// 2026-10-15: (label, k, seq_len).
const CASES: [(&str, usize, usize); 4] = [
    ("first 8208", 8208, 0),
    ("cont 8192", 8192, 8208),
    ("tail 335", 335, 16400),
    ("prefix-hit 8208", 8208, 4133),
];

struct Dest([DevicePtr; 4]);

impl Dest {
    fn new(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self([
            g.alloc(MAX_ROWS * 8)?,
            g.alloc(MAX_ROWS * 4)?,
            g.alloc(BT_CAP * 4)?,
            g.alloc(MAX_ROWS * 4)?,
        ]))
    }

    fn parts<'a>(&self, r: &'a HostRowBytes) -> [(&'a [u8], DevicePtr); 4] {
        r.parts(self.0[0], self.0[1], self.0[2], self.0[3])
    }

    fn read(&self, g: &dyn GpuBackend, r: &HostRowBytes) -> Result<Vec<Vec<u8>>> {
        let lens = [r.slot.len(), r.q_pos.len(), r.bt.len(), r.sl.len()];
        let mut out = Vec::new();
        for (p, n) in self.0.iter().zip(lens) {
            let mut v = vec![0u8; n];
            g.copy_d2h(*p, &mut v)?;
            out.push(v);
        }
        Ok(out)
    }
}

/// 2026-10-15: A block table that is a fixed permutation of `0..BT_CAP` (not the identity,
/// so a wrong block index shows in the slots).
fn table() -> Vec<u32> {
    (0..BT_CAP as u32)
        .map(|i| (i * 389 + 17) % BT_CAP as u32)
        .collect()
}

fn rows(k: usize, seq_len: usize, bt: &[u32]) -> Result<HostRowBytes> {
    HostRowBytes::new(0, k, seq_len, bt, BLOCK, BLOCK, BT_CAP)
}

struct Busy {
    src: DevicePtr,
    dst: DevicePtr,
}

impl Busy {
    fn enqueue(&self, g: &dyn GpuBackend, copies: usize, stream: u64) -> Result<()> {
        for _ in 0..copies {
            g.copy_d2d_async(self.src, self.dst, BUSY_BYTES, stream)?;
        }
        Ok(())
    }
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

/// 2026-10-15: Host microseconds of every upload and of every sub-chunk's eleven, over
/// `RUNS` sub-chunks, each call behind `busy` copies.
fn time_mode(
    g: &dyn GpuBackend,
    staging: Option<&HostRowsStaging>,
    busy: &Busy,
    copies: usize,
    bt: &[u32],
    dest: &Dest,
    stream: u64,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let (mut calls, mut chunks) = (Vec::new(), Vec::new());
    for run in 0..RUNS {
        // 2026-10-15: A new sub-chunk's bytes every run, as consecutive sub-chunks.
        let r = rows(8208, (run % 2) * 8208, bt)?;
        let mut chunk = 0.0;
        for _ in 0..LAYERS {
            busy.enqueue(g, copies, stream)?;
            let t = Instant::now();
            upload_rows(g, staging, &dest.parts(&r), stream)?;
            let us = t.elapsed().as_secs_f64() * 1e6;
            calls.push(us);
            chunk += us;
        }
        chunks.push(chunk);
        g.synchronize(stream)?;
    }
    Ok((calls, chunks))
}

fn report(label: &str, mut calls: Vec<f64>, mut chunks: Vec<f64>) {
    let n = calls.len();
    println!(
        "{label}: per call median {:.1} us p10 {:.1} p90 {:.1} max {:.1} (n={n}); per sub-chunk \
         median {:.1} us p10 {:.1} p90 {:.1} (n={RUNS})",
        pct(&mut calls, 0.5),
        pct(&mut calls, 0.1),
        pct(&mut calls, 0.9),
        pct(&mut calls, 1.0),
        pct(&mut chunks, 0.5),
        pct(&mut chunks, 0.1),
        pct(&mut chunks, 0.9),
    );
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &[])?;
    let g: &dyn GpuBackend = &gpu;
    let stream = g.create_stream()?;
    let bt = table();
    let staging = HostRowsStaging::alloc(g, MAX_ROWS, BT_CAP)?;
    let busy = Busy {
        src: g.alloc(BUSY_BYTES)?,
        dst: g.alloc(BUSY_BYTES)?,
    };
    g.memset(busy.src, 0x5a, BUSY_BYTES)?;
    let (off, on, neg) = (Dest::new(g)?, Dest::new(g)?, Dest::new(g)?);

    // 2026-10-15: Bytes, case by case, both paths behind a little device work.
    let mut fails = 0usize;
    for (label, k, seq_len) in CASES {
        let r = rows(k, seq_len, &bt)?;
        busy.enqueue(g, 2, stream)?;
        upload_rows(g, None, &off.parts(&r), stream)?;
        upload_rows(g, Some(&staging), &on.parts(&r), stream)?;
        let shifted = rows(k, seq_len + 1, &bt)?;
        upload_rows(g, Some(&staging), &neg.parts(&shifted), stream)?;
        g.synchronize(stream)?;
        let (want, got, bad) = (off.read(g, &r)?, on.read(g, &r)?, neg.read(g, &r)?);
        let eq = want == got;
        let negctl = bad[0] != want[0] && bad[1] != want[1] && bad[3] != want[3];
        println!(
            "bytes {label} (k={k} seq_len={seq_len}, {} B): staged == pageable {eq}; negctl \
             (seq_len+1) differs {negctl}",
            r.slot.len() + r.q_pos.len() + r.bt.len() + r.sl.len()
        );
        fails += usize::from(!eq) + usize::from(!negctl);
    }

    // 2026-10-15: Ring under pending copies: three different staged calls behind ~50 ms of
    // device work; the third rewrites the first's region once its event has fired.
    let ring_dests = [Dest::new(g)?, Dest::new(g)?, Dest::new(g)?];
    let ring_rows = [
        rows(8208, 0, &bt)?,
        rows(8208, 8208, &bt)?,
        rows(335, 16400, &bt)?,
    ];
    busy.enqueue(g, 25, stream)?;
    for (d, r) in ring_dests.iter().zip(&ring_rows) {
        upload_rows(g, Some(&staging), &d.parts(r), stream)?;
    }
    g.synchronize(stream)?;
    for (i, (d, r)) in ring_dests.iter().zip(&ring_rows).enumerate() {
        upload_rows(g, None, &off.parts(r), stream)?;
        g.synchronize(stream)?;
        let eq = d.read(g, r)? == off.read(g, r)?;
        println!("ring call {i}: staged bytes landed intact {eq}");
        fails += usize::from(!eq);
    }

    // 2026-10-15: Busy chunk calibration, then the timing.
    let copies = 2;
    let t = Instant::now();
    busy.enqueue(g, copies, stream)?;
    g.synchronize(stream)?;
    println!(
        "busy work per call: {copies} x {} MiB D2D = {:.2} ms",
        BUSY_BYTES >> 20,
        t.elapsed().as_secs_f64() * 1e3
    );
    let (c_off, k_off) = time_mode(g, None, &busy, copies, &bt, &off, stream)?;
    let (c_on, k_on) = time_mode(g, Some(&staging), &busy, copies, &bt, &on, stream)?;
    report("pageable (lever off)", c_off, k_off);
    report("page-locked (lever on)", c_on, k_on);

    ensure!(fails == 0, "{fails} check(s) failed");
    println!("RESULT: PASS");
    Ok(())
}
