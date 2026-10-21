// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Identity gate and microbench of the screened LM head
//! (`kernels/gb10/common/screen_head.cu`) on GLM-5.3-Flash's real BF16 head (154,880 x 4,096)
//! under the serve's three-rank vocab split.
//!
//! - `gate [rows]`: for every activation row, the screened token (three shards keyed apart and
//!   combined by `screen_head_resolve`'s peer max, as the one-shot exchange hands them) must be
//!   `argmax_bf16` over the full head's logits (`dense_gemv_bf16_batchm` at M = 2,
//!   `dense_gemv_bf16` at M = 1). Also: 4,096 recomputed logits must equal the full head's bytes.
//!   Prints the candidate-set distribution. Every M = 1 row also takes a masked pick
//!   (`masked.rs`). Exit status 1 on any difference.
//! - `sabotage half|noquant [rows]`: the same with the bound halved, or with only its rounding
//!   terms; it must go RED on the adversarial family.
//! - `bench`: the old shard head against the screen at M = 1 and 2, one rank's shard.
//! - `census [rows]`: host only, the candidate sets of a 4-bit screen beside the INT8 one's
//!   (`census.rs`).
//!
//!   METRALE_TARGET_MODEL=glm-5.3-flash cargo run -p metrale-gpu-examples --release \
//!       --example screen_head_gate --features cuda,gpu-examples -- gate 100000

mod bench;
mod census;
mod masked;
mod rows;

use std::io::{Read, Seek, SeekFrom};
use std::time::Instant;

use anyhow::{Context, Result, bail, ensure};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::ops::{
    self, SCREEN_MAX_M, SCREEN_STATE_BYTES, SCREEN_STATE_COUNTS, ScreenKernels, ScreenShard,
};
use metrale_model_layers::weight_map::DenseWeight;
use rows::{Family, REALISTIC, Rng};

const CKPT: &str = "/srv/models/glm-5.3-flash-nvfp4";
const V: usize = 154_880;
const K: usize = 4096;
/// 2026-10-21: The serve's split (`lm_head_vocab_tp.rs`, 154,880 rows over 3 ranks).
const SPLIT: [(usize, usize); 3] = [(0, 51_648), (51_648, 51_648), (103_296, 51_584)];

/// 2026-10-21: A tensor's bytes from the checkpoint's sharded safetensors.
fn tensor(name: &str) -> Result<Vec<u8>> {
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!(
        "{CKPT}/model.safetensors.index.json"
    ))?)?;
    let file = idx["weight_map"][name]
        .as_str()
        .context("not in the index")?;
    let mut f = std::fs::File::open(format!("{CKPT}/{file}"))?;
    let mut n = [0u8; 8];
    f.read_exact(&mut n)?;
    let n = u64::from_le_bytes(n);
    let mut head = vec![0u8; n as usize];
    f.read_exact(&mut head)?;
    let head: serde_json::Value = serde_json::from_slice(&head)?;
    ensure!(
        head[name]["dtype"] == "BF16",
        "{name}: {}",
        head[name]["dtype"]
    );
    let a = head[name]["data_offsets"][0].as_u64().context("offset")?;
    let b = head[name]["data_offsets"][1].as_u64().context("offset")?;
    f.seek(SeekFrom::Start(8 + n + a))?;
    let mut out = vec![0u8; (b - a) as usize];
    f.read_exact(&mut out)?;
    Ok(out)
}

fn bytes_of(x: &[f32]) -> Vec<u8> {
    x.iter()
        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
        .collect()
}

struct Rig<'a> {
    g: &'a dyn GpuBackend,
    w: DevicePtr,
    shards: Vec<ScreenShard>,
    /// 2026-10-21: The three state blocks back to back: shard 0's, then the peers' slots.
    states: DevicePtr,
    k: ScreenKernels,
    k_quant: KernelHandle,
    k_exact: KernelHandle,
    k_batchm: KernelHandle,
    k_gemv: KernelHandle,
    k_argmax: KernelHandle,
    x: DevicePtr,
    logits: DevicePtr,
    probe: DevicePtr,
    out: DevicePtr,
}

impl Rig<'_> {
    fn u32s(&self, p: DevicePtr, n: usize) -> Vec<u32> {
        let mut b = vec![0u8; n * 4];
        self.g.copy_d2h(p, &mut b).unwrap();
        b.chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// 2026-10-21: The full head's tokens for the `m` rows at `x` (M = 1 takes dense_gemv_bf16).
    fn reference(&self, m: usize) -> Result<Vec<u32>> {
        let s = self.g.default_stream();
        let wt = DenseWeight { weight: self.w };
        if m == 1 {
            ops::dense_gemv(
                self.g,
                self.k_gemv,
                self.x,
                &wt,
                self.logits,
                V as u32,
                K as u32,
                s,
            )?;
        } else {
            ops::dense_gemv_batchm(
                self.g,
                self.k_batchm,
                self.x,
                &wt,
                self.logits,
                m as u32,
                V as u32,
                K as u32,
                V as u32,
                s,
            )?;
        }
        for t in 0..m {
            ops::argmax_bf16(
                self.g,
                self.k_argmax,
                self.logits.offset(t * V * 2),
                self.out.offset(16 + t * 4),
                V as u32,
                s,
            )?;
        }
        self.g.synchronize(s)?;
        Ok(self.u32s(self.out.offset(16), m))
    }

    /// 2026-10-21: The screened tokens, and the candidates per row (all shards).
    fn screened(&self, m: usize, kappa: f32) -> Result<(Vec<u32>, Vec<u32>)> {
        let s = self.g.default_stream();
        for sh in &self.shards {
            ops::screen_head_keys(
                self.g,
                &self.k,
                sh,
                self.x,
                m as u32,
                DevicePtr::NULL,
                kappa,
                s,
            )?;
        }
        ops::screen_head_resolve(
            self.g,
            self.k.resolve,
            self.states,
            self.states.offset(SCREEN_STATE_BYTES),
            SCREEN_STATE_BYTES as u64,
            2,
            m as u32,
            self.out,
            s,
        )?;
        self.g.synchronize(s)?;
        let mut cand = vec![0u32; m];
        for i in 0..3 {
            let c = self.u32s(
                self.states
                    .offset(i * SCREEN_STATE_BYTES + SCREEN_STATE_COUNTS),
                m,
            );
            for t in 0..m {
                cand[t] += c[t];
            }
        }
        Ok((self.u32s(self.out, m), cand))
    }

    /// 2026-10-21: Recomputed logits at `ids` (after `reference` on the same rows) must be the
    /// full head's bytes; returns how many differ.
    fn exact_ids(&self, m: usize, ids: &[u32]) -> Result<usize> {
        let s = self.g.default_stream();
        for (n, &id) in ids.iter().enumerate() {
            let row = (n % m) as u32;
            ops::screen_head_exact_id(
                self.g,
                self.k_exact,
                self.w,
                self.x,
                K as u32,
                (row, id, V as u32),
                self.logits,
                DevicePtr::NULL,
                self.probe.offset(n * 2),
                s,
            )?;
        }
        self.g.synchronize(s)?;
        let mut full = vec![0u8; m * V * 2];
        let mut got = vec![0u8; ids.len() * 2];
        self.g.copy_d2h(self.logits, &mut full)?;
        self.g.copy_d2h(self.probe, &mut got)?;
        Ok(ids
            .iter()
            .enumerate()
            .filter(|&(n, &id)| {
                let at = ((n % m) * V + id as usize) * 2;
                full[at..at + 2] != got[n * 2..n * 2 + 2]
            })
            .count())
    }
}

fn pct(v: &mut [u32], p: f64) -> u32 {
    v.sort_unstable();
    v[((v.len() - 1) as f64 * p).round() as usize]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mode = args.first().map(String::as_str).unwrap_or("gate");
    let t0 = Instant::now();
    let head_bytes = tensor("lm_head.weight")?;
    ensure!(head_bytes.len() == V * K * 2);
    let gamma: Vec<f32> = tensor("model.language_model.norm.weight")?
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let head: Vec<bf16> = head_bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    if mode == "census" {
        let rows = args.get(1).map_or(Ok(500), |s| s.parse())?;
        return census::census(&head, &gamma, rows);
    }
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;
    let w = g.alloc(V * K * 2)?;
    g.copy_h2d(&head_bytes, w)?;
    drop(head_bytes);
    eprintln!("head loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let k = ScreenKernels {
        scan: g.kernel("screen_head", "screen_head_scan")?,
        rescore: g.kernel("screen_head", "screen_head_rescore")?,
        resolve: g.kernel("screen_head", "screen_head_resolve")?,
    };
    let k_quant = g.kernel("screen_head", "screen_head_quant_int8")?;
    let states = g.alloc(3 * SCREEN_STATE_BYTES)?;
    let bad = g.alloc(4)?;
    g.memset(bad, 0, 4)?;
    let mut shards = Vec::new();
    for (i, &(b, n)) in SPLIT.iter().enumerate() {
        let sh = ScreenShard {
            w: w.offset(b * K * 2),
            q: g.alloc(n * K)?,
            s: g.alloc(n * K / 32)?,
            upper: g.alloc(SCREEN_MAX_M as usize * n * 4)?,
            state: states.offset(i * SCREEN_STATE_BYTES),
            rows: n as u32,
            k: K as u32,
            base: b as u32,
        };
        ops::screen_head_quant_int8(g, k_quant, sh.w, sh.q, sh.s, bad, sh.rows, sh.k, 0)?;
        shards.push(sh);
    }
    g.synchronize(0)?;
    let mut b = [0u8; 4];
    g.copy_d2h(bad, &mut b)?;
    ensure!(
        u32::from_le_bytes(b) == 0,
        "screen build flagged {:#x}",
        u32::from_le_bytes(b)
    );
    let rig = Rig {
        g,
        w,
        shards,
        states,
        k,
        k_quant,
        k_exact: g.kernel("screen_head", "screen_head_exact_id")?,
        k_batchm: g.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?,
        k_gemv: g.kernel("gemv", "dense_gemv_bf16")?,
        k_argmax: g.kernel("argmax", "argmax_bf16")?,
        x: g.alloc(SCREEN_MAX_M as usize * K * 2)?,
        logits: g.alloc(2 * V * 2)?,
        probe: g.alloc(2 * V * 2)?,
        out: g.alloc(64)?,
    };
    let _ = rig.k_quant;
    let rows: usize = args
        .get(if mode == "sabotage" { 2 } else { 1 })
        .map_or(Ok(100_000), |s| s.parse())?;
    match mode {
        "gate" => gate(
            &rig,
            &head,
            &gamma,
            rows,
            ops::screen_kappa(K as u32),
            "exact",
        ),
        "sabotage" => {
            let (quant, rounding) = ops::screen_kappa_parts(K as u32);
            let (kappa, name) = match args.get(1).map(String::as_str) {
                Some("half") => (ops::f32_up((quant + rounding) / 2.0), "half"),
                Some("noquant") => (ops::f32_up(rounding), "noquant"),
                other => bail!("sabotage half|noquant, got {other:?}"),
            };
            gate(&rig, &head, &gamma, rows, kappa, name)
        }
        "bench" => bench::bench(&rig, &head, &gamma),
        other => bail!("mode gate|sabotage|bench, got {other}"),
    }
}

fn gate(
    rig: &Rig,
    head: &[bf16],
    gamma: &[f32],
    rows: usize,
    kappa: f32,
    name: &str,
) -> Result<()> {
    let mut g = Rng(0x5eed_2026_1021);
    let fams: Vec<Family> = REALISTIC
        .iter()
        .copied()
        .chain([Family::Adversarial])
        .collect();
    let mut stats: Vec<(Family, usize, usize, Vec<u32>)> =
        fams.iter().map(|&f| (f, 0, 0, Vec::new())).collect();
    let (mut done, mut byte_diffs, mut probes) = (0usize, 0usize, 0usize);
    let mut masked = masked::Masked::new(rig.g)?;
    let t0 = Instant::now();
    let mut batch = 0usize;
    while done < rows {
        let fi = batch % fams.len();
        let m = if batch % 20 == 19 { 1 } else { 2 };
        let xs: Vec<f32> = (0..m)
            .flat_map(|_| match fams[fi] {
                Family::Adversarial => rows::adversarial(&mut g, head, K, V),
                f => rows::realistic(f, &mut g, head, K, gamma),
            })
            .collect();
        rig.g.copy_h2d(&bytes_of(&xs), rig.x)?;
        let want = rig.reference(m)?;
        let (got, cand) = rig.screened(m, kappa)?;
        if m == 1 {
            masked.check(rig, &mut g, want[0], kappa)?;
        }
        if batch.is_multiple_of(50) && name == "exact" {
            let ids: Vec<u32> = (0..64)
                .map(|_| g.below(V) as u32)
                .chain(want.iter().copied())
                .collect();
            byte_diffs += rig.exact_ids(m, &ids)?;
            probes += ids.len();
        }
        let st = &mut stats[fi];
        for t in 0..m {
            st.1 += 1;
            if got[t] != want[t] {
                st.2 += 1;
                if st.2 <= 3 {
                    eprintln!(
                        "  {name}: {:?} row: screened {} != full head {}",
                        fams[fi], got[t], want[t]
                    );
                }
            }
            st.3.push(cand[t]);
        }
        done += m;
        batch += 1;
    }
    println!(
        "screen_head_gate {name}: kappa {kappa:e}, {done} rows, {:.0}s",
        t0.elapsed().as_secs_f64()
    );
    println!(
        "{:<12} {:>7} {:>6} {:>6} {:>6} {:>6} {:>7}",
        "family", "rows", "diff", "p50", "p99", "max", "mean"
    );
    let mut all = Vec::new();
    let mut diffs = 0;
    for (f, n, d, mut c) in stats {
        let mean = c.iter().map(|&v| f64::from(v)).sum::<f64>() / c.len().max(1) as f64;
        all.extend_from_slice(&c);
        diffs += d;
        let (p50, p99, mx) = (pct(&mut c, 0.5), pct(&mut c, 0.99), pct(&mut c, 1.0));
        println!(
            "{:<12} {n:>7} {d:>6} {p50:>6} {p99:>6} {mx:>6} {mean:>7.1}",
            format!("{f:?}")
        );
    }
    println!(
        "all: candidates (3 shards summed) p50 {} p99 {} max {}",
        pct(&mut all.clone(), 0.5),
        pct(&mut all.clone(), 0.99),
        pct(&mut all, 1.0)
    );
    if name == "exact" {
        println!(
            "recomputed logits: {probes} probes, {byte_diffs} differ from the full head's bytes"
        );
    }
    println!(
        "masked one-row picks: {}, {} decided by the masked key, {} differ",
        masked.picks, masked.q_decided, masked.diffs
    );
    diffs += masked.diffs;
    if diffs != 0 || byte_diffs != 0 {
        println!("RED: {diffs} tokens differ ({name})");
        std::process::exit(1);
    }
    println!("GREEN: every token identical ({name})");
    Ok(())
}
