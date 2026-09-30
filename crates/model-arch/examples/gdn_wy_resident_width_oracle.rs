// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-30: Width oracle for the register-resident K = 2 and 3 WY verify GDN twins:
//! at every batch width 1..=32, in the batched verify's table form (`state_is_table` = 1),
//! eager and replayed from a captured CUDA graph, each resident twin must equal its base
//! kernel bit for bit on the output, every intermediate state and the final state.
//!
//! `wy_select::wy_resident_min_width` decides from which width the twins serve a launch;
//! each CTA is one (value head, sequence) pair and reads only its own row of the state
//! table, so a twin's bits cannot depend on the width. This oracle checks that claim on the
//! launches themselves, for both GB10 GDN shapes (Qwen3.8-27B: 16 key / 48 value heads;
//! Qwen3.6-35B-A3B: 16 / 32) and both state storages (FP32, FP16).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless every leg is bit-identical AND the negative control (one state element of
//!   one sequence perturbed before the twin's launch) is reported as a mismatch.
//!
//!   cargo run -p metrale-model-arch --release --example gdn_wy_resident_width_oracle \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::{bf16, f16};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const MAX_WIDTH: usize = 32;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

/// 2026-09-30: One verify's host inputs: `n` sequences of `k` tokens.
struct Inputs {
    nv: usize,
    n: usize,
    k: usize,
    f16_state: bool,
    /// 2026-09-30: `[n][nv * KD * VD]` raw state bytes in the storage format.
    h0: Vec<Vec<u8>>,
    q: Vec<u8>,
    key: Vec<u8>,
    val: Vec<u8>,
    gate: Vec<u8>,
    beta: Vec<u8>,
}

fn bf16_bytes(v: impl Iterator<Item = f64>) -> Vec<u8> {
    v.flat_map(|x| bf16::from_f64(x).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(v: impl Iterator<Item = f64>) -> Vec<u8> {
    v.flat_map(|x| (x as f32).to_le_bytes()).collect()
}

impl Inputs {
    fn new(rng: &mut Lcg, nv: usize, n: usize, k: usize, f16_state: bool) -> Self {
        let h_numel = nv * KD * VD;
        let h0 = (0..n)
            .map(|_| {
                (0..h_numel)
                    .flat_map(|_| {
                        let x = rng.r(-0.25, 0.25) as f32;
                        if f16_state {
                            f16::from_f32(x).to_bits().to_le_bytes().to_vec()
                        } else {
                            x.to_le_bytes().to_vec()
                        }
                    })
                    .collect()
            })
            .collect();
        let rows = n * k;
        Self {
            nv,
            n,
            k,
            f16_state,
            h0,
            q: bf16_bytes((0..rows * NK * KD).map(|_| rng.r(-1.0, 1.0))),
            key: bf16_bytes((0..rows * NK * KD).map(|_| rng.r(-1.0, 1.0))),
            val: bf16_bytes((0..rows * nv * VD).map(|_| rng.r(-1.0, 1.0))),
            gate: f32_bytes((0..rows * nv).map(|_| rng.r(0.80, 0.999))),
            beta: f32_bytes((0..rows * nv).map(|_| rng.r(0.0, 1.0))),
        }
    }
}

/// 2026-09-30: Everything a launch wrote: output, then each intermediate state and the
/// final state of every sequence, as raw bytes.
type Written = (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>);

fn upload(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn download(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut v = vec![0u8; n];
    g.copy_d2h(p, &mut v)?;
    Ok(v)
}

fn table(g: &dyn GpuBackend, ptrs: &[DevicePtr]) -> Result<DevicePtr> {
    let b: Vec<u8> = ptrs.iter().flat_map(|p| p.0.to_le_bytes()).collect();
    upload(g, &b)
}

/// 2026-09-30: Run `kernel` once on `inp` in table form, eagerly or as a captured graph
/// replayed once on `stream`. `poke` flips the low byte of sequence `poke`'s first state
/// element before the launch (the negative control).
fn run(
    g: &dyn GpuBackend,
    kernel: KernelHandle,
    inp: &Inputs,
    graphed: bool,
    stream: u64,
    poke: Option<usize>,
) -> Result<Written> {
    let (nv, n, k) = (inp.nv, inp.n, inp.k);
    let h_bytes = nv * KD * VD * if inp.f16_state { 2 } else { 4 };
    let mut owned = Vec::new();
    let mut hs = Vec::with_capacity(n);
    for (b, h) in inp.h0.iter().enumerate() {
        let mut h = h.clone();
        if poke == Some(b) {
            h[0] ^= 1;
        }
        hs.push(upload(g, &h)?);
    }
    let inters: Vec<Vec<DevicePtr>> = (0..k - 1)
        .map(|_| (0..n).map(|_| g.alloc(h_bytes)).collect::<Result<_>>())
        .collect::<Result<_>>()?;
    let h_tab = table(g, &hs)?;
    let i_tabs: Vec<DevicePtr> = inters.iter().map(|v| table(g, v)).collect::<Result<_>>()?;
    let (qp, kp, vp) = (
        upload(g, &inp.q)?,
        upload(g, &inp.key)?,
        upload(g, &inp.val)?,
    );
    let (gp, bp) = (upload(g, &inp.gate)?, upload(g, &inp.beta)?);
    let out_bytes = n * k * nv * VD * 2;
    let op = g.alloc(out_bytes)?;
    owned.extend([h_tab, qp, kp, vp, gp, bp, op]);
    owned.extend(i_tabs.iter().copied());

    let launch = |s: u64| -> Result<()> {
        let mut l = KernelLaunch::new(g, kernel)
            .grid([nv as u32, n as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(h_tab)
            .arg_ptr(qp)
            .arg_ptr(kp)
            .arg_ptr(vp)
            .arg_ptr(gp)
            .arg_ptr(bp)
            .arg_ptr(op);
        for &t in &i_tabs {
            l = l.arg_ptr(t);
        }
        l.arg_u32(n as u32)
            .arg_u32(NK as u32)
            .arg_u32(nv as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32((NK * KD) as u32)
            .arg_u32((nv * VD) as u32)
            .arg_u32(nv as u32)
            .arg_u32(1)
            .launch(s)
    };
    if graphed {
        g.begin_capture(stream)?;
        if let Err(e) = launch(stream) {
            g.abort_capture_if_active(stream);
            return Err(e);
        }
        let graph = g.end_capture(stream)?;
        anyhow::ensure!(graph.0 != 0, "graph capture returned no graph");
        g.launch_graph(graph, stream)?;
        g.synchronize(stream)?;
        g.destroy_graph(graph)?;
    } else {
        launch(stream)?;
        g.synchronize(stream)?;
    }

    let out = download(g, op, out_bytes)?;
    let mut states = Vec::with_capacity(k - 1);
    for v in &inters {
        states.push(
            v.iter()
                .map(|&p| download(g, p, h_bytes))
                .collect::<Result<Vec<_>>>()?
                .concat(),
        );
    }
    let finals = hs
        .iter()
        .map(|&p| download(g, p, h_bytes))
        .collect::<Result<Vec<_>>>()?;
    for p in owned
        .into_iter()
        .chain(hs)
        .chain(inters.into_iter().flatten())
    {
        let _ = g.free(p);
    }
    Ok((out, states, finals))
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let stream = g.create_stream()?;

    // 2026-09-30: (K, FP16 state, base kernel, resident twin).
    let legs: [(usize, bool, KernelHandle, KernelHandle); 4] = [
        (
            2,
            false,
            g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            g.kernel(
                "gated_delta_rule_wy2_resident",
                "gated_delta_rule_wy2_resident",
            )?,
        ),
        (
            3,
            false,
            g.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            g.kernel(
                "gated_delta_rule_wy3_resident",
                "gated_delta_rule_wy3_resident",
            )?,
        ),
        (
            2,
            true,
            g.kernel("gated_delta_rule_wy_f16", "gated_delta_rule_wy2_f16")?,
            g.kernel(
                "gated_delta_rule_wy2_resident_f16",
                "gated_delta_rule_wy2_resident_f16",
            )?,
        ),
        (
            3,
            true,
            g.kernel("gated_delta_rule_wy3_f16", "gated_delta_rule_wy3_f16")?,
            g.kernel(
                "gated_delta_rule_wy3_resident_f16",
                "gated_delta_rule_wy3_resident_f16",
            )?,
        ),
    ];

    let mut rng = Lcg(0x0DC8_2026_0930_0001);
    let (mut checked, mut failed) = (0usize, 0usize);
    for nv in [48usize, 32] {
        for &(k, f16_state, base, resident) in &legs {
            for n in 1..=MAX_WIDTH {
                let inp = Inputs::new(&mut rng, nv, n, k, f16_state);
                for graphed in [false, true] {
                    let want = run(g, base, &inp, graphed, stream, None)?;
                    let got = run(g, resident, &inp, graphed, stream, None)?;
                    checked += 1;
                    // 2026-09-30: A launch that did not run would leave the output unwritten
                    // and every state at its initial value; both kernels must have written.
                    let ran = |w: &Written| w.0.iter().any(|&x| x != 0) && w.2 != inp.h0;
                    if !ran(&want) || !ran(&got) {
                        failed += 1;
                        println!(
                            "NO-OP LAUNCH nv={nv} K={k} f16={f16_state} n={n} graphed={graphed}"
                        );
                    }
                    if want != got {
                        failed += 1;
                        println!("MISMATCH nv={nv} K={k} f16={f16_state} n={n} graphed={graphed}");
                    }
                }
            }
            println!("nv={nv} K={k} f16={f16_state}: widths 1..={MAX_WIDTH}, eager + graphed");
        }
    }

    // 2026-09-30: Negative control: the comparison must see a one-bit change in one
    // sequence's initial state, at a width the threshold change newly admits.
    let (k, f16_state, base, resident) = legs[2];
    let inp = Inputs::new(&mut rng, 48, 8, k, f16_state);
    let want = run(g, base, &inp, false, stream, None)?;
    let poked = run(g, resident, &inp, false, stream, Some(5))?;
    let control_seen = want != poked;
    println!(
        "negative control (sequence 5 of 8, one state bit flipped): {}",
        if control_seen { "DETECTED" } else { "MISSED" }
    );

    println!("{checked} launches compared, {failed} mismatched");
    if failed > 0 || !control_seen {
        std::process::exit(1);
    }
    println!(
        "PASS: every resident twin is bit-identical to its base kernel at widths 1..={MAX_WIDTH}"
    );
    Ok(())
}
