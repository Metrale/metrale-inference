// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Byte gate and microbench of `METRALE_GLM_KDA_ONORM_IN_SCAN`: the 64-token chunked
//! scan that applies the KDA output's sigmoid-gated RMSNorm itself (`kda_chunk64_scan_onorm`)
//! against the path it replaces (`kda_chunk64_scan`, then `kda_o_norm_gated_bf16` over every core
//! row), at GLM-5.3's TP=3 per-rank shape (22 heads of 128, eps 1e-5), behind both prepares: the
//! plain one on the packed rows and the conv-fused one (`METRALE_GLM_KDA_CHUNK64_CONV`) on the
//! three projections.
//!
//! Arms per case: BASE (the base tree's `kda_chunk64_scan`, loaded from `GLM_PARITY_BASE_PTX` as
//! a second module, then the o_norm launch), PLAIN (this tree's `kda_chunk64_scan`, the o_norm
//! launch) and ONORM (`kda_chunk64_scan_onorm`, no o_norm launch).
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants:
//! - Exits with an error on the first byte of PLAIN or ONORM that differs from BASE: the BF16
//!   o_norm output (poison-filled before every arm, so a row the scan skips shows), the core
//!   output, the recurrent state, the conv state, or any slab's six factor buffers. The output
//!   gate is noise of scale 3 with every 1013th element +-40 (both ends of the sigmoid); the
//!   weight is noise.
//! - Exits with an error unless BASE's core output is finite, and unless both negative controls
//!   on ONORM change the o_norm output: one output-gate element of row T - 2 (head 5, channel 3)
//!   moved 4 across zero, and eps 1e-5 -> 1e-3.
//! - Exits with an error unless every arrival counter is zero after the timing loops, and a
//!   rerun of ONORM still equals BASE byte for byte.
//! - `GLM_PARITY_BASE_PTX` is required; `GLM_PARITY_FIDELITY_PTX` (this tree's `kda_chunk64.cu`
//!   compiled by the same command as the base) must equal the embedded module byte for byte.
//! - Times PLAIN (scan + o_norm launch) against ONORM as captured CUDA graphs, median [min..max]
//!   of 7 samples (`common/graph_timing.rs`).
//!
//!   GLM_PARITY_BASE_PTX=base.ptx GLM_PARITY_FIDELITY_PTX=new.ptx \
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --example glm5next_kda_onorm_in_scan_parity \
//!       --features cuda,gpu-examples

#[path = "common/glm5next_kda_chunk64_conv_arms.rs"]
mod arms;
#[path = "common/glm5next_kda_chunk64_case.rs"]
mod cases;
#[path = "common/glm5next_rows_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use anyhow::{Result, ensure};
use arms::{Arm, Gate, H, In, Kernels, Layout, QKV, Run, ScanNorm, fused, unfused};
use cases::{
    bumped_copy, case_rows, compare, ensure_finite, free_inputs, inputs, modules_with_base, run_arm,
};
use fixture::{Lcg, same, up_bf16};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda::chunk64::{CHUNK64 as C, CHUNK64_D as D, CHUNK64_SLAB};

/// 2026-10-14: GLM-5.3-Flash's `rms_norm_eps`, and the negative control's.
const EPS: f32 = 1e-5;
const EPS_CONTROL: f32 = 1e-3;

/// 2026-10-14: The arms' scans and the o_norm launch.
struct NormKernels {
    o_norm: KernelHandle,
    scan_onorm: KernelHandle,
    base_scan: KernelHandle,
}

/// 2026-10-14: The o_norm's inputs, output and the fused scan's scratch for one case.
struct Io {
    gate: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    part: DevicePtr,
    count: DevicePtr,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Path {
    Base,
    Plain,
    Onorm,
}

/// 2026-10-14: One case, one prepare: what every path launches on.
struct Ctx<'a> {
    g: &'a dyn GpuBackend,
    k: &'a Kernels,
    nk: &'a NormKernels,
    x: &'a In,
    a: &'a Arm,
    io: &'a Io,
    layout: Layout,
    conv_fused: bool,
}

impl Ctx<'_> {
    /// 2026-10-14: The prepare and scan per slab (`k`'s scan), the layout's conv path.
    fn chunks(&self, k: &Kernels, s: u64, rec: Option<&mut Vec<[Vec<u8>; 6]>>) -> Result<()> {
        let (g, x, a, layout) = (self.g, self.x, self.a, self.layout);
        if self.conv_fused {
            fused(
                g,
                k,
                k.prep_conv,
                Gate::Rows,
                x,
                a,
                layout,
                x.src(layout),
                s,
                rec,
            )
        } else {
            unfused(g, k, k.prep, Gate::Rows, x, a, layout, s, rec)
        }
    }

    /// 2026-10-14: `path` with the output gate `gate` and `eps`.
    fn go(
        &self,
        path: Path,
        gate: DevicePtr,
        eps: f32,
        s: u64,
        rec: Option<&mut Vec<[Vec<u8>; 6]>>,
    ) -> Result<()> {
        let k = match path {
            Path::Base => Kernels {
                scan: self.nk.base_scan,
                ..*self.k
            },
            Path::Plain => *self.k,
            Path::Onorm => Kernels {
                norm: Some(ScanNorm {
                    scan: self.nk.scan_onorm,
                    gate,
                    weight: self.io.weight,
                    out: self.io.out,
                    part: self.io.part,
                    count: self.io.count,
                    eps,
                }),
                ..*self.k
            },
        };
        self.chunks(&k, s, rec)?;
        if path == Path::Onorm {
            return Ok(());
        }
        KernelLaunch::new(self.g, self.nk.o_norm)
            .grid([(self.x.t * H) as u32, 1, 1])
            .block([D as u32, 1, 1])
            .arg_ptr(self.a.core)
            .arg_ptr(gate)
            .arg_ptr(self.io.weight)
            .arg_ptr(self.io.out)
            .arg_u32(D as u32)
            .arg_f32(eps)
            .launch(s)
    }

    /// 2026-10-14: `path` from the case's initial state with the output poison-filled: the run
    /// and the o_norm output's bytes.
    fn run(&self, s: u64, path: Path, gate: DevicePtr, eps: f32) -> Result<(Run, Vec<u8>)> {
        let n = self.x.t * QKV * 2;
        self.g.memset(self.io.out, 0xFF, n)?;
        let r = run_arm(self.g, s, self.x, self.a, &mut |rec| {
            self.go(path, gate, eps, s, rec)
        })?;
        Ok((r, fixture::read(self.g, self.io.out, n)?))
    }
}

/// 2026-10-14: The first differing byte of the run or the o_norm output, as an error.
fn compare_all(what: &str, want: &(Run, Vec<u8>), got: &(Run, Vec<u8>)) -> Result<()> {
    compare(what, &want.0, &got.0)?;
    same(&format!("{what} o_norm output"), &want.1, &got.1)
}

fn case(
    g: &dyn GpuBackend,
    s: u64,
    k: &Kernels,
    nk: &NormKernels,
    rng: &mut Lcg,
    t: usize,
) -> Result<()> {
    let x = inputs(g, rng, t)?;
    let a = Arm::new(g, &x)?;
    let mut graw = rng.vec(t * QKV, 3.0);
    for (i, v) in graw.iter_mut().enumerate() {
        if i % 1013 == 0 {
            *v = if i % 2 == 0 { 40.0 } else { -40.0 };
        }
    }
    let (part_n, count_n) = (CHUNK64_SLAB * H * C * (D / 32) * 4, CHUNK64_SLAB * H * 4);
    let io = Io {
        gate: up_bf16(g, &graw)?,
        weight: up_bf16(g, &rng.vec(D, 1.0))?,
        out: g.alloc(t * QKV * 2)?,
        part: g.alloc(part_n)?,
        count: g.alloc(count_n)?,
    };
    g.memset(io.part, 0, part_n)?;
    g.memset(io.count, 0, count_n)?;
    for (layout, conv_fused) in [(Layout::Packed, false), (Layout::Parts, true)] {
        let what = format!("T={t} {layout:?} conv_fused={conv_fused}");
        let c = Ctx {
            g,
            k,
            nk,
            x: &x,
            a: &a,
            io: &io,
            layout,
            conv_fused,
        };
        let base = c.run(s, Path::Base, io.gate, EPS)?;
        ensure_finite(&what, &base.0)?;
        for path in [Path::Plain, Path::Onorm] {
            let got = c.run(s, path, io.gate, EPS)?;
            compare_all(&format!("{what} {path:?} vs BASE"), &base, &got)?;
        }
        // 2026-10-14: Negative controls on ONORM: one output-gate element of row T - 2 (head 5,
        // channel 3) moved 4 across zero; eps 1e-5 -> 1e-3. The o_norm output must move.
        let bad = bumped_copy(g, io.gate, t * QKV * 2, ((t - 2) * QKV + 5 * D + 3) * 2)?;
        let ctl_gate = c.run(s, Path::Onorm, bad, EPS)?;
        g.free(bad)?;
        let ctl_eps = c.run(s, Path::Onorm, io.gate, EPS_CONTROL)?;
        for (name, ctl) in [("output gate", &ctl_gate), ("eps", &ctl_eps)] {
            let caught = compare_all(&format!("{what} CONTROL {name}"), &base, ctl).is_err();
            ensure!(
                caught && ctl.1 != base.1,
                "{what}: the {name} negative control was not caught (the o_norm output must differ)"
            );
        }
        println!(
            "{what}: BASE = PLAIN = ONORM byte for byte (o_norm output, core, states, {} slabs); NEG CONTROL: differs (o_norm output) on output gate 4 across zero and on eps - caught",
            base.0.slabs.len()
        );
        let copies = if t <= 300 { 8 } else { 2 };
        let t_old = timing::time_graph(g, s, copies, &mut |s| {
            c.go(Path::Plain, io.gate, EPS, s, None)
        })?;
        let t_new = timing::time_graph(g, s, copies, &mut |s| {
            c.go(Path::Onorm, io.gate, EPS, s, None)
        })?;
        timing::report(&format!("scan+o_norm {layout:?}"), t, t_old, t_new);
        let counts = fixture::read(g, io.count, count_n)?;
        ensure!(
            counts.iter().all(|&b| b == 0),
            "{what}: an arrival counter is not zero after the timing loops"
        );
        compare_all(
            &format!("{what} ONORM rerun vs BASE"),
            &base,
            &c.run(s, Path::Onorm, io.gate, EPS)?,
        )?;
    }
    for p in [io.gate, io.weight, io.out, io.part, io.count] {
        g.free(p)?;
    }
    a.free(g)?;
    free_inputs(g, x)
}

fn main() -> Result<()> {
    let modules = modules_with_base("kda_chunk64", "kda_chunk64_base")?;
    let gpu = MetraleCudaBackend::new(0, &modules)?;
    let g: &dyn GpuBackend = &gpu;
    let s = g.create_stream()?;
    let k = Kernels::load(g)?;
    let nk = NormKernels {
        o_norm: g.kernel("kda_layer_ops", "kda_o_norm_gated_bf16")?,
        scan_onorm: g.kernel("kda_chunk64", "kda_chunk64_scan_onorm")?,
        base_scan: g.kernel("kda_chunk64_base", "kda_chunk64_scan")?,
    };
    let mut rng = Lcg(0x6f6e_6f72_6d5f_7363);
    for t in case_rows()? {
        case(g, s, &k, &nk, &mut rng, t)?;
    }
    println!(
        "PASS: the o_norm inside the chunk64 scan is byte-identical to the scan then the o_norm launch"
    );
    Ok(())
}
