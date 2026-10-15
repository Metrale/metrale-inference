// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Can an NCCL send/recv CTA start while a prefill compute kernel holds every SM?
//! The SP pipeline (overlapping one sub-chunk's exchange with the next one's compute) needs the
//! answer to be yes under the MoE gate/up GEMM, whose two CTAs per SM leave no room for the
//! 98.5 KB `ncclDevKernel_SendRecv` CTA; the NCCL CTA must win an SM as a GEMM CTA retires.
//!
//! Per load (`load.rs`), per priority arm, per transfer size:
//! 1. solo: the grouped self send/recv alone on the comm stream, and the load chain alone on
//!    the main stream;
//! 2. co-run: the chain on the main stream; once its first launch has finished and the host
//!    has waited `GLM_CORUN_DELAY_US` more (the second launch now fills the SMs), the transfer
//!    on the comm stream. The verdict (`verdict.rs`) compares when the transfer ends against
//!    when the chain ends.
//!
//! Arms: `comm>main` is the serve with `METRALE_MAIN_STREAM_UNDER_GREATEST=1` (main stream one
//! level under the greatest, the SP comm stream at the greatest); `tie` puts both at the
//! greatest. The transfer is a one-rank communicator sending to itself through the serve's
//! `NcclBackend` (`send_to`/`recv_from` in one group): the same kernel and CTA shape as the
//! cross-node exchange, with a device-local copy in place of the network wait, so it measures
//! when the CTA gets an SM, not the link.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on any driver, NCCL or launch failure; prints one
//! `CORUN_VERDICT` line per (load, arm, size).
//!
//! ```text
//! METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=glm-5.3-flash METRALE_TARGET_QUANT=nvfp4 \
//! METRALE_MAIN_STREAM_HIGH_PRIORITY=1 METRALE_MAIN_STREAM_UNDER_GREATEST=1 \
//! GLM_BENCH_GPU_ORDINAL=0 GLM_CORUN_NCCL_PORT=<free port> NCCL_MAX_NCHANNELS=8 \
//! NCCL_MIN_NCHANNELS=8 cargo run -p metrale-model-arch --release \
//!     --example glm5next_comm_corun_probe --features cuda,nccl-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_comm::{CommBackend, NcclBackend};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use std::time::{Duration, Instant};

mod cuda;
#[allow(dead_code)]
#[path = "../glm5next_moe_wide_bench/device.rs"]
mod device;
#[allow(dead_code)]
#[path = "../common/glm5next_dsa_hb_fixture.rs"]
mod fixture;
#[allow(dead_code)]
#[path = "../glm5next_moe_prefill_w4a4_bench/launch.rs"]
mod launch;
mod load;
#[allow(dead_code)]
#[path = "../common/glm5next_dsa_prefill_fixture.rs"]
mod prefill;
mod verdict;

use device::env_or;
use launch::Kern;
use load::{Gu, Load, Phi};
use verdict::{Corun, judge};

/// 2026-10-15: The SP exchange piece per peer (684 rows x 4096 x BF16) and the SP shard per
/// peer (2736 rows), the two message sizes of the 32k prefill.
const BYTES: [usize; 2] = [684 * 4096 * 2, 2736 * 4096 * 2];

struct Rig<'a> {
    gpu: &'a dyn GpuBackend,
    nccl: NcclBackend,
    comm: u64,
    send: DevicePtr,
    recv: DevicePtr,
    trials: usize,
    chain: usize,
    delay: Duration,
    ev: [u64; 5],
}

impl Rig<'_> {
    fn transfer(&self, bytes: usize) -> Result<()> {
        self.nccl.group_start()?;
        self.nccl.send_to(self.send.0, bytes, 0, self.comm)?;
        self.nccl.recv_from(self.recv.0, bytes, 0, self.comm)?;
        self.nccl.group_end()
    }

    fn comm_solo(&self, bytes: usize) -> Result<f64> {
        let [c0, c1, ..] = self.ev;
        cuda::record(c0, self.comm)?;
        self.transfer(bytes)?;
        cuda::record(c1, self.comm)?;
        cuda::sync(c1)?;
        cuda::elapsed_us(c0, c1)
    }

    fn load_solo(&self, l: &dyn Load, main: u64) -> Result<f64> {
        let [.., e0, _, e1] = self.ev;
        cuda::record(e0, main)?;
        for _ in 0..self.chain {
            l.launch(self.gpu, main)?;
        }
        cuda::record(e1, main)?;
        cuda::sync(e1)?;
        cuda::elapsed_us(e0, e1)
    }

    fn corun(&self, l: &dyn Load, main: u64, bytes: usize) -> Result<Corun> {
        let [c0, c1, e0, ea, e1] = self.ev;
        cuda::record(e0, main)?;
        l.launch(self.gpu, main)?;
        cuda::record(ea, main)?;
        for _ in 1..self.chain {
            l.launch(self.gpu, main)?;
        }
        cuda::record(e1, main)?;
        cuda::sync(ea)?;
        let t = Instant::now();
        while t.elapsed() < self.delay {
            std::hint::spin_loop();
        }
        cuda::record(c0, self.comm)?;
        self.transfer(bytes)?;
        cuda::record(c1, self.comm)?;
        cuda::sync(c1)?;
        cuda::sync(e1)?;
        Ok(Corun {
            comm_us: cuda::elapsed_us(c0, c1)?,
            tail_us: cuda::elapsed_us(c0, e1)?,
            load_us: cuda::elapsed_us(e0, e1)?,
        })
    }

    fn arm(&self, l: &dyn Load, arm: &str, main: u64, bytes: usize) -> Result<()> {
        let n = self.trials;
        let comm: Vec<f64> = (0..n)
            .map(|_| self.comm_solo(bytes))
            .collect::<Result<_>>()?;
        let load: Vec<f64> = (0..n)
            .map(|_| self.load_solo(l, main))
            .collect::<Result<_>>()?;
        let co: Vec<Corun> = (0..n)
            .map(|_| self.corun(l, main, bytes))
            .collect::<Result<_>>()?;
        let r = judge(&comm, &load, &co)?;
        let s = |x: verdict::Stats| format!("{:9.1} [{:9.1},{:9.1}]", x.median, x.p10, x.p90);
        println!(
            "{} arm={arm:9} bytes={bytes:9} n={n}: comm solo {} co-run {} | tail {} | load solo {} co-run {} | comm/tail {:.3} [{:.3},{:.3}]",
            l.name(),
            s(r.comm_solo),
            s(r.comm_corun),
            s(r.tail),
            s(r.load_solo),
            s(r.load_corun),
            r.ratio.median,
            r.ratio.p10,
            r.ratio.p90
        );
        println!(
            "CORUN_VERDICT load={} arm={arm} bytes={bytes} verdict={:?} comm_corun_over_solo={:.2} load_corun_over_solo={:.3}",
            l.name(),
            r.verdict,
            r.comm_corun.median / r.comm_solo.median,
            r.load_corun.median / r.load_solo.median
        );
        Ok(())
    }
}

fn main() -> Result<()> {
    let ordinal = std::env::var("GLM_BENCH_GPU_ORDINAL")
        .context("GLM_BENCH_GPU_ORDINAL names the GPU to run on")?
        .parse()?;
    let port: u16 = std::env::var("GLM_CORUN_NCCL_PORT")
        .context("GLM_CORUN_NCCL_PORT names a free local port for the NCCL id bootstrap")?
        .parse()?;
    // 2026-10-15: The serve's ALLU priorities: the process stream one level under the greatest
    // (`comm>main`), the SP comm stream and the `tie` main stream at the greatest.
    for v in [
        "METRALE_MAIN_STREAM_HIGH_PRIORITY",
        "METRALE_MAIN_STREAM_UNDER_GREATEST",
    ] {
        ensure!(
            std::env::var(v).as_deref() == Ok("1"),
            "{v}=1 is required: the comm>main arm runs on the process stream it places"
        );
    }
    let target = metrale_kernels::ptx_for_exact_target("glm-5.3-flash", "nvfp4")
        .context("glm-5.3-flash nvfp4 target")?;
    let gpu = MetraleCudaBackend::new(ordinal, &target.modules)?;
    let g: &dyn GpuBackend = &gpu;
    let under = gpu.registry().raw_stream();
    let tie = g.create_stream_greatest_priority()?;
    let comm = g.create_stream_greatest_priority()?;
    println!(
        "streams: main {under:#x} (under the greatest), tie {tie:#x} and comm {comm:#x} (greatest)"
    );
    let nccl = NcclBackend::new(0, 1, "127.0.0.1", port, under, 0, 0)?;
    let max = BYTES[1];
    let (send, recv) = (g.alloc(max)?, g.alloc(max)?);
    g.copy_h2d(&vec![0x3c; max], send)?;
    let m = "w4a4_grouped_mx";
    let q = "w4a4_gemv_mx_moe";
    let kn = Kern {
        sort: g.kernel("moe", "moe_sort_by_expert")?,
        tiles: g.kernel(m, "w4a4_grouped_tiles")?,
        quant: g.kernel(q, "w4a4_quant_rows_static")?,
        quant_k64: g.kernel(q, "w4a4_quant_rows_static_k64")?,
        swiglu: g.kernel("glm5next_ffn", "glm5next_swiglu_clamp")?,
        slots: g.kernel(q, "w4a4_gemv_mx8_moe_slots")?,
        slots_k64: g.kernel(q, "w4a4_gemv_mx8_moe_slots_k64")?,
        mx: [g.kernel(m, "w4a4_grouped_mx_m128")?; 4],
        mx_k64: [g.kernel(m, "w4a4_grouped_mx_m128_k64")?; 4],
        mx32: g.kernel("w4a4_gemv_mx", "w4a4_gemv_mx32")?,
        w4a16: g.kernel("moe_w4a16", "moe_w4a16_grouped_gemm_ptrtable_bt_m16_k128")?,
    };
    let gu = Gu::new(
        g,
        &kn,
        g.kernel(m, "w4a4_grouped_gu_m128")?,
        env_or("GLM_CORUN_GU_ROWS", 4096),
    )?;
    let phi = Phi::new(g)?;
    let rig = Rig {
        gpu: g,
        nccl,
        comm,
        send,
        recv,
        trials: env_or("GLM_CORUN_TRIALS", 15),
        chain: env_or("GLM_CORUN_CHAIN", 6),
        delay: Duration::from_micros(env_or("GLM_CORUN_DELAY_US", 1000) as u64),
        ev: [
            cuda::event()?,
            cuda::event()?,
            cuda::event()?,
            cuda::event()?,
            cuda::event()?,
        ],
    };
    ensure!(rig.chain >= 3, "GLM_CORUN_CHAIN must be at least 3");
    let loads: [&dyn Load; 2] = [&gu, &phi];
    for l in loads {
        for main in [under, tie] {
            l.launch(g, main)?;
        }
        rig.transfer(BYTES[0])?;
        g.synchronize(under)?;
        g.synchronize(tie)?;
        g.synchronize(comm)?;
        for (arm, main) in [("comm>main", under), ("tie", tie)] {
            for bytes in BYTES {
                rig.arm(l, arm, main, bytes)?;
            }
        }
    }
    let got = device::read(g, recv, max)?;
    ensure!(
        got.iter().all(|&b| b == 0x3c),
        "the self transfer did not deliver the sent bytes"
    );
    println!("CORUN_DONE");
    Ok(())
}
