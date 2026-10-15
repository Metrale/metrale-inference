// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Cross-node NCCL exchange and all-reduce time at the GLM-5.3 32k prefill's SP
//! message sizes (`plan.rs`), one process per rank, through the serve's `NcclBackend`. Run
//! once per NCCL transport setting (`NCCL_NET_GDR_LEVEL`, `NCCL_NET_GDR_C2C`) and with
//! `GLM_XCHG_REGISTER=1` (the send, receive and all-reduce buffers registered with
//! `ncclCommRegister` before the first operation) to compare a host-staged transport against
//! GPUDirect RDMA and registered user buffers on the same boxes.
//!
//! Every rank times its own stream with events (median of `GLM_XCHG_REPS` after 3 warm-ups)
//! and prints one `XCHG` line per pattern; after the timed patterns it checks one exchange's
//! received bytes and one all-reduce of ones, then prints `XCHG_DONE`.
//!
//! Owner: model-arch examples (GLM-5.3).
//! Invariants: exits with an error on any driver or NCCL failure, a failed registration, or a
//! received byte that differs from its sender's fill.
//!
//! ```text
//! GLM_BENCH_GPU_ORDINAL=0 GLM_XCHG_RANK=<r> GLM_XCHG_WORLD=3 GLM_XCHG_MASTER=<rank-0 address> \
//! GLM_XCHG_PORT=<port> GLM_XCHG_TAG=<label> GLM_XCHG_REGISTER=0 <NCCL env> \
//! cargo run -p metrale-model-arch --release --example glm5next_comm_xchg_probe \
//!     --features cuda,nccl-examples
//! ```

use anyhow::{Context, Result, ensure};
use metrale_comm::{CommBackend, NcclBackend};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

#[path = "../glm5next_comm_corun_probe/cuda.rs"]
mod cuda;
mod plan;
#[allow(dead_code)]
#[path = "../glm5next_comm_corun_probe/verdict.rs"]
mod verdict;

use plan::{Kind, PATTERNS, Pattern, ROW_BYTES};

/// 2026-10-15: Untimed repetitions before each pattern's timed ones (connection setup and
/// the first-use registration of NCCL's own buffers land here).
const WARMUP: usize = 3;
/// 2026-10-15: One contention copy (512 MiB each way), four per timed repetition: longer than
/// the contended section, so the DRAM traffic spans it.
const CONTENTION_BYTES: usize = 512 << 20;
const CONTENTION_COPIES: usize = 4;

fn var(name: &str) -> Result<String> {
    std::env::var(name).with_context(|| format!("{name} is required"))
}

fn parsed<T: std::str::FromStr>(name: &str) -> Result<T>
where
    T::Err: std::error::Error + Send + Sync + 'static,
{
    var(name)?
        .parse()
        .with_context(|| format!("{name} does not parse"))
}

struct Rig<'a> {
    gpu: &'a dyn GpuBackend,
    nccl: NcclBackend,
    rank: usize,
    world: usize,
    stream: u64,
    side: u64,
    send: DevicePtr,
    recv: DevicePtr,
    ar: DevicePtr,
    copy: [DevicePtr; 2],
    ev: [u64; 2],
}

impl Rig<'_> {
    fn xchg(&self, bytes: usize) -> Result<()> {
        self.nccl.group_start()?;
        for q in (0..self.world).filter(|&q| q != self.rank) {
            let off = (q * bytes) as u64;
            self.nccl
                .send_to(self.send.0 + off, bytes, q, self.stream)?;
            self.nccl
                .recv_from(self.recv.0 + off, bytes, q, self.stream)?;
        }
        self.nccl.group_end()
    }

    fn op(&self, p: &Pattern) -> Result<()> {
        for _ in 0..p.times {
            match p.kind {
                Kind::Xchg => self.xchg(p.bytes())?,
                Kind::AllReduce => self.nccl.all_reduce(self.ar.0, p.bytes())?,
            }
        }
        Ok(())
    }

    fn timed(&self, p: &Pattern) -> Result<f64> {
        if p.contended {
            for _ in 0..CONTENTION_COPIES {
                self.gpu
                    .copy_d2d_async(self.copy[0], self.copy[1], CONTENTION_BYTES, self.side)?;
            }
        }
        let [e0, e1] = self.ev;
        cuda::record(e0, self.stream)?;
        self.op(p)?;
        cuda::record(e1, self.stream)?;
        cuda::sync(e1)?;
        self.gpu.synchronize(self.side)?;
        Ok(cuda::elapsed_us(e0, e1)? / 1e3)
    }

    fn pattern(&self, tag: &str, p: &Pattern, reps: usize) -> Result<()> {
        for _ in 0..WARMUP {
            self.timed(p)?;
        }
        let ms: Vec<f64> = (0..reps).map(|_| self.timed(p)).collect::<Result<_>>()?;
        let s = verdict::stats(&ms)?;
        println!(
            "XCHG tag={tag} rank={} pattern={} bytes={} times={} n={} median_ms={:.3} p10_ms={:.3} p90_ms={:.3} gbps={:.2}",
            self.rank,
            p.name,
            p.bytes(),
            p.times,
            s.n,
            s.median,
            s.p10,
            s.p90,
            p.gbps(self.world, s.median)
        );
        Ok(())
    }

    fn verify(&self) -> Result<()> {
        let bytes = 2736 * ROW_BYTES;
        self.gpu.memset(self.recv, 0, bytes * self.world)?;
        self.xchg(bytes)?;
        self.gpu.synchronize(self.stream)?;
        let mut got = vec![0u8; bytes * self.world];
        self.gpu.copy_d2h(self.recv, &mut got)?;
        plan::check_recv(&got, self.world, self.rank, bytes)?;
        let n = plan::all_reduce_buffer_bytes();
        let one = plan::bf16_bits(1.0).to_le_bytes();
        let ones: Vec<u8> = (0..n / 2).flat_map(|_| one).collect();
        self.gpu.copy_h2d(&ones, self.ar)?;
        self.nccl.all_reduce(self.ar.0, n)?;
        self.gpu.synchronize(self.stream)?;
        let mut out = vec![0u8; n];
        self.gpu.copy_d2h(self.ar, &mut out)?;
        plan::check_all_reduce_ones(&out, self.world)
    }
}

fn main() -> Result<()> {
    let ordinal: usize = parsed("GLM_BENCH_GPU_ORDINAL")?;
    let rank: usize = parsed("GLM_XCHG_RANK")?;
    let world: usize = parsed("GLM_XCHG_WORLD")?;
    let master = var("GLM_XCHG_MASTER")?;
    let port: u16 = parsed("GLM_XCHG_PORT")?;
    let tag = var("GLM_XCHG_TAG")?;
    let register = match var("GLM_XCHG_REGISTER")?.as_str() {
        "0" => false,
        "1" => true,
        other => anyhow::bail!("GLM_XCHG_REGISTER={other:?}: expected 0 or 1"),
    };
    // 2026-10-15: 20 timed repetitions keep a 7-pattern run under a minute per setting.
    let reps: usize = std::env::var("GLM_XCHG_REPS")
        .ok()
        .map(|v| v.parse())
        .transpose()?
        .unwrap_or(20);
    ensure!(world >= 2 && rank < world, "rank {rank} of world {world}");
    ensure!(reps >= 1, "GLM_XCHG_REPS must be at least 1");

    let gpu = MetraleCudaBackend::new(ordinal, &[])?;
    let g: &dyn GpuBackend = &gpu;
    // 2026-10-15: The SP comm stream's priority, as the serve makes it.
    let stream = g.create_stream_greatest_priority()?;
    let side = g.create_stream()?;
    let xb = plan::xchg_buffer_bytes(world);
    let nccl = NcclBackend::new(rank, world, &master, port, stream, xb, 0)?;
    let (send, recv) = (g.alloc(xb)?, g.alloc(xb)?);
    g.copy_h2d(&vec![plan::fill(rank)?; xb], send)?;
    g.memset(recv, 0, xb)?;
    let ab = plan::all_reduce_buffer_bytes();
    let ar = g.alloc(ab)?;
    g.memset(ar, 0, ab)?;
    let copy = [g.alloc(CONTENTION_BYTES)?, g.alloc(CONTENTION_BYTES)?];
    if register {
        for (name, ptr, bytes) in [
            ("send", send, xb),
            ("recv", recv, xb),
            ("all-reduce", ar, ab),
        ] {
            let h = nccl
                .register_buffer(ptr.0, bytes)
                .with_context(|| format!("ncclCommRegister of the {name} buffer"))?;
            println!("XCHG_REGISTER rank={rank} buffer={name} bytes={bytes} handle={h:#x}");
        }
    }
    let rig = Rig {
        gpu: g,
        nccl,
        rank,
        world,
        stream,
        side,
        send,
        recv,
        ar,
        copy,
        ev: [cuda::event()?, cuda::event()?],
    };
    for p in &PATTERNS {
        rig.pattern(&tag, p, reps)?;
    }
    rig.verify()?;
    println!(
        "XCHG_DONE tag={tag} rank={rank} register={}",
        u8::from(register)
    );
    Ok(())
}
