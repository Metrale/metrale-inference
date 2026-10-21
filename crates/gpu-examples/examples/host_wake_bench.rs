// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: Microbench for the host wake keeper (`metrale_gpu_runtime::host_wake`): how long
//! the first NCCL-shaped host task after D ms of GPU work waits on an otherwise idle host, with
//! and without a keeper holding one core out of deep idle.
//!
//! Per iteration, on one stream: a one-block kernel spins D ms (writes its end time); a host
//! function (as NCCL's proxy-upload callback) records its thread and CPU and wakes an emulated
//! NCCL progress thread (named `NCCL Progress 0`, sleeping on a condition variable as NCCL's does
//! between collectives); a one-thread "collective" kernel writes its start time and spins until
//! the progress thread has written the iteration number into mapped host memory, then writes that
//! time. All times are device globaltimer. The host thread sleeps through the iteration.
//! - `dispatch` = collective start - busy end: the host function's wake (GPU interrupt, driver
//!   thread) and the completion signal back.
//! - `total` = flag seen - busy end: plus the progress thread's wake (two serialised wakes, the
//!   shape of the layer-0 all-reduce wait).
//!
//! Owner: gpu-examples.
//! Invariants: refuses to run on a host with load average >= 2 (an idle-host measurement).
//!
//!   cargo run -p metrale-gpu-examples --release --example host_wake_bench \
//!       --features cuda,gpu-examples [-- <rounds>]

use std::sync::atomic::{AtomicI32, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::host_wake::{
    KeeperConfig, KeeperMode, Role, WakeKeeper, os, select,
};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_settings::groups::device::DeviceSettings;

#[path = "common/pdl_probe_ptx.rs"]
mod probe_ptx;

/// 2026-10-21: The "collective": writes its start, spins on a system-scope acquire load of the
/// mapped flag until it equals `want`, writes the time it saw it.
const WAIT_PTX: &str = r#"
.version 9.0
.target sm_90
.address_size 64
.visible .entry wake_probe_wait(.param .u64 p_flag, .param .u32 p_want, .param .u64 p_ts)
{
	.reg .pred %p<2>;
	.reg .b32 %r<3>;
	.reg .b64 %rd<6>;
	ld.param.u64 %rd1, [p_flag];
	ld.param.u32 %r1, [p_want];
	ld.param.u64 %rd2, [p_ts];
	cvta.to.global.u64 %rd1, %rd1;
	cvta.to.global.u64 %rd2, %rd2;
	mov.u64 %rd3, %globaltimer;
	st.global.u64 [%rd2], %rd3;
$L_spin:
	ld.acquire.sys.global.u32 %r2, [%rd1];
	setp.ne.s32 %p1, %r2, %r1;
	@%p1 bra $L_spin;
	mov.u64 %rd4, %globaltimer;
	st.global.u64 [%rd2+8], %rd4;
	ret;
}
"#;

const THREADS: u32 = 128;
const ITERS: usize = 60;
const DS_US: [u32; 4] = [500, 1000, 2500, 5000];

/// 2026-10-21: The host function's thread and CPU, last call.
static CB_TID: AtomicI32 = AtomicI32::new(0);
static CB_CPU: AtomicI32 = AtomicI32::new(-1);
/// 2026-10-21: The mapped flag's host address.
static FLAG: AtomicUsize = AtomicUsize::new(0);
/// 2026-10-21: (generation posted, stop) for the emulated progress thread.
static POST: Mutex<(u32, bool)> = Mutex::new((0, false));
static POSTED: Condvar = Condvar::new();

unsafe extern "C" fn host_task(generation: *mut std::ffi::c_void) {
    CB_TID.store(os::gettid(), Ordering::Relaxed);
    // SAFETY: sched_getcpu takes no arguments.
    CB_CPU.store(unsafe { libc_getcpu() }, Ordering::Relaxed);
    if let Ok(mut g) = POST.lock() {
        g.0 = generation as usize as u32;
        POSTED.notify_one();
    }
}

unsafe fn libc_getcpu() -> i32 {
    unsafe extern "C" {
        fn sched_getcpu() -> i32;
    }
    // SAFETY: plain query.
    unsafe { sched_getcpu() }
}

/// 2026-10-21: The emulated NCCL progress thread: sleeps until a generation is posted, then
/// writes it to the mapped flag.
fn progress_thread() -> Result<std::thread::JoinHandle<()>> {
    Ok(std::thread::Builder::new()
        .name("NCCL Progress 0".into())
        .spawn(|| {
            let mut seen = 0;
            loop {
                let Ok(mut g) = POST.lock() else { return };
                while g.0 == seen && !g.1 {
                    let Ok(n) = POSTED.wait(g) else { return };
                    g = n;
                }
                if g.1 {
                    return;
                }
                seen = g.0;
                drop(g);
                // SAFETY: FLAG holds a live page-locked u32 for the bench's lifetime.
                let f = unsafe { &*(FLAG.load(Ordering::Relaxed) as *const AtomicU32) };
                f.store(seen, Ordering::Release);
            }
        })?)
}

struct Rig<'a> {
    gpu: &'a MetraleCudaBackend,
    busy: KernelHandle,
    wait: KernelHandle,
    data: DevicePtr,
    t: DevicePtr,
    flag_dev: DevicePtr,
    generation: u32,
}

/// 2026-10-21: One (D, iterations) sample: dispatch and total waits (us) and the host
/// function's CPUs.
#[derive(Default)]
struct Sample {
    dispatch: Vec<f64>,
    total: Vec<f64>,
    cpus: std::collections::BTreeMap<i32, usize>,
}

impl Rig<'_> {
    fn iter(&mut self, d_us: u32, s: Option<&mut Sample>) -> Result<()> {
        let g: &dyn GpuBackend = self.gpu;
        let st = g.default_stream();
        self.generation += 1;
        g.memset(self.t, 0, 32)?;
        KernelLaunch::new(g, self.busy)
            .block([THREADS, 1, 1])
            .arg_ptr(self.data)
            .arg_ptr(self.t)
            .arg_u32(7)
            .arg_u32(d_us * 1000)
            .launch(st)?;
        // SAFETY: host_task only touches statics; the generation is passed by value.
        let rc = unsafe {
            sys::cuLaunchHostFunc(st as sys::CUstream, Some(host_task), self.generation as usize as *mut _)
        };
        ensure!(rc == sys::CUresult::CUDA_SUCCESS, "cuLaunchHostFunc: {rc:?}");
        KernelLaunch::new(g, self.wait)
            .block([1, 1, 1])
            .arg_ptr(self.flag_dev)
            .arg_u32(self.generation)
            .arg_ptr(self.t.offset(8))
            .launch(st)?;
        std::thread::sleep(Duration::from_micros(u64::from(d_us) + 3000));
        g.synchronize(st)?;
        let mut b = [0u8; 24];
        g.copy_d2h(self.t, &mut b)?;
        let at = |i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().expect("8"));
        ensure!(at(0) != 0 && at(1) != 0 && at(2) != 0, "missing timestamps");
        if let Some(s) = s {
            s.dispatch.push((at(1) as f64 - at(0) as f64) / 1e3);
            s.total.push((at(2) as f64 - at(0) as f64) / 1e3);
            *s.cpus.entry(CB_CPU.load(Ordering::Relaxed)).or_default() += 1;
        }
        Ok(())
    }
}

fn pct(v: &[f64], p: f64) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    s[((s.len() - 1) as f64 * p).round() as usize]
}

/// 2026-10-21: What an arm sets up before its samples.
#[derive(Clone, Copy, Debug)]
enum Arm {
    Base,
    PinOnly,
    /// 2026-10-21: A keeper on the core; `pin`: the driver and proxy threads pinned to it.
    Keep(KeeperMode, bool),
    /// 2026-10-21: A keeper on `cpu` instead of the interrupt's core, threads pinned there.
    KeepOn(KeeperMode, u32),
    /// 2026-10-21: A keeper on the interrupt's core (nothing pinned) and one on `cpu` with the
    /// threads pinned there: the threads off a core shared with other interrupts.
    Split(KeeperMode, u32),
    /// 2026-10-21: A keeper on every allowed core, nothing pinned.
    AllCores(KeeperMode),
}

struct Setup {
    keepers: Vec<WakeKeeper>,
    pinned: Vec<i32>,
}

fn roles() -> Vec<Role> {
    vec![Role::CudaDriver, Role::NcclProxy]
}

fn setup(arm: Arm, core: u32, allowed: &[u32]) -> Result<Setup> {
    let tids = vec![CB_TID.load(Ordering::Relaxed)];
    let keeper = |mode, cpu, roles: Vec<Role>| {
        WakeKeeper::start(KeeperConfig {
            mode,
            cpu,
            tids: if roles.is_empty() { vec![] } else { tids.clone() },
            roles,
            rescan: Duration::from_secs(1),
        })
    };
    let mut s = Setup {
        keepers: vec![],
        pinned: vec![],
    };
    match arm {
        Arm::Base => {}
        Arm::PinOnly => {
            for p in select(&os::tasks()?, &roles(), &tids, &mut Default::default()) {
                os::pin(p.tid, core)?;
                s.pinned.push(p.tid);
            }
        }
        Arm::Keep(m, pin) => {
            let k = keeper(m, core, if pin { roles() } else { vec![] })?;
            s.pinned = k.pinned().iter().map(|p| p.tid).collect();
            s.keepers.push(k);
        }
        Arm::KeepOn(m, cpu) => {
            let k = keeper(m, cpu, roles())?;
            s.pinned = k.pinned().iter().map(|p| p.tid).collect();
            s.keepers.push(k);
        }
        Arm::Split(m, cpu) => {
            s.keepers.push(keeper(m, core, vec![])?);
            let k = keeper(m, cpu, roles())?;
            s.pinned = k.pinned().iter().map(|p| p.tid).collect();
            s.keepers.push(k);
        }
        Arm::AllCores(m) => {
            for &c in allowed {
                s.keepers.push(keeper(m, c, vec![])?);
            }
        }
    }
    Ok(s)
}

/// 2026-10-21: Stop the keepers (their summed share of one core) and unpin.
fn teardown(s: Setup, allowed: &[u32]) -> Result<f64> {
    let mut share = 0.0;
    for k in s.keepers {
        share += k.stop()?.core_share();
    }
    for t in s.pinned {
        os::set_affinity(t, allowed)?;
    }
    Ok(share)
}

fn loadavg() -> Result<f64> {
    std::fs::read_to_string("/proc/loadavg")?
        .split_whitespace()
        .next()
        .context("loadavg")?
        .parse()
        .context("loadavg")
}

fn main() -> Result<()> {
    let rounds: usize = std::env::args().nth(1).map_or(Ok(1), |a| a.parse())?;
    let la = loadavg()?;
    println!("host load average (1 min) at start: {la:.2}");
    ensure!(la < 2.0, "host not idle (load {la:.2} >= 2): rerun later");
    let modules = [
        ("pdl_probe", probe_ptx::PTX.as_bytes()),
        ("wake_probe", WAIT_PTX.as_bytes()),
    ];
    let gpu = MetraleCudaBackend::with_settings(0, &modules, &DeviceSettings::declared()?)?;
    let flag_host = gpu.alloc_host_pinned(64)?;
    FLAG.store(flag_host as usize, Ordering::Relaxed);
    let mut rig = Rig {
        gpu: &gpu,
        busy: gpu.kernel("pdl_probe", "pdl_probe_producer")?,
        wait: gpu.kernel("wake_probe", "wake_probe_wait")?,
        data: gpu.alloc(THREADS as usize * 4)?,
        t: gpu.alloc(64)?,
        flag_dev: gpu.host_ptr_to_device(flag_host)?,
        generation: 0,
    };
    let progress = progress_thread()?;
    for _ in 0..5 {
        rig.iter(500, None)?;
    }
    let core = os::gpu_irq_cpu()?;
    let allowed = os::affinity(0)?;
    let cb = CB_TID.load(Ordering::Relaxed);
    let comm = os::tasks()?
        .into_iter()
        .find(|(t, _)| *t == cb)
        .map(|(_, c)| c);
    println!("GPU interrupt core {core}; host function runs on tid {cb} ({comm:?})");
    for (t, c) in os::tasks()? {
        println!("  thread {t} {c:?} role {:?}", metrale_gpu_runtime::host_wake::classify(&c));
    }
    let neighbour = core + 1;
    let t1000 = KeeperMode::Timer { period_us: 1000 };
    let arms = [
        Arm::Base,
        Arm::PinOnly,
        Arm::Keep(t1000, true),
        Arm::Split(t1000, neighbour),
        Arm::KeepOn(t1000, neighbour),
        Arm::Keep(KeeperMode::Timer { period_us: 2000 }, true),
        Arm::AllCores(KeeperMode::SpinPause),
    ];
    println!(
        "arm | D us | dispatch p10/p50/p90 us | total p10/p50/p90 us | host-fn CPUs | keeper cost (cores)"
    );
    for round in 0..rounds {
        for arm in arms {
            let s = setup(arm, core, &allowed)?;
            std::thread::sleep(Duration::from_millis(200));
            let mut rows = Vec::new();
            for d in DS_US {
                let mut smp = Sample::default();
                for _ in 0..ITERS {
                    rig.iter(d, Some(&mut smp))?;
                }
                rows.push((d, smp));
            }
            let cost = teardown(s, &allowed)?;
            for (d, smp) in rows {
                println!(
                    "r{round} {arm:?} | {d} | {:.0}/{:.0}/{:.0} | {:.0}/{:.0}/{:.0} | {:?} | {cost:.2}",
                    pct(&smp.dispatch, 0.1),
                    pct(&smp.dispatch, 0.5),
                    pct(&smp.dispatch, 0.9),
                    pct(&smp.total, 0.1),
                    pct(&smp.total, 0.5),
                    pct(&smp.total, 0.9),
                    smp.cpus,
                );
            }
        }
    }
    println!("host load average (1 min) at end: {:.2}", loadavg()?);
    if let Ok(mut g) = POST.lock() {
        g.1 = true;
        POSTED.notify_one();
    }
    let _ = progress.join();
    gpu.free_host_pinned(flag_host, 64)?;
    Ok(())
}
