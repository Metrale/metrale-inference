// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: `kernel_energy_loop`: the steady-state energy harness of DESIGN-kernel-energy.md
//! §3.1 / §5.2 (no serve; OWAY). One kernel at one plan shape and row count is looped between
//! interleaved idle baselines while EVERY available power meter is sampled concurrently (the NVML
//! counter, the NVML sensor, an `nvidia-smi` child, read-only hwmon; each one that cannot open is
//! recorded in the trace with its reason). The raw trace is written to `--out`, the row is
//! derived from it alone (`metrale_power::derive`), and one JSON line on stdout carries the
//! kernel's time, bytes, FLOPs, the trace's hash and the row's energy lines for the writer
//! (`metrale-bench`'s `kernel_energy write`).
//!
//! Sequence: 2 s warm-up, then `I L I L … I` (`metrale_stats::design::Order::abba(1, n)`): every
//! load flanked by idle windows. The load is a CUDA graph of 16 launches over weights rotated
//! across a 128 MB pool (≥ 64 MB, so every launch streams DRAM; a hot-L2 rotation is refused),
//! replayed with a host sync after each replay (launch-paced, not queue-paced). Launch
//! checkpoints are recorded every 100 ms.
//!
//! Kernels (`--kernel`): `w4a16_gemv_pipe` at a GLM-5.3 W4A16 projection group (`--shape`, a name
//! from `common/w4a16_seg_fixture.rs` GROUPS) and `--rows` 1 or 2, dispatched exactly as the
//! serve dispatches it (`ops::w4a16_gemv_pipe`). A further kernel is one more arm of `Arm`.
//!
//! Every argument is required (PCND): `--kernel --shape --rows --repeats --idle-s --load-s
//! --sample-ms --seed --bootstrap-reps --ci-level --platform --fingerprint --energy-source --out`.
//!
//! Run (GB10): cargo build --release -p metrale-gpu-examples --features cuda,gpu-examples
//!   --example kernel_energy_loop, then run the binary.
//!
//! Owner: model-arch examples (energy).
//! Invariants: the harness never changes a host setting (no clocks, limits or sysfs writes).

#[path = "common/w4a16_seg_fixture.rs"]
mod fixture;
#[path = "common/graph_timing.rs"]
mod timing;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail, ensure};
use fixture::{GROUPS, POOL, Pool, Rng, bf16_rows, nvfp4_bytes, segs, upload};
use metrale_abstract_io::Clock;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_io_host_power::{HwmonProbe, NvidiaSmiProbe, ThreadedMeter};
use metrale_io_host_std::SystemClock;
use metrale_model_layers::layers::ops;
use metrale_power::derive::derive;
use metrale_power::multi::Sampler;
use metrale_power::policy::{OF_RECORD, WindowPolicy};
use metrale_power::trace::{Checkpoint, Phase, PowerTrace, Window, sha256_hex};
use metrale_stats::design::Order;
use metrale_telemetry::power_nvml::{
    GB10_COUNTER_PERIOD_MS, GB10_SENSOR_PERIOD_MS, NvmlCounterProbe, NvmlPowerConfig,
    NvmlSensorProbe,
};

/// 2026-10-22: Launches per graph replay (one host sync per replay).
const COPIES: usize = 16;
/// 2026-10-22: The smallest rotation footprint that keeps every launch on DRAM (GB10 L2 24 MB).
const MIN_ROTATION_BYTES: usize = 64 << 20;
/// 2026-10-22: Warm-up before the first baseline.
const WARMUP: Duration = Duration::from_secs(2);
/// 2026-10-22: Launch checkpoint cadence.
const CHECKPOINT: Duration = Duration::from_millis(100);

fn args() -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let key = k
            .strip_prefix("--")
            .with_context(|| format!("expected --key, got {k}"))?;
        let v = it
            .next()
            .with_context(|| format!("--{key} needs a value"))?;
        out.insert(key.to_string(), v);
    }
    Ok(out)
}

fn need<'a>(a: &'a BTreeMap<String, String>, k: &str) -> Result<&'a str> {
    a.get(k)
        .map(String::as_str)
        .with_context(|| format!("--{k} is required (no default)"))
}

fn num<T: std::str::FromStr>(a: &BTreeMap<String, String>, k: &str) -> Result<T> {
    need(a, k)?
        .parse()
        .map_err(|_| anyhow::anyhow!("--{k} is not a number"))
}

/// 2026-10-22: Opens every meter this box can offer; the ones that cannot open are returned with
/// their reasons (recorded in the trace, never skipped silently).
fn meters(sample_ms: u32) -> (Vec<ThreadedMeter>, Vec<(String, String)>) {
    let cfg = NvmlPowerConfig {
        index: 0,
        rail: metrale_abstract_io::power::Rail::GpuCore,
        counter_period_ms: GB10_COUNTER_PERIOD_MS,
        sensor_period_ms: GB10_SENSOR_PERIOD_MS,
    };
    let mut ok = Vec::new();
    let mut absent = Vec::new();
    let mut add = |name: &str, m: metrale_abstract_io::Result<ThreadedMeter>| match m {
        Ok(m) => ok.push(m),
        Err(e) => absent.push((name.to_string(), e.to_string())),
    };
    add(
        "nvml.counter",
        NvmlCounterProbe::open(cfg).and_then(|p| ThreadedMeter::spawn(p, 1)),
    );
    add(
        "nvml.sensor",
        NvmlSensorProbe::open(cfg).and_then(|p| ThreadedMeter::spawn(p, 1)),
    );
    add(
        "nvidia-smi",
        NvidiaSmiProbe::open("nvidia-smi", 0, sample_ms.max(100))
            .and_then(|p| ThreadedMeter::spawn(p, 1)),
    );
    add(
        "hwmon",
        HwmonProbe::open(std::path::Path::new("/sys/class/hwmon"), &[])
            .and_then(|p| ThreadedMeter::spawn(p, 1)),
    );
    (ok, absent)
}

fn main() -> Result<()> {
    let a = args()?;
    let kernel = need(&a, "kernel")?;
    ensure!(
        kernel == "w4a16_gemv_pipe",
        "--kernel: this harness has the arm w4a16_gemv_pipe"
    );
    let shape = need(&a, "shape")?;
    let rows: u32 = num(&a, "rows")?;
    let repeats: usize = num(&a, "repeats")?;
    let idle_s: f64 = num(&a, "idle-s")?;
    let load_s: f64 = num(&a, "load-s")?;
    let sample_ms: u32 = num(&a, "sample-ms")?;
    let seed: u64 = num(&a, "seed")?;
    let reps: usize = num(&a, "bootstrap-reps")?;
    let level: f64 = num(&a, "ci-level")?;
    let out = need(&a, "out")?;
    let policy = WindowPolicy {
        min_load_s: load_s.min(OF_RECORD.min_load_s),
        min_idle_s: idle_s.min(OF_RECORD.min_idle_s),
        ..OF_RECORD
    };
    policy
        .check()
        .context("the windows are below the refusal floor")?;
    ensure!(
        rows == 1 || rows == 2,
        "--rows: w4a16_gemv_pipe runs 1 or 2 rows"
    );
    let (_, ns, k, _) = GROUPS
        .iter()
        .find(|g| g.0 == shape)
        .with_context(|| format!("--shape {shape}: not a fixture group"))?;
    let per_launch: usize = ns.iter().map(|&n| n as usize * *k as usize / 2).sum();
    ensure!(
        POOL >= MIN_ROTATION_BYTES && per_launch * COPIES >= MIN_ROTATION_BYTES.min(POOL),
        "the rotation would stay in L2: {} B per graph",
        per_launch * COPIES
    );

    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let s = g.create_stream()?;
    let sms = g.sm_count()?;
    let mut rng = Rng(seed);
    let pool = Pool::new(&g, &mut rng)?;
    let outs: Vec<DevicePtr> = (0..ns.len())
        .map(|_| g.alloc(4 * 11264 * 2))
        .collect::<Result<_>>()?;
    let ad = upload(&g, &bf16_rows(&mut rng, rows as usize, *k as usize))?;
    let mut i = 0usize;
    let mut body = |st: u64| -> Result<()> {
        let ws: Vec<_> = (0..ns.len())
            .map(|j| pool.weight(i * 7 + j, ns[j], *k))
            .collect();
        i += 1;
        ops::w4a16_gemv_pipe(&g, ad, &segs(&ws, &outs, ns), rows, *k, sms, st)
    };
    let (time_us, lo, hi) = timing::time_graph(&g, s, COPIES, &mut body)?;
    g.begin_capture(s)?;
    for _ in 0..COPIES {
        body(s)?;
    }
    let graph = g.end_capture(s)?;

    let clock = SystemClock::new();
    let (set, absent) = meters(sample_ms);
    ensure!(!set.is_empty(), "no power meter opened: {absent:?}");
    let stop = Arc::new(AtomicBool::new(false));
    let stop2 = stop.clone();
    let sampler = std::thread::spawn(
        move || -> Result<(PowerTrace, metrale_power::multi::SamplerStats)> {
            let mut set = set;
            let mut trace = PowerTrace::default();
            let mut sm = Sampler::new(&set, &mut trace);
            let period = Duration::from_millis(u64::from(sample_ms));
            let mut next = Instant::now();
            while !stop2.load(Ordering::Relaxed) {
                sm.tick(&mut set, clock.monotonic_ns(), &mut trace)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                next += period;
                std::thread::sleep(next.saturating_duration_since(Instant::now()));
            }
            Ok((trace, sm.stats))
        },
    );

    let mut windows = Vec::new();
    let mut checkpoints = Vec::new();
    let mut launches_total = 0u64;
    let mut run_load = |dur: Duration,
                        record: bool,
                        windows: &mut Vec<Window>,
                        checkpoints: &mut Vec<Checkpoint>,
                        rep: u32|
     -> Result<()> {
        let from = clock.monotonic_ns();
        let t0 = Instant::now();
        let mut last_cp = Instant::now();
        let mut n = 0u64;
        while t0.elapsed() < dur {
            g.launch_graph(graph, s)?;
            g.synchronize(s)?;
            n += COPIES as u64;
            if record && last_cp.elapsed() >= CHECKPOINT {
                checkpoints.push(Checkpoint {
                    at: clock.monotonic_ns(),
                    launches: launches_total + n,
                });
                last_cp = Instant::now();
            }
        }
        let to = clock.monotonic_ns();
        if record {
            launches_total += n;
            checkpoints.push(Checkpoint {
                at: to,
                launches: launches_total,
            });
            windows.push(Window {
                phase: Phase::Load,
                box_id: 0,
                repeat: rep,
                from,
                to,
                launches: Some(n),
            });
        }
        Ok(())
    };
    let idle = |windows: &mut Vec<Window>, rep: u32| {
        let from = clock.monotonic_ns();
        std::thread::sleep(Duration::from_secs_f64(idle_s));
        windows.push(Window {
            phase: Phase::Idle,
            box_id: 0,
            repeat: rep,
            from,
            to: clock.monotonic_ns(),
            launches: None,
        });
    };
    run_load(WARMUP, false, &mut windows, &mut checkpoints, 0)?;
    idle(&mut windows, 0);
    for r in 0..repeats {
        let r32 = u32::try_from(r)?;
        run_load(
            Duration::from_secs_f64(load_s),
            true,
            &mut windows,
            &mut checkpoints,
            r32,
        )?;
        idle(&mut windows, r32 + 1);
    }
    stop.store(true, Ordering::Relaxed);
    let (mut trace, stats) = sampler
        .join()
        .map_err(|_| anyhow::anyhow!("the sampler thread panicked"))??;
    g.destroy_graph(graph)?;

    trace.windows = windows;
    trace.checkpoints = checkpoints;
    let order = Order::abba(1, repeats);
    let flops =
        2.0 * f64::from(rows) * ns.iter().map(|&n| f64::from(n)).sum::<f64>() * f64::from(*k);
    let n_sum = ns.iter().map(|&n| f64::from(n)).sum::<f64>();
    // 2026-10-22: Weights and scales, the BF16 input rows and the BF16 output rows (as the nsys rows count).
    let bytes = nvfp4_bytes(ns, *k) + f64::from(rows) * (f64::from(*k) + n_sum) * 2.0;
    for (key, v) in [
        ("platform", need(&a, "platform")?.to_string()),
        ("scope", "gpu_rail".to_string()),
        ("policy", policy.encode()),
        ("seed", seed.to_string()),
        ("order", format!("{}:{}", order.label(), order.render())),
        ("ci_level", level.to_string()),
        ("bootstrap_reps", reps.to_string()),
        ("fingerprint", need(&a, "fingerprint")?.to_string()),
        ("energy_source", need(&a, "energy-source")?.to_string()),
        ("kernel", format!("{kernel} rows={rows} shape={shape}")),
        ("sampler", format!("{stats:?}")),
    ] {
        trace.meta.insert(key.to_string(), v);
    }
    for (name, why) in &absent {
        trace
            .meta
            .insert(format!("absent.{name}"), why.replace('\n', " "));
    }
    let text = trace.encode();
    if let Some(dir) = std::path::Path::new(out).parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, &text)?;
    let sha = sha256_hex(text.as_bytes());
    let row = match derive(&trace, &sha) {
        Ok(d) => d.row.toml_lines(),
        Err(e) => bail!("trace {out} written ({sha}); the row was refused: {e}"),
    };
    let summary = serde_json::json!({
        "kernel": kernel,
        "entry": format!("w4a16_gemv_pipe_m{rows}"),
        "shape": shape,
        "rows": rows,
        "n": ns.iter().sum::<u32>(),
        "k": k,
        "time_us": time_us,
        "time_us_min": lo,
        "time_us_max": hi,
        "bytes": bytes,
        "flops": flops,
        "launches": launches_total,
        "trace": out,
        "trace_sha256": sha,
        "absent": absent,
        "sampler": format!("{stats:?}"),
        "row": row,
    });
    println!("{summary}");
    Ok(())
}
