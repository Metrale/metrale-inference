// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Synthetic traces with known truth for the attribution tests: a GB10-like NVML
//! counter (1 mJ, updated every 100 ms) and power sensor (updated every 250 ms, optionally biased),
//! both read at 40 Hz; static power with an optional linear drift; a load adding a known dynamic
//! power; AR(1) noise per 100 ms tick; idle A / load / idle B windows with shared baselines.

use alloc::string::ToString;
use alloc::vec::Vec;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerSample, Rail, RailCaps, RailReading,
    SourceId, SourceKind,
};
use metrale_stats::rng::SplitMix64;

use crate::policy::{OF_RECORD, WindowPolicy};
use crate::trace::{Checkpoint, Phase, PowerTrace, Window};

/// 2026-10-22: The ground truth of one synthetic run.
#[derive(Clone, Copy, Debug)]
pub struct Synth {
    pub static_w: f64,
    /// 2026-10-22: Static power change over the whole run (end − start), linear.
    pub static_drift_w: f64,
    pub dyn_w: f64,
    pub noise_w: f64,
    pub phi: f64,
    /// 2026-10-22: Sensor reads `(1 + bias) ×` true power.
    pub sensor_bias: f64,
    pub repeats: usize,
    pub idle_s: f64,
    pub load_s: f64,
    pub launch_us: f64,
    pub sm_mhz: u32,
    pub seed: u64,
}

pub const BASE: Synth = Synth {
    static_w: 12.5,
    static_drift_w: 0.0,
    dyn_w: 5.3,
    noise_w: 0.4,
    phi: 0.6,
    sensor_bias: 0.04,
    repeats: 3,
    idle_s: 10.0,
    load_s: 10.0,
    launch_us: 4613.0,
    sm_mhz: 2435,
    seed: 11,
};

const TICK_NS: u64 = 100_000_000;
const SENSOR_NS: u64 = 250_000_000;
const SAMPLE_NS: u64 = 25_000_000;

fn s2ns(s: f64) -> u64 {
    (s * 1e9) as u64
}

pub fn caps_counter() -> PowerCapabilities {
    PowerCapabilities::builder(
        SourceId("nvml0.counter".to_string()),
        SourceKind::GpuDriverCounter,
        "test".to_string(),
    )
    .counter(
        Rail::GpuCore,
        CounterSpec {
            unit_j: 0.001,
            update_period_ms: 100.0,
        },
    )
    .field(Field::SmClock)
    .field(Field::Throttle)
    .build(&[
        Field::Energy(Rail::GpuCore),
        Field::SmClock,
        Field::Throttle,
    ])
    .unwrap()
}

pub fn caps_sensor() -> PowerCapabilities {
    PowerCapabilities::builder(
        SourceId("nvml0.sensor".to_string()),
        SourceKind::GpuDriverSensor,
        "test".to_string(),
    )
    .sensor(
        Rail::GpuCore,
        RailCaps {
            update_period_ms: 250.0,
            average_window_ms: None,
        },
    )
    .build(&[Field::InstantPower(Rail::GpuCore)])
    .unwrap()
}

impl Synth {
    fn total_s(&self) -> f64 {
        self.idle_s + self.repeats as f64 * (self.load_s + self.idle_s)
    }

    /// 2026-10-22: Load windows `(from_ns, to_ns)`.
    pub fn loads(&self) -> Vec<(u64, u64)> {
        (0..self.repeats)
            .map(|r| {
                let from = self.idle_s + r as f64 * (self.load_s + self.idle_s);
                (s2ns(from), s2ns(from + self.load_s))
            })
            .collect()
    }

    fn true_w(&self, t_ns: u64, noise: f64) -> f64 {
        let t = t_ns as f64 * 1e-9;
        let s = self.static_w + self.static_drift_w * t / self.total_s();
        let busy = self.loads().iter().any(|(a, b)| t_ns >= *a && t_ns < *b);
        s + if busy { self.dyn_w } else { 0.0 } + noise
    }

    /// 2026-10-22: Expected (W under load, J/launch total, J/launch dynamic) of repeat `r`.
    pub fn truth(&self, r: usize) -> (f64, f64, f64) {
        let (a, b) = self.loads()[r];
        let mid = (a + b) as f64 * 0.5e-9;
        let w = self.static_w + self.static_drift_w * mid / self.total_s() + self.dyn_w;
        let launches = self.launches() as f64;
        (
            w,
            w * self.load_s / launches,
            self.dyn_w * self.load_s / launches,
        )
    }

    pub fn launches(&self) -> u64 {
        (self.load_s * 1e6 / self.launch_us) as u64
    }

    /// 2026-10-22: The trace, with windows, checkpoints and metadata (policy `p`).
    pub fn trace(&self, p: &WindowPolicy) -> PowerTrace {
        let mut t = PowerTrace::default();
        let c = t.add_source(caps_counter());
        let s = t.add_source(caps_sensor());
        let mut rng = SplitMix64::new(self.seed);
        let end = s2ns(self.total_s());
        let (mut noise, mut cum_j, mut counter) = (0.0f64, 1.0e6f64, 1_000_000_000u64);
        let (mut sensor, mut sensor_acc, mut sensor_n) = (self.static_w, 0.0, 0u32);
        // 2026-10-22: Counter ticks at 37 ms + k·100 ms, sensor at 61 ms + k·250 ms, samples at
        // 5 ms + k·25 ms: no instrument aligned to a window edge.
        let mut next_tick = 37_000_000u64;
        let mut next_sensor = 61_000_000u64;
        let mut at = 5_000_000u64;
        while at < end {
            while next_tick <= at {
                noise = self.phi * noise + self.noise_w * rng.normal();
                let w = self.true_w(next_tick, noise);
                cum_j += w * (TICK_NS as f64 * 1e-9);
                counter = (cum_j * 1000.0) as u64;
                sensor_acc += w;
                sensor_n += 1;
                next_tick += TICK_NS;
            }
            while next_sensor <= at {
                if sensor_n > 0 {
                    sensor = sensor_acc / f64::from(sensor_n) * (1.0 + self.sensor_bias);
                }
                sensor_acc = 0.0;
                sensor_n = 0;
                next_sensor += SENSOR_NS;
            }
            let mut x = PowerSample::at(MonoNs(at));
            x.energy.insert(Rail::GpuCore, EnergyCount(counter));
            x.clocks.sm_mhz = Some(self.sm_mhz);
            x.throttle = Some(0);
            t.push(c, x).unwrap();
            let mut y = PowerSample::at(MonoNs(at));
            y.rails.insert(
                Rail::GpuCore,
                RailReading {
                    instant_w: Some(sensor),
                    ..RailReading::default()
                },
            );
            t.push(s, y).unwrap();
            at += SAMPLE_NS;
        }
        let mut idle_from = 0u64;
        for (r, (a, b)) in self.loads().into_iter().enumerate() {
            let r = r as u32;
            t.windows.push(Window {
                phase: Phase::Idle,
                box_id: 0,
                repeat: r,
                from: MonoNs(idle_from),
                to: MonoNs(a),
                launches: None,
            });
            t.windows.push(Window {
                phase: Phase::Load,
                box_id: 0,
                repeat: r,
                from: MonoNs(a),
                to: MonoNs(b),
                launches: Some(self.launches()),
            });
            t.checkpoints.push(Checkpoint {
                at: MonoNs(b),
                launches: self.launches() * (u64::from(r) + 1),
            });
            idle_from = b;
        }
        t.windows.push(Window {
            phase: Phase::Idle,
            box_id: 0,
            repeat: self.repeats as u32,
            from: MonoNs(idle_from),
            to: MonoNs(end),
            launches: None,
        });
        for (k, v) in [
            ("platform", "gb10".to_string()),
            ("scope", "gpu_rail".to_string()),
            ("policy", p.encode()),
            ("seed", "7".to_string()),
            ("order", "I L0 I L0 I L0 I".to_string()),
            ("ci_level", "0.95".to_string()),
            ("bootstrap_reps", "400".to_string()),
            ("fingerprint", "synthetic".to_string()),
            ("energy_source", "synth_tests".to_string()),
        ] {
            t.meta.insert(k.to_string(), v);
        }
        t
    }

    /// 2026-10-22: [`Synth::trace`] under [`OF_RECORD`].
    pub fn record(&self) -> PowerTrace {
        self.trace(&OF_RECORD)
    }
}
