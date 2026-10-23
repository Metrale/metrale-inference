// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Concurrent sampling never blocks (a slow meter costs only its own samples), every
//! sample passes its source's declaration, tickets complete once, and a trace replayed through the
//! meters and the sampler is the same trace (HPAR).

use alloc::vec;
use alloc::vec::Vec;
use core::task::Poll;

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    EnergyCount, PowerMeter, PowerSample, Rail, RailReading, SampleTicket,
};
use metrale_error::{ErrorCode, error};

use crate::derive::derive;
use crate::meter::{ReplayPowerMeter, Scripted, ScriptedPowerMeter};
use crate::multi::Sampler;
use crate::synth_tests::{BASE, caps_counter, caps_sensor};
use crate::trace::{PowerTrace, sha256_hex};

fn counter_sample(raw: u64) -> PowerSample {
    let mut s = PowerSample::at(MonoNs(0));
    s.energy.insert(Rail::GpuCore, EnergyCount(raw));
    s
}

fn sensor_sample(w: f64) -> PowerSample {
    let mut s = PowerSample::at(MonoNs(0));
    s.rails.insert(
        Rail::GpuCore,
        RailReading {
            instant_w: Some(w),
            ..RailReading::default()
        },
    );
    s
}

fn script(n: usize, pending: u32, f: impl Fn(usize) -> PowerSample) -> Vec<Scripted> {
    (0..n)
        .map(|i| Scripted {
            result: Ok(f(i)),
            pending_polls: pending,
        })
        .collect()
}

#[test]
fn a_slow_meter_never_holds_back_a_fast_one() {
    let fast = ScriptedPowerMeter::new(
        caps_counter(),
        script(200, 0, |i| counter_sample(i as u64)),
        1,
    );
    let slow = ScriptedPowerMeter::new(
        caps_sensor(),
        script(200, 9, |i| sensor_sample(i as f64)),
        1,
    );
    let mut set = (fast, slow);
    let mut trace = PowerTrace::default();
    let mut sampler = Sampler::new(&set, &mut trace);
    for tick in 0..100u64 {
        sampler
            .tick(&mut set, MonoNs(tick * 25_000_000), &mut trace)
            .unwrap();
    }
    // 2026-10-22: The fast meter answered every tick; the slow one every tenth.
    assert_eq!(trace.sources[0].samples.len(), 100);
    assert_eq!(trace.sources[1].samples.len(), 10);
    assert_eq!(sampler.stats.idle_ticks, 0);
    assert_eq!(sampler.in_flight(), 0);
    // 2026-10-22: Samples carry the caller's clock at issue.
    assert_eq!(trace.sources[1].samples[1].at, MonoNs(10 * 25_000_000));
}

#[test]
fn ticks_with_nothing_ready_are_counted_as_idle() {
    let slow = ScriptedPowerMeter::new(caps_sensor(), script(5, 3, |_| sensor_sample(1.0)), 1);
    let mut set = vec![slow];
    let mut trace = PowerTrace::default();
    let mut sampler = Sampler::new(&set, &mut trace);
    for t in 0..8 {
        sampler.tick(&mut set, MonoNs(t), &mut trace).unwrap();
    }
    assert_eq!(trace.sources[0].samples.len(), 2);
    assert_eq!(sampler.stats.idle_ticks, 6);
}

#[test]
fn a_sample_with_an_undeclared_field_is_refused() {
    let mut bad = counter_sample(1);
    bad.rails.insert(
        Rail::Module,
        RailReading {
            instant_w: Some(3.0),
            ..RailReading::default()
        },
    );
    let m = ScriptedPowerMeter::new(
        caps_counter(),
        vec![Scripted {
            result: Ok(bad),
            pending_polls: 0,
        }],
        1,
    );
    let mut set = vec![m];
    let mut trace = PowerTrace::default();
    let mut sampler = Sampler::new(&set, &mut trace);
    let e = sampler.tick(&mut set, MonoNs(0), &mut trace).unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerFieldUndeclared);
}

#[test]
fn a_failed_read_is_counted_and_reissued() {
    let fail = error!(ErrorCode::PowerReadFailed, "s", "transient");
    let m = ScriptedPowerMeter::new(
        caps_counter(),
        vec![
            Scripted {
                result: Err(fail),
                pending_polls: 0,
            },
            Scripted {
                result: Ok(counter_sample(3)),
                pending_polls: 0,
            },
        ],
        1,
    );
    let mut set = vec![m];
    let mut trace = PowerTrace::default();
    let mut sampler = Sampler::new(&set, &mut trace);
    sampler.tick(&mut set, MonoNs(0), &mut trace).unwrap();
    sampler.tick(&mut set, MonoNs(1), &mut trace).unwrap();
    assert_eq!(sampler.stats.read_errors, 1);
    assert_eq!(trace.sources[0].samples.len(), 1);
}

#[test]
fn tickets_complete_once_and_a_full_queue_refuses() {
    let mut m = ScriptedPowerMeter::new(
        caps_counter(),
        script(3, 0, |i| counter_sample(i as u64)),
        1,
    );
    let t = m.issue(MonoNs(0)).unwrap();
    assert_eq!(
        m.issue(MonoNs(0)).unwrap_err().code(),
        ErrorCode::PowerQueueFull
    );
    assert!(matches!(m.complete(t), Poll::Ready(Ok(_))));
    match m.complete(t) {
        Poll::Ready(Err(e)) => assert_eq!(e.code(), ErrorCode::PowerTicketUnknown),
        other => panic!("{other:?}"),
    }
    match m.complete(SampleTicket(99)) {
        Poll::Ready(Err(e)) => assert_eq!(e.code(), ErrorCode::PowerTicketUnknown),
        other => panic!("{other:?}"),
    }
}

#[test]
fn a_trace_replayed_through_the_sampler_is_the_same_trace_and_row() {
    let original = BASE.record();
    let mut set = (
        ReplayPowerMeter::new(&original.sources[0]),
        ReplayPowerMeter::new(&original.sources[1]),
    );
    let mut replayed = PowerTrace {
        meta: original.meta.clone(),
        windows: original.windows.clone(),
        checkpoints: original.checkpoints.clone(),
        ..PowerTrace::default()
    };
    let mut sampler = Sampler::new(&set, &mut replayed);
    while set.0.remaining() > 0 || set.1.remaining() > 0 || sampler.in_flight() > 0 {
        if sampler
            .tick(&mut set, MonoNs(u64::MAX), &mut replayed)
            .is_err()
        {
            break;
        }
    }
    assert_eq!(replayed.encode(), original.encode());
    let sha = sha256_hex(original.encode().as_bytes());
    assert_eq!(
        derive(&replayed, &sha).unwrap().row.toml_lines(),
        derive(&original, &sha).unwrap().row.toml_lines()
    );
}
