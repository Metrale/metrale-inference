// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-22: Four slow probes sampled concurrently through tickets never block the caller;
//! the queue bound, the ticket rules and the capability check hold.

use std::task::Poll;
use std::time::{Duration, Instant};

use metrale_abstract_io::MonoNs;
use metrale_abstract_io::power::{
    CounterSpec, EnergyCount, Field, PowerCapabilities, PowerMeter, PowerProbe, PowerSample, Rail,
    RailReading, SampleTicket, SourceId, SourceKind,
};
use metrale_error::ErrorCode;

use super::ThreadedMeter;

struct SlowProbe {
    caps: PowerCapabilities,
    delay: Duration,
    reads: u64,
    undeclared: bool,
}

fn caps(name: &str) -> PowerCapabilities {
    PowerCapabilities::builder(
        SourceId(name.to_string()),
        SourceKind::External,
        "test".to_string(),
    )
    .counter(
        Rail::Wall,
        CounterSpec {
            unit_j: 1.0,
            update_period_ms: 50.0,
        },
    )
    .rest_unavailable(&Field::universe(), "test probe reports wall energy only")
    .build(&Field::universe())
    .unwrap()
}

fn probe(name: &str, delay_ms: u64, undeclared: bool) -> SlowProbe {
    SlowProbe {
        caps: caps(name),
        delay: Duration::from_millis(delay_ms),
        reads: 0,
        undeclared,
    }
}

impl PowerProbe for SlowProbe {
    fn capabilities(&self) -> &PowerCapabilities {
        &self.caps
    }

    fn read(&mut self, at: MonoNs) -> metrale_abstract_io::Result<PowerSample> {
        std::thread::sleep(self.delay);
        self.reads += 1;
        let mut s = PowerSample::at(at);
        s.energy.insert(Rail::Wall, EnergyCount(self.reads));
        if self.undeclared {
            s.rails.insert(
                Rail::Wall,
                RailReading {
                    instant_w: Some(1.0),
                    ..RailReading::default()
                },
            );
        }
        Ok(s)
    }
}

fn timed<T>(max: &mut Duration, f: impl FnOnce() -> T) -> T {
    let t0 = Instant::now();
    let r = f();
    *max = (*max).max(t0.elapsed());
    r
}

#[test]
fn four_slow_meters_sampled_concurrently_never_block_and_every_sample_arrives() {
    let mut meters: Vec<ThreadedMeter> = (0..4)
        .map(|i| ThreadedMeter::spawn(probe(&format!("m{i}"), 50, false), 3).unwrap())
        .collect();
    let mut slowest = Duration::ZERO;
    let start = Instant::now();
    let mut pending: Vec<(usize, SampleTicket, u64)> = Vec::new();
    for (i, m) in meters.iter_mut().enumerate() {
        for k in 0..3u64 {
            let t = timed(&mut slowest, || m.issue(MonoNs(k)).unwrap());
            pending.push((i, t, k));
        }
    }
    let mut got = Vec::new();
    while !pending.is_empty() {
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "samples did not arrive"
        );
        pending.retain(
            |&(i, t, k)| match timed(&mut slowest, || meters[i].complete(t)) {
                Poll::Pending => true,
                Poll::Ready(r) => {
                    let s = r.unwrap();
                    assert_eq!(s.at, MonoNs(k));
                    got.push((i, s.energy[&Rail::Wall].0));
                    false
                }
            },
        );
        std::thread::sleep(Duration::from_millis(1));
    }
    assert_eq!(got.len(), 12);
    assert!(
        slowest < Duration::from_millis(1),
        "a call took {slowest:?}"
    );
    // 2026-10-22: Concurrency, not just non-blocking calls: 3 serial reads of 50 ms per meter,
    // the 4 meters in parallel, is ~150 ms; serial would be 600 ms.
    assert!(
        start.elapsed() < Duration::from_millis(450),
        "{:?}",
        start.elapsed()
    );
}

#[test]
fn the_queue_bound_is_enforced_and_frees_on_completion() {
    let mut m = ThreadedMeter::spawn(probe("q", 20, false), 1).unwrap();
    let t = m.issue(MonoNs(0)).unwrap();
    let e = m.issue(MonoNs(1)).unwrap_err();
    assert_eq!(e.code(), ErrorCode::PowerQueueFull);
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        if let Poll::Ready(r) = m.complete(t) {
            r.unwrap();
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(m.issue(MonoNs(2)).is_ok());
}

#[test]
fn zero_depth_is_refused() {
    let e = ThreadedMeter::spawn(probe("z", 0, false), 0).err().unwrap();
    assert_eq!(e.code(), ErrorCode::InvalidArgument);
}

#[test]
fn an_unknown_or_already_completed_ticket_is_refused() {
    let mut m = ThreadedMeter::spawn(probe("t", 0, false), 2).unwrap();
    let Poll::Ready(Err(e)) = m.complete(SampleTicket(99)) else {
        panic!("an unissued ticket must be refused");
    };
    assert_eq!(e.code(), ErrorCode::PowerTicketUnknown);
    let t = m.issue(MonoNs(0)).unwrap();
    while m.complete(t).is_pending() {
        std::thread::sleep(Duration::from_millis(1));
    }
    let Poll::Ready(Err(e)) = m.complete(t) else {
        panic!("a completed ticket must be refused");
    };
    assert_eq!(e.code(), ErrorCode::PowerTicketUnknown);
}

#[test]
fn a_probe_reporting_an_undeclared_field_is_refused() {
    let mut m = ThreadedMeter::spawn(probe("u", 0, true), 1).unwrap();
    let t = m.issue(MonoNs(0)).unwrap();
    let r = loop {
        if let Poll::Ready(r) = m.complete(t) {
            break r;
        }
        std::thread::sleep(Duration::from_millis(1));
    };
    assert_eq!(r.unwrap_err().code(), ErrorCode::PowerFieldUndeclared);
}

#[test]
fn drop_with_reads_in_flight_returns() {
    let mut m = ThreadedMeter::spawn(probe("d", 30, false), 3).unwrap();
    for k in 0..3 {
        m.issue(MonoNs(k)).unwrap();
    }
    let t0 = Instant::now();
    drop(m);
    assert!(t0.elapsed() < Duration::from_secs(1));
}
