// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `EarlyLaunch` against a stream-semantics model: every queued op is a node of a
//! happens-before graph (stream order, event records resolved when a wait is queued or a graph is
//! launched, a graph's nodes after its launch point). The properties are read off that graph, so
//! a reordering inside `EarlyLaunch` that breaks one fails here, whatever calls it makes.
//!
//! Owner: gpu-runtime.
//! Invariants: the model resolves an external wait at launch, as CUDA does (verified on GB10 by
//! the serial-order emulation in `early_launch.rs`).

use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap};

use anyhow::{Result, bail, ensure};

use super::{EarlyLaunch, LaunchDevice};
use crate::gpu::GraphHandle;

/// 2026-10-21: A node: a kernel, a graph's start, its NCCL host-callback root, or its end.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Kind {
    Op(&'static str),
    GraphStart,
    HostRoot,
    GraphEnd,
    GateWait,
}

#[derive(Default)]
struct Model {
    kinds: Vec<Kind>,
    deps: Vec<Vec<usize>>,
    tail: HashMap<u64, usize>,
    events: HashMap<u64, usize>,
    /// 2026-10-21: The capture in progress: its stream, whether it opened with the gate wait (and
    /// on which event), and its kernels in order.
    capture: Option<(u64, Option<u64>, Vec<&'static str>)>,
    graphs: Vec<(Option<u64>, Vec<&'static str>)>,
}

impl Model {
    fn node(&mut self, kind: Kind, deps: Vec<usize>) -> usize {
        self.kinds.push(kind);
        self.deps.push(deps);
        self.kinds.len() - 1
    }

    /// 2026-10-21: A node at the end of `stream`.
    fn push(&mut self, stream: u64, kind: Kind, mut extra: Vec<usize>) -> usize {
        if let Some(&t) = self.tail.get(&stream) {
            extra.push(t);
        }
        let n = self.node(kind, extra);
        self.tail.insert(stream, n);
        n
    }

    fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = BTreeSet::new();
        let mut stack = vec![to];
        while let Some(x) = stack.pop() {
            if x == from {
                return true;
            }
            if seen.insert(x) {
                stack.extend(self.deps[x].iter().copied());
            }
        }
        false
    }

    fn find(&self, k: &Kind) -> Vec<usize> {
        (0..self.kinds.len())
            .filter(|&i| &self.kinds[i] == k)
            .collect()
    }
}

/// 2026-10-21: `LaunchDevice` over [`Model`]; `kernel` queues (or captures) a named kernel.
#[derive(Default)]
struct Sim(RefCell<Model>);

impl Sim {
    fn kernel(&self, stream: u64, name: &'static str) {
        let mut m = self.0.borrow_mut();
        if let Some((s, _, ks)) = m.capture.as_mut()
            && *s == stream
        {
            ks.push(name);
            return;
        }
        m.push(stream, Kind::Op(name), Vec::new());
    }

    fn end_capture(&self) -> GraphHandle {
        let mut m = self.0.borrow_mut();
        let (_, gate, ks) = m.capture.take().expect("no capture");
        m.graphs.push((gate, ks));
        GraphHandle(m.graphs.len() as u64)
    }

    fn op(&self, name: &'static str) -> usize {
        let m = self.0.borrow();
        let v = m.find(&Kind::Op(name));
        assert_eq!(v.len(), 1, "{name} queued {} times", v.len());
        v[0]
    }
}

impl LaunchDevice for Sim {
    fn begin_capture(&self, stream: u64) -> Result<()> {
        let mut m = self.0.borrow_mut();
        ensure!(m.capture.is_none(), "nested capture");
        m.capture = Some((stream, None, Vec::new()));
        Ok(())
    }
    fn abort_capture(&self, _stream: u64) {
        self.0.borrow_mut().capture = None;
    }
    fn wait_external(&self, stream: u64, event: u64) -> Result<()> {
        let mut m = self.0.borrow_mut();
        match m.capture.as_mut() {
            Some((s, gate, ks)) if *s == stream && ks.is_empty() && gate.is_none() => {
                *gate = Some(event);
                Ok(())
            }
            _ => bail!("an external wait outside a capture's first node"),
        }
    }
    fn record(&self, event: u64, stream: u64) -> Result<()> {
        let mut m = self.0.borrow_mut();
        ensure!(m.capture.is_none(), "record under capture (not modelled)");
        // 2026-10-21: A record on an empty stream is complete at once: a wait on it waits for
        // nothing.
        match m.tail.get(&stream).copied() {
            Some(t) => m.events.insert(event, t),
            None => m.events.remove(&event),
        };
        Ok(())
    }
    fn wait(&self, stream: u64, event: u64) -> Result<()> {
        let mut m = self.0.borrow_mut();
        let e = m.events.get(&event).copied().into_iter().collect();
        m.push(stream, Kind::Op("wait"), e);
        Ok(())
    }
    fn launch(&self, graph: GraphHandle, stream: u64) -> Result<()> {
        let mut m = self.0.borrow_mut();
        let (gate, ks) = m.graphs[graph.0 as usize - 1].clone();
        let start = m.push(stream, Kind::GraphStart, Vec::new());
        // 2026-10-21: NCCL's host-callback chain roots at the graph's start, never at its kernels.
        let host = m.node(Kind::HostRoot, vec![start]);
        let mut prev = start;
        if let Some(ev) = gate {
            let mut d = vec![start];
            d.extend(m.events.get(&ev).copied());
            prev = m.node(Kind::GateWait, d);
        }
        for k in ks {
            // 2026-10-21: The first kernel is the collective that waits for the host callback.
            let d = if prev == start || m.kinds[prev] == Kind::GateWait {
                vec![prev, host]
            } else {
                vec![prev]
            };
            prev = m.node(Kind::Op(k), d);
        }
        let end = m.node(Kind::GraphEnd, vec![prev, host]);
        m.tail.insert(stream, end);
        Ok(())
    }
}

const COMPUTE: u64 = 1;

/// 2026-10-21: One step: the drafter on the compute stream, the verify graph launched early, the
/// next step's readback after it. Returns the model and the early launch.
fn step(sim: &Sim) -> EarlyLaunch {
    let el = EarlyLaunch {
        side: 2,
        gate: 10,
        done: 11,
    };
    sim.kernel(COMPUTE, "warm");
    el.begin_capture(sim, COMPUTE).unwrap();
    sim.kernel(COMPUTE, "layer0_collective");
    sim.kernel(COMPUTE, "layer1");
    let g = sim.end_capture();
    sim.kernel(COMPUTE, "drafter");
    sim.kernel(COMPUTE, "uploads");
    el.launch(sim, g, COMPUTE).unwrap();
    sim.kernel(COMPUTE, "readback");
    el
}

/// 2026-10-21: The graph's kernels run after everything queued on the compute stream before the
/// launch (the drafter, the uploads), and everything queued after it runs after the graph.
#[test]
fn the_graph_runs_between_the_work_before_and_after_its_launch() {
    let sim = Sim::default();
    step(&sim);
    let m = sim.0.borrow();
    let first = m.find(&Kind::Op("layer0_collective"))[0];
    let last = m.find(&Kind::Op("layer1"))[0];
    drop(m);
    let (drafter, uploads, readback) = (sim.op("drafter"), sim.op("uploads"), sim.op("readback"));
    let m = sim.0.borrow();
    assert!(m.reaches(drafter, first), "graph kernel before the drafter");
    assert!(m.reaches(uploads, first), "graph kernel before its uploads");
    assert!(m.reaches(last, readback), "readback before the graph ended");
    let end = m.find(&Kind::GraphEnd)[0];
    assert!(m.reaches(end, readback), "readback before the graph's end");
}

/// 2026-10-21: The graph's host-callback root does not wait for the compute stream's work, so the
/// first collective's proxy upload is done before its producer: the reason for the launch.
#[test]
fn the_host_callback_root_starts_before_the_compute_work_ends() {
    let sim = Sim::default();
    step(&sim);
    let drafter = sim.op("drafter");
    let m = sim.0.borrow();
    let host = m.find(&Kind::HostRoot)[0];
    assert!(
        !m.reaches(drafter, host),
        "the host root waits for the drafter"
    );
    let first = m.find(&Kind::Op("layer0_collective"))[0];
    assert!(
        m.reaches(host, first),
        "the collective no longer waits for its host callback"
    );
}

/// 2026-10-21: A failed gate capture ends the capture, so the stream runs work again.
#[test]
fn a_failed_gate_capture_ends_the_capture() {
    struct NoExternal(Sim);
    impl LaunchDevice for NoExternal {
        fn begin_capture(&self, s: u64) -> Result<()> {
            self.0.begin_capture(s)
        }
        fn abort_capture(&self, s: u64) {
            self.0.abort_capture(s)
        }
        fn wait_external(&self, _: u64, _: u64) -> Result<()> {
            bail!("no external waits")
        }
        fn record(&self, e: u64, s: u64) -> Result<()> {
            self.0.record(e, s)
        }
        fn wait(&self, s: u64, e: u64) -> Result<()> {
            self.0.wait(s, e)
        }
        fn launch(&self, g: GraphHandle, s: u64) -> Result<()> {
            self.0.launch(g, s)
        }
    }
    let dev = NoExternal(Sim::default());
    let el = EarlyLaunch {
        side: 2,
        gate: 10,
        done: 11,
    };
    assert!(el.begin_capture(&dev, COMPUTE).is_err());
    assert!(dev.0.0.borrow().capture.is_none(), "capture left open");
}
