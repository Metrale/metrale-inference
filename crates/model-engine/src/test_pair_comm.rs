// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: A two-rank communicator for host tests, moved here from the snapshot-agreement
//! tests so the rank-agreement tests share it: `PairComm` hands a broadcast root's device bytes
//! to the other rank through a condition variable, and [`run_pair`] runs one closure per rank on
//! two threads, each with its own `MockGpuBackend`. Every broadcast is recorded per rank with
//! its root and whether it was a rendezvous (`CommBackend::broadcast_rendezvous`).
//!
//! Owner: model-engine tests.
//! Invariants: only broadcasts are implemented; any other collective panics.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: Two-rank rendezvous broadcast: the root hands its device bytes to the
/// other rank, which writes them at its own rank-local pointer.
pub(crate) struct Link {
    /// 2026-09-25: `(root, bytes)`, tagged with the root so that a root waiting for its
    /// bytes to be taken does not mistake the next broadcast's post, already issued by
    /// the faster peer, for its own unread one.
    slot: Mutex<Option<(usize, Vec<u8>)>>,
    cv: Condvar,
}

/// 2026-10-10: One recorded broadcast: `(rendezvous, root)`.
pub(crate) type Op = (bool, usize);

pub(crate) struct PairComm {
    rank: usize,
    gpu: Arc<MockGpuBackend>,
    link: Arc<Link>,
    ops: Mutex<Vec<Op>>,
    /// 2026-10-21: Set by `poison`, read by `is_healthy`.
    unhealthy: AtomicBool,
}

impl PairComm {
    /// 2026-10-10: The broadcasts this rank issued, in order.
    pub(crate) fn ops(&self) -> Vec<Op> {
        self.ops.lock().unwrap().clone()
    }

    fn transfer(
        &self,
        ptr: u64,
        bytes: usize,
        root: usize,
        rendezvous: bool,
    ) -> anyhow::Result<()> {
        self.ops.lock().unwrap().push((rendezvous, root));
        let dev = metrale_gpu_runtime::gpu::DevicePtr(ptr);
        let mut slot = self.link.slot.lock().unwrap();
        if self.rank == root {
            let mut out = vec![0u8; bytes];
            self.gpu.copy_d2h(dev, &mut out)?;
            *slot = Some((root, out));
            self.link.cv.notify_all();
            // 2026-09-25: Wait until the peer has taken the bytes: a broadcast completes
            // on every rank together.
            while slot.as_ref().is_some_and(|(r, _)| *r == root) {
                slot = self.link.cv.wait(slot).unwrap();
            }
        } else {
            while slot.as_ref().is_none_or(|(r, _)| *r != root) {
                slot = self.link.cv.wait(slot).unwrap();
            }
            let (_, bytes) = slot.take().unwrap();
            self.gpu.copy_h2d(&bytes, dev)?;
            self.link.cv.notify_all();
        }
        Ok(())
    }
}

impl CommBackend for PairComm {
    fn all_reduce(&self, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!("PairComm implements broadcasts only")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!()
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        unreachable!()
    }
    fn broadcast(&self, ptr: u64, bytes: usize, root: usize) -> anyhow::Result<()> {
        self.transfer(ptr, bytes, root, false)
    }
    fn broadcast_rendezvous(&self, ptr: u64, bytes: usize, root: usize) -> anyhow::Result<()> {
        self.transfer(ptr, bytes, root, true)
    }
    fn barrier(&self) -> anyhow::Result<()> {
        Ok(())
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        unreachable!()
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        unreachable!()
    }
    fn is_healthy(&self) -> bool {
        !self.unhealthy.load(Ordering::Acquire)
    }
    fn poison(&self) {
        self.unhealthy.store(true, Ordering::Release);
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
}

impl PairComm {
    fn new(rank: usize, gpu: &Arc<MockGpuBackend>, link: &Arc<Link>) -> Self {
        Self {
            rank,
            gpu: Arc::clone(gpu),
            link: Arc::clone(link),
            ops: Mutex::new(Vec::new()),
            unhealthy: AtomicBool::new(false),
        }
    }
}

fn new_link() -> Arc<Link> {
    Arc::new(Link {
        slot: Mutex::new(None),
        cv: Condvar::new(),
    })
}

/// 2026-10-21: How long [`run_pair_two_links`] waits for both ranks: a message sent on the wrong
/// link, or received out of order, leaves a rank blocked forever, which must fail the test, not
/// hang it.
pub(crate) const PAIR_DEADLINE: std::time::Duration = std::time::Duration::from_secs(10);

/// 2026-10-21: [`run_pair`] over two independent links, `f(rank, gpu, main, ctrl)` (the control
/// channel beside the main communicator); returns each rank's result and the broadcasts it
/// issued on each link. Panics naming the stuck rank when a rank has not finished within
/// [`PAIR_DEADLINE`] (its thread is left blocked; the test binary exits past it).
pub(crate) type TwoLinkOut<T> = (T, Vec<Op>, Vec<Op>);
pub(crate) fn run_pair_two_links<T: Send + 'static>(
    f: impl Fn(usize, &Arc<MockGpuBackend>, Arc<PairComm>, Arc<PairComm>) -> T + Send + Sync + 'static,
) -> [TwoLinkOut<T>; 2] {
    let (main_link, ctrl_link) = (new_link(), new_link());
    let f = Arc::new(f);
    let (tx, rx) = std::sync::mpsc::channel();
    for rank in 0..2 {
        let (main_link, ctrl_link, f, tx) = (
            Arc::clone(&main_link),
            Arc::clone(&ctrl_link),
            Arc::clone(&f),
            tx.clone(),
        );
        std::thread::spawn(move || {
            let gpu = Arc::new(MockGpuBackend::new());
            let main = Arc::new(PairComm::new(rank, &gpu, &main_link));
            let ctrl = Arc::new(PairComm::new(rank, &gpu, &ctrl_link));
            let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(rank, &gpu, Arc::clone(&main), Arc::clone(&ctrl))
            }));
            let _ = tx.send((rank, out.map(|o| (o, main.ops(), ctrl.ops()))));
        });
    }
    drop(tx);
    let deadline = std::time::Instant::now() + PAIR_DEADLINE;
    let mut done: [Option<TwoLinkOut<T>>; 2] = [None, None];
    while done.iter().any(Option::is_none) {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        let Ok((rank, out)) = rx.recv_timeout(left) else {
            let stuck: Vec<usize> = (0..2).filter(|&r| done[r].is_none()).collect();
            panic!(
                "pair exchange deadlock: rank(s) {stuck:?} still blocked after {PAIR_DEADLINE:?} \
                 (a message on the wrong link, or a receive with no matching send)"
            );
        };
        match out {
            Ok(o) => done[rank] = Some(o),
            Err(p) => std::panic::resume_unwind(p),
        }
    }
    let [a, b] = done;
    [a.unwrap(), b.unwrap()]
}

/// 2026-10-10: Run `f(rank, gpu, comm)` for ranks 0 and 1 on two threads over one link; returns
/// each rank's result and the broadcasts it issued, indexed by rank.
pub(crate) fn run_pair<T: Send + 'static>(
    f: impl Fn(usize, &dyn GpuBackend, &PairComm) -> T + Send + Sync + 'static,
) -> [(T, Vec<Op>); 2] {
    let link = new_link();
    let f = Arc::new(f);
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (link, f) = (Arc::clone(&link), Arc::clone(&f));
            std::thread::spawn(move || {
                let gpu = Arc::new(MockGpuBackend::new());
                let comm = PairComm::new(rank, &gpu, &link);
                let out = f(rank, gpu.as_ref(), &comm);
                (out, comm.ops())
            })
        })
        .collect();
    let mut out = handles.into_iter().map(|h| h.join().unwrap());
    [out.next().unwrap(), out.next().unwrap()]
}
