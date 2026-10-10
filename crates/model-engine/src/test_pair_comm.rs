// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-10: A two-rank communicator for host tests, moved here from the snapshot-agreement
//! tests so the rank-agreement tests share it: `PairComm` hands a broadcast root's device bytes
//! to the other rank through a condition variable, and [`run_pair`] runs one closure per rank on
//! two threads, each with its own `MockGpuBackend`. Every broadcast is recorded per rank with
//! its root and whether it was a rendezvous (`CommBackend::broadcast_rendezvous`).
//!
//! Owner: model-engine tests.
//! Invariants: only broadcasts are implemented; any other collective panics.

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
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
}

/// 2026-10-10: Run `f(rank, gpu, comm)` for ranks 0 and 1 on two threads over one link; returns
/// each rank's result and the broadcasts it issued, indexed by rank.
pub(crate) fn run_pair<T: Send + 'static>(
    f: impl Fn(usize, &dyn GpuBackend, &PairComm) -> T + Send + Sync + 'static,
) -> [(T, Vec<Op>); 2] {
    let link = Arc::new(Link {
        slot: Mutex::new(None),
        cv: Condvar::new(),
    });
    let f = Arc::new(f);
    let handles: Vec<_> = (0..2)
        .map(|rank| {
            let (link, f) = (Arc::clone(&link), Arc::clone(&f));
            std::thread::spawn(move || {
                let gpu = Arc::new(MockGpuBackend::new());
                let comm = PairComm {
                    rank,
                    gpu: Arc::clone(&gpu),
                    link,
                    ops: Mutex::new(Vec::new()),
                };
                let out = f(rank, gpu.as_ref(), &comm);
                (out, comm.ops())
            })
        })
        .collect();
    let mut out = handles.into_iter().map(|h| h.join().unwrap());
    [out.next().unwrap(), out.next().unwrap()]
}
