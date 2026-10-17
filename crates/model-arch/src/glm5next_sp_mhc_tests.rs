// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-12: SP-mHC: the shards, the exchange plans against the one-shot
//! slot layout and the ring's byte count, and the enqueued exchange on a recording communicator.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;

/// 2026-10-12: The SP body takes a sub-chunk only with the lever on, two ranks and `SP_MIN_ROWS`.
#[test]
fn applies_needs_the_lever_two_ranks_and_the_rows() {
    assert!(applies(true, 3, SP_MIN_ROWS));
    assert!(!applies(true, 3, SP_MIN_ROWS - 1));
    assert!(!applies(true, 1, 8208));
    assert!(!applies(false, 3, 8208));
}

/// 2026-10-12: The shards cover `0..k` once in order, differ by at most one row, the first
/// `k % world` ranks own the extra row, and the widest is `max_shard`.
#[test]
fn shards_cover_every_row_once_in_order() {
    for world in 2..=4 {
        for k in [192, 4096, 4137, 8208, 16400] {
            let s = shards(k, world);
            assert_eq!(s.len(), world);
            assert_eq!(s[0].0, 0);
            assert_eq!(s[world - 1].1, k, "k {k} world {world}");
            for w in s.windows(2) {
                assert_eq!(w[0].1, w[1].0);
            }
            let sizes: Vec<usize> = s.iter().map(|(lo, hi)| hi - lo).collect();
            let (mn, mx) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
            assert!(mx - mn <= 1, "{sizes:?}");
            assert_eq!(mx, max_shard(k, world));
            assert!(sizes[..k % world].iter().all(|&n| n == mx));
        }
    }
    assert_eq!(shards(4137, 3), vec![(0, 1379), (1379, 2758), (2758, 4137)]);
}

/// 2026-10-12: The reduce-scatter sends every peer exactly its shard's rows of the partial and
/// receives this rank's rows from every peer into the one-shot slot layout
/// (`glm5next_hc_post_rank_sum` reads peer r at slot r < rank ? r : r - 1); the all-gather sends
/// this rank's rows to every peer and places each peer's rows at their own offset. Per rank, the
/// bytes on the wire of one reduce-scatter plus one all-gather equal the ring all-reduce's
/// `2 (n - 1) / n` of the buffer when the shards are equal.
#[test]
fn plans_match_the_slot_layout_and_the_rings_bytes() {
    let (k, world, row) = (4098usize, 3usize, 4096 * 2usize);
    let s = shards(k, world);
    let slot = max_shard(k, world) * row;
    for rank in 0..world {
        let (lo, hi) = s[rank];
        let rs = reduce_scatter_plan(rank, &s, row, slot, 0);
        let ag = all_gather_plan(rank, &s, row);
        assert_eq!(rs.len(), world - 1);
        assert_eq!(ag.len(), world - 1);
        for x in &rs {
            let (plo, phi) = s[x.peer];
            assert_eq!((x.send_off, x.send_bytes), (plo * row, (phi - plo) * row));
            assert_eq!(x.recv_bytes, (hi - lo) * row);
            assert_eq!(x.recv_off, peer_slot(rank, x.peer).unwrap() * slot);
            assert!(x.recv_off + x.recv_bytes <= slot * (world - 1));
        }
        for x in &ag {
            let (plo, phi) = s[x.peer];
            assert_eq!((x.send_off, x.send_bytes), (lo * row, (hi - lo) * row));
            assert_eq!((x.recv_off, x.recv_bytes), (plo * row, (phi - plo) * row));
        }
        let wire: usize = rs.iter().map(|x| x.send_bytes).sum::<usize>()
            + ag.iter().map(|x| x.send_bytes).sum::<usize>();
        let ring = 2 * (world - 1) * k * row / world;
        assert_eq!(wire, ring, "rank {rank}");
    }
    // 2026-10-12: Slots and the kernel's rule agree for every peer.
    for rank in 0..5 {
        for peer in 0..5usize {
            let kernel = if peer < rank {
                peer
            } else {
                peer.wrapping_sub(1)
            };
            assert_eq!(peer_slot(rank, peer), (peer != rank).then_some(kernel));
        }
    }
}

/// 2026-10-12: A communicator that records what is enqueued: `(op, ptr, bytes, peer, stream)`,
/// with group markers, so the exchange's order and grouping are checked without NCCL.
pub(crate) struct Recorder {
    pub(crate) rank: usize,
    pub(crate) world: usize,
    pub(crate) ops: Mutex<Vec<(&'static str, u64, usize, usize, u64)>>,
}

impl Recorder {
    pub(crate) fn push(&self, op: &'static str, ptr: u64, bytes: usize, peer: usize, stream: u64) {
        self.ops
            .lock()
            .unwrap()
            .push((op, ptr, bytes, peer, stream));
    }
}

impl CommBackend for Recorder {
    fn all_reduce(&self, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("not an exchange")
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("not an exchange")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("not an exchange")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("not an exchange")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("not an exchange")
    }
    fn send_to(&self, ptr: u64, bytes: usize, dest: usize, stream: u64) -> Result<()> {
        self.push("send", ptr, bytes, dest, stream);
        Ok(())
    }
    fn recv_from(&self, ptr: u64, bytes: usize, src: usize, stream: u64) -> Result<()> {
        self.push("recv", ptr, bytes, src, stream);
        Ok(())
    }
    fn group_start(&self) -> Result<()> {
        self.push("group_start", 0, 0, 0, 0);
        Ok(())
    }
    fn group_end(&self) -> Result<()> {
        self.push("group_end", 0, 0, 0, 0);
        Ok(())
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
}

/// 2026-10-12: On rank 1 of 3 with 300 rows (shards 100/100/100): the reduce-scatter is one
/// group of a send and a receive per peer on the caller's stream, peer 0's partial of rows
/// 100..200 in slot 0 and peer 2's in slot 1, and the returned partials point at the workspace;
/// the all-gather sends rows 100..200 and receives rows 0..100 and 200..300 in place. A shard
/// wider than the workspace, or a workspace of another world, is refused before anything is
/// enqueued.
#[test]
fn the_exchanges_enqueue_one_group_of_paired_transfers_on_the_stream() {
    let comm = Recorder {
        rank: 1,
        world: 3,
        ops: Mutex::new(Vec::new()),
    };
    let row = 4096 * 2;
    let ws = SpWorkspace {
        recv: DevicePtr(0x9000_0000),
        slot_bytes: 100 * row,
        world: 3,
    };
    let s = shards(300, 3);
    let p = DevicePtr(0x1000_0000);
    let partials = reduce_scatter_rows(&comm, &ws, p, &s, 0, row, 7).unwrap();
    assert_eq!(
        (
            partials.recv,
            partials.slot_elems,
            partials.world_size,
            partials.rank
        ),
        (ws.recv.0, 100 * 4096, 3, 1)
    );
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    let r = row as u64;
    assert_eq!(
        ops,
        vec![
            ("group_start", 0, 0, 0, 0),
            ("send", p.0, 100 * row, 0, 7),
            ("recv", ws.recv.0, 100 * row, 0, 7),
            ("send", p.0 + 200 * r, 100 * row, 2, 7),
            ("recv", ws.recv.0 + 100 * r, 100 * row, 2, 7),
            ("group_end", 0, 0, 0, 0),
        ]
    );
    all_gather_rows(&comm, p, &s, row, 7).unwrap();
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    assert_eq!(
        ops,
        vec![
            ("group_start", 0, 0, 0, 0),
            ("send", p.0 + 100 * r, 100 * row, 0, 7),
            ("recv", p.0, 100 * row, 0, 7),
            ("send", p.0 + 100 * r, 100 * row, 2, 7),
            ("recv", p.0 + 200 * r, 100 * row, 2, 7),
            ("group_end", 0, 0, 0, 0),
        ]
    );
    let wide = shards(303, 3);
    let e = reduce_scatter_rows(&comm, &ws, p, &wide, 0, row, 7).unwrap_err();
    assert!(e.to_string().contains("exceeds the workspace"), "{e}");
    let other_world = SpWorkspace { world: 2, ..ws };
    assert!(reduce_scatter_rows(&comm, &other_world, p, &s, 0, row, 7).is_err());
    assert!(comm.ops.lock().unwrap().is_empty(), "nothing enqueued");
}

/// 2026-10-14: At every world 2..=5 and uneven shards: each rank's ring steps 1..world together
/// are exactly its reduce-scatter plan's sends and receives, and at every step the peer a rank
/// sends to receives from that rank the same byte count, so the stepped exchange pairs up.
#[test]
fn ring_steps_are_the_plan_and_pair_up_at_every_step() {
    let row = 4096 * 2;
    for world in 2..=5usize {
        for k in [301usize, 4098] {
            let s = shards(k, world);
            let slot = max_shard(k, world) * row;
            let step = |rank, st| ring_step_plan(rank, &s, row, slot, st).unwrap();
            for rank in 0..world {
                let plan = reduce_scatter_plan(rank, &s, row, slot, 0);
                let mut sends: Vec<_> = (1..world)
                    .map(|st| step(rank, st))
                    .map(|x| (x.to, x.send_off, x.send_bytes))
                    .collect();
                let mut recvs: Vec<_> = (1..world)
                    .map(|st| step(rank, st))
                    .map(|x| (x.from, x.recv_off, x.recv_bytes))
                    .collect();
                sends.sort();
                recvs.sort();
                let mut want_s: Vec<_> = plan
                    .iter()
                    .map(|x| (x.peer, x.send_off, x.send_bytes))
                    .collect();
                let mut want_r: Vec<_> = plan
                    .iter()
                    .map(|x| (x.peer, x.recv_off, x.recv_bytes))
                    .collect();
                want_s.sort();
                want_r.sort();
                assert_eq!(
                    (sends, recvs),
                    (want_s, want_r),
                    "world {world} rank {rank}"
                );
                for st in 1..world {
                    let x = step(rank, st);
                    let y = step(x.to, st);
                    assert_eq!(y.from, rank, "world {world} step {st}");
                    assert_eq!(y.recv_bytes, x.send_bytes, "world {world} step {st}");
                }
            }
            assert!(ring_step_plan(0, &s, row, slot, 0).is_err());
            assert!(ring_step_plan(0, &s, row, slot, world).is_err());
        }
    }
}

/// 2026-10-14: On rank 1 of 3 with 300 rows: step 1 is one group sending rows 200..300 to rank 2
/// and receiving rank 0's partial into slot 0, and returns nothing; step 2 sends rows 0..100 to
/// rank 0, receives rank 2's into slot 1, and returns the partials `reduce_scatter_rows` returns.
/// A step outside 1..3 or a shard wider than the workspace is refused before anything is enqueued.
#[test]
fn the_ring_steps_enqueue_one_pair_each_and_the_last_returns_the_partials() {
    let comm = Recorder {
        rank: 1,
        world: 3,
        ops: Mutex::new(Vec::new()),
    };
    let row = 4096 * 2;
    let ws = SpWorkspace {
        recv: DevicePtr(0x9000_0000),
        slot_bytes: 100 * row,
        world: 3,
    };
    let s = shards(300, 3);
    let p = DevicePtr(0x1000_0000);
    let r = row as u64;
    assert!(
        reduce_scatter_step(&comm, &ws, p, &s, 1, row, 7)
            .unwrap()
            .is_none()
    );
    let last = reduce_scatter_step(&comm, &ws, p, &s, 2, row, 7)
        .unwrap()
        .unwrap();
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    assert_eq!(
        ops,
        vec![
            ("group_start", 0, 0, 0, 0),
            ("send", p.0 + 200 * r, 100 * row, 2, 7),
            ("recv", ws.recv.0, 100 * row, 0, 7),
            ("group_end", 0, 0, 0, 0),
            ("group_start", 0, 0, 0, 0),
            ("send", p.0, 100 * row, 0, 7),
            ("recv", ws.recv.0 + 100 * r, 100 * row, 2, 7),
            ("group_end", 0, 0, 0, 0),
        ]
    );
    let whole = reduce_scatter_rows(&comm, &ws, p, &s, 0, row, 7).unwrap();
    comm.ops.lock().unwrap().clear();
    assert_eq!(
        (last.recv, last.slot_elems, last.world_size, last.rank),
        (whole.recv, whole.slot_elems, whole.world_size, whole.rank)
    );
    for bad in [0, 3] {
        assert!(reduce_scatter_step(&comm, &ws, p, &s, bad, row, 7).is_err());
    }
    assert!(reduce_scatter_step(&comm, &ws, p, &shards(303, 3), 1, row, 7).is_err());
    assert!(reduce_scatter_step(&comm, &ws, p, &shards(300, 2), 1, row, 7).is_err());
    assert!(comm.ops.lock().unwrap().is_empty(), "nothing enqueued");
}

/// 2026-10-12: Off, `reserve` allocates nothing (the declared levers) and
/// `workspace_get` reports that nothing was reserved.
#[test]
fn off_nothing_is_reserved() {
    let gpu = MockGpuBackend::new();
    reserve(
        &gpu,
        8208,
        4096,
        3,
        &crate::glm5next_mhc_levers::MhcLevers::declared(),
    )
    .unwrap();
    let e = workspace_get().unwrap_err();
    assert!(e.to_string().contains("no receive workspace"), "{e}");
}
