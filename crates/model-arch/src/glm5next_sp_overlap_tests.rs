// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: SP overlap: the pieces, the piece exchanges against the
//! whole-shard ones on simulated ranks, the schedule's cross-stream ordering under a
//! happens-before model, and the enqueued section on a recording communicator.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::sync::Mutex;

use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

use super::*;
use crate::glm5next_sp_mhc::tests::Recorder;
use crate::glm5next_sp_mhc::{all_gather_plan, reduce_scatter_plan, shards};

/// 2026-10-14: Off is one piece; on, the narrowest shard over `MIN_PIECE_ROWS`, clamped to
/// `1..=MAX_PIECES`, so a pipelined piece never has fewer than `MIN_PIECE_ROWS` rows.
#[test]
fn piece_count_follows_the_narrowest_shard() {
    assert_eq!(piece_count(false, 8208, 3), 1);
    assert_eq!(piece_count(true, 8208, 3), 4);
    assert_eq!(piece_count(true, 2048, 3), 2);
    assert_eq!(piece_count(true, 192, 3), 1);
    assert_eq!(piece_count(true, 1535, 3), 1);
    assert_eq!(piece_count(true, 1536, 3), 2);
    assert_eq!(piece_count(true, 1 << 20, 2), MAX_PIECES);
    for world in 2..=4 {
        for k in [192, 1536, 2048, 4137, 8208, 16400] {
            let m = piece_count(true, k, world);
            if m >= 2 {
                let p = pieces(&shards(k, world), m);
                let narrowest = p.iter().flatten().map(|(lo, hi)| hi - lo).min().unwrap();
                assert!(narrowest >= MIN_PIECE_ROWS, "k {k} world {world}");
            }
        }
    }
}

/// 2026-10-14: Each rank's pieces cover its shard once, in order, sizes within one row; one
/// piece is the shards themselves.
#[test]
fn pieces_cover_each_shard_in_order() {
    for world in 2..=4 {
        for k in [1536, 4137, 8208] {
            let s = shards(k, world);
            assert_eq!(pieces(&s, 1), vec![s.clone()]);
            for m in 2..=MAX_PIECES {
                let p = pieces(&s, m);
                assert_eq!(p.len(), m);
                for (r, &(lo, hi)) in s.iter().enumerate() {
                    assert_eq!(p[0][r].0, lo);
                    assert_eq!(p[m - 1][r].1, hi);
                    for j in 1..m {
                        assert_eq!(p[j - 1][r].1, p[j][r].0);
                    }
                    let sizes: Vec<usize> = p.iter().map(|q| q[r].1 - q[r].0).collect();
                    let (mn, mx) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
                    assert!(mx - mn <= 1, "{sizes:?}");
                }
            }
        }
    }
    assert_eq!(
        pieces(&shards(1536, 3), 2),
        vec![
            vec![(0, 256), (512, 768), (1024, 1280)],
            vec![(256, 512), (768, 1024), (1280, 1536)],
        ]
    );
}

/// 2026-10-14: Run every rank's plans against the paired peer's: each receive copies the bytes
/// the peer's matching send names (the sizes must agree). Returns every rank's receive
/// workspace and gathered buffer.
fn simulate(
    k: usize,
    world: usize,
    row: usize,
    ranges: &[Vec<(usize, usize)>],
    slot: usize,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let shard = shards(k, world);
    let partial = |r: usize| -> Vec<u8> {
        (0..k * row)
            .map(|i| (i as u32 * 31 + r as u32 * 7 + 1) as u8)
            .collect()
    };
    let mut out: Vec<(Vec<u8>, Vec<u8>)> = (0..world)
        .map(|r| (vec![0u8; slot * (world - 1)], partial(r)))
        .collect();
    for piece in ranges {
        for r in 0..world {
            let off = (piece[r].0 - shard[r].0) * row;
            for x in reduce_scatter_plan(r, piece, row, slot, off) {
                let back = reduce_scatter_plan(x.peer, piece, row, slot, 0);
                let y = back.iter().find(|y| y.peer == r).unwrap();
                assert_eq!(y.send_bytes, x.recv_bytes, "rs {r} <- {}", x.peer);
                let src = partial(x.peer);
                out[r].0[x.recv_off..x.recv_off + x.recv_bytes]
                    .copy_from_slice(&src[y.send_off..y.send_off + y.send_bytes]);
            }
        }
        let before: Vec<Vec<u8>> = out.iter().map(|o| o.1.clone()).collect();
        for r in 0..world {
            for x in all_gather_plan(r, piece, row) {
                let back = all_gather_plan(x.peer, piece, row);
                let y = back.iter().find(|y| y.peer == r).unwrap();
                assert_eq!(y.send_bytes, x.recv_bytes, "ag {r} <- {}", x.peer);
                let src = &before[x.peer];
                out[r].1[x.recv_off..x.recv_off + x.recv_bytes]
                    .copy_from_slice(&src[y.send_off..y.send_off + y.send_bytes]);
            }
        }
    }
    out
}

/// 2026-10-14: The piece exchanges leave every rank's receive slots and gathered rows byte for
/// byte as the whole-shard exchanges do, for every piece count.
#[test]
fn piece_exchanges_move_the_whole_shards_bytes() {
    let row = 6;
    for world in 2..=4 {
        for k in [1536, 4137] {
            let s = shards(k, world);
            let slot = crate::glm5next_sp_mhc::max_shard(k, world) * row;
            let whole = simulate(k, world, row, std::slice::from_ref(&s), slot);
            for m in 2..=MAX_PIECES {
                let piecewise = simulate(k, world, row, &pieces(&s, m), slot);
                assert!(whole == piecewise, "k {k} world {world} m {m}");
            }
        }
    }
}

/// 2026-10-14: Events are distinct indices below `EVENTS`; a piece past `MAX_PIECES` is refused.
#[test]
fn event_indices_are_distinct() {
    let mut seen = std::collections::HashSet::new();
    let mut evs = vec![Ev::In, Ev::Out];
    for j in 0..MAX_PIECES {
        evs.extend([Ev::Received(j), Ev::Computed(j)]);
    }
    for ev in evs {
        let i = event_index(ev).unwrap();
        assert!(i < EVENTS && seen.insert(i), "{ev:?}");
    }
    assert!(event_index(Ev::Received(MAX_PIECES)).is_err());
    assert!(event_index(Ev::Computed(MAX_PIECES)).is_err());
}

/// 2026-10-14: Off or one piece, the serial body on the caller's stream; no exchange, no events.
#[test]
fn one_piece_is_the_serial_body() {
    use On::Main;
    assert_eq!(
        schedule(1, true, true),
        vec![
            Op::ReduceScatter(0, Main),
            Op::Compute(0),
            Op::AllGather(0, Main)
        ]
    );
    assert_eq!(
        schedule(1, false, true),
        vec![Op::Compute(0), Op::AllGather(0, Main)]
    );
    assert_eq!(
        schedule(1, true, false),
        vec![Op::ReduceScatter(0, Main), Op::Compute(0)]
    );
    assert_eq!(
        schedule(3, false, false),
        vec![Op::Compute(0), Op::Compute(1), Op::Compute(2)]
    );
}

/// 2026-10-14: A device model of `ops`: each stream runs its ops in order, a wait orders the
/// stream after the last record of its event in host order. Node 0 is the caller stream's
/// earlier work and the last node its later work; returns `hb[a][b]` (a happens before b) and
/// each op's node.
fn happens_before(ops: &[Op]) -> (Vec<Vec<bool>>, Vec<usize>) {
    let n = ops.len() + 2;
    let mut edges: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut last: HashMap<&'static str, usize> = HashMap::from([("main", 0)]);
    let mut recorded: HashMap<usize, usize> = HashMap::new();
    let name = |s: On| if s == On::Main { "main" } else { "comm" };
    for (i, op) in ops.iter().enumerate() {
        let node = i + 1;
        let s = match *op {
            Op::Record(_, s) | Op::Wait(s, _) | Op::ReduceScatter(_, s) | Op::AllGather(_, s) => s,
            Op::Compute(_) => On::Main,
        };
        if let Some(&prev) = last.get(name(s)) {
            edges[prev].push(node);
        }
        last.insert(name(s), node);
        match *op {
            Op::Record(ev, _) => {
                recorded.insert(event_index(ev).unwrap(), node);
            }
            Op::Wait(_, ev) => {
                let r = recorded.get(&event_index(ev).unwrap());
                edges[*r.unwrap_or_else(|| panic!("wait on unrecorded {ev:?}"))].push(node);
            }
            _ => {}
        }
    }
    edges[last["main"]].push(n - 1);
    let mut hb = vec![vec![false; n]; n];
    for a in 0..n {
        let mut stack = edges[a].clone();
        while let Some(b) = stack.pop() {
            if !hb[a][b] {
                hb[a][b] = true;
                stack.extend(edges[b].iter().copied());
            }
        }
    }
    (hb, (1..=ops.len()).collect())
}

/// 2026-10-14: Pipelined, for every piece count and exchange mix: every exchange runs on the
/// comm stream after the caller's earlier work and before its later work; a piece computes
/// after its reduce-scatter and is gathered after it computes; every reduce-scatter precedes
/// every all-gather; and the overlap is real: piece `j` computes without waiting for piece
/// `j + 1`'s reduce-scatter, and is gathered without waiting for piece `j + 1`'s compute.
#[test]
fn pipelined_schedule_orders_the_streams_by_events() {
    for m in 2..=MAX_PIECES {
        for (rs, ag) in [(true, true), (true, false), (false, true)] {
            let ops = schedule(m, rs, ag);
            let (hb, node) = happens_before(&ops);
            let end = ops.len() + 1;
            let at = |want: Op| node[ops.iter().position(|&o| o == want).unwrap()];
            for (i, op) in ops.iter().enumerate() {
                if matches!(op, Op::ReduceScatter(..) | Op::AllGather(..)) {
                    assert!(matches!(
                        op,
                        Op::ReduceScatter(_, On::Comm) | Op::AllGather(_, On::Comm)
                    ));
                    assert!(hb[0][node[i]] && hb[node[i]][end], "{op:?} m {m}");
                }
            }
            for j in 0..m {
                let c = at(Op::Compute(j));
                if rs {
                    let r = at(Op::ReduceScatter(j, On::Comm));
                    assert!(hb[r][c], "compute {j} after its reduce-scatter");
                    if j + 1 < m {
                        assert!(
                            !hb[at(Op::ReduceScatter(j + 1, On::Comm))][c],
                            "overlap {j}"
                        );
                    }
                }
                if ag {
                    let g = at(Op::AllGather(j, On::Comm));
                    assert!(hb[c][g], "gather {j} after its compute");
                    if j + 1 < m {
                        assert!(!hb[at(Op::Compute(j + 1))][g], "overlap {j}");
                    }
                    if rs {
                        for i in 0..m {
                            assert!(hb[at(Op::ReduceScatter(i, On::Comm))][g]);
                        }
                    }
                }
            }
        }
    }
}

/// 2026-10-14: Rank 1 of 3, 1536 rows (shards of 512, two pieces of 256), both exchanges: on
/// the comm stream one group per piece's reduce-scatter (peer pieces sent, this rank's piece
/// received at its place in each slot), then the computes on the caller's stream with each
/// piece's partials, then one group per piece's all-gather. Without the comm stream a
/// pipelined section is refused before anything is enqueued; one piece runs serially on the
/// caller's stream.
#[test]
fn a_section_enqueues_piece_groups_on_the_comm_stream() {
    let comm = Recorder {
        rank: 1,
        world: 3,
        ops: Mutex::new(Vec::new()),
    };
    let gpu = MockGpuBackend::new();
    let row = 4096 * 2;
    let r = row as u64;
    let ws = sp::SpWorkspace {
        recv: DevicePtr(0x9000_0000),
        slot_bytes: 512 * row,
        world: 3,
    };
    let (src, buf) = (DevicePtr(0x1000_0000), DevicePtr(0x5000_0000));
    let ov = OverlapStreams {
        comm: 11,
        events: [0; EVENTS],
    };
    let s = shards(1536, 3);
    let two = pieces(&s, 2);
    let sec = |p| Section {
        pieces: p,
        rs: Some((src, None)),
        ag: Some((buf, None)),
        row_bytes: row,
        received: None,
        defer: None,
    };
    let mut compute = |lo: usize, hi: usize, p: Option<&PeerPartials>| -> Result<()> {
        comm.push("compute", p.unwrap().recv, hi - lo, lo, 7);
        Ok(())
    };
    run_section(&gpu, &comm, &ws, Some(&ov), 7, &sec(&two), &mut compute).unwrap();
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    let (g0, g1) = (("group_start", 0, 0, 0, 0), ("group_end", 0, 0, 0, 0));
    let n = 256 * row;
    let slot = ws.slot_bytes as u64;
    assert_eq!(
        ops,
        vec![
            g0,
            ("send", src.0, n, 0, 11),
            ("recv", ws.recv.0, n, 0, 11),
            ("send", src.0 + 1024 * r, n, 2, 11),
            ("recv", ws.recv.0 + slot, n, 2, 11),
            g1,
            g0,
            ("send", src.0 + 256 * r, n, 0, 11),
            ("recv", ws.recv.0 + 256 * r, n, 0, 11),
            ("send", src.0 + 1280 * r, n, 2, 11),
            ("recv", ws.recv.0 + slot + 256 * r, n, 2, 11),
            g1,
            ("compute", ws.recv.0, 256, 512, 7),
            ("compute", ws.recv.0 + 256 * r, 256, 768, 7),
            g0,
            ("send", buf.0 + 512 * r, n, 0, 11),
            ("recv", buf.0, n, 0, 11),
            ("send", buf.0 + 512 * r, n, 2, 11),
            ("recv", buf.0 + 1024 * r, n, 2, 11),
            g1,
            g0,
            ("send", buf.0 + 768 * r, n, 0, 11),
            ("recv", buf.0 + 256 * r, n, 0, 11),
            ("send", buf.0 + 768 * r, n, 2, 11),
            ("recv", buf.0 + 1280 * r, n, 2, 11),
            g1,
        ]
    );
    let e = run_section(&gpu, &comm, &ws, None, 7, &sec(&two), &mut |_, _, _| Ok(())).unwrap_err();
    assert!(e.to_string().contains("without"), "{e}");
    assert!(comm.ops.lock().unwrap().is_empty(), "nothing enqueued");
    let one = pieces(&s, 1);
    run_section(&gpu, &comm, &ws, None, 7, &sec(&one), &mut |_, _, _| Ok(())).unwrap();
    let ops = std::mem::take(&mut *comm.ops.lock().unwrap());
    assert_eq!(ops.len(), 12);
    assert!(ops.iter().all(|o| o.4 == 7 || o.0.starts_with("group")));
}
