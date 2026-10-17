// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: The split decision and the split run against a mock GPU and a recording
//! communicator: which passes split, that a split rank launches only its own rows with the
//! unsplit pass's per-row pointers, that its two gathers move exactly the slabs the plan names,
//! and that a decode pass never reaches a gather.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use std::sync::Mutex;

use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

use super::super::select;
use super::super::select::passes_tests::{
    EXPAND, EXPAND_FOLDED, KPOOL_COMPRESS, SEL_FOLD, TOPK_POOLS, cfg, inputs, kernels,
};
use super::super::select::{DsaSelectGeometry, DsaSelectScratch, select_tokens_in_passes};
use super::lever::Payload;
use super::run::{SlabPlan, SplitStreams, plan, plan_with, run_split, select_with};
use super::*;
use crate::glm5next_sp_mhc::tests::Recorder;

fn table(s: &str) -> MinPoolsTable {
    lever::parse(Some(s)).unwrap().unwrap()
}

fn recorder(rank: usize, world: usize) -> Recorder {
    Recorder {
        rank,
        world,
        ops: Mutex::new(Vec::new()),
    }
}

/// 2026-10-15: The lever off, a decode pass, one rank, a pass below the table, or too few rows
/// for the slabs: no split. On a prefill pass at the table's threshold: this rank's slabs.
#[test]
fn a_pass_splits_only_on_prefill_with_the_lever_and_at_the_table() {
    let t = table("4096:1024,8208:2052");
    let wide = usize::MAX;
    for rank in 0..3 {
        let own = owned(8208, 3, rank).unwrap();
        assert_eq!(
            plan(Some(&t), 3, rank, 8208, 2052, true, wide).unwrap(),
            Some(own)
        );
    }
    assert_eq!(plan(None, 3, 0, 8208, 9000, true, wide).unwrap(), None);
    assert_eq!(plan(Some(&t), 3, 0, 8208, 2051, true, wide).unwrap(), None);
    assert_eq!(plan(Some(&t), 1, 0, 8208, 9000, true, wide).unwrap(), None);
    assert_eq!(plan(Some(&t), 3, 0, 4095, 9000, true, wide).unwrap(), None);
    assert_eq!(plan(Some(&t), 3, 0, 9000, 9000, true, wide).unwrap(), None);
    assert!(
        plan(Some(&t), 3, 0, 5000, 1024, true, wide)
            .unwrap()
            .is_some()
    );
    assert_eq!(plan(Some(&t), 3, 0, 8208, 2052, true, 0).unwrap(), None);
    for k in [1, 6, 287, 4096, 8208, 9000] {
        for pools in [0, 1024, 2052, 16384] {
            for per in [1, 2, 1368, wide] {
                for world in 1..=4 {
                    assert_eq!(
                        plan(Some(&t), world, 0, k, pools, false, per).unwrap(),
                        None,
                        "decode k {k} pools {pools} per {per} world {world}"
                    );
                }
            }
        }
    }
}

/// 2026-10-15: A 1-row sub-pass, in the unsplit plan or in any rank's slab, keeps the pass
/// unsplit; the same pass with every sub-pass at least 2 rows wide splits (the control).
/// 2026-10-17: slabs take `slab_passes` (even sizes), so a slab ends in a 1-row sub-pass only
/// at `per` 2 (or 1); the stepped plan's refusal at `per` 1367 is kept as the negative control.
#[test]
fn a_one_row_sub_pass_anywhere_keeps_the_pass_unsplit() {
    let t = table("4096:1024,8208:2052");
    let p = |k, per| plan(Some(&t), 3, 1, k, 4100, true, per).unwrap();
    // 2026-10-15: 8207 + 1 rows unsplit.
    assert_eq!(p(8208, 8207), None);
    // 2026-10-17: 8202 rows: 4101 x 2 unsplit, but every 1367-row slab as 683 x 2 + 1.
    assert_eq!(p(8202, 2), None);
    // 2026-10-17: controls: 2734 x 3 unsplit and 455 x 3 + 2 per slab; every 1368-row slab as
    // 684 + 684, which the stepped plan ran as 1367 + 1 and refused.
    assert_eq!(p(8202, 3), Some(owned(8202, 3, 1).unwrap()));
    assert_eq!(p(8208, 1367), Some(owned(8208, 3, 1).unwrap()));
    let stepped: SlabPlan = |slab, per| select::row_passes(&[slab], per);
    let old = plan_with(stepped, Some(&t), 3, 1, 8208, 4100, true, 1367).unwrap();
    assert_eq!(old, None);
    // 2026-10-15: controls: 1368-row sub-passes exactly; 1366 + 2 per slab and 6 x 1366 + 12
    // unsplit; 4104 + 4104 unsplit and whole slabs.
    for per in [1368, 1366, 4104] {
        assert_eq!(p(8208, per), Some(owned(8208, 3, 1).unwrap()), "per {per}");
    }
}

fn of(launches: &[MockLaunch], func: u64) -> Vec<MockLaunch> {
    launches
        .iter()
        .filter(|l| l.func == func)
        .cloned()
        .collect()
}

/// 2026-10-15: Rank 1 of 3, 96 rows (slabs of 16; rank 1 owns rows 16..32 and 64..80): one pool
/// compression; the expansion runs exactly those rows, each at the unsplit pass's output
/// offset, on the caller's stream; then two grouped exchanges on the comm stream, the first
/// sending rows 16..32 to ranks 0 and 2 and receiving their slabs 0..16 and 32..48 in place,
/// the second sending 64..80 and receiving 80..96 (rank 0) and 48..64 (rank 2). The process
/// counter rose by at least the two gathers.
#[test]
fn a_split_rank_selects_its_slabs_and_gathers_the_peers_in_place() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4096, 96).unwrap();
    let gpu = MockGpuBackend::new();
    let inp = inputs(&gpu, &c, 4096, 96);
    let scratch = DsaSelectScratch::alloc(&gpu, &c, &geom).unwrap();
    let comm = recorder(1, 3);
    let s = SplitStreams {
        comm: 99,
        slab_a: 1,
        slab_b: 2,
        done: 3,
    };
    let before = run::gathers();
    let own = owned(96, 3, 1).unwrap();
    assert_eq!(own, [(16, 32), (64, 80)]);
    let n = run_split(
        &gpu,
        &kernels(),
        &c,
        &geom,
        &inp,
        &scratch,
        &comm,
        &s,
        own,
        Payload::Tokens,
        7,
    )
    .unwrap();
    assert_eq!(n, 2);
    assert!(run::gathers() >= before + 2);
    let all = gpu.launches_snapshot();
    assert_eq!(all[0].func, KPOOL_COMPRESS);
    assert_eq!(of(&all, KPOOL_COMPRESS).len(), 1);
    assert!(all.iter().all(|l| l.stream == 7));
    let rb = c.out_width() * 4;
    let expand = of(&all, EXPAND);
    assert_eq!(expand.len(), 2);
    for (e, (lo, hi)) in expand.iter().zip(own) {
        assert_eq!(e.grid, [(hi - lo) as u32, 1, 1]);
        assert_eq!(e.args[6], MockArg::Buffer(scratch.tokens().offset(lo * rb)));
        assert_eq!(e.args[4], MockArg::Buffer(inp.q_pos.offset(lo * 4)));
    }
    let base = scratch.tokens().0;
    let at = |row: usize| base + (row * rb) as u64;
    let ops = comm.ops.lock().unwrap().clone();
    let group = |send_row: usize, peers: [(usize, usize); 2]| {
        let mut g = vec![("group_start", 0, 0, 0, 0)];
        for (peer, recv_row) in peers {
            g.push(("send", at(send_row), 16 * rb, peer, 99));
            g.push(("recv", at(recv_row), 16 * rb, peer, 99));
        }
        g.push(("group_end", 0, 0, 0, 0));
        g
    };
    let mut want = group(16, [(0, 0), (2, 32)]);
    want.extend(group(64, [(0, 80), (2, 48)]));
    assert_eq!(ops, want);
}

/// 2026-10-15: With the table on and a communicator of 3, under either payload: a decode pass
/// (and a prefill pass with no communicator) enqueues no exchange and launches exactly the
/// unsplit pass; the same
/// prefill pass with the communicator splits (the control: the recorder is wired).
#[test]
fn a_decode_pass_never_reaches_a_gather() {
    let c = cfg();
    let t = table("96:0");
    let geom = DsaSelectGeometry::plan(&c, 4096, 96).unwrap();
    let launches = |route: &dyn Fn(&MockGpuBackend, &DsaSelectScratch) -> usize| {
        let gpu = MockGpuBackend::new();
        let scratch = DsaSelectScratch::alloc(&gpu, &c, &geom).unwrap();
        let n = route(&gpu, &scratch);
        (n, gpu.launches_snapshot())
    };
    let unsplit = launches(&|gpu, scratch| {
        let inp = inputs(gpu, &c, 4096, 96);
        select_tokens_in_passes(gpu, &kernels(), &c, &geom, &inp, scratch, 7).unwrap()
    });
    let comm = recorder(0, 3);
    let routes = [(false, true), (true, false)];
    for (payload, (is_prefill, with_comm)) in [Payload::Tokens, Payload::Selected]
        .into_iter()
        .flat_map(|p| routes.map(|r| (p, r)))
    {
        let got = launches(&|gpu, scratch| {
            let inp = inputs(gpu, &c, 4096, 96);
            let cm: Option<&dyn metrale_comm::CommBackend> = with_comm.then_some(&comm);
            select_with(
                Some(&t),
                payload,
                gpu,
                &kernels(),
                &c,
                &geom,
                &inp,
                scratch,
                cm,
                is_prefill,
                7,
            )
            .unwrap()
        });
        assert_eq!(got.0, unsplit.0);
        assert_eq!(got.1.len(), unsplit.1.len());
        for (a, b) in got.1.iter().zip(&unsplit.1) {
            assert_eq!(
                (a.func, a.grid, &a.args, a.stream),
                (b.func, b.grid, &b.args, b.stream)
            );
        }
        assert!(
            comm.ops.lock().unwrap().is_empty(),
            "prefill {is_prefill} payload {payload:?}"
        );
    }
    let split = launches(&|gpu, scratch| {
        let inp = inputs(gpu, &c, 4096, 96);
        select_with(
            Some(&t),
            Payload::Tokens,
            gpu,
            &kernels(),
            &c,
            &geom,
            &inp,
            scratch,
            Some(&comm),
            true,
            7,
        )
        .unwrap()
    });
    assert_eq!(split.0, 2);
    assert_eq!(comm.ops.lock().unwrap().len(), 12);
}

/// 2026-10-15: The selected-pools payload, rank 1 of 3, 96 rows (slabs 16..32 and 64..80): one
/// compression; each slab's top-k writes its selected pools at its own rows (stride `select_k`)
/// and the fold runs exactly those rows there, with no token expansion before the gathers; the
/// two gathers move `select_k * 4`-byte rows of the selected region (rows 16..32 to ranks 0 and
/// 2, receiving 0..16 and 32..48; then 64..80, receiving 80..96 and 48..64); last, one folded
/// expansion of all 96 rows from the selected region into the token rows, on the caller's
/// stream. The token payload of the same split (the control) folds nothing and expands twice.
#[test]
fn a_selected_payload_split_folds_its_slabs_gathers_pools_and_expands_every_row() {
    let c = cfg();
    let geom = DsaSelectGeometry::plan(&c, 4096, 96).unwrap();
    let s = SplitStreams {
        comm: 99,
        slab_a: 1,
        slab_b: 2,
        done: 3,
    };
    let own = owned(96, 3, 1).unwrap();
    let run = |payload| {
        let gpu = MockGpuBackend::new();
        let inp = inputs(&gpu, &c, 4096, 96);
        let scratch = DsaSelectScratch::alloc(&gpu, &c, &geom).unwrap();
        let comm = recorder(1, 3);
        let k = &kernels();
        let n = run_split(
            &gpu, k, &c, &geom, &inp, &scratch, &comm, &s, own, payload, 7,
        )
        .unwrap();
        let ops = comm.ops.lock().unwrap().clone();
        (n, gpu.launches_snapshot(), ops, scratch, inp)
    };
    let (n, all, ops, scratch, inp) = run(Payload::Selected);
    assert_eq!(n, 2);
    assert!(all.iter().all(|l| l.stream == 7));
    assert_eq!(of(&all, KPOOL_COMPRESS).len(), 1);
    assert!(of(&all, EXPAND).is_empty());
    let sk = geom.select_k * 4;
    let sel = scratch.selected();
    let (topk, fold) = (of(&all, TOPK_POOLS), of(&all, SEL_FOLD));
    assert_eq!((topk.len(), fold.len()), (2, 2));
    for ((t, f), (lo, hi)) in topk.iter().zip(&fold).zip(own) {
        assert_eq!(t.args[1], MockArg::Buffer(sel.offset(lo * sk)));
        assert_eq!(f.grid, [(hi - lo) as u32, 1, 1]);
        assert_eq!(f.args[0], MockArg::Buffer(sel.offset(lo * sk)));
        assert_eq!(f.args[2], MockArg::Buffer(inp.q_pos.offset(lo * 4)));
        assert_eq!(f.args[3], MockArg::Buffer(inp.q_mask.offset(lo)));
    }
    let last = all.last().unwrap();
    assert_eq!(last.func, EXPAND_FOLDED);
    assert_eq!(of(&all, EXPAND_FOLDED).len(), 1);
    assert_eq!(last.grid, [96, 1, 1]);
    assert_eq!(last.args[0], MockArg::Buffer(sel));
    assert_eq!(last.args[3], MockArg::Buffer(inp.q_pos));
    assert_eq!(last.args[5], MockArg::Buffer(scratch.tokens()));
    let at = |row: usize| sel.0 + (row * sk) as u64;
    let group = |send_row: usize, peers: [(usize, usize); 2]| {
        let mut g = vec![("group_start", 0, 0, 0, 0)];
        for (peer, recv_row) in peers {
            g.push(("send", at(send_row), 16 * sk, peer, 99));
            g.push(("recv", at(recv_row), 16 * sk, peer, 99));
        }
        g.push(("group_end", 0, 0, 0, 0));
        g
    };
    let mut want = group(16, [(0, 0), (2, 32)]);
    want.extend(group(64, [(0, 80), (2, 48)]));
    assert_eq!(ops, want);
    let (_, ctl, ctl_ops, ..) = run(Payload::Tokens);
    assert!(of(&ctl, SEL_FOLD).is_empty() && of(&ctl, EXPAND_FOLDED).is_empty());
    assert_eq!(of(&ctl, EXPAND).len(), 2);
    assert_ne!(ctl_ops, ops);
}
