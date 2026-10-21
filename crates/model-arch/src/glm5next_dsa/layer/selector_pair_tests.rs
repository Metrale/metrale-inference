// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `selector_query_rows` under `METRALE_GLM_DSA_SELECTOR_PAIR`: one pair launch with
//! both operand sets where the few-row FP32-out entry would run both, else the two launches.
//! The bytes are checked on a GPU by `examples/glm5next_dsa_proj_in_gate`.

use metrale_gpu_runtime::gpu::mock::MockArg;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::decode_rows_fixture::{BATCHM_FEW_F32, Rig};

const PAIR: u64 = 0x10D;

fn u32a(v: u32) -> MockArg {
    MockArg::Bytes(v.to_le_bytes().to_vec())
}

/// 2026-10-21: The launches (kernel, grid x, args) of one `selector_query_rows` of `n` rows.
fn run(lever: bool, pair: bool, few: bool, n: usize) -> (Vec<(u64, u32, Vec<MockArg>)>, Vec<DevicePtr>) {
    let rig = Rig::new();
    let mut l = rig.layer();
    l.levers.dsa_selector_pair = lever;
    l.levers.dense.batchm_few = few;
    l.kernels.selector_pair = KernelHandle(if pair { PAIR } else { 0 });
    let hidden = rig.gpu.alloc(64).unwrap();
    let from = rig.gpu.launches_snapshot().len();
    l.selector_query_rows(&rig.gpu, hidden, n, false, 0).unwrap();
    let w = &l.workspace;
    let ptrs = vec![
        w.q_resid,
        l.weights.wq_b,
        w.q_idx_rows,
        hidden,
        l.weights.weights_proj,
        w.head_weights_rows,
    ];
    let got = rig.gpu.launches_snapshot()[from..]
        .iter()
        .map(|x| (x.func, x.grid[0], x.args.clone()))
        .collect();
    (got, ptrs)
}

#[test]
fn two_to_eight_rows_take_one_pair_launch_with_both_operand_sets() {
    let c = super::super::decode_rows_fixture::cfg();
    let (heads, idx_row) = (c.index_heads as u32, (c.index_heads * c.index_head_dim) as u32);
    for n in [2usize, 5, 8] {
        let (got, p) = run(true, true, true, n);
        assert_eq!(got.len(), 1, "n={n}: {got:?}");
        let (f, grid, a) = &got[0];
        assert_eq!(*f, PAIR);
        assert_eq!(*grid, idx_row.div_ceil(4) + heads.div_ceil(4));
        let b = |i: usize| MockArg::Buffer(p[i]);
        assert_eq!(
            a,
            &vec![
                b(0),
                b(1),
                b(2),
                u32a(idx_row),
                u32a(c.q_lora_rank as u32),
                u32a(idx_row),
                b(3),
                b(4),
                b(5),
                u32a(heads),
                u32a(c.hidden as u32),
                u32a(heads),
                u32a(n as u32),
            ]
        );
    }
}

/// 2026-10-21: Lever off, the pair entry absent, the few entry off, or 9+ rows: the two launches.
#[test]
fn otherwise_the_two_launches_run() {
    for (lever, pair, few, n) in [
        (false, true, true, 2),
        (true, false, true, 2),
        (true, true, false, 2),
        (true, true, true, 9),
    ] {
        let (got, _) = run(lever, pair, few, n);
        assert_eq!(got.len(), 2, "{lever} {pair} {few} {n}: {got:?}");
        assert!(got.iter().all(|(f, ..)| *f != PAIR));
        if few && n <= 8 {
            assert!(got.iter().all(|(f, ..)| *f == BATCHM_FEW_F32));
        }
    }
}
