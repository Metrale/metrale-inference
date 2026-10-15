// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-15: Host tests of the unabsorbed-prefill routing on the mock backend: a prefill pass
//! of a layer holding unabsorbed weights runs `q_b_proj`, `glm5next_mla_head_bmm_q`,
//! `glm5next_mla_head_bmm_o` and `o_proj` on the right buffers, in order, and never reads
//! `q_absorb` or `o_absorb`; a decode pass, or a layer without them, launches exactly what it
//! launched before; more rows than the scratch are refused before the half's first launch.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::{MockArg, MockLaunch};
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use crate::glm5next_dsa::layer::Glm5NextDsaLayer;
use crate::glm5next_dsa::unabsorb::lever::HEAD_DIM;
use crate::glm5next_dsa::unabsorb::{HeadBmmKernels, MlaUnabsorb, UnabsorbWeights};

use super::super::decode_rows_fixture::*;

const HB_Q: u64 = 0x301;
const HB_O: u64 = 0x302;
const MID_ROWS: usize = 8;

fn u32_arg(v: usize) -> MockArg {
    MockArg::Bytes((v as u32).to_le_bytes().to_vec())
}

fn with_unabsorb(rig: &Rig) -> Glm5NextDsaLayer {
    let mut layer = rig.layer();
    let d = layer.cfg.local_heads * HEAD_DIM;
    layer.weights.unabsorb = Some(UnabsorbWeights {
        mode: MlaUnabsorb::Declared,
        q_b: rig.buf(64),
        o_proj: rig.buf(64),
        w_uk_t: rig.buf(64),
        w_uv: rig.buf(64),
        mid: rig.buf(MID_ROWS * d * 2),
        mid_rows: MID_ROWS,
        kernels: HeadBmmKernels {
            q: KernelHandle(HB_Q),
            o: KernelHandle(HB_O),
        },
    });
    layer
}

/// 2026-10-15: The index of the first launch reading `p`, if any.
fn reads(l: &[MockLaunch], p: DevicePtr) -> Option<usize> {
    l.iter().position(|x| x.args.contains(&ptr(p)))
}

fn run_in(rig: &Rig, layer: &Glm5NextDsaLayer, k: usize, prefill: bool) -> Vec<MockLaunch> {
    let hidden = rig.buf(16 * HIDDEN * 2);
    let from = rig.gpu.launch_count();
    layer
        .project_in(&rig.gpu, hidden, k, prefill, 7)
        .unwrap();
    rig.since(from)
}

fn run_out(rig: &Rig, layer: &Glm5NextDsaLayer, k: usize, prefill: bool) -> Vec<MockLaunch> {
    let hidden = rig.buf(16 * HIDDEN * 2);
    let from = rig.gpu.launch_count();
    layer
        .project_out(&rig.gpu, hidden, k, prefill, 7)
        .unwrap();
    rig.since(from)
}

/// 2026-10-15: Prefill, query side: `q_b_proj` reads `q_resid` into the scratch, then one
/// `head_bmm_q` reads the scratch and `W_UK^T` into `q_abs` over `k` rows and the rank's heads,
/// one row tile of four 128-wide column tiles; `q_absorb` is never read.
#[test]
fn prefill_query_runs_q_b_then_head_bmm_q() {
    let rig = Rig::new();
    let layer = with_unabsorb(&rig);
    let (u, w) = (layer.weights.unabsorb.as_ref().unwrap(), &layer.workspace);
    for k in [1usize, 4, MID_ROWS] {
        let l = run_in(&rig, &layer, k, true);
        assert_eq!(reads(&l, layer.weights.q_absorb), None, "k {k}");
        let qb = reads(&l, u.q_b).expect("q_b_proj launched");
        assert!(l[qb].args.contains(&ptr(w.q_resid)) && l[qb].args.contains(&ptr(u.mid)));
        let hb = of(&l, HB_Q);
        assert_eq!(hb.len(), 1, "k {k}");
        assert_eq!(
            hb[0].args,
            vec![
                ptr(u.mid),
                ptr(u.w_uk_t),
                ptr(w.q_abs),
                u32_arg(k),
                u32_arg(layer.cfg.local_heads)
            ]
        );
        assert_eq!(hb[0].grid, [4, 1, layer.cfg.local_heads as u32]);
        assert_eq!(hb[0].block, [256, 1, 1]);
        let at = l.iter().position(|x| x.func == HB_Q).unwrap();
        assert!(qb < at, "q_b_proj before head_bmm_q");
        assert!(of(&l, HB_O).is_empty());
    }
}

/// 2026-10-15: Prefill, output side: one `head_bmm_o` reads `attn_out` and `W_UV` into the
/// scratch, then `o_proj` reads the scratch; `o_absorb` is never read.
#[test]
fn prefill_output_runs_head_bmm_o_then_o_proj() {
    let rig = Rig::new();
    let layer = with_unabsorb(&rig);
    let u = layer.weights.unabsorb.as_ref().unwrap();
    for k in [1usize, 5, MID_ROWS] {
        let l = run_out(&rig, &layer, k, true);
        assert_eq!(reads(&l, layer.weights.o_absorb), None, "k {k}");
        let hb = of(&l, HB_O);
        assert_eq!(hb.len(), 1);
        assert_eq!(
            hb[0].args,
            vec![
                ptr(layer.workspace.attn_out),
                ptr(u.w_uv),
                ptr(u.mid),
                u32_arg(k),
                u32_arg(layer.cfg.local_heads)
            ]
        );
        assert_eq!(hb[0].grid, [2, 1, layer.cfg.local_heads as u32]);
        let at = l.iter().position(|x| x.func == HB_O).unwrap();
        let op = reads(&l, u.o_proj).expect("o_proj launched");
        assert!(at < op, "head_bmm_o before o_proj");
        assert!(l[op].args.contains(&ptr(u.mid)));
        assert!(of(&l, HB_Q).is_empty());
    }
}

/// 2026-10-15: A decode (or verify) pass of a layer holding unabsorbed weights launches exactly
/// what the same layer without them launches: the absorbed products, no `head_bmm`, nothing
/// that reads an unabsorbed weight or the scratch.
#[test]
fn decode_passes_are_unchanged() {
    let rig = Rig::new();
    let on = with_unabsorb(&rig);
    let u = on.weights.unabsorb.as_ref().unwrap();
    for k in [1usize, 4] {
        let l_in = run_in(&rig, &on, k, false);
        let l_out = run_out(&rig, &on, k, false);
        for l in [&l_in, &l_out] {
            assert!(of(l, HB_Q).is_empty() && of(l, HB_O).is_empty());
            for p in [u.q_b, u.o_proj, u.w_uk_t, u.w_uv, u.mid] {
                assert_eq!(reads(l, p), None, "k {k}");
            }
        }
        assert!(reads(&l_in, on.weights.q_absorb).is_some());
        assert!(reads(&l_out, on.weights.o_absorb).is_some());
        let off = rig.layer();
        let shape = |l: &[MockLaunch]| -> Vec<(u64, [u32; 3])> {
            l.iter().map(|x| (x.func, x.grid)).collect()
        };
        assert_eq!(shape(&l_in), shape(&run_in(&rig, &off, k, false)), "k {k}");
        assert_eq!(shape(&l_out), shape(&run_out(&rig, &off, k, false)), "k {k}");
    }
}

/// 2026-10-15: More rows than the scratch holds: each half refuses before its first launch.
#[test]
fn rows_past_the_scratch_are_refused_before_launch() {
    let rig = Rig::new();
    let layer = with_unabsorb(&rig);
    let u = layer.weights.unabsorb.as_ref().unwrap();
    let hidden = rig.buf(16 * HIDDEN * 2);
    let k = MID_ROWS + 1;
    let from = rig.gpu.launch_count();
    let e = layer
        .project_in(&rig.gpu, hidden, k, true, 7)
        .unwrap_err();
    assert!(e.to_string().contains("for a scratch of 8"), "{e}");
    let l = rig.since(from);
    assert_eq!(reads(&l, u.q_b), None);
    assert!(of(&l, HB_Q).is_empty());
    let from = rig.gpu.launch_count();
    assert!(layer.project_out(&rig.gpu, hidden, k, true, 7).is_err());
    assert!(rig.since(from).is_empty(), "nothing launched");
}
