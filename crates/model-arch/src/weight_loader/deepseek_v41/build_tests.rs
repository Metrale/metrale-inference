// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-10-19: The engine configs a rank plans from the real config: its local heads, groups and
//! expert width, and an FP8 stage exactly as large as its largest projection.

use anyhow::bail;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_layers::layers::ops::ResidentMat;
use metrale_model_layers::lever_equivalence::defaults;

use super::super::source::{EngramWkv, T, V41Source};
use super::super::tp_cut::tests::v41_rank;
use super::*;

/// 2026-10-19: `plan` asks a source only which layers carry the compressor and the indexer; this
/// one answers for layer 2 and refuses every read, so a plan that read a tensor fails.
struct Carries;

impl V41Source for Carries {
    fn has(&self, l: usize, t: T) -> bool {
        l == 2 && matches!(t, T::CompWkv | T::IdxWqB)
    }
    fn mat(&self, _: &dyn GpuBackend, _: usize, _: T) -> Result<ResidentMat> {
        bail!("plan reads no tensor")
    }
    fn bf16(&self, _: &dyn GpuBackend, _: usize, _: T) -> Result<DevicePtr> {
        bail!("plan reads no tensor")
    }
    fn host_f32(&self, _: &dyn GpuBackend, _: usize, _: T) -> Result<Vec<f32>> {
        bail!("plan reads no tensor")
    }
    fn engram_wkv(&self, _: &dyn GpuBackend, _: usize) -> Result<EngramWkv> {
        bail!("plan reads no tensor")
    }
}

fn levers() -> Ds41LoadLevers {
    Ds41LoadLevers::from_settings(&defaults())
}

#[test]
fn each_rank_plans_its_own_heads_groups_and_expert_width() {
    let want = [(24, 3), (24, 3), (16, 2)];
    for (rank, &(heads, groups)) in want.iter().enumerate() {
        let c = v41_rank(3, rank);
        let p = plan(&c, &levers(), &Carries, &Local::of(&c).unwrap(), true).unwrap();
        let a = &p.attn_cfg;
        assert_eq!(
            (a.n_heads, a.groups, a.tp_world),
            (heads, groups, 3),
            "rank {rank}"
        );
        assert_eq!(p.moe_cfg.inter, 768, "rank {rank}");
        assert_eq!(
            (p.kv_sources.clone(), p.index_sources.clone()),
            (vec![2], vec![2])
        );
        assert_eq!(p.ratios.len(), c.num_hidden_layers);
    }
}

#[test]
fn the_fp8_stage_holds_the_largest_projection_and_no_more() {
    let c = v41_rank(3, 0);
    let p = plan(&c, &levers(), &Carries, &Local::of(&c).unwrap(), true).unwrap();
    let a = &p.attn_cfg;
    // 2026-10-19: wq_b keeps 24 heads x 512 rows of the 1280-wide q rank; wo_b keeps 3 groups x
    // 1024 columns of the 5120 rows. Both are 24 * 512 * 1280 values, the largest site.
    assert_eq!(a.fp8_stage.weight_elems, 24 * 512 * 1280);
    assert_eq!(a.fp8_stage.weight_elems, 5120 * 3 * 1024);
    assert_eq!(a.fp8_stage.act_elems, p.max_tokens * 5120);
    let m = &p.moe_cfg.fp8_stage;
    assert_eq!(
        (m.weight_elems, m.act_elems),
        (768 * 5120, p.max_tokens * 5120)
    );
}

#[test]
fn the_gguf_plan_is_one_whole_rank_without_a_stage() {
    let c = v41_rank(1, 0);
    let p = plan(&c, &levers(), &Carries, &Local::whole(&c), false).unwrap();
    assert_eq!(
        (p.attn_cfg.n_heads, p.attn_cfg.groups, p.attn_cfg.tp_world),
        (64, 8, 1)
    );
    assert_eq!(p.moe_cfg.inter, 2304);
    assert_eq!(p.attn_cfg.fp8_stage, Fp8StageSize::NONE);
    assert_eq!(p.moe_cfg.fp8_stage, Fp8StageSize::NONE);
}

#[test]
fn a_checkpoint_without_a_compressor_is_refused() {
    struct Bare;
    impl V41Source for Bare {
        fn has(&self, _: usize, _: T) -> bool {
            false
        }
        fn mat(&self, g: &dyn GpuBackend, l: usize, t: T) -> Result<ResidentMat> {
            Carries.mat(g, l, t)
        }
        fn bf16(&self, g: &dyn GpuBackend, l: usize, t: T) -> Result<DevicePtr> {
            Carries.bf16(g, l, t)
        }
        fn host_f32(&self, g: &dyn GpuBackend, l: usize, t: T) -> Result<Vec<f32>> {
            Carries.host_f32(g, l, t)
        }
        fn engram_wkv(&self, g: &dyn GpuBackend, l: usize) -> Result<EngramWkv> {
            Carries.engram_wkv(g, l)
        }
    }
    let c = v41_rank(1, 0);
    let e = plan(&c, &levers(), &Bare, &Local::whole(&c), false)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("no compressor"), "{e}");
}

/// 2026-10-19: The serve's bounds cap the per-step workspace and the position tables; a bound of
/// 0 (unset) leaves the levers, and a bound above a lever leaves the lever. A plan that ignored
/// either bound would size every reduce, score and activation buffer for rows no step carries.
#[test]
fn the_serve_bounds_cap_the_plan_and_never_raise_it() {
    let lv = levers();
    let at = |rows: usize, seq: usize| {
        let mut c = v41_rank(3, 0);
        c.serve_max_batch_tokens = rows;
        c.serve_max_seq_len = seq;
        let p = plan(&c, &lv, &Carries, &Local::of(&c).unwrap(), true).unwrap();
        let (a, m) = (&p.attn_cfg, &p.moe_cfg);
        assert_eq!((a.max_tokens, m.max_tokens), (p.max_tokens, p.max_tokens));
        assert_eq!(a.max_seq, p.max_seq);
        (p.max_tokens, p.max_seq)
    };
    assert_eq!(at(0, 0), (lv.max_tokens, lv.max_seq));
    assert_eq!(at(513, 4096), (513, 4096));
    assert_eq!(
        at(lv.max_tokens + 1, lv.max_seq + 1),
        (lv.max_tokens, lv.max_seq)
    );
    assert_eq!(at(0, 300), (300, 300));
}
