// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the DSA decode launcher's guards, geometry only. The
//! kernel's numerics are checked on a GPU by the `glm5next_dsa_decode_gate`
//! example.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use super::*;
use crate::glm5next_dsa::select::DsaSelectGeometry;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

fn paging() -> DsaDecodePaging {
    DsaDecodePaging {
        num_seqs: 1,
        num_q_heads: 64,
        num_kv_heads: 1,
        max_blocks_per_seq: 256,
        block_size: 64,
        cache_stride_bytes: 64 * 512,
    }
}

/// 2026-09-25: Pools are built over absolute positions, so a `block_size` that
/// is not a multiple of `index_kpool` would split a pool across two pages.
#[test]
fn a_block_size_that_splits_a_pool_is_refused() {
    let c = cfg();
    let mut p = paging();
    assert!(p.validate(&c).is_ok(), "64 is a multiple of kpool 4");
    p.block_size = 66;
    let e = p.validate(&c).unwrap_err().to_string();
    assert!(e.contains("straddle"), "{e}");
}

/// 2026-09-25: MLA has one latent KV head; `validate` refuses any other count.
#[test]
fn more_than_one_kv_head_is_refused() {
    let c = cfg();
    let mut p = paging();
    p.num_kv_heads = 8;
    assert!(p.validate(&c).is_err());
}

#[test]
fn degenerate_launches_are_refused() {
    let c = cfg();
    for mutate in [
        (|p: &mut DsaDecodePaging| p.num_seqs = 0) as fn(&mut DsaDecodePaging),
        |p: &mut DsaDecodePaging| p.num_q_heads = 0,
        |p: &mut DsaDecodePaging| p.block_size = 0,
    ] {
        let mut p = paging();
        mutate(&mut p);
        assert!(p.validate(&c).is_err(), "{p:?} must be refused");
    }
}

/// 2026-09-25: `decode_attention` refuses a selection planned for a different row
/// count than it decodes. The launch needs a GPU, so this checks only that the
/// refused condition is reachable from `DsaSelectGeometry::plan`.
#[test]
fn a_row_count_mismatch_between_selection_and_launch_is_refused() {
    let c = cfg();
    let p = paging();
    let two_rows = DsaSelectGeometry::plan(&c, 4_096, 2).unwrap();
    assert_eq!(two_rows.q_rows, 2);
    assert_ne!(two_rows.q_rows, p.num_seqs);
    assert!(
        two_rows.q_rows != p.num_seqs,
        "the guarded condition must be reachable"
    );
    let one_row = DsaSelectGeometry::plan(&c, 4_096, 1).unwrap();
    assert_eq!(one_row.q_rows, p.num_seqs);
}

/// 2026-09-25: NoPE: the softmax scale is over the latent width, which is the whole
/// cache token, not over a 576-wide token with a 64-dim rope tail (`MLA_CACHE_DIM`
/// in the DeepSeek-V4-Flash decode kernels).
#[test]
fn the_score_scale_is_the_latent_width_not_the_v4_cache_width() {
    let c = cfg();
    assert_eq!(c.kv_cache_dim(), 512, "NoPE: cache token is pure latent");
    assert_eq!(c.kv_cache_dim(), c.kv_lora_rank);
    let ours = (c.kv_lora_rank as f32).powf(-0.5);
    let v4 = 576f32.powf(-0.5);
    assert!((ours - 0.044_194_173).abs() < 1e-9, "1/sqrt(512)");
    assert!(
        (ours - v4).abs() > 1e-4,
        "the two scales must not be interchangeable by accident"
    );
}

/// 2026-10-14: A head-batched kernel at 8 splits with every twin present.
fn hb8() -> HeadBatched {
    HeadBatched {
        decode: KernelHandle(11),
        decode_p_hi: KernelHandle(13),
        prefill_pipe: KernelHandle(14),
        prefill_tc: KernelHandle(15),
        prefill_tc_nanor: KernelHandle(16),
        merge: KernelHandle(12),
        splits: 8,
        smem: 0,
        block: HB_BLOCK,
        offset_table: false,
    }
}

fn per_head_only() -> Glm5NextDsaDecodeKernel {
    Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: None,
    }
}

/// 2026-10-12: The prefill attend takes one split only when asked and only on the head-batched
/// kernel; decode keeps the lever's split count, and the per-head kernel stays per-head.
#[test]
fn the_prefill_attend_drops_to_one_split_only_when_asked() {
    let k = Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: Some(hb8()),
    };
    assert_eq!(k.for_prefill(true).hb_splits(), Some(1));
    assert_eq!(k.for_prefill(false).hb_splits(), Some(8));
    assert_eq!(k.hb_splits(), Some(8));
    assert_eq!(per_head_only().for_prefill(true).hb_splits(), None);
}

/// 2026-10-12: The P-hi twin replaces the head-batched entry only when asked and present: off,
/// the kernel is unchanged; on without the head-batched kernel, or with a target that lacks the
/// twin, it is an error rather than the hi + lo entry.
#[test]
fn the_prefill_p_hi_twin_is_swapped_in_only_when_asked_and_present() {
    let hb = hb8();
    let k = Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: Some(hb),
    };
    let off = k
        .for_prefill_p_hi(false)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!(off.decode.0, 11);
    let on = k.for_prefill_p_hi(true).ok().unwrap().head_batched.unwrap();
    assert_eq!(
        (on.decode.0, on.merge.0, on.splits, on.smem),
        (13, 12, 8, 0)
    );
    let no_twin = Glm5NextDsaDecodeKernel {
        head_batched: Some(HeadBatched {
            decode_p_hi: KernelHandle(0),
            ..hb
        }),
        ..k
    };
    let e = no_twin.for_prefill_p_hi(true).err().unwrap().to_string();
    assert!(e.contains("not in this target"), "{e}");
    let e = per_head_only()
        .for_prefill_p_hi(true)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("needs METRALE_GLM_DSA_DECODE_HB"), "{e}");
}

/// 2026-10-14: The pipelined twin replaces the P-hi entry only when asked, only on top of the
/// P-hi entry, and carries its dynamic shared memory; off, nothing changes (smem stays 0, the
/// other entries' launch); every refusal is an error, never a silent P-hi or hi + lo launch.
#[test]
fn the_prefill_pipe_twin_is_swapped_in_only_over_p_hi_and_carries_its_smem() {
    let hb = hb8();
    let k = Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: Some(hb),
    };
    let p_hi = k.for_prefill_p_hi(true).ok().unwrap();
    let off = p_hi
        .for_prefill_pipe(false)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!((off.decode.0, off.smem), (13, 0));
    let on = p_hi
        .for_prefill_pipe(true)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!(
        (on.decode.0, on.merge.0, on.splits, on.smem),
        (14, 12, 8, PIPE_SMEM_BYTES)
    );
    // 2026-10-14: the kernel's PP_SMEM_BYTES: two 32 KB tiles + four 24 x 36 f32 planes.
    assert_eq!(PIPE_SMEM_BYTES, 79_360);
    let one = p_hi.for_prefill(true).for_prefill_pipe(true).ok().unwrap();
    assert_eq!(one.hb_splits(), Some(1));

    let e = k.for_prefill_pipe(true).err().unwrap().to_string();
    assert!(e.contains("needs METRALE_GLM_DSA_HB_PREFILL_P_HI=1"), "{e}");
    let no_twin = Glm5NextDsaDecodeKernel {
        head_batched: Some(HeadBatched {
            prefill_pipe: KernelHandle(0),
            ..hb
        }),
        ..k
    };
    let e = no_twin
        .for_prefill_p_hi(true)
        .and_then(|k| k.for_prefill_pipe(true))
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("not in this target"), "{e}");
    let e = per_head_only()
        .for_prefill_pipe(true)
        .err()
        .unwrap()
        .to_string();
    assert!(e.contains("needs METRALE_GLM_DSA_DECODE_HB"), "{e}");
}

/// 2026-10-14: The tc twin replaces the P-hi entry only when asked, only at one split and only
/// over the P-hi entry (not over the pipe twin), and carries its block, shared memory and
/// offset-table limits; off, nothing changes. Every refusal is an error naming what is missing.
#[test]
fn the_prefill_tc_twin_is_swapped_in_only_over_p_hi_at_one_split() {
    let hb = hb8();
    let k = Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: Some(hb),
    };
    let p_hi1 = k.for_prefill(true).for_prefill_p_hi(true).ok().unwrap();
    let off = p_hi1
        .for_prefill_tc(false)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!(
        (off.decode.0, off.smem, off.block, off.offset_table),
        (13, 0, HB_BLOCK, false)
    );
    let on = p_hi1
        .for_prefill_tc(true)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!(
        (on.decode.0, on.splits, on.smem, on.block, on.offset_table),
        (15, 1, TC_SMEM_BYTES, TC_BLOCK, true)
    );
    // 2026-10-14: the kernel's TC_SMEM_BYTES: two 32 KB tiles, four 24 x 36 f32 planes and a
    // 2064-slot u32 offset table.
    assert_eq!(TC_SMEM_BYTES, 87_616);

    let err = |r: Result<Glm5NextDsaDecodeKernel>| r.err().unwrap().to_string();
    let e = err(k.for_prefill(true).for_prefill_tc(true));
    assert!(e.contains("needs METRALE_GLM_DSA_HB_PREFILL_P_HI=1"), "{e}");
    let e = err(p_hi1
        .for_prefill_pipe(true)
        .and_then(|k| k.for_prefill_tc(true)));
    assert!(
        e.contains("excludes METRALE_GLM_DSA_HB_PREFILL_PIPE=1"),
        "{e}"
    );
    let e = err(k
        .for_prefill_p_hi(true)
        .and_then(|k| k.for_prefill_tc(true)));
    assert!(e.contains("one split, got 8"), "{e}");
    let no_twin = Glm5NextDsaDecodeKernel {
        head_batched: Some(HeadBatched {
            prefill_tc: KernelHandle(0),
            ..hb
        }),
        ..k
    };
    let e = err(no_twin
        .for_prefill(true)
        .for_prefill_p_hi(true)
        .and_then(|k| k.for_prefill_tc(true)));
    assert!(e.contains("not in this target"), "{e}");
    let e = err(per_head_only().for_prefill_tc(true));
    assert!(e.contains("needs METRALE_GLM_DSA_DECODE_HB"), "{e}");
}

/// 2026-10-15: The NaN-test twin replaces only the tc entry, keeping its launch shape (block,
/// shared memory, offset table, one split); off, nothing changes. Refused, as an error naming
/// what is missing, over any other entry or without the twin in the build.
#[test]
fn the_tc_nanor_twin_is_swapped_in_only_over_the_tc_entry() {
    let hb = hb8();
    let k = Glm5NextDsaDecodeKernel {
        per_head: KernelHandle(10),
        head_batched: Some(hb),
    };
    let tc = k
        .for_prefill(true)
        .for_prefill_p_hi(true)
        .and_then(|k| k.for_prefill_tc(true))
        .ok()
        .unwrap();
    let off = tc.for_prefill_tc_nanor(false).ok().unwrap();
    assert_eq!(off.head_batched.unwrap().decode.0, 15);
    let on = tc
        .for_prefill_tc_nanor(true)
        .ok()
        .unwrap()
        .head_batched
        .unwrap();
    assert_eq!(
        (on.decode.0, on.splits, on.smem, on.block, on.offset_table),
        (16, 1, TC_SMEM_BYTES, TC_BLOCK, true)
    );

    let err = |r: Result<Glm5NextDsaDecodeKernel>| r.err().unwrap().to_string();
    let p_hi1 = k.for_prefill(true).for_prefill_p_hi(true).ok().unwrap();
    let e = err(p_hi1.for_prefill_tc_nanor(true));
    assert!(e.contains("needs METRALE_GLM_DSA_HB_PREFILL_TC=1"), "{e}");
    let e = err(p_hi1
        .for_prefill_pipe(true)
        .and_then(|k| k.for_prefill_tc_nanor(true)));
    assert!(e.contains("needs METRALE_GLM_DSA_HB_PREFILL_TC=1"), "{e}");
    let no_tc = Glm5NextDsaDecodeKernel {
        head_batched: Some(HeadBatched {
            prefill_tc: KernelHandle(0),
            ..hb
        }),
        ..k
    };
    let e = err(no_tc.for_prefill(true).for_prefill_tc_nanor(true));
    assert!(e.contains("needs METRALE_GLM_DSA_HB_PREFILL_TC=1"), "{e}");
    let no_twin = Glm5NextDsaDecodeKernel {
        head_batched: Some(HeadBatched {
            prefill_tc_nanor: KernelHandle(0),
            ..hb
        }),
        ..k
    };
    let e = err(no_twin
        .for_prefill(true)
        .for_prefill_p_hi(true)
        .and_then(|k| k.for_prefill_tc(true))
        .and_then(|k| k.for_prefill_tc_nanor(true)));
    assert!(e.contains("not in this target"), "{e}");
    let e = err(per_head_only().for_prefill_tc_nanor(true));
    assert!(e.contains("needs METRALE_GLM_DSA_DECODE_HB"), "{e}");
}
