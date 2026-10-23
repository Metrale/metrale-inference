// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: The one-field sabotages of a bound DSA plan, shared by the text-layer gate
//! (`dsa_tests.rs`) and the draft head's (`dsa_mtp_tests.rs`), moved out of `dsa_tests.rs`
//! (exact copy, made a function): each plan differs from the good one in one field.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::KernelHandle;

use crate::glm5next_dsa::plan::{AttendStep, DsaPlan, DsaSteps, KNormStep, ScoresStep};

/// 2026-10-23: `good` with one field changed, by field: the latent write, the store, the pool
/// key, the geometry, the ceiling (pools, CTAs, SMs), the key norm, the scores, a top-k pass, the
/// collect, the expansion, the attend entry, its merge and its split count; `other` is the
/// kernel a changed launch names.
pub(crate) fn sabotaged(good: &DsaPlan, other: KernelHandle) -> Vec<(&'static str, DsaPlan)> {
    let good = *good;
    let s = good.steps;
    let with = |steps: DsaSteps<KernelHandle>| DsaPlan { steps, ..good };
    let mut bad: Vec<(&'static str, DsaPlan)> = vec![
        (
            "latent",
            with(DsaSteps {
                latent_write: other,
                ..s
            }),
        ),
        ("store", with(DsaSteps { store: other, ..s })),
        (
            "pool key",
            with(DsaSteps {
                pool_keys: other,
                ..s
            }),
        ),
        (
            "geometry",
            with(DsaSteps {
                write_geom: other,
                ..s
            }),
        ),
        (
            "pools",
            DsaPlan {
                max_pools: good.max_pools / 2,
                ..good
            },
        ),
        (
            "ctas",
            DsaPlan {
                topk_ctas: good.topk_ctas - 1,
                ..good
            },
        ),
        (
            "sms",
            DsaPlan {
                sm_count: good.sm_count - 1,
                ..good
            },
        ),
    ];
    let k_norm = match s.k_norm {
        KNormStep::Row(_) => KNormStep::Row(other),
        KNormStep::Rows(_) => KNormStep::Rows(other),
    };
    bad.push(("key norm", with(DsaSteps { k_norm, ..s })));
    let mut sel = s.select;
    sel.scores = match sel.scores {
        ScoresStep::HeadsPaged(_) => ScoresStep::HeadsPaged(other),
        ScoresStep::HeadsWindow(_) => ScoresStep::HeadsWindow(other),
    };
    bad.push(("scores", with(DsaSteps { select: sel, ..s })));
    for (what, f) in [("top-k pass", 0), ("collect", 1), ("expansion", 2)] {
        let mut sel = s.select;
        *[&mut sel.topk_pass, &mut sel.topk_collect, &mut sel.expand][f] = other;
        bad.push((what, with(DsaSteps { select: sel, ..s })));
    }
    let AttendStep::HeadBatched {
        decode,
        merge,
        splits,
    } = s.attend
    else {
        panic!("the served attend is head-batched")
    };
    for (what, attend) in [
        (
            "attend entry",
            AttendStep::HeadBatched {
                decode: other,
                merge,
                splits,
            },
        ),
        (
            "attend merge",
            AttendStep::HeadBatched {
                decode,
                merge: other,
                splits,
            },
        ),
        (
            "attend splits",
            AttendStep::HeadBatched {
                decode,
                merge,
                splits: splits / 2,
            },
        ),
    ] {
        bad.push((what, with(DsaSteps { attend, ..s })));
    }
    bad
}
