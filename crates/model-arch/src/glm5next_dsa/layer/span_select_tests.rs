// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: Host tests of `METRALE_GLM_DSA_SPAN_SELECT` on the mock backend, which records
//! every launch with its arguments: a captured speculative verify stores and selects the rows of
//! its paged spans in one launch per stage, through metadata rows from its base, and a flat
//! span after them keeps the per-row launches; off, every row keeps them. The bytes are checked
//! on a GPU by `examples/glm5next_dsa_span_select_microtest.rs`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::mock::MockLaunch;
use metrale_model_layers::layer::LayerState;

use crate::glm5next_dsa::layer::DsaRowSpan;
use crate::glm5next_dsa::state::Glm5NextDsaState;

use super::super::decode_rows_fixture::*;

fn paged(len: usize) -> Box<dyn LayerState> {
    let mut s = Glm5NextDsaState::paged(&cfg()).unwrap();
    s.advance(len).unwrap();
    Box::new(s)
}

/// 2026-10-14: The per-row store, geometry and compression launches, and the rows store's grid.
fn counts(l: &[MockLaunch]) -> (usize, usize, usize, Option<[u32; 3]>) {
    (
        of(l, STORE).len(),
        of(l, GEOM).len(),
        of(l, KPOOL).len(),
        of(l, STORE_ROWS).first().map(|x| x.grid),
    )
}

/// 2026-10-14: Spans of 2 and 3 rows on paged caches, then a 1-row span on a flat cache,
/// captured and staged. On: the five paged rows store and select in one launch per stage
/// (metadata rows 0..5), the flat row alone runs per row at metadata row 5, and each state grows
/// by its span. Off: no rows launch, six per-row stores. A bug that counted spans as rows (2
/// rows), or ran past the flat span, fails here.
#[test]
fn verify_spans_store_and_select_in_one_launch_per_stage() {
    for on in [true, false] {
        let rig = Rig::new();
        let layer = rig.layer();
        let meta = rig.meta(6);
        let mut boxes = vec![paged(5), paged(9)];
        boxes.extend(rig.states(&[2]));
        let spans =
            [(5, 2), (9, 3), (2, 1)].map(|(first_pos, rows)| DsaRowSpan { first_pos, rows });
        let mut kv = rig.kv();
        let hidden = rig.buf(6 * HIDDEN * 2);
        let mut refs: Vec<&mut (dyn LayerState + 'static)> =
            boxes.iter_mut().map(|b| b.as_mut()).collect();
        let from = rig.gpu.launch_count();
        layer
            .decode_spans_with(
                hidden,
                &mut refs,
                &spans,
                &mut kv,
                &meta,
                0,
                &rig.ctx(true),
                7,
                true,
                on,
            )
            .unwrap();
        let l = rig.since(from);
        if on {
            assert_eq!(counts(&l), (1, 1, 1, Some([1, 5, 1])));
            let store = &of(&l, STORE_ROWS)[0];
            assert_eq!(store.args[2], ptr(meta.positions));
            assert_eq!(store.args[6], ptr(meta.block_table));
            assert_eq!(of(&l, GEOM_ROWS)[0].args[0], ptr(meta.seq_len));
            assert_eq!(of(&l, SCORES_ROWS)[0].grid[2], 5, "grid z is the row");
            assert_eq!(of(&l, STORE)[0].args[2], ptr(meta.positions.offset(5 * 4)));
        } else {
            assert_eq!(counts(&l), (6, 6, 6, None));
        }
        let after: Vec<usize> = boxes.iter().map(|b| dsa(b.as_ref()).len()).collect();
        assert_eq!(after, vec![7, 12, 3], "on={on}");
    }
}

/// 2026-10-14: A captured 3-row verify of one paged sequence through `decode_k_with`. On: one
/// latent write for the 3 rows, then one launch per stage over the metadata's 3 rows and no
/// per-row store, geometry or selection. Off (and no `METRALE_GLM_DSA_VERIFY_ROWS`): three of
/// each. The state grows by 3 either way.
#[test]
fn a_one_sequence_verify_stores_and_selects_in_one_launch_per_stage() {
    for on in [true, false] {
        let rig = Rig::new();
        let layer = rig.layer();
        let meta = rig.meta(3);
        let mut st = paged(20);
        let mut kv = rig.kv();
        let hidden = rig.buf(3 * HIDDEN * 2);
        let mut ctx = rig.ctx(true);
        ctx.attn_metadata = Some(meta);
        let mut bt = vec![0u32; MB];
        let from = rig.gpu.launch_count();
        layer
            .decode_k_with(
                hidden,
                3,
                st.as_mut(),
                &mut kv,
                20,
                &mut bt,
                &ctx,
                7,
                false,
                true,
                on,
            )
            .unwrap();
        let l = rig.since(from);
        let latent = of(&l, LATENT);
        if on {
            assert_eq!(counts(&l), (0, 0, 0, Some([1, 3, 1])));
            assert_eq!((latent.len(), latent[0].grid[0]), (1, 3));
            assert_eq!(latent[0].args[3], ptr(meta.slot));
            assert_eq!(of(&l, STORE_ROWS)[0].args[2], ptr(meta.positions));
            assert_eq!(of(&l, GEOM_ROWS)[0].args[0], ptr(meta.seq_len));
        } else {
            assert_eq!(counts(&l), (3, 3, 3, None));
            assert_eq!(latent.len(), 3);
        }
        assert_eq!(dsa(st.as_ref()).len(), 23, "on={on}");
    }
}
