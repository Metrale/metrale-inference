// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_PREFILL_PIECE_ROWS=P` (`glm5next_prefill_piece`): a prefill
//! sub-chunk of one sequence run as the calls of `plan_calls`. A one-piece call is `forward_k`.
//! A call of several pieces is `forward_k`'s body over all its rows, except that the mixer runs
//! once per piece (each at its own positions, as `forward_k` of that piece runs it) and the MLP
//! and the non-SP all-reduces keep a launch per piece (`row_pieces`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Every rank plans from `k`, `P`, the levers and the layer alone, so all ranks issue the same
//!   collectives.
//! - Pieces share a call only where the routed path is row-invariant (the grouped W4A16 GEMM, or
//!   the grouped W4A4 GEMM with the same tile for the piece and the call) and the SP-mHC choice
//!   is the same for the piece and the call ([`Glm5NextLayer::piece_share`]).

use super::*;
use crate::glm5next_mlp::forward::{
    ExpertRoute, expert_route, w4a4_grouped_tile_index, w4a4_takes_grouped,
};

/// 2026-10-14: The row-count-dependent decisions a call makes, which every piece of a shared
/// call must make the same: SP-mHC, and the routed path (with the grouped W4A4 tile).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::glm5next_layer) struct PieceSig {
    sp: bool,
    route: Option<(ExpertRoute, usize)>,
}

impl Glm5NextLayer {
    /// 2026-10-14: [`PieceSig`] of a `rows`-row prefill launch, or `None` when its routed path is
    /// not row-invariant, so pieces of that size never share a call.
    pub(in crate::glm5next_layer) fn piece_share(
        &self,
        ctx: &ForwardContext,
        rows: usize,
    ) -> Result<Option<PieceSig>> {
        let sp = self.sp_takes(ctx, rows, true)?;
        let route = match &self.mlp {
            Glm5NextMlpSite::Dense(_) => None,
            Glm5NextMlpSite::Moe(w) => {
                let r = expert_route(&self.mlp_kernels, &self.mlp_cfg, w, &self.mlp_ws, rows);
                match r {
                    ExpertRoute::Grouped => Some((r, 0)),
                    ExpertRoute::W4a4 if w4a4_takes_grouped(rows)? => Some((
                        r,
                        w4a4_grouped_tile_index(
                            rows * self.mlp_cfg.top_k,
                            self.mlp_cfg.num_experts,
                        ),
                    )),
                    _ => return Ok(None),
                }
            }
        };
        Ok(Some(PieceSig { sp, route }))
    }

    /// 2026-10-14: `k` rows of one prefill sub-chunk at positions from `seq_len`, highway slots
    /// from `slot_base`, in `P`-row pieces (module header). `chunk` is the layer dump's sub-chunk
    /// counter, advanced once per call.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::glm5next_layer) fn forward_k_pieces(
        &self,
        hidden: DevicePtr,
        k: usize,
        piece: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
        slot_base: usize,
        chunk: &mut usize,
        last_pos: Option<usize>,
    ) -> Result<()> {
        let Some(mhc) = self.mhc.as_ref() else {
            bail!("GLM layer {}: no hyper-connection bound", self.layer_idx);
        };
        let row_bytes = self.hidden * 2;
        let calls =
            crate::glm5next_prefill_piece::plan_calls(k, piece, |n| self.piece_share(ctx, n))?;
        let mut off = 0usize;
        for pieces in calls {
            let m: usize = pieces.iter().sum();
            let call = dump::Call {
                path: dump::Path::Prefill,
                chunk: *chunk,
                last_pos,
            };
            *chunk += 1;
            let x = hidden.offset(off * row_bytes);
            if let [n] = pieces[..] {
                self.forward_k(
                    x,
                    n,
                    state,
                    kv_cache,
                    seq_len + off,
                    block_table,
                    ctx,
                    stream,
                    false,
                    slot_base + off,
                    // 2026-10-14: A one-piece call IS a prefill sub-chunk (`is_prefill`).
                    true,
                    call,
                    &[],
                )?;
                off += n;
                continue;
            }
            let sp = self.sp_takes(ctx, m, true)?;
            let own = ctx
                .comm
                .filter(|_| sp)
                .map(|c| crate::glm5next_sp_mhc::shards(m, c.world_size())[c.rank()]);
            let first = seq_len + off;
            // 2026-10-14: The DSA selection output holds the last piece's rows only.
            let tap = dump::begin(ctx, stream, self.dump_layer(mhc, false), call, own, || {
                let mut r0 = 0usize;
                pieces
                    .iter()
                    .map(|&n| {
                        let w = dump::Window {
                            row0: r0,
                            rows: n,
                            seq: 0,
                            first_pos: first + r0,
                            last_pos,
                        };
                        r0 += n;
                        w
                    })
                    .collect()
            })?;
            let (st, kv, bt) = (&mut *state, &mut *kv_cache, &mut *block_table);
            let p = &pieces;
            let mixer = move |normed: DevicePtr| -> Result<DevicePtr> {
                let mut r0 = 0usize;
                for &n in p {
                    let dst = normed.offset(r0 * row_bytes);
                    let out = self.mix(
                        dst,
                        n,
                        st,
                        kv,
                        first + r0,
                        bt,
                        ctx,
                        stream,
                        false,
                        true,
                        &[],
                    )?;
                    // 2026-10-14: KDA leaves its output in `ws.final_out`, which the next piece
                    // reuses; DSA writes over its input.
                    if out.0 != dst.0 {
                        ctx.gpu.copy_d2d_async(out, dst, n * row_bytes, stream)?;
                    }
                    r0 += n;
                }
                Ok(normed)
            };
            if sp {
                self.forward_rows_sp(
                    mhc,
                    x,
                    m,
                    slot_base + off,
                    ctx,
                    stream,
                    &pieces,
                    tap.as_ref(),
                    mixer,
                )?;
            } else {
                self.forward_rows_with(
                    mhc,
                    x,
                    m,
                    slot_base + off,
                    ctx,
                    stream,
                    &pieces,
                    tap.as_ref(),
                    mixer,
                )?;
            }
            off += m;
        }
        Ok(())
    }
}
