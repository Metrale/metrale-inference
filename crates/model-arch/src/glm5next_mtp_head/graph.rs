// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-11: The drafter forward as a CUDA graph (`METRALE_GLM_MTP_GRAPH=1`): one capture per
//! proposer state, replayed for every later draft from the target's hidden row.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - The graph runs `forward_body` with the block on its decode-step path: the DSA layer reads
//!   the row's position, KV slot, length and block table from this state's device metadata
//!   (`DrafterGraph`), places its indexer row on the device and plans its selection from the
//!   device geometry, so nothing a replay depends on is a host value frozen at capture.
//! - Before every capture or replay the host writes that metadata and copies the token's
//!   embedding row into `embed_in`, the fixed address the graph reads.
//! - Every rank captures and replays at the same propose: eligibility depends only on the
//!   lever, the state and the kernels, which the ranks share, so the graph's collectives pair.
//!
//! An eager propose launches ~60-100 kernels (norms, `eh_proj`, the DSA projections, indexer,
//! selection and attend, the MoE, the head) and stages the DSA row metadata from the host.

use super::*;
use metrale_gpu_runtime::gpu::GraphHandle;
use metrale_model_layers::layer::AttnMetadataDev;

/// 2026-10-11: The lever; read once.
pub(super) fn graph_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_MTP_GRAPH").as_deref() == Ok("1"))
}

/// 2026-10-11: Device bytes `DrafterGraph::alloc` takes per proposer state (0 with the lever
/// off), for the per-sequence reserve: the row metadata, the block table of a drafter sized
/// for `max_seq_len` rows (an upper bound: the drafter caps its rows at the DSA reservation),
/// and the embedding row.
pub fn state_bytes(hidden: usize, max_seq_len: usize) -> usize {
    if !graph_enabled() {
        return 0;
    }
    4 + 8 + 4 + (max_seq_len / 16 + 2) * 4 + hidden * 2
}

/// 2026-10-11: A proposer state's graph and the device metadata it reads.
pub(super) struct DrafterGraph {
    /// 2026-10-11: `[1]` u32 position, `[1]` i64 KV slot, `[1]` i32 length, `[blocks]` i32
    /// block table (the state's identity table, written once at allocation).
    positions: DevicePtr,
    slot: DevicePtr,
    seq_len: DevicePtr,
    block_table: DevicePtr,
    blocks: u32,
    /// 2026-10-11: `[hidden]` BF16: the token's embedding row, copied in before each launch.
    embed_in: DevicePtr,
    /// 2026-10-11: The captured graph and the target-hidden address it was captured with.
    graph: Option<(GraphHandle, DevicePtr)>,
}

impl DrafterGraph {
    /// 2026-10-11: Allocate the metadata and write the identity block table `0..blocks`.
    pub(super) fn alloc(gpu: &dyn GpuBackend, hidden: usize, blocks: usize) -> Result<Self> {
        let bt: Vec<u8> = (0..blocks as i32).flat_map(i32::to_le_bytes).collect();
        let g = Self {
            positions: gpu.alloc(4)?,
            slot: gpu.alloc(8)?,
            seq_len: gpu.alloc(4)?,
            block_table: gpu.alloc(bt.len().max(4))?,
            blocks: blocks as u32,
            embed_in: gpu.alloc(hidden * 2)?,
            graph: None,
        };
        gpu.copy_h2d(&bt, g.block_table)?;
        Ok(g)
    }

    /// 2026-10-11: Destroy the graph and release every buffer.
    pub(super) fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some((g, _)) = self.graph.take() {
            gpu.destroy_graph(g)?;
        }
        for p in [
            self.positions,
            self.slot,
            self.seq_len,
            self.block_table,
            self.embed_in,
        ] {
            gpu.free(p)?;
        }
        Ok(())
    }

    /// 2026-10-11: The row metadata the block's decode step reads: one row at `positions`,
    /// written per draft by [`Self::stage`].
    fn meta(&self) -> AttnMetadataDev {
        AttnMetadataDev {
            positions: self.positions,
            positions_h: self.positions,
            positions_w: self.positions,
            slot: self.slot,
            seq_len: self.seq_len,
            block_table: self.block_table,
            max_blocks_per_seq: self.blocks,
            num_seqs: 1,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        }
    }

    /// 2026-10-11: Stream-ordered writes for the draft at row `row` (the state's `seq_len`): its
    /// position and KV slot are the row (`rows_impl`: the row space is dense, slot == key ==
    /// RoPE position, the block table the identity), its length `row + 1`; and the embedding row.
    fn stage(
        &self,
        gpu: &dyn GpuBackend,
        row: usize,
        embed_row: DevicePtr,
        hidden: usize,
        stream: u64,
    ) -> Result<()> {
        let w = staged_words(row);
        gpu.copy_h2d_async(&w.position.to_le_bytes(), self.positions, stream)?;
        gpu.copy_h2d_async(&w.slot.to_le_bytes(), self.slot, stream)?;
        gpu.copy_h2d_async(&w.seq_len.to_le_bytes(), self.seq_len, stream)?;
        gpu.copy_d2d_async(embed_row, self.embed_in, hidden * 2, stream)
    }
}

/// 2026-10-11: The metadata words of the draft at dense row `row`.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct StagedWords {
    pub position: u32,
    pub slot: i64,
    pub seq_len: i32,
}

pub(super) fn staged_words(row: usize) -> StagedWords {
    StagedWords {
        position: row as u32,
        slot: row as i64,
        seq_len: row as i32 + 1,
    }
}

/// 2026-10-11: Whether this draft runs through the graph: the first draft of a propose (later
/// drafts read the drafter's own output, another input address), a state that allocated the
/// metadata (the lever), no timing arm, and a DSA block whose rows can be placed and planned on
/// the device.
pub(super) fn eligible(head: &Glm5NextMtpHead, st: &Glm5NextMtpProposerState, first: bool) -> bool {
    first
        && st.graph.is_some()
        && !skip_block()
        && !skip_head()
        && match &head.module.layer.mixer {
            crate::glm5next_layer::Glm5NextMixer::Dsa(l) => {
                l.select_kernels.indexer_store.0 != 0 && l.select_kernels.write_geom.0 != 0
            }
            _ => false,
        }
}

impl Glm5NextMtpHead {
    /// 2026-10-11: [`Self::forward_body`] through the state's graph: stage the metadata and the
    /// embedding row, then replay the graph, or capture it on first use (and after the target
    /// hidden moved). A replay runs only kernels, so the host bookkeeping a capture run did is
    /// redone here: the DSA indexer length (`sync_replayed_step`) and `seq_len`.
    pub(super) fn forward_graphed(
        &self,
        embed_row: DevicePtr,
        hidden_in: DevicePtr,
        st: &mut Glm5NextMtpProposerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let layer = &self.module.layer;
        let (meta, embed_in, cached) = {
            let g = st
                .graph
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("GLM MTP graph: state has no graph metadata"))?;
            g.stage(gpu, st.seq_len, embed_row, self.hidden, stream)?;
            let cached = match g.graph {
                Some((graph, h)) if h == hidden_in => Some(graph),
                Some((graph, _)) => {
                    gpu.destroy_graph(graph)?;
                    g.graph = None;
                    None
                }
                None => None,
            };
            (g.meta(), g.embed_in, cached)
        };
        if let Some(graph) = cached {
            use metrale_model_layers::layer::LayerGraphHooks;
            layer.check_replay_room(&st.dsa, st.seq_len, 1)?;
            gpu.launch_graph(graph, stream)?;
            layer.sync_replayed_step(&mut st.dsa, st.seq_len, 1)?;
            st.seq_len += 1;
            return Ok(());
        }
        let gctx = ForwardContext {
            buffers: ctx.buffers,
            hc_row_offset: ctx.hc_row_offset,
            gpu: ctx.gpu,
            config: ctx.config,
            dispatch: ctx.dispatch,
            derived: ctx.derived,
            levers: ctx.levers,
            stats: ctx.stats,
            attn_metadata: Some(meta),
            profile: false,
            comm: ctx.comm,
            graph_capture: true,
            decode_step: true,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: ctx.moe_lora_route,
        };
        gpu.begin_capture(stream)?;
        let body = self.forward_body(embed_in, hidden_in, st, &gctx, stream);
        let graph = gpu.end_capture(stream);
        if let (Err(e), Ok(g)) = (&body, &graph) {
            if g.0 != 0 {
                gpu.destroy_graph(*g)?;
            }
            anyhow::bail!("GLM MTP graph: capture failed: {e:#}");
        }
        let graph = graph?;
        if graph.0 == 0 {
            anyhow::bail!("GLM MTP graph: the capture returned no graph");
        }
        gpu.launch_graph(graph, stream)?;
        if let Some(g) = st.graph.as_mut() {
            g.graph = Some((graph, hidden_in));
        }
        tracing::info!("GLM MTP: captured the drafter graph");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-10-11: The metadata of dense row `r`: position and KV slot `r` (the identity block
    /// table maps slot `r` to block `r / 16`, offset `r % 16`), length `r + 1`, as the host path
    /// computes them (`row_slots`, `stage_host_rows`) for the drafter's row space.
    #[test]
    fn a_draft_row_stages_its_own_position_slot_and_length() {
        for r in [0usize, 1, 15, 16, 75, 8191] {
            assert_eq!(
                staged_words(r),
                StagedWords {
                    position: r as u32,
                    slot: r as i64,
                    seq_len: r as i32 + 1,
                }
            );
        }
    }
}
