// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: `METRALE_GLM_MTP_STATE_POOL=1` (`ModelLevers::glm_mtp_state_pool`): a freed
//! drafter state keeps its device buffers, KV region and drafter graph in this pool, and the
//! next sequence takes it instead of allocating. A flat DSA indexer cache is two
//! `[capacity, index_head_dim]` BF16 buffers; allocating them cost ~1.7 ms of `cuMemAlloc` per
//! request ahead of the prefill (nsys, cold c256/c1k).
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants:
//! - A pooled state still holds its KV region, so live plus pooled states never exceed the
//!   region count, and a region is never in two states.
//! - A taken state starts at zero rows (`seq_len`, `last_drafted`, the DSA counter), as a fresh
//!   one does; the rows past zero are unreachable until rewritten, as after a rewind.

use super::*;

/// 2026-10-14: The idle drafter states, most recently freed first.
pub(super) struct StatePool {
    idle: Mutex<Vec<Glm5NextMtpProposerState>>,
}

impl StatePool {
    pub(super) fn new() -> Self {
        Self {
            idle: Mutex::new(Vec::new()),
        }
    }

    /// 2026-10-14: An idle state reset to zero rows, or `None` when the pool is empty.
    pub(super) fn take(&self) -> Result<Option<Glm5NextMtpProposerState>> {
        let Some(mut st) = self.idle.lock().pop() else {
            return Ok(None);
        };
        st.seq_len = 0;
        st.last_drafted = 0;
        st.dsa.rewind_to(0)?;
        st.released = false;
        Ok(Some(st))
    }

    /// 2026-10-14: Move `st`'s buffers, region and graph into the pool. `st` is left released
    /// and empty (a placeholder DSA state that owns nothing, no region, no graph), so a second
    /// `free_state` on it is a no-op and cannot release the region twice.
    pub(super) fn keep(
        &self,
        cfg: &crate::glm5next_dsa::Glm5NextDsaConfig,
        st: &mut Glm5NextMtpProposerState,
    ) -> Result<()> {
        let kept = Glm5NextMtpProposerState {
            dsa: std::mem::replace(&mut st.dsa, Glm5NextDsaState::paged(cfg)?),
            seq_len: 0,
            block_table: std::mem::take(&mut st.block_table),
            last_drafted: 0,
            concat: st.concat,
            x: st.x,
            logits: st.logits,
            arg: st.arg,
            head_xchg: st.head_xchg,
            released: true,
            kv_region: st.kv_region.take(),
            graph: st.graph.take(),
        };
        st.released = true;
        st.seq_len = 0;
        self.idle.lock().push(kept);
        Ok(())
    }
}

#[cfg(test)]
#[path = "state_pool_tests.rs"]
mod tests;
