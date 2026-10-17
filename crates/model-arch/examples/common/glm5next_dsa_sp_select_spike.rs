// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-17: The SP select gate's spike leg (`glm5next_dsa_sp_select_gate spike`): the
//! 8,192-row chunks of a 1,048,576-token prefill that end at 196,608 * m tokens (m = 1..=5), on
//! the scratch the 1M serve's workspace allocates, from a random flat cache of `CEILING_1M`
//! keys. Before 2026-10-17 a rank stepped each slab in `per` rows, a 1,366- or 1,365-row slab
//! ended in a 1-row sub-pass there, `sp_select::run::plan` refused the split and every rank
//! selected the whole chunk.
//!
//! Per chunk: the stepped slab plan refuses and the even one (`slab_passes`) splits on this
//! scratch (both required), then `shape`'s byte checks without timing: each rank's slabs in
//! their even sub-passes write only its rows, the rows put together equal the unsplit pass's
//! (0 rows differ), and the shifted-input control differs.
//!
//! Owner: model-arch examples.
//! Invariants: errors unless every plan decision, byte check and control holds. Under
//! `METRALE_GLM_DSA_SELECT_FUSED=1` a pass has no sub-passes (`per` covers the chunk), so the
//! plans agree and only the bytes are checked.

use super::*;
use metrale_model_arch::glm5next_dsa::select::row_passes;
use metrale_model_arch::glm5next_dsa::sp_select::run::{SlabPlan, plan_with};

/// 2026-10-17: The 1M serve's `--max-seq-len`, the cache the leg reads, and its prefill chunk.
const CEILING_1M: usize = 1 << 20;
const CHUNK: usize = 8192;
/// 2026-10-17: The first chunk end the stepped slab plan refused on 3 ranks.
const SPIKE: usize = 196_608;

pub fn spike_gate(gpu: &dyn GpuBackend, kern: &Glm5NextDsaKernels) -> Result<()> {
    let c = cfg(CEILING_1M);
    let inp = inputs(gpu, &c, CEILING_1M)?;
    let scratch = serve_scratch(gpu, &c)?;
    let t = MinPoolsTable::new(vec![(CHUNK, 1)])?;
    let stepped: SlabPlan = |slab, per| row_passes(&[slab], per);
    for m in 1..=5 {
        let seq = SPIKE * m;
        let pools = seq / c.index_kpool;
        let per = scratch.rows_per_pass(&DsaSelectGeometry::plan(&c, seq, CHUNK)?);
        let old = plan_with(stepped, Some(&t), WORLD, 0, CHUNK, pools, true, per)?;
        let new = run::plan(Some(&t), WORLD, 0, CHUNK, pools, true, per)?;
        let word = |d: &Option<_>| if d.is_some() { "split" } else { "unsplit" };
        println!(
            "spike seq={seq} per={per}: stepped slab plan {}, even slab plan {}",
            word(&old),
            word(&new)
        );
        ensure!(
            new.is_some(),
            "seq {seq}: the even slab plan keeps the pass unsplit"
        );
        ensure!(
            old.is_none() || per >= CHUNK,
            "seq {seq}: the stepped slab plan splits, so this chunk is not a spike point"
        );
        shape(gpu, kern, &c, &inp, &scratch, None, CHUNK, pools, false)?;
    }
    Ok(())
}
