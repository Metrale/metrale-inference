// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: `ModelCircuit::state_digest` for `TransformerModel`: the parity instruments'
//! read of a sequence's state (LIFECYCLE-DESIGN.md 7.1 and 15.4: "byte-identical final state").
//!
//! Owner: model-engine.
//! Invariants:
//! - The device is synchronized first, so the digest covers every queued write.
//! - Entries are in layer order: `ssm<i>.h`, `ssm<i>.conv` per recurrent layer, `kv<i>` per
//!   attention layer (K then V of each block, in block-table order).

use anyhow::Result;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;
use crate::traits::model::circuit::{FNV_START, fnv1a};

impl TransformerModel {
    pub(super) fn state_digest_impl(&self, seq: &SequenceState) -> Result<Vec<(String, u64)>> {
        let gpu = self.gpu.as_ref();
        gpu.synchronize(gpu.default_stream())?;
        let mut out = Vec::new();
        let pool = &self.ssm_pool;
        for i in 0..pool.num_ssm_layers {
            let mut h = vec![0u8; pool.h_stored_bytes];
            gpu.copy_d2h(pool.h_state(i, seq.slot_idx), &mut h)?;
            out.push((format!("ssm{i}.h"), fnv1a(FNV_START, &h)));
            let mut c = vec![0u8; pool.conv_bytes];
            gpu.copy_d2h(pool.conv_state(i, seq.slot_idx), &mut c)?;
            out.push((format!("ssm{i}.conv"), fnv1a(FNV_START, &c)));
        }
        let cache = self.kv_cache.lock();
        for l in 0..cache.num_layers() {
            let mut acc = FNV_START;
            for &b in &seq.block_table {
                let (k, v) = cache.read_block(l, b, gpu)?;
                acc = fnv1a(fnv1a(acc, &k), &v);
            }
            out.push((format!("kv{l}"), acc));
        }
        Ok(out)
    }
}
