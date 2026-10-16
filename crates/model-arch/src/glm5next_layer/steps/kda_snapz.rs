// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-16: SNAPZ (`METRALE_GLM_KDA_SNAPZ=1`) on the GLM layer: the binding of the model's
//! flag words (`kda_snapz_bind`) and, per batched verify, each sequence's deferred-commit
//! source ([`KdaDefer`]): its flag word and its first snapshot slot. When a verdict keeps only
//! a sequence's first row, the model skips the commit's copy of that slot into the live state
//! and sets the flag for the sequence's position in the next batched verify, whose KDA launches
//! then read the state from the slot (`kda_seqs_tokens.cu`, `kda_seqs_src`; a group without
//! the seqs kernels resolves it first, `snapz_resolve`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - A position gets a defer only when it is below `VERIFY_WY_TABLE_SEQS` and its first
//!   snapshot pointers are its own (no other position of the pass shares them); the model sets
//!   a flag only for such a position (model-engine `kda_snapz`, the same rule).
//! - Unbound (the lever off), no position gets a defer, so every launch reads the live state.

use super::*;
use crate::glm5next_kda::seqs_tokens::KdaDefer;
use metrale_model_layers::layer::{KdaSnapzBinding, VERIFY_WY_TABLE_SEQS};

/// 2026-10-16: The defer of each position from its first snapshot pair (`None`: the pool keeps
/// no snapshot for it): the flag word at `flags + 4 * i`, then `None` for positions past the
/// flag table and for every position whose recurrent or conv snapshot another position shares
/// (a padded batch's dummy slot).
pub(in crate::glm5next_layer) fn defers_from(
    flags: DevicePtr,
    firsts: &[Option<(DevicePtr, DevicePtr)>],
) -> Vec<Option<KdaDefer>> {
    let shared = |i: usize, p: (DevicePtr, DevicePtr)| {
        firsts
            .iter()
            .enumerate()
            .any(|(j, q)| j != i && q.is_some_and(|q| q.0 == p.0 || q.1 == p.1))
    };
    firsts
        .iter()
        .enumerate()
        .map(|(i, f)| match *f {
            Some(p) if i < VERIFY_WY_TABLE_SEQS && !shared(i, p) => Some(KdaDefer {
                flag: flags.offset(4 * i),
                h: p.0,
                conv: p.1,
            }),
            _ => None,
        })
        .collect()
}

impl Glm5NextLayer {
    /// 2026-10-16: `LayerWriteOnAccept::kda_snapz_bind`: a KDA layer with the resolve kernel
    /// whose state sizes are the pool's binds (again with the same binding is a no-op); a DSA
    /// layer, a layer without the kernel, a size mismatch or a different binding refuses.
    pub(in crate::glm5next_layer) fn snapz_bind(&self, b: KdaSnapzBinding) -> bool {
        let Glm5NextMixer::Kda { layer, cfg, .. } = &self.mixer else {
            return false;
        };
        if b.flags.0 == 0
            || layer.kernels.seq.snapz_resolve.0 == 0
            || cfg.recurrent_state_elems() * 4 != b.h_bytes
            || cfg.conv_state_elems() * 4 != b.conv_bytes
        {
            return false;
        }
        self.snapz.set(b).is_ok() || self.snapz.get() == Some(&b)
    }

    /// 2026-10-16: Each sequence's defer for a batched verify over `states` (empty when
    /// unbound): see [`defers_from`].
    pub(in crate::glm5next_layer) fn snapz_defers(
        &self,
        states: &mut [&mut (dyn LayerState + 'static)],
    ) -> Result<Vec<Option<KdaDefer>>> {
        let Some(b) = self.snapz.get() else {
            return Ok(Vec::new());
        };
        let firsts = states
            .iter_mut()
            .map(|s| {
                let st = self.kda_state(&mut **s)?;
                Ok(st
                    .h_state_intermediates
                    .first()
                    .zip(st.conv_state_intermediates.first())
                    .map(|(&h, &c)| (h, c)))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(defers_from(b.flags, &firsts))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(i: u64) -> Option<(DevicePtr, DevicePtr)> {
        Some((DevicePtr(0x1000 + i), DevicePtr(0x2000 + i)))
    }

    /// 2026-10-16: Distinct snapshots each get their own flag word and slots, in order; a
    /// position without a snapshot gets none and keeps its flag offset for the next.
    #[test]
    fn each_own_snapshot_gets_its_positions_flag() {
        let f = DevicePtr(0x9000);
        let d = defers_from(f, &[pair(0), None, pair(2)]);
        let want = |i: usize, p: u64| {
            Some(KdaDefer {
                flag: DevicePtr(0x9000 + 4 * i as u64),
                h: DevicePtr(0x1000 + p),
                conv: DevicePtr(0x2000 + p),
            })
        };
        assert_eq!(d, vec![want(0, 0), None, want(2, 2)]);
    }

    /// 2026-10-16: Positions sharing either snapshot pointer (the dummy slot of a padded batch)
    /// get none, while an unshared one beside them keeps its defer; so does none past the flag
    /// table. The control: the same pass with the sharing removed defers every position.
    #[test]
    fn shared_snapshots_and_positions_past_the_table_get_none() {
        let f = DevicePtr(0x9000);
        let h_shared = Some((DevicePtr(0x1000), DevicePtr(0x2005)));
        let d = defers_from(f, &[pair(0), pair(1), h_shared]);
        assert_eq!(
            d.iter().map(Option::is_some).collect::<Vec<_>>(),
            [false, true, false]
        );
        let conv_shared = Some((DevicePtr(0x1005), DevicePtr(0x2001)));
        let d = defers_from(f, &[pair(0), pair(1), conv_shared]);
        assert_eq!(
            d.iter().map(Option::is_some).collect::<Vec<_>>(),
            [true, false, false]
        );
        let d = defers_from(f, &[pair(0), pair(1), pair(5)]);
        assert!(d.iter().all(Option::is_some));
        let many: Vec<_> = (0..VERIFY_WY_TABLE_SEQS as u64 + 1).map(pair).collect();
        let d = defers_from(f, &many);
        assert!(d[VERIFY_WY_TABLE_SEQS - 1].is_some());
        assert!(d[VERIFY_WY_TABLE_SEQS].is_none());
    }
}
