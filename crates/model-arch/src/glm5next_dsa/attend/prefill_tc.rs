// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-14: The warp-specialized prefill twin of the P-hi DSA attend
//! (`glm5next_dsa_mla_prefill_tc_phi`, lever `METRALE_GLM_DSA_HB_PREFILL_TC`): its launch shape
//! and the swap that selects it.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: the tc entry is only ever swapped in over the P-hi entry at one split, never over
//! the pipe entry; every refusal is an error, never a silent fallback to another entry.

use anyhow::{Result, bail};

use super::{Glm5NextDsaDecodeKernel, HeadBatched};

/// 2026-10-14: Threads per block of the tc entry: eight consumer warps (QK, softmax, PV) and four
/// producer warps (gather + widen), its `TC_THREADS`.
pub const TC_BLOCK: u32 = 384;

/// 2026-10-14: The widest selection row the tc entry takes: its per-row offset table holds this
/// many slots (`TC_MAX_WIDTH`); a wider row traps in the kernel and is refused here first.
pub const TC_MAX_WIDTH: usize = 2064;

/// 2026-10-14: Dynamic shared memory of the tc entry: two bf16 tiles (32 tokens x 512 dims), four
/// f32 QK chain planes (24 heads x 36) and the offset table, its `TC_SMEM_BYTES`. The kernel
/// traps when launched with less, so a drift between the two fails loudly.
pub const TC_SMEM_BYTES: u32 = 2 * (32 * 512 * 2) + 4 * (24 * 36 * 4) + TC_MAX_WIDTH as u32 * 4;

impl Glm5NextDsaDecodeKernel {
    /// 2026-10-14: This kernel with the P-hi entry replaced by its warp-specialized twin when
    /// `tc` ([`super::hb_prefill_tc`]) is set; unchanged when it is not. An error when
    /// it is set but the entry is not the P-hi one ([`Self::for_prefill_p_hi`] first; the pipe
    /// entry is another twin of it, so the two levers are exclusive), the split count is not 1,
    /// or the twin is not in the build.
    pub fn for_prefill_tc(self, tc: bool) -> Result<Self> {
        if !tc {
            return Ok(self);
        }
        let Some(h) = self.head_batched else {
            bail!("METRALE_GLM_DSA_HB_PREFILL_TC=1 needs METRALE_GLM_DSA_DECODE_HB");
        };
        if h.decode_p_hi.0 == 0 || h.decode.0 != h.decode_p_hi.0 {
            bail!(
                "METRALE_GLM_DSA_HB_PREFILL_TC=1 needs METRALE_GLM_DSA_HB_PREFILL_P_HI=1 and \
                 excludes METRALE_GLM_DSA_HB_PREFILL_PIPE=1"
            );
        }
        if h.splits != 1 {
            bail!(
                "METRALE_GLM_DSA_HB_PREFILL_TC=1 runs at one split, got {} (set \
                 METRALE_GLM_DSA_HB_PREFILL_ONE_SPLIT=1)",
                h.splits
            );
        }
        if h.prefill_tc.0 == 0 {
            bail!(
                "METRALE_GLM_DSA_HB_PREFILL_TC=1: glm5next_dsa_mla_prefill_tc_phi is not in this \
                 target's kernels"
            );
        }
        Ok(Self {
            head_batched: Some(HeadBatched {
                decode: h.prefill_tc,
                smem: TC_SMEM_BYTES,
                block: TC_BLOCK,
                offset_table: true,
                ..h
            }),
            ..self
        })
    }
}
