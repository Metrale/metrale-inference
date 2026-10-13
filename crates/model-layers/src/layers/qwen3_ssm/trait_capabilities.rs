// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-13: `LayerCapabilities` for [`Qwen3SsmLayer`]: graph eligibility, the
//! replay-tail split and the SSM-layer flag. Moved out of `trait_layer.rs` at its line
//! cap.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::Qwen3SsmLayer;
use crate::layer::LayerCapabilities;

impl LayerCapabilities for Qwen3SsmLayer {
    /// 2026-09-25: True when this layer carries PLE. The multi-seq decode
    /// (`forward_row`) hashes ids on the host inside the forward, which a graph
    /// capture refuses. The model ORs this into `decode_graph_veto`, which keeps
    /// decode eager.
    fn decode_graph_unsupported(&self) -> bool {
        self.ple.is_some()
    }

    /// 2026-10-13: The single-sequence decode hashes PLE ids in `decode_prestage`, before
    /// any capture or replay, when the token id is on the host; a fed row's id is only on
    /// the device, so that step runs eagerly.
    fn decode_graph_unsupported_at(&self, _seq_len: usize, host_token: bool) -> bool {
        self.ple.is_some() && !host_token
    }

    /// 2026-10-13: PLE's per-sequence prestage buffers are allocated per sequence, so a
    /// captured graph's addresses die with it.
    fn graph_stale_on_new_sequence(&self) -> bool {
        self.ple.is_some()
    }

    fn is_ssm_layer(&self) -> bool {
        self.is_ssm_layer_inner()
    }

    /// 2026-10-01: Both prefill entry paths (`prefill_inner`, `prefill_inner_hc`) run
    /// `prefill_block`, whose conv (`conv1d_prefill_capture`) and recurrence
    /// (`prefill_gdn_recurrence`) split at the capture point on every build, the
    /// recurrence's tail on the exact-replay arm.
    fn supports_replay_tail_split(&self) -> bool {
        true
    }
}
