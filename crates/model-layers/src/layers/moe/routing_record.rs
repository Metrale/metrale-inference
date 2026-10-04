// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-03: The expert-load recorder behind `met serve --record-routing`: per MoE layer, how
//! many routed rows each expert received in prefill. Its snapshot is the routing profile a mock
//! checkpoint reproduces (`metrale_ml_utils::routing`), and the same recorder on a mock serve
//! measures whether it does.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Unarmed (the default), [`record`] returns after one atomic load: no synchronize, no copy.
//! - Never under graph capture: a synchronize inside a capture invalidates it.
//! - Layers are keyed by the `MoeLayer` they belong to and listed in first-touch order, which is
//!   layer order for the main stack (the first prefill runs every layer in order), followed by a
//!   draft head's layer.
//! - It changes no computation: the counts are read from `expert_offsets` after the kernel that
//!   wrote them.

use std::sync::{Mutex, OnceLock};

use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-10-03: The counts so far, per layer key in first-touch order.
#[derive(Debug, Default)]
pub struct RoutingRecorder {
    layers: Mutex<Vec<(usize, Vec<u64>)>>,
}

static RECORDER: OnceLock<RoutingRecorder> = OnceLock::new();

/// 2026-10-03: Arm the recorder for this process (idempotent).
pub fn arm() {
    let _ = RECORDER.get_or_init(RoutingRecorder::default);
}

/// 2026-10-03: The counts recorded so far, one row per layer in first-touch order; `None` when
/// the recorder is not armed.
pub fn snapshot() -> Option<Vec<Vec<u64>>> {
    let r = RECORDER.get()?;
    let layers = r.layers.lock().unwrap_or_else(|p| p.into_inner());
    Some(layers.iter().map(|(_, c)| c.clone()).collect())
}

impl RoutingRecorder {
    /// 2026-10-03: Add `counts` to layer `key`.
    pub fn add(&self, key: usize, counts: &[u64]) {
        let mut layers = self.layers.lock().unwrap_or_else(|p| p.into_inner());
        match layers.iter_mut().find(|(k, _)| *k == key) {
            Some((_, c)) if c.len() == counts.len() => {
                for (a, b) in c.iter_mut().zip(counts) {
                    *a += b;
                }
            }
            Some((_, c)) => tracing::warn!(
                "routing record: layer {key:#x} changed expert count {} -> {}; ignored",
                c.len(),
                counts.len()
            ),
            None => layers.push((key, counts.to_vec())),
        }
    }
}

/// 2026-10-03: Per-expert counts from `ne + 1` cumulative u32 offsets.
pub fn counts_from_offsets(raw: &[u8]) -> Vec<u64> {
    let eo: Vec<u32> = raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    eo.windows(2)
        .map(|w| u64::from(w[1].saturating_sub(w[0])))
        .collect()
}

/// 2026-10-03: Record one prefill MoE step of layer `key` from its `expert_offsets` (`ne + 1`
/// u32 on the device, written on `stream`).
pub(super) fn record(
    ctx: &crate::layer::ForwardContext<'_>,
    key: usize,
    expert_offsets: DevicePtr,
    ne: usize,
    stream: u64,
) {
    let Some(r) = RECORDER.get() else {
        return;
    };
    let gpu = ctx.gpu;
    if ctx.graph_capture || gpu.stream_is_capturing(stream) {
        return;
    }
    let mut raw = vec![0u8; (ne + 1) * 4];
    if let Err(e) = gpu
        .synchronize(stream)
        .and_then(|()| gpu.copy_d2h(expert_offsets, &mut raw))
    {
        tracing::warn!("routing record: reading expert offsets failed: {e:#}");
        return;
    }
    r.add(key, &counts_from_offsets(&raw));
}

#[cfg(test)]
#[path = "routing_record_tests.rs"]
mod routing_record_tests;
