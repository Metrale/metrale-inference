// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-21: `METRALE_GLM_MOE_SCALE_DICT` at load: each routed layer's packing decision
//! (`glm5next_mlp::scale_dict::plan`, made once from the lever, the expert layout, the layer's
//! precision plan and the `_cs` twins) and the [`ScalePack`] the `tp` binder packs through
//! (`expert_tp::bind_routed_expert`), with its counts for the load log.
//!
//! Owner: model-arch weight loader.
//! Invariants: a layer packs only under `ScaleDictPlan::Compact`; the binder of any other layer
//! (and of the MTP layer) leaves every scale plain.

use std::cell::Cell;
use std::sync::Once;
use std::time::{Duration, Instant};

use anyhow::Result;

use super::expert_tp::HostProj;
use crate::glm5next_mlp::forward::MlpLevers;
use crate::glm5next_mlp::precision::GroupPrecision;
use crate::glm5next_mlp::scale_dict::{self, ScaleDictPlan};
use crate::glm5next_mlp::weights::ScaleLayout;
use crate::glm5next_mlp::{ExpertShard, Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-21: One routed layer's block-scale packing (`glm5next_mlp::scale_dict`): whether the
/// layer packs (its `scale_dict::plan`), and for the load log the tensors packed, the tensors
/// kept plain (codes spanning more than 32) and the host time the packing took.
pub(super) struct ScalePack {
    plan: ScaleDictPlan,
    packed: Cell<usize>,
    plain: Cell<usize>,
    time: Cell<Duration>,
}

impl ScalePack {
    pub(super) fn new(plan: ScaleDictPlan) -> Self {
        Self {
            plan,
            packed: Cell::new(0),
            plain: Cell::new(0),
            time: Cell::new(Duration::ZERO),
        }
    }

    /// 2026-10-21: Whether the layer packs.
    pub(super) fn on(&self) -> bool {
        self.plan == ScaleDictPlan::Compact
    }

    /// 2026-10-21: Pack `h`'s scales (`groups` per row) when the layer packs; their layout.
    pub(super) fn apply(&self, h: &mut HostProj, groups: usize) -> ScaleLayout {
        if !self.on() {
            return ScaleLayout::Plain;
        }
        let t = Instant::now();
        let scales = std::mem::take(&mut h.1);
        let rows = scales.len() / groups.max(1);
        let (bytes, layout) = scale_dict::pack(scales, rows, groups);
        h.1 = bytes;
        let n = if layout == ScaleLayout::Plain {
            &self.plain
        } else {
            &self.packed
        };
        n.set(n.get() + 1);
        self.time.set(self.time.get() + t.elapsed());
        layout
    }

    /// 2026-10-21: `(packed, plain, host time)` so far.
    pub(super) fn counts(&self) -> (usize, usize, Duration) {
        (self.packed.get(), self.plain.get(), self.time.get())
    }
}

/// 2026-10-21: Layer packing of a routed layer under `levers`. The precision plan is resolved
/// with `has_scales = true` before the experts exist: its kernel per row count does not depend on
/// it, and a plan that fails without the scales fails again after binding (`build_moe`).
pub(super) fn of(
    levers: &MlpLevers,
    k: &Glm5NextMlpKernels,
    cfg: &Glm5NextMlpConfig,
    precision: &dyn Fn(bool) -> Result<GroupPrecision>,
) -> Result<ScalePack> {
    let sliced = matches!(cfg.expert_shard, ExpertShard::Sliced(_));
    let every_row = levers.scale_dict && sliced && precision(true)?.w4a4_at_every_row();
    let plan = scale_dict::plan(levers.scale_dict, sliced, every_row, k.scale_dict.ready());
    Ok(ScalePack::new(plan))
}

/// 2026-10-21: The load log: a packing layer's counts and host time, or once the reason a load
/// keeps plain scales.
pub(super) fn announce(layer: usize, p: &ScalePack) {
    static PLAIN: Once = Once::new();
    let (packed, plain, t) = p.counts();
    if p.on() {
        tracing::info!(
            "glm5_next layer {layer}: METRALE_GLM_MOE_SCALE_DICT packed {packed} of {} routed \
             expert tensors ({plain} keep plain scales: codes span more than 32) in {:.3} s",
            packed + plain,
            t.as_secs_f64()
        );
    } else {
        PLAIN.call_once(|| tracing::info!("METRALE_GLM_MOE_SCALE_DICT: {}", p.plan.describe()));
    }
}
