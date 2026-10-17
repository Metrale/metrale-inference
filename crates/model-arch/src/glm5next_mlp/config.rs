// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GLM-5.3 MLP geometry of one rank: `Glm5NextMlpConfig` and its refusals.
//! 2026-10-09: Moved here whole from `mod.rs` (file size), then given the routed-expert layout.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: those of the module header in `mod.rs`.

use anyhow::{Context, Result, bail};
use metrale_config::{Glm5NextRouterMode, ModelConfig, TpSlice};

use super::expert_tp::{EXPERT_TP_UNIT, ExpertShard, expert_slice};
use super::{BF16_GEMM_K_ALIGN, KERNEL_MAX_TOP_K};

/// 2026-09-25: Which MLP a layer runs; the same variants as [`crate::glm5next_skeleton::Mlp`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm5NextMlpKind {
    Dense,
    RoutedMoe,
}

/// 2026-09-25: GLM MLP geometry for one rank, built by [`Self::from_config`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm5NextMlpConfig {
    pub hidden: usize,
    /// 2026-09-25: `intermediate_size / tp_world_size`: this rank's share of the dense FFN width.
    /// 2026-10-08: The length of this rank's `tp_split` of `intermediate_size`.
    pub local_dense_intermediate: usize,
    /// 2026-10-08: The first dense FFN column this rank owns.
    pub dense_start: usize,
    /// 2026-09-25: `moe_intermediate_size`, one routed expert's width. Not divided by TP: an
    /// expert is owned whole by one EP rank. 2026-10-09: Under `ExpertShard::Sliced`, the width
    /// this rank runs of every expert (`ExpertSlice::len`, padding included).
    pub moe_intermediate: usize,
    /// 2026-09-25: `shared_expert_intermediate_size / tp_world_size`: this rank's share of the
    /// shared expert. 2026-10-08: The length of this rank's `tp_split` of it.
    pub local_shared_intermediate: usize,
    /// 2026-10-08: The first shared-expert column this rank owns.
    pub shared_start: usize,
    /// 2026-09-25: The full routed-expert count, not this rank's share.
    pub num_experts: usize,
    /// 2026-09-25: `num_experts / ep_world_size`; this rank owns ids
    /// `[ep_rank * local_experts, (ep_rank + 1) * local_experts)`. 2026-10-09: `num_experts`
    /// under `ExpertShard::Sliced`, where `ep_world_size` is 1.
    pub local_experts: usize,
    pub ep_rank: usize,
    pub top_k: usize,
    /// 2026-09-25: `routed_scaling_factor`. Applied to the top-k weights, not to the shared
    /// expert.
    pub routed_scale: f32,
    /// 2026-09-25: `norm_topk_prob`: divide the top-k weights by their sum plus `1e-20`.
    pub renormalize: bool,
    /// 2026-09-25: The asymmetric SwiGLU clamp bound; see the module header.
    pub swiglu_limit: f32,
    /// 2026-09-25: True for `Glm5NextRouterMode::VllmBf16`: the router kernel then rounds the
    /// scores, the running sum and each weight to BF16, which can change which experts are
    /// selected. The parser yields `HfFp32` (false) when the checkpoint names no router dtype.
    pub router_bf16_ladder: bool,
    /// 2026-09-25: TP ranks the dense/shared FFN is split over. Above 1, the site output is a
    /// partial sum.
    pub tp_world_size: usize,
    /// 2026-09-25: EP ranks the routed experts are split over. Above 1, the routed sum is a
    /// partial sum.
    pub ep_world_size: usize,
    /// 2026-10-09: Whole experts over EP, or every expert sliced over TP
    /// (`--moe-expert-layout`); see `expert_tp`.
    pub expert_shard: ExpertShard,
}

impl Glm5NextMlpConfig {
    /// 2026-09-25: Divides the global widths by TP and the expert set by EP (a world size of 0
    /// counts as 1), then runs [`Self::validate`]. 2026-10-08: The widths split by
    /// `metrale_config::tp_split` in [`BF16_GEMM_K_ALIGN`] units, so they need not divide by
    /// `tp_world_size`; errors when a width is not a multiple of the unit or has fewer units
    /// than ranks, or when the expert count does not divide over EP. 2026-10-09: Under
    /// `--moe-expert-layout tp` every expert's width splits by `expert_tp::expert_slice` and
    /// this rank holds all of them; the layout's topology rule
    /// (`MoeExpertLayout::check_topology`) is checked first.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let tp = config.tp_world_size.max(1);
        let ep = config.ep_world_size.max(1);
        config.moe_expert_layout.check_topology(tp, ep)?;
        let expert_shard = match config.moe_expert_layout {
            metrale_config::MoeExpertLayout::Ep => ExpertShard::Whole,
            metrale_config::MoeExpertLayout::Tp => ExpertShard::Sliced(expert_slice(
                config.moe_intermediate_size,
                tp,
                config.tp_rank,
            )?),
        };
        let moe_intermediate = match expert_shard {
            ExpertShard::Whole => config.moe_intermediate_size,
            ExpertShard::Sliced(s) => s.len,
        };
        // 2026-10-09: The shared width splits in the FP8 dense tier's unit when it is on.
        let fp8_unit = crate::glm5next_fp8_dense::shared_split_unit(BF16_GEMM_K_ALIGN);
        let unit = |shared: bool| if shared { fp8_unit } else { BF16_GEMM_K_ALIGN };
        let split = |name: &str, total: usize, shared: bool| {
            metrale_config::tp_split(total, tp, config.tp_rank, unit(shared))
                .with_context(|| format!("GLM MLP: {name} {total} over tp_world_size {tp}"))
        };
        let dense = split("intermediate_size", config.intermediate_size, false)?;
        let shared = split(
            "shared_expert_intermediate_size",
            config.shared_expert_intermediate_size,
            true,
        )?;
        if !config.num_experts.is_multiple_of(ep) {
            bail!(
                "GLM MLP: num_experts {} does not divide over ep_world_size {ep}; a ragged \
                 expert split would leave some ids owned by nobody",
                config.num_experts
            );
        }
        let c = Self {
            hidden: config.hidden_size,
            local_dense_intermediate: dense.len,
            dense_start: dense.start,
            moe_intermediate,
            local_shared_intermediate: shared.len,
            shared_start: shared.start,
            num_experts: config.num_experts,
            local_experts: config.num_experts / ep,
            ep_rank: config.ep_rank,
            top_k: config.num_experts_per_tok,
            routed_scale: config.routed_scaling_factor as f32,
            renormalize: config.norm_topk_prob,
            swiglu_limit: config.swiglu_limit,
            router_bf16_ladder: matches!(config.glm5next_router_mode, Glm5NextRouterMode::VllmBf16),
            tp_world_size: tp,
            ep_world_size: ep,
            expert_shard,
        };
        c.validate()?;
        Ok(c)
    }

    /// 2026-10-08: This rank's columns of the dense FFN width.
    pub fn dense_slice(&self) -> TpSlice {
        TpSlice {
            start: self.dense_start,
            len: self.local_dense_intermediate,
        }
    }

    /// 2026-10-08: This rank's columns of the shared-expert width.
    pub fn shared_slice(&self) -> TpSlice {
        TpSlice {
            start: self.shared_start,
            len: self.local_shared_intermediate,
        }
    }

    /// 2026-09-25: The half-open global expert-id range this rank owns.
    pub fn local_expert_range(&self) -> std::ops::Range<usize> {
        let start = self.ep_rank * self.local_experts;
        start..start + self.local_experts
    }

    /// 2026-09-25: Global expert id to local slot, or `None` when another rank owns it. A
    /// remote id contributes zero; `experts` is indexed by this slot, never by the global id.
    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        let r = self.local_expert_range();
        r.contains(&global_id).then(|| global_id - r.start)
    }

    /// 2026-09-25: Whether the site output leaves this rank as a partial sum needing
    /// `all_reduce(SUM)`.
    pub fn needs_all_reduce(&self) -> bool {
        self.tp_world_size > 1 || self.ep_world_size > 1
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0 {
            bail!("GLM MLP: hidden_size is 0");
        }
        if self.swiglu_limit <= 0.0 {
            bail!(
                "GLM MLP: swiglu_limit is {}. GLM-5.3 clamps its SwiGLU and the clamp is \
                 asymmetric; a zero limit is not 'no clamp', it is a gate forced to <= 0. \
                 The glm5_next parser reads the real value (10.0) and refuses to default it.",
                self.swiglu_limit
            );
        }
        if self.top_k == 0 || self.top_k > KERNEL_MAX_TOP_K {
            bail!(
                "GLM MLP: num_experts_per_tok {} is outside the {}-slot bound \
                 glm5next_router_topk keeps in registers (`float best_w[16]`)",
                self.top_k,
                KERNEL_MAX_TOP_K
            );
        }
        if self.top_k > self.num_experts {
            bail!(
                "GLM MLP: top_k {} exceeds num_experts {}",
                self.top_k,
                self.num_experts
            );
        }
        if self.moe_intermediate == 0 {
            bail!("GLM MLP: moe_intermediate_size is 0 — a routed layer would compute nothing");
        }
        // 2026-10-08: Expert parallelism without tensor parallelism replicates the dense layers
        // and the shared expert on every rank, and the site's all-reduce then sums those
        // replicated outputs `ep_world_size` times: measured on the three-box GLM serve at EP=3,
        // TP=1, whose greedy output was word salad. EP runs only beside TP.
        if self.ep_world_size > 1 && self.tp_world_size == 1 {
            bail!(
                "GLM MLP: expert parallelism (ep {}) without tensor parallelism would sum the \
                 replicated dense and shared-expert outputs {} times in the MLP all-reduce; \
                 run --tp-size equal to --ep-size",
                self.ep_world_size,
                self.ep_world_size
            );
        }
        if let ExpertShard::Sliced(s) = self.expert_shard {
            // 2026-10-09: Every rank holds every expert, each `moe_intermediate` wide.
            if self.ep_world_size != 1
                || self.tp_world_size < 2
                || self.local_experts != self.num_experts
                || self.moe_intermediate != s.len
                || !s.len.is_multiple_of(EXPERT_TP_UNIT)
            {
                bail!(
                    "GLM MLP: the tp expert layout needs ep_world_size 1 (got {}), tp_world_size \
                     >= 2 (got {}), all {} experts local (got {}) and a {EXPERT_TP_UNIT}-aligned \
                     slice width equal to moe_intermediate (slice {}, moe_intermediate {})",
                    self.ep_world_size,
                    self.tp_world_size,
                    self.num_experts,
                    self.local_experts,
                    s.len,
                    self.moe_intermediate
                );
            }
        }
        if self.ep_rank >= self.ep_world_size {
            bail!(
                "GLM MLP: ep_rank {} is outside ep_world_size {}",
                self.ep_rank,
                self.ep_world_size
            );
        }
        Ok(())
    }
}
