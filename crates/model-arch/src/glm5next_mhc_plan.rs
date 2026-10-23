// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: An mHC site run from its plan groups instead of the layer's levers: the fused
//! pre-mix + norm entry and its grid, and the post-mix that sums the one-shot exchange's
//! partials (`glm_ar_fused_post`), as typed steps over plan kernel ids ([`MhcSteps`], built by
//! `glm5next_plan::mhc`), bound to this device's handles and launched through the launchers the
//! hand-written site uses (`glm5next_mhc_pre_launch`, `glm_hc_post_rank_sum_with`).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Running reads no lever: the entry, its grid and the post kernel are the plan's; the
//!   post-mix grid is the uncapped one (an installed plan refuses `METRALE_GLM_MHC_POST_SUM_CTAS`).
//! - The plan's exchange must take the communicator's one-shot path; a payload it refuses is an
//!   error, never the all-reduce the hand-written site would fall back to.
//! - One step per row count serves both sites of a layer: their groups name the same rule.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use parking_lot::RwLock;

use crate::glm5next_mhc::{Glm5NextMhcSiteWeights, mix_hc};

/// 2026-10-23: The fused pre-mix + norm: its entry and blocks per token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreStep<K> {
    pub kernel: K,
    pub grid_y: u32,
}

/// 2026-10-23: The post-mix that sums the exchanged partials.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PostStep<K> {
    pub kernel: K,
}

/// 2026-10-23: An mHC site's plan at `rows` rows, with the dims it was planned at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MhcSteps<K> {
    pub rows: usize,
    pub hidden: usize,
    pub hc_mult: usize,
    pub pre: PreStep<K>,
    pub post: PostStep<K>,
}

/// 2026-10-23: The blocks per token of a per-row pre-norm entry: `glm5next_hc_pre_norm_c1`
/// runs the per-row grid (`glm_hc_pre_norm_with`), one block per mix row.
pub fn c1_grid_y(hc_mult: usize) -> Result<u32> {
    u32::try_from(mix_hc(hc_mult)).context("mix rows overflow u32")
}

impl<K> MhcSteps<K> {
    /// 2026-10-23: Every kernel through `f`.
    pub fn map<L>(self, mut f: impl FnMut(K) -> Result<L>) -> Result<MhcSteps<L>> {
        Ok(MhcSteps {
            rows: self.rows,
            hidden: self.hidden,
            hc_mult: self.hc_mult,
            pre: PreStep {
                kernel: f(self.pre.kernel)?,
                grid_y: self.pre.grid_y,
            },
            post: PostStep {
                kernel: f(self.post.kernel)?,
            },
        })
    }
}

impl MhcSteps<KernelHandle> {
    /// 2026-10-23: The pre-mix + norm of `k` rows; `outs` is `[y, post, comb, normed]`, `eps`
    /// `[norm, hc, rms]`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_pre(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        w: &Glm5NextMhcSiteWeights,
        norm_w: DevicePtr,
        outs: [DevicePtr; 4],
        k: usize,
        sinkhorn_iters: usize,
        eps: [f32; 3],
        stream: u64,
    ) -> Result<()> {
        ensure!(
            k == self.rows,
            "GLM mHC plan: compiled for {} rows, run at {k}",
            self.rows
        );
        let dims = [
            self.hidden as u32,
            self.hc_mult as u32,
            sinkhorn_iters as u32,
        ];
        crate::glm5next_mhc_fused::pre_norm_checks(w, k as u32, dims[0], dims[1])?;
        crate::glm5next_mhc_pre_launch::launch(
            gpu,
            (self.pre.kernel, self.pre.grid_y),
            streams,
            w,
            norm_w,
            outs,
            k as u32,
            dims,
            eps,
            stream,
        )
    }

    /// 2026-10-23: The post-mix of `k` rows of `x` summed with the exchanged `partials`;
    /// `io` is `[residual, post, comb, streams_out]`.
    pub fn run_post(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        partials: &metrale_comm::PeerPartials,
        io: [DevicePtr; 4],
        k: usize,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            k == self.rows,
            "GLM mHC plan: compiled for {} rows, run at {k}",
            self.rows
        );
        crate::glm5next_mhc_fused::glm_hc_post_rank_sum_with(
            gpu,
            self.post.kernel,
            x,
            partials,
            io,
            [k as u32, self.hidden as u32, self.hc_mult as u32],
            None,
            stream,
        )
    }
}

/// 2026-10-23: The mHC plans one text layer runs, by row count, with their executed sites.
#[derive(Default)]
pub struct InstalledMhc {
    by_rows: RwLock<Vec<(MhcSteps<KernelHandle>, Vec<String>)>>,
}

impl std::fmt::Debug for InstalledMhc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstalledMhc")
            .field("sites", &self.sites().len())
            .finish()
    }
}

impl InstalledMhc {
    /// 2026-10-23: Replace the installed plans (empty clears).
    pub fn set(&self, plans: Vec<(MhcSteps<KernelHandle>, Vec<String>)>) {
        *self.by_rows.write() = plans;
    }

    /// 2026-10-23: The plan at `rows` rows.
    pub fn get(&self, rows: usize) -> Option<MhcSteps<KernelHandle>> {
        self.by_rows
            .read()
            .iter()
            .find(|(p, _)| p.rows == rows)
            .map(|(p, _)| *p)
    }

    /// 2026-10-23: The installed sites.
    pub fn sites(&self) -> Vec<String> {
        self.by_rows
            .read()
            .iter()
            .flat_map(|(_, s)| s.clone())
            .collect()
    }
}
