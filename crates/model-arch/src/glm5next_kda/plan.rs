// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: A KDA mixer run from its plan groups instead of the layer's levers: the pack, the
//! gate chain, the conv (one row, or a snapshotting verify's window advances, token conv and
//! window advance), the recurrence and (one row) the output norm, as typed steps over plan kernel
//! ids ([`KdaSteps`], built by `glm5next_plan::kda`), bound to this device's handles and launched
//! through the launchers the hand-written path uses. The W4A16 projections (the six-projection
//! group, `o`, and at two rows the output norm with `o`) are the plan's `glm_w4a16` sites: they
//! stay the layer's own code.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - Running reads no KDA lever: every kernel and launch form is the plan's. The projection
//!   groups are the dense tier's own decision; a plan whose six-projection launch the tier
//!   declines, or whose gate kernel the weights refuse, is an error, never the unfused chain.
//! - The profile buckets are the hand-written path's ([`Glm5NextKdaLayer::spans`]).

use anyhow::ensure;
use parking_lot::RwLock;

use super::decode::RowIo;
use super::seq_tokens::SnapKernels;
use super::*;

/// 2026-10-23: The conv of a planned KDA mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConvStep<K> {
    /// 2026-10-23: One row: `causal_conv1d_update_l2norm`.
    Row(K),
    /// 2026-10-23: A verify keeping the state after each row but the last: one `advance_to` per
    /// snapshot, the token conv over the packed rows, then the live window's advance.
    TokensSnap { advance_to: K, tokens: K, window: K },
}

/// 2026-10-23: The recurrence of a planned KDA mixer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurStep<K> {
    /// 2026-10-23: One row on the register-resident kernel at T = 1.
    Row(K),
    /// 2026-10-23: The rows of one sequence, storing each snapshot as it passes.
    Snap(K),
}

/// 2026-10-23: A KDA mixer's plan at `rows` rows, with the dims it was planned at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdaSteps<K> {
    pub rows: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub pack: K,
    /// 2026-10-23: The rows entry of the W8A8 gate chain (`kda_gates_w8a8_k128_rows`).
    pub gates: K,
    pub conv: ConvStep<K>,
    pub recur: RecurStep<K>,
    /// 2026-10-23: The output norm when it is its own group (one row); `None` when it belongs
    /// to the `glm_w4a16` group with `o`, which the layer runs.
    pub out_norm: Option<K>,
}

impl<K> KdaSteps<K> {
    /// 2026-10-23: Every kernel through `f`.
    pub fn map<L>(self, mut f: impl FnMut(K) -> Result<L>) -> Result<KdaSteps<L>> {
        Ok(KdaSteps {
            rows: self.rows,
            heads: self.heads,
            head_dim: self.head_dim,
            pack: f(self.pack)?,
            gates: f(self.gates)?,
            conv: match self.conv {
                ConvStep::Row(k) => ConvStep::Row(f(k)?),
                ConvStep::TokensSnap {
                    advance_to,
                    tokens,
                    window,
                } => ConvStep::TokensSnap {
                    advance_to: f(advance_to)?,
                    tokens: f(tokens)?,
                    window: f(window)?,
                },
            },
            recur: match self.recur {
                RecurStep::Row(k) => RecurStep::Row(f(k)?),
                RecurStep::Snap(k) => RecurStep::Snap(f(k)?),
            },
            out_norm: self.out_norm.map(f).transpose()?,
        })
    }
}

impl<K> KdaSteps<K> {
    /// 2026-10-23: Whether this plan runs `k` rows that keep `snapshots` intermediate states:
    /// one row keeping none, or a verify keeping one per row but the last.
    pub fn takes(&self, k: usize, snapshots: usize) -> bool {
        k == self.rows
            && match self.conv {
                ConvStep::Row(_) => snapshots == 0,
                ConvStep::TokensSnap { .. } => snapshots + 1 == k,
            }
    }
}

impl Glm5NextKdaLayer {
    /// 2026-10-23: The q|k|v projections and the f_a|b|g_a low-rank group of `t` rows, as
    /// `(weight, output, n)`.
    pub(super) fn proj_groups(
        &self,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
    ) -> (
        Vec<(DevicePtr, DevicePtr, usize)>,
        [(DevicePtr, DevicePtr, usize); 3],
    ) {
        let (c, w) = (&self.cfg, &self.weights);
        let (qkv, hd) = (c.qkv_dim(), c.head_dim);
        let qkv_group = [&w.q_proj, &w.k_proj, &w.v_proj]
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p.weight, ws.qkv_parts.offset(i * t * qkv * 2), qkv))
            .collect();
        let low_group = [
            (w.f_a.weight, ws.lowrank, hd),
            (w.b_proj.weight, ws.beta_bf16, c.heads),
            (w.g_a.weight, ws.lowrank_g, hd),
        ];
        (qkv_group, low_group)
    }

    /// 2026-10-10: One row under `METRALE_GLM_W4A16_GEMV_SEG`: all six projections in one launch.
    /// 2026-10-11: Also 2..=4 GEMV-tier rows (`glm5next_w4a16_dense::group_of_six`). `Ok(false)`,
    /// launching nothing, where the dense tier does not group them.
    pub(super) fn six_group(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
    ) -> Result<bool> {
        let dense = &self.levers.dense;
        if !crate::glm5next_w4a16_dense::group_of_six(dense, t)? {
            return Ok(false);
        }
        let (qkv_group, low_group) = self.proj_groups(t, ws);
        let all: Vec<_> = qkv_group.iter().chain(&low_group).copied().collect();
        crate::glm5next_w4a16_dense::proj_group(
            gpu,
            dense,
            &all,
            hidden,
            t,
            self.cfg.hidden,
            stream,
        )
    }

    /// 2026-10-08: `front`, then `middle` (the stateful half), then `back`, each span in its
    /// profile bucket. Refuses `k == 0` or `k` above the workspace before any launch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn spans(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        ws: &Glm5NextKdaWorkspace,
        stream: u64,
        front: impl FnOnce() -> Result<()>,
        middle: impl FnOnce() -> Result<()>,
        back: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        if k == 0 || k > ws.max_tokens {
            bail!(
                "KDA decode of {k} rows does not fit a workspace built for {}",
                ws.max_tokens
            );
        }
        use crate::glm5next_layer::profile;
        // 2026-09-25: Three profile buckets (front, recurrence, back), each closed where its span
        // ends.
        let t_front = self.levers.profile.start();
        front()?;
        profile::end(profile::KDA_FRONT, t_front, gpu, stream);
        let t_recur = self.levers.profile.start();
        middle()?;
        profile::end(profile::KDA_RECUR, t_recur, gpu, stream);
        let t_back = self.levers.profile.start();
        let r = back();
        profile::end(profile::KDA_BACK, t_back, gpu, stream);
        r
    }

    /// 2026-10-23: `k` rows of one sequence from `plan`, as [`Self::decode_k`]; `snapshots` as
    /// there (a verify passes one per row but the last, a one-row plan none).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_k_plan(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        plan: &KdaSteps<KernelHandle>,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        ensure!(
            (k, plan.heads, plan.head_dim) == (plan.rows, c.heads, c.head_dim),
            "GLM KDA plan: compiled for {} rows of {} x {} heads, run at {k} rows of {} x {}",
            plan.rows,
            plan.heads,
            plan.head_dim,
            c.heads,
            c.head_dim
        );
        self.spans(
            gpu,
            k,
            ws,
            stream,
            || self.front_plan(gpu, hidden, k, ws, plan, stream),
            || self.stateful_plan(gpu, k, state, ws, snapshots, plan, stream),
            || match plan.out_norm {
                Some(norm) => {
                    self.o_norm_launch(gpu, norm, k, ws, stream)?;
                    self.o_proj(gpu, k, ws, stream)
                }
                None => self.back_end(gpu, k, ws, stream),
            },
        )
    }

    /// 2026-10-23: The plan's front: the dense tier's six-projection launch, the pack, the gate
    /// chain.
    fn front_plan(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        t: usize,
        ws: &Glm5NextKdaWorkspace,
        plan: &KdaSteps<KernelHandle>,
        stream: u64,
    ) -> Result<()> {
        let _stable = crate::glm5next_fp8_dense::stable_input(hidden);
        ensure!(
            self.six_group(gpu, hidden, t, ws, stream)?,
            "GLM KDA plan: the dense tier did not run the six projections of {t} rows as the one \
             launch the plan names"
        );
        self.pack_launch(gpu, plan.pack, t, ws, stream)?;
        let w = self.gate_weights();
        if let Some(why) = self.gates_refusal(plan.gates, true, w) {
            bail!("GLM KDA plan: the gate chain the plan names cannot run: {why}");
        }
        self.launch_gates(gpu, plan.gates, Some(t), w, ws, stream)
    }

    /// 2026-10-23: The plan's conv and recurrence.
    #[allow(clippy::too_many_arguments)]
    fn stateful_plan(
        &self,
        gpu: &dyn GpuBackend,
        k: usize,
        state: &KdaSeqState,
        ws: &Glm5NextKdaWorkspace,
        snapshots: &[(DevicePtr, DevicePtr)],
        plan: &KdaSteps<KernelHandle>,
        stream: u64,
    ) -> Result<()> {
        match (plan.conv, plan.recur) {
            (ConvStep::Row(conv), RecurStep::Row(recur)) => {
                ensure!(
                    k == 1 && snapshots.is_empty(),
                    "GLM KDA plan: the one-row conv runs one row without snapshots ({k} rows, {} \
                     snapshots)",
                    snapshots.len()
                );
                let io = RowIo::workspace(&self.cfg, ws, 0);
                self.conv_row_with(gpu, conv, io, state, stream)?;
                self.recurrent_reg_launch(gpu, recur, io, state, stream)
            }
            (
                ConvStep::TokensSnap {
                    advance_to,
                    tokens,
                    window,
                },
                RecurStep::Snap(recurrent_snap),
            ) => {
                let kernels = SnapKernels {
                    advance_to: Some(advance_to),
                    conv_tokens: tokens,
                    conv_window: window,
                    recurrent_snap,
                    conv_one: None,
                };
                self.stateful_tokens_snap_with(gpu, kernels, k, state, ws, snapshots, stream)
            }
            (conv, recur) => bail!("GLM KDA plan: conv {conv:?} with recurrence {recur:?}"),
        }
    }
}

/// 2026-10-23: The KDA plans one layer runs, by row count, with their executed sites.
#[derive(Default)]
pub struct InstalledKda {
    by_rows: RwLock<Vec<(KdaSteps<KernelHandle>, Vec<String>)>>,
}

impl std::fmt::Debug for InstalledKda {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InstalledKda")
            .field("sites", &self.sites().len())
            .finish()
    }
}

impl InstalledKda {
    /// 2026-10-23: Replace the installed plans (empty clears).
    pub fn set(&self, plans: Vec<(KdaSteps<KernelHandle>, Vec<String>)>) {
        *self.by_rows.write() = plans;
    }

    /// 2026-10-23: The plan at `rows` rows.
    pub fn get(&self, rows: usize) -> Option<KdaSteps<KernelHandle>> {
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
