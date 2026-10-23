// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-23: One rank's routed MoE and mHC sites on the recording device for the CPU launch
//! gates: the served levers' kernels, the rank's MLP config and workspace, synthetic layer
//! weights, a recording communicator, and the gate comparison itself (stream and event graph
//! first, then every operation). The MoE part moved here from `plan_moe_tests.rs` (exact copy).
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use metrale_config::{ActivationQuantization, Nvfp4Act, ProjFamily};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::gpu::recording::{Op, RecordingGpu, stream_graph};

use super::fixture::{mlp_config_at, served_levers, served_world};
use crate::glm5next_mlp::forward::Glm5NextMlpWorkspace;
use crate::glm5next_mlp::precision::{GroupPrecision, MlpGroup};
use crate::glm5next_mlp::weights::{
    Glm5NextDenseMlpWeights, Glm5NextExpertPtrTable, Glm5NextExpertWeights, Glm5NextMoePtrTables,
    Glm5NextMoeWeights, Nvfp4Proj, ScaleLayout, W4a4ActScales,
};
use crate::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels};

/// 2026-10-23: GB10's SM count (the sweep's persistent grid).
pub(crate) const SMS: u32 = 48;
pub(crate) const STREAM: u64 = 7;
/// 2026-10-23: The checkpoint's routed experts, all local under the served `tp` layout (the
/// site check pins it to `cfg.num_experts`).
pub(crate) const EXPERTS: u64 = 288;
pub(crate) const X: DevicePtr = DevicePtr(0x9000_0000);
pub(crate) const OUT: DevicePtr = DevicePtr(0x9100_0000);
pub(crate) const SCALES: W4a4ActScales = W4a4ActScales {
    gate_up: 0.000642,
    down: 0.000115,
};

pub(crate) struct Site {
    pub gpu: RecordingGpu,
    pub k: Glm5NextMlpKernels,
    pub cfg: Glm5NextMlpConfig,
    pub ws: Glm5NextMlpWorkspace,
}

pub(crate) fn site(rank: usize) -> Site {
    site_at(rank, served_world())
}

/// 2026-10-23: Rank `rank` of `world`'s site.
pub(crate) fn site_at(rank: usize, world: usize) -> Site {
    let gpu = RecordingGpu::new(SMS);
    let k = Glm5NextMlpKernels::resolve(&gpu, served_levers()).unwrap();
    let cfg = mlp_config_at(rank, world);
    let ws = Glm5NextMlpWorkspace::new(&gpu, &cfg, 16, &k.levers.mlp).unwrap();
    gpu.take();
    Site { gpu, k, cfg, ws }
}

/// 2026-10-23: Layer `l`'s weights at distinct synthetic addresses, every expert table carrying
/// a base table (compact scales, as the served load packs them) unless `plain`. The per-expert
/// projections are bound (`forward_moe` checks their count) but no W4A4 launch reads them.
pub(crate) fn weights(l: usize, plain: bool) -> Glm5NextMoeWeights {
    let tag = |s: u64| DevicePtr(0x4_0000_0000 + ((l as u64) << 24) + (s << 12));
    let proj = |s: u64| Nvfp4Proj {
        packed: tag(s),
        scale: tag(s + 1),
        scale_2: 0.5,
        input_scale: None,
        scale_layout: ScaleLayout::Plain,
    };
    let experts = (0..EXPERTS)
        .map(|e| Glm5NextExpertWeights {
            gate_proj: proj(64 + 6 * e),
            up_proj: proj(66 + 6 * e),
            down_proj: proj(68 + 6 * e),
        })
        .collect();
    let table = |s: u64| Glm5NextExpertPtrTable {
        packed_ptrs: tag(s),
        scale_ptrs: tag(s + 1),
        scale2_vals: tag(s + 2),
        scale_bases: if plain { DevicePtr::NULL } else { tag(s + 3) },
    };
    Glm5NextMoeWeights {
        router: tag(1),
        router_bias: tag(2),
        shared: Glm5NextDenseMlpWeights {
            gate_proj: tag(3),
            up_proj: tag(4),
            down_proj: tag(5),
        },
        experts,
        ptrs: Glm5NextMoePtrTables {
            gate: table(16),
            up: table(32),
            down: table(48),
        },
        precision: GroupPrecision::resolve(
            MlpGroup::RoutedExperts,
            ActivationQuantization::default()
                .ladder(ProjFamily::Moe)
                .clone(),
            Nvfp4Act::A4,
            16,
            true,
        )
        .unwrap(),
        act_scales: Some(SCALES),
    }
}

/// 2026-10-23: `None` when the two logs are equal, else what differs first: the stream and event
/// graph (which stream each operation is ordered on, every fork, join and collective), then the
/// first differing operation of each.
pub(crate) fn gate(a: &[Op], b: &[Op]) -> Option<String> {
    let (ga, gb) = (stream_graph(a), stream_graph(b));
    if ga != gb {
        let i = ga.iter().zip(&gb).take_while(|(x, y)| x == y).count();
        return Some(format!(
            "stream graph differs at {i} of {} / {}:\n  legacy {:?}\n  plan   {:?}",
            ga.len(),
            gb.len(),
            ga.get(i),
            gb.get(i)
        ));
    }
    if a == b {
        return None;
    }
    let i = a.iter().zip(b).take_while(|(x, y)| x == y).count();
    Some(format!(
        "op {i} of {} / {}:\n  legacy {:?}\n  plan   {:?}",
        a.len(),
        b.len(),
        a.get(i),
        b.get(i)
    ))
}

/// 2026-10-23: A communicator that logs its calls into the recording device's log, in order
/// with the launches. The one-shot exchange answers with peer slots at `recv` when `oneshot`
/// (the served payloads at decode rows), else declines as a backend without the path does.
pub(crate) struct RecordingComm<'a> {
    pub gpu: &'a RecordingGpu,
    pub rank: usize,
    pub world: usize,
    pub oneshot: bool,
}

/// 2026-10-23: Where the recording communicator says the peers' partials landed, and its slot.
pub(crate) const PEER_RECV: u64 = 0x7700_0000;
pub(crate) const PEER_SLOT_ELEMS: usize = 16 * 4096;

impl metrale_comm::CommBackend for RecordingComm<'_> {
    fn all_reduce(&self, ptr: u64, bytes: usize) -> anyhow::Result<()> {
        self.gpu
            .record_comm(format!("all_reduce({ptr:#x}, {bytes})"), None);
        Ok(())
    }
    fn all_reduce_async(&self, ptr: u64, bytes: usize, stream: u64) -> anyhow::Result<()> {
        self.gpu
            .record_comm(format!("all_reduce({ptr:#x}, {bytes})"), Some(stream));
        Ok(())
    }
    fn exchange_partials(
        &self,
        ptr: u64,
        bytes: usize,
        stream: Option<u64>,
    ) -> anyhow::Result<Option<metrale_comm::PeerPartials>> {
        if !self.oneshot {
            return Ok(None);
        }
        self.gpu
            .record_comm(format!("exchange({ptr:#x}, {bytes})"), stream);
        Ok(Some(metrale_comm::PeerPartials {
            recv: PEER_RECV,
            slot_elems: PEER_SLOT_ELEMS,
            world_size: self.world,
            rank: self.rank,
        }))
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not broadcast")
    }
    fn barrier(&self) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> anyhow::Result<()> {
        anyhow::bail!("the mHC site does not receive")
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        self.world
    }
}
